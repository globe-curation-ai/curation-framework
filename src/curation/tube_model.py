import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import truncnorm


class BayesianTubeModel:
    """
    Bayesian hierarchical model for Transparency Tube measurements.
    Applies Truncated-Normal distributions (bounded by 0 and max tube length)
    and continuous linear temporal weighting.

    Right-censored observations (water perfectly clear to tube bottom) are
    included via a censored likelihood that contributes log P(X >= observed)
    instead of the standard log P(X = observed).
    """

    def __init__(self, config: dict):
        self.config = config

        bayesian_params = config.get('bayesian_model', config)
        val_params = config.get('validation', {})

        self.target_col = bayesian_params.get('tube_target_col', 'tube_image_disappearance_cm')
        self.active_tube_length_col = bayesian_params.get('active_tube_length_col', 'tube_len_mle_site')
        self.date_col = config.get('date_col', 'measured_on')
        self.site_col = config.get('site_col', 'site_id')
        self.max_tube_cm = val_params.get('max_tube_cm', 130.0)

        self.water_source_col = config.get('water_type_col', 'water_body_source')

        self.draws = bayesian_params.get('draws', 4000)
        self.tune = bayesian_params.get('tune', 1000)
        self.chains = bayesian_params.get('chains', 4)
        self.target_accept = bayesian_params.get('target_accept', 0.95)

        self.df_full = None
        self.df_valid = None
        self.model = None
        self.trace = None

    def evaluate(self, df: pd.DataFrame, window_years: int = 10) -> pd.DataFrame:
        """
        Executes the full Bayesian evaluation pipeline.
        """
        self.df_full = df.copy()

        if 'passed_heuristics' not in self.df_full.columns:
            self.df_full['passed_heuristics'] = True

        self.df_valid = self.df_full[self.df_full['passed_heuristics']].copy()

        # Ternary outlier status: True (outlier), False (not outlier), None (censored, not evaluated)
        self.df_full['is_statistical_outlier'] = None

        if self.df_valid.empty:
            print("No records passed heuristic gates. Bypassing Bayesian evaluation.")
            return self.df_full

        self._prepare_data(window_years)

        if self.df_valid.empty:
            return self.df_full

        self.build_model()
        self.sample()
        self._flag_outliers()

        self.df_full.update(self.df_valid[['is_statistical_outlier']])
        return self.df_full

    def _prepare_data(self, window_years: int):
        """
        Prepares data for modeling, including temporal weighting.
        """
        val_col = self.target_col
        date_col = self.date_col

        self.df_valid[val_col] = pd.to_numeric(self.df_valid[val_col], errors='coerce')
        self.df_valid = self.df_valid.dropna(subset=[val_col]).copy()

        # Ensure is_censored column exists
        if 'is_censored' not in self.df_valid.columns:
            self.df_valid['is_censored'] = False

        # Prepare the dynamic max tube length
        if self.active_tube_length_col not in self.df_valid.columns:
            self.df_valid[self.active_tube_length_col] = self.max_tube_cm
            
        self.df_valid[self.active_tube_length_col] = pd.to_numeric(self.df_valid[self.active_tube_length_col], errors='coerce')
        # Fall back to max_tube_cm if estimate is missing or 0.0
        self.df_valid.loc[self.df_valid[self.active_tube_length_col].isna() | (self.df_valid[self.active_tube_length_col] <= 0), self.active_tube_length_col] = self.max_tube_cm

        epsilon = 1.0

        # For UNCENSORED observations: clamp to valid range for TruncatedNormal
        uncensored_mask = ~self.df_valid['is_censored']
        self.df_valid.loc[uncensored_mask & (self.df_valid[val_col] <= 0), val_col] = epsilon
        self.df_valid['dynamic_limit'] = self.df_valid[self.active_tube_length_col]
        self.df_valid.loc[uncensored_mask & (self.df_valid[val_col] >= self.df_valid['dynamic_limit']), val_col] = self.df_valid['dynamic_limit'] - epsilon

        # For CENSORED observations: the value IS the detection limit (lower bound).
        # Ensure it is strictly positive
        censored_mask = self.df_valid['is_censored']
        self.df_valid.loc[censored_mask & (self.df_valid[val_col] <= 0), val_col] = epsilon

        self.df_valid[date_col] = pd.to_datetime(self.df_valid[date_col])

        now = pd.Timestamp.now().normalize()
        safe_dates = self.df_valid[self.df_valid[date_col] <= now][date_col]
        current_time = safe_dates.max() if not safe_dates.empty else now

        raw_age_years = (current_time - self.df_valid[date_col]) / pd.Timedelta(days=365.2425)
        self.df_valid['age_years'] = np.maximum(0.0, raw_age_years)

        self.df_valid['weight'] = np.maximum(0.0, 1.0 - (self.df_valid['age_years'] / window_years))

        # Filter out zero weights
        self.df_valid = self.df_valid[self.df_valid['weight'] > 0.0].copy()
        self.df_valid['weight'] = self.df_valid['weight'].round(5)

    def build_model(self):
        """
        Constructs the PyMC hierarchical model with a split likelihood:
        - Uncensored observations: standard log-density from TruncatedNormal
        - Censored observations: log-survival function (log P(X >= observed))
        """
        self.site_idx, self.sites = pd.factorize(self.df_valid[self.site_col])

        site_mapping = self.df_valid.drop_duplicates(subset=[self.site_col]).set_index(self.site_col)
        site_water_sources = site_mapping.loc[self.sites, self.water_source_col].fillna('unknown')
        self.site_water_source_idx, self.water_sources = pd.factorize(site_water_sources)

        coords = {
            "site": self.sites,
            "water_source": self.water_sources,
            "obs_id": np.arange(len(self.df_valid))
        }

        init_mu = self.df_valid[self.target_col].mean()

        # Prepare censoring indicator (1.0 = censored, 0.0 = uncensored)
        censored_indicator = self.df_valid['is_censored'].astype(float).values

        with pm.Model(coords=coords) as self.model:
            site_idx_data = pm.Data("site_idx", self.site_idx)
            water_source_idx_data = pm.Data("water_source_idx", self.site_water_source_idx)
            obs_data = pm.Data("obs", self.df_valid[self.target_col].values)
            weights = pm.Data("weights", self.df_valid['weight'].values)
            upper_bounds = pm.Data("upper_bounds", self.df_valid['dynamic_limit'].values)
            is_censored_data = pm.Data("is_censored", censored_indicator)

            global_mu = pm.Normal("global_mu", mu=self.max_tube_cm / 2, sigma=self.max_tube_cm / 4, initval=init_mu)
            global_sigma = pm.HalfNormal("global_sigma", sigma=20.0, initval=10.0)

            water_source_offset = pm.Normal("water_source_offset", mu=0.0, sigma=1.0, dims="water_source")

            water_source_sigma = pm.HalfNormal(
                "water_source_sigma",
                sigma=10.0,
                dims="water_source",
                initval=np.full(len(self.water_sources), 5.0)
            )

            water_source_mu = pm.Deterministic(
                "water_source_mu",
                global_mu + (water_source_offset * global_sigma),
                dims="water_source"
            )

            site_offset = pm.Normal("site_offset", mu=0.0, sigma=1.0, dims="site")

            site_mu = pm.Deterministic(
                "site_mu",
                water_source_mu[water_source_idx_data] + 
                (site_offset * water_source_sigma[water_source_idx_data]),
                dims="site"
            )

            obs_sigma = pm.HalfNormal("obs_sigma", sigma=10.0, initval=5.0)

            # Build the truncated distribution for uncensored observations
            dist = pm.TruncatedNormal.dist(
                mu=site_mu[site_idx_data],
                sigma=obs_sigma,
                lower=0.0,
                upper=upper_bounds
            )

            # Uncensored contribution: standard log-density
            logp_val = pm.logp(dist, obs_data)

            # Censored contribution: log P(true_clarity >= tube_length)
            # IMPORTANT: We use the UNDERLYING Normal distribution (not TruncatedNormal)
            # for the survival function. The TruncatedNormal has CDF = 1.0 at its upper
            # bound by definition, which would give log(0) = -inf. The Normal survival
            # function correctly represents the probability that true clarity extends
            # beyond the tube's physical measurement limit.
            #
            # Uses log1mexp for numerical stability when CDF ≈ 1.0 (avoids
            # gradient cliffs that cause NUTS divergences).
            # Clipped to -30 (≈ P = 1e-13) as a hard floor.
            latent_dist = pm.Normal.dist(mu=site_mu[site_idx_data], sigma=obs_sigma)
            logcdf_val = pm.logcdf(latent_dist, obs_data)
            log_sf_val = pt.maximum(pt.log1mexp(logcdf_val), -30.0)

            # Select the appropriate likelihood per observation
            log_lik = pt.switch(pt.eq(is_censored_data, 1.0), log_sf_val, logp_val)

            pm.Deterministic("log_lik", weights * log_lik)
            pm.Potential("weighted_logp", pm.math.sum(weights * log_lik))

    def sample(self):
        """Executes the MCMC sampler and generates posterior predictive samples."""
        with self.model:
            self.trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                target_accept=self.target_accept,
                progressbar=True,
                return_inferencedata=True,
                nuts_sampler_kwargs={"max_treedepth": 15}
            )
            self.trace.add_groups({"log_likelihood": {"obs": self.trace.posterior["log_lik"]}})

    def _flag_outliers(self):
        """
        Generates simulated observations from the posterior and flags true
        observations that fall outside the 95% predictive interval.

        Censored observations are marked as None (not evaluated) since their
        true value is unknown.
        """
        post = self.trace.posterior

        mu_flat = post["site_mu"].values.reshape(-1, post["site_mu"].shape[-1])
        sigma_flat = post["obs_sigma"].values.flatten()

        total_draws = mu_flat.shape[0]
        sample_size = min(500, total_draws)
        sample_idx = np.random.choice(total_draws, size=sample_size, replace=False)

        mu_samples = mu_flat[sample_idx, :]
        sigma_samples = sigma_flat[sample_idx]

        mu_obs = mu_samples[:, self.site_idx]

        a = (0.0 - mu_obs) / sigma_samples[:, None]
        dynamic_limits = self.df_valid['dynamic_limit'].values
        b = (dynamic_limits - mu_obs) / sigma_samples[:, None]
        simulated_obs = truncnorm.rvs(a, b, loc=mu_obs, scale=sigma_samples[:, None])

        lower_bound, upper_bound = np.percentile(simulated_obs, [2.5, 97.5], axis=0)

        # Only evaluate outlier status for uncensored observations
        is_censored = self.df_valid['is_censored'].values
        outlier_mask = (self.df_valid[self.target_col] < lower_bound) | (self.df_valid[self.target_col] > upper_bound)

        # Ternary assignment: True/False for uncensored, None for censored
        self.df_valid['is_statistical_outlier'] = None  # default: not evaluated
        self.df_valid.loc[~is_censored, 'is_statistical_outlier'] = outlier_mask[~is_censored]