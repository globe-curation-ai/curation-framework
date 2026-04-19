import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")
import pandas as pd
import numpy as np
import pymc as pm
from scipy.stats import truncnorm


class BayesianTubeModel:
    """
    Bayesian hierarchical model for Transparency Tube measurements.
    Applies Truncated-Normal distributions (bounded by 0 and max tube length)
    and continuous linear temporal weighting.
    """

    def __init__(self, config: dict):
        self.config = config

        bayesian_params = config.get('bayesian_model', config)
        val_params = config.get('validation', {})

        self.target_col = bayesian_params.get('tube_target_col', 'tube_image_disappearance_cm')
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
        self.df_full['is_statistical_outlier'] = False

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

        epsilon = 0.1
        self.df_valid.loc[self.df_valid[val_col] <= 0, val_col] = epsilon
        self.df_valid.loc[self.df_valid[val_col] >= self.max_tube_cm, val_col] = self.max_tube_cm - epsilon

        self.df_valid[date_col] = pd.to_datetime(self.df_valid[date_col])

        now = pd.Timestamp.now().normalize()
        safe_dates = self.df_valid[self.df_valid[date_col] <= now][date_col]
        current_time = safe_dates.max() if not safe_dates.empty else now

        raw_age_years = (current_time - self.df_valid[date_col]) / pd.Timedelta(days=365.2425)
        self.df_valid['age_years'] = np.maximum(0.0, raw_age_years)

        self.df_valid['weight'] = np.maximum(0.0, 1.0 - (self.df_valid['age_years'] / window_years))
        self.df_valid['weight'] = self.df_valid['weight'].round(5)

        self.df_valid = self.df_valid[self.df_valid['weight'] > 0.0].copy()

    def build_model(self):
        """
        Constructs the PyMC hierarchical model using Truncated Normal likelihoods.
        Utilizes Non-Centered Parameterization.
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

        with pm.Model(coords=coords) as self.model:
            site_idx_data = pm.Data("site_idx", self.site_idx)
            water_source_idx_data = pm.Data("water_source_idx", self.site_water_source_idx)
            obs_data = pm.Data("obs", self.df_valid[self.target_col].values)
            weights = pm.Data("weights", self.df_valid['weight'].values)

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
                water_source_mu[water_source_idx_data] + (site_offset * water_source_sigma[water_source_idx_data]),
                dims="site"
            )

            obs_sigma = pm.HalfNormal("obs_sigma", sigma=10.0, initval=5.0)

            dist = pm.TruncatedNormal.dist(
                mu=site_mu[site_idx_data],
                sigma=obs_sigma,
                lower=0.0,
                upper=self.max_tube_cm
            )
            logp = pm.logp(dist, obs_data)
            pm.Deterministic("log_lik", weights * logp)
            pm.Potential("weighted_logp", pm.math.sum(weights * logp))

    def sample(self):
        """Executes the MCMC sampler and generates posterior predictive checks."""
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
            # Instructs the sampler to simulate data based on the trained posteriors
            pm.sample_posterior_predictive(self.trace, extend_inferencedata=True)
            self.trace.add_groups({"log_likelihood": {"obs": self.trace.posterior["log_lik"]}})

    def _flag_outliers(self):
        """
        Generates simulated observations from the posterior and flags true
        observations that fall outside the 95% predictive interval.
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
        b = (self.max_tube_cm - mu_obs) / sigma_samples[:, None]
        simulated_obs = truncnorm.rvs(a, b, loc=mu_obs, scale=sigma_samples[:, None])

        lower_bound, upper_bound = np.percentile(simulated_obs, [2.5, 97.5], axis=0)

        outlier_mask = (self.df_valid[self.target_col] < lower_bound) | (self.df_valid[self.target_col] > upper_bound)
        self.df_valid['is_statistical_outlier'] = outlier_mask