import numpy as np
import pandas as pd

class BaseTubeLengthEstimator:
    def __init__(self, common_lengths=None):
        if common_lengths is None:
            common_lengths = [27.0, 45.0, 60.0, 80.0, 100.0, 120.0]
        self.common_lengths = sorted(common_lengths)
    
    def estimate(self, values: list[float]) -> tuple[list[float], list]:
        """
        Given a list of chronological measurements, returns a tuple of:
          - estimated tube lengths (list[float])
          - estimation metadata (list): reason strings or posterior PMFs,
            depending on the estimator
        """
        raise NotImplementedError

class AlgorithmicEstimator(BaseTubeLengthEstimator):
    def estimate(self, values: list[float]) -> tuple[list[float], list]:
        estimated_lengths = []
        reasons = []
        num_times_taken_max = 0
        
        valid_counts = {length: 0 for length in self.common_lengths}
        
        def get_mode():
            max_count = -1
            mode = None
            for length in self.common_lengths:
                if valid_counts[length] > max_count and valid_counts[length] > 0:
                    max_count = valid_counts[length]
                    mode = length
            return mode
            
        for sample in values:
            if sample in self.common_lengths:
                estimated_lengths.append(sample)
                reasons.append('Exact length')
                valid_counts[sample] += 1
            elif pd.isna(sample) or sample <= 0.0:
                estimated_lengths.append(0.0)
                reasons.append('Measurement is zero or missing')
            else:
                longer_count = sum(1 for length in self.common_lengths if length >= sample)
                if longer_count == 1:
                    estimated_lengths.append(self.common_lengths[-1])
                    reasons.append('Longest common length')
                    valid_counts[self.common_lengths[-1]] += 1
                elif longer_count == 0:
                    estimated_lengths.append(0.0)
                    reasons.append('Too long')
                else:
                    mode_length = get_mode()
                    
                    if mode_length is None:
                        estimated_lengths.append(self.common_lengths[-1])
                        reasons.append('First sample, assume max')
                        num_times_taken_max += 1
                        valid_counts[self.common_lengths[-1]] += 1

                    elif sample > mode_length:
                        for length in self.common_lengths:
                            if sample <= length:
                                estimated_lengths.append(length)
                                reasons.append('Long enough tube length')
                                valid_counts[length] += 1
                                break
                    else:
                        estimated_lengths.append(mode_length)
                        reasons.append('Is Mode')
                        valid_counts[mode_length] += 1
                        
        return estimated_lengths, reasons

class MLEEstimator(BaseTubeLengthEstimator):
    def __init__(self, common_lengths=None, only_update_exact_matches=False):
        super().__init__(common_lengths)
        self.only_update_exact_matches = only_update_exact_matches

    def estimate(self, values: list[float]) -> tuple[list[float], list]:
        alpha = [1] * len(self.common_lengths)
        pmfs = []
        estimated_lengths = []
        
        for sample in values:
            if pd.isna(sample) or sample <= 0.0 or sample > self.common_lengths[-1]:
                estimated_lengths.append(0.0)
                pmfs.append([])
            else:
                n = sum(alpha)
                # MAP estimate of categorical probabilities from Dirichlet prior
                pmf = [(a - 1) / (n - 1) if n > 1 else 1/len(self.common_lengths) for a in alpha]
                pmfs.append(pmf)
                
                longer_pmf = [p if l >= sample else 0.0 for p, l in zip(pmf, self.common_lengths)]
                # Reverse to handle argmax tie-breaking properly (selects shortest valid tube on ties)
                reversed_pmf = list(reversed(longer_pmf))
                mle_ind = len(self.common_lengths) - np.argmax(reversed_pmf) - 1
                estimated_length = self.common_lengths[mle_ind]
                
                if self.only_update_exact_matches and sample in self.common_lengths:
                    update_ind = self.common_lengths.index(sample)
                    alpha[update_ind] += 1
                elif not self.only_update_exact_matches:
                    update_ind = self.common_lengths.index(estimated_length)
                    alpha[update_ind] += 1
                    
                estimated_lengths.append(estimated_length)
                
        return estimated_lengths, pmfs

def apply_all_tube_estimations(df: pd.DataFrame, target_col: str = 'tube_image_disappearance_cm', site_col: str = 'site_id', country_col: str = 'countryCode', date_col: str = 'measured_on') -> pd.DataFrame:
    df_result = df.copy()
    
    # Needs chronological ordering to apply algorithms statefully
    if date_col in df_result.columns:
        df_result = df_result.sort_values(by=date_col)
    
    algo_est = AlgorithmicEstimator()
    mle_est = MLEEstimator(only_update_exact_matches=True)
    
    new_cols = {
        'tube_len_algo_site': np.nan,
        'tube_len_algo_country': np.nan,
        'tube_len_mle_site': np.nan,
        'tube_len_mle_country': np.nan
    }
    
    for col in new_cols:
        if col not in df_result.columns:
            df_result[col] = np.nan
            
    # Apply Grouping by Site
    if site_col in df_result.columns:
        for site, group in df_result.groupby(site_col):
            vals = group[target_col].tolist()
            
            algo_lengths, _ = algo_est.estimate(vals)
            df_result.loc[group.index, 'tube_len_algo_site'] = algo_lengths
            
            mle_lengths, _ = mle_est.estimate(vals)
            df_result.loc[group.index, 'tube_len_mle_site'] = mle_lengths
            
    # Apply Grouping by Country
    if country_col in df_result.columns:
        for country, group in df_result.groupby(country_col):
            vals = group[target_col].tolist()
            
            algo_lengths, _ = algo_est.estimate(vals)
            df_result.loc[group.index, 'tube_len_algo_country'] = algo_lengths
            
            mle_lengths, _ = mle_est.estimate(vals)
            df_result.loc[group.index, 'tube_len_mle_country'] = mle_lengths
            
    return df_result
