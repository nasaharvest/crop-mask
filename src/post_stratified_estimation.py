import numpy as np
import pandas as pd


class StehmanStratifiedEstimators:
    """Class to calculate accuracy metrics for map evaluation.
    https://www.tandfonline.com/doi/full/10.1080/01431161.2014.930207
    """

    def __init__(
        self,
        reference_class: np.ndarray,
        map_class: np.ndarray,
        stratum: np.ndarray,
        stratum_size: np.ndarray,
    ):
        """
        Initialize the calculator with data arrays.

        Args:
            reference_class: Reference classification data
            map_class: Map classification data
            stratum: Stratum assignments for each pixel
            stratum_size: Size of each stratum
        """
        self.reference_class = reference_class
        self.map_class = map_class
        self.stratum = stratum
        self.stratum_size = stratum_size
        self.sample_size = pd.Series(stratum).value_counts().values

    def get_indicators(self, target_class: int = 1) -> pd.DataFrame:
        """
        Calculate indicator variables for accuracy assessment.

        Args:
            target_class: Target class to evaluate (default: 1 for crop)

        Returns:
            DataFrame with indicator variables for each pixel
        """
        # Overall accuracy indicator
        oa_yu = np.where(self.reference_class == self.map_class, 1, 0)

        # Area proportion indicator
        area_yu = np.where(self.reference_class == target_class, 1, 0)

        # User's accuracy indicators
        ua_yu = np.where(
            (self.reference_class == self.map_class) & (self.reference_class == target_class), 1, 0
        )
        ua_xu = np.where(self.map_class == target_class, 1, 0)

        # Producer's accuracy indicators
        pa_yu = np.where(
            (self.reference_class == self.map_class) & (self.reference_class == target_class), 1, 0
        )
        pa_xu = np.where(self.reference_class == target_class, 1, 0)

        return pd.DataFrame(
            {
                "stratum": self.stratum,
                "oa_yu": oa_yu,
                "area_prop": area_yu,
                "ua_yu": ua_yu,
                "ua_xu": ua_xu,
                "pa_yu": pa_yu,
                "pa_xu": pa_xu,
            }
        )

    def _get_summary_stats(self, data: pd.DataFrame) -> tuple:
        """
        Calculate summary statistics for each stratum.

        Args:
            data: DataFrame with indicator variables

        Returns:
            Tuple of (mean, variance, UA covariance, PA covariance)
        """
        mean = data.groupby("stratum").mean()
        var = data.groupby("stratum").var()

        # Calculate covariances
        cov = data.groupby("stratum").cov()
        cov_ua = cov.loc[(slice(None), "ua_yu"), "ua_xu"].values
        cov_pa = cov.loc[(slice(None), "pa_yu"), "pa_xu"].values

        return mean, var, cov_ua, cov_pa

    def _calculate_ratio_accuracy(self, mean: pd.DataFrame, y_col: str, x_col: str) -> float:
        """
        Calculate ratio-based accuracy (user's or producer's accuracy).

        Args:
            mean: Mean values by stratum
            y_col: Numerator column name
            x_col: Denominator column name

        Returns:
            Calculated accuracy ratio
        """
        y_hat = mean[y_col] * self.stratum_size
        x_hat = mean[x_col] * self.stratum_size
        return y_hat.sum() / x_hat.sum()

    def calculate_user_accuracy(self, mean: pd.DataFrame) -> float:
        """Calculate user's accuracy (precision)."""
        return self._calculate_ratio_accuracy(mean, "ua_yu", "ua_xu")

    def calculate_producer_accuracy(self, mean: pd.DataFrame) -> float:
        """Calculate producer's accuracy (recall)."""
        return self._calculate_ratio_accuracy(mean, "pa_yu", "pa_xu")

    def calculate_overall_accuracy(self, mean: pd.DataFrame) -> float:
        """Calculate overall accuracy."""
        y_hat = mean["oa_yu"] * self.stratum_size
        return y_hat.sum() / self.stratum_size.sum()

    def calculate_area_proportion(self, mean: pd.DataFrame) -> float:
        """Calculate area proportion."""
        y_hat = mean["area_prop"] * self.stratum_size
        return y_hat.sum() / self.stratum_size.sum()

    def _calculate_standard_error(
        self,
        var: pd.DataFrame,
        mean: pd.DataFrame,
        cov: np.ndarray,
        y_col: str,
        x_col: str,
        accuracy_value: float = None,
    ) -> float:
        """
        Calculate standard error for accuracy metrics.

        Args:
            var: Variance values by stratum
            mean: Mean values by stratum
            cov: Covariance values
            y_col: Numerator column name
            x_col: Denominator column name
            accuracy_value: Pre-calculated accuracy value (for ratio metrics)

        Returns:
            Standard error value
        """
        if accuracy_value is None:
            # For non-ratio metrics (overall accuracy, area proportion)
            lhs = 1 / self.stratum_size.sum() ** 2
            rhs = (var[y_col] / self.sample_size * self.stratum_size**2).sum()
            return np.sqrt(rhs * lhs)
        else:
            # For ratio metrics (user's/producer's accuracy)
            x_hat = sum(mean[x_col] * self.stratum_size)
            lhs = 1 / x_hat**2
            rhs = sum(
                self.stratum_size**2
                * (var[y_col] + accuracy_value**2 * var[x_col] - 2 * accuracy_value * cov)
                / self.sample_size
            )
            return np.sqrt(rhs * lhs)

    def calculate_standard_errors(
        self,
        var: pd.DataFrame,
        mean: pd.DataFrame,
        cov_ua: np.ndarray,
        cov_pa: np.ndarray,
        ua: float,
        pa: float,
    ) -> dict:
        """
        Calculate all standard errors.

        Args:
            var: Variance values by stratum
            mean: Mean values by stratum
            cov_ua: UA covariance values
            cov_pa: PA covariance values
            ua: User's accuracy value
            pa: Producer's accuracy value

        Returns:
            Dictionary of standard errors
        """
        return {
            "ua_se": self._calculate_standard_error(var, mean, cov_ua, "ua_yu", "ua_xu", ua),
            "pa_se": self._calculate_standard_error(var, mean, cov_pa, "pa_yu", "pa_xu", pa),
            "oa_se": self._calculate_standard_error(var, mean, None, "oa_yu", None),
            "area_se": self._calculate_standard_error(var, mean, None, "area_prop", None),
        }

    def compute_f1_standard_error(
        self, recall: float, precision: float, std_recall: float, std_precision: float
    ) -> float:
        """
        Calculate F1 score standard error using error propagation.

        Args:
            recall: Recall value (producer's accuracy)
            precision: Precision value (user's accuracy)
            std_recall: Standard deviation of recall
            std_precision: Standard deviation of precision

        Returns:
            Standard deviation of F1 score
        """
        term1 = 2 * (recall * std_precision + precision * std_recall) / (precision + recall)
        term2 = (2 * precision * recall * (std_precision + std_recall)) / (
            (precision + recall) ** 2
        )
        return term1 + term2

    def generate_report(self, dataset_name: str, country: str) -> pd.DataFrame:
        """
        Generate comprehensive accuracy report.

        Args:
            dataset_name: Name of the dataset
            country: Country being evaluated

        Returns:
            DataFrame with all accuracy metrics and standard errors
        """
        # Calculate crop class metrics
        crop_indicators = self.get_indicators(target_class=1)
        crop_mean, crop_var, crop_cov_ua, crop_cov_pa = self._get_summary_stats(crop_indicators)

        ua = self.calculate_user_accuracy(crop_mean)
        pa = self.calculate_producer_accuracy(crop_mean)
        oa = self.calculate_overall_accuracy(crop_mean)
        area_prop = self.calculate_area_proportion(crop_mean)

        # Calculate standard errors
        se_dict = self.calculate_standard_errors(
            crop_var, crop_mean, crop_cov_ua, crop_cov_pa, ua, pa
        )

        # Calculate F1 score and its standard error
        f1_score = 2 * (ua * pa) / (ua + pa)
        f1_se = self.compute_f1_standard_error(pa, ua, se_dict["pa_se"], se_dict["ua_se"])

        # Calculate non-crop class metrics
        nc_indicators = self.get_indicators(target_class=0)
        nc_mean, nc_var, nc_cov_ua, nc_cov_pa = self._get_summary_stats(nc_indicators)
        return pd.DataFrame(
            {
                "dataset": dataset_name,
                "country": country,
                "crop_ua": ua,
                "crop_ua_se": se_dict["ua_se"],
                "crop_pa": pa,
                "crop_pa_se": se_dict["pa_se"],
                "oa": oa,
                "oa_se": se_dict["oa_se"],
                "crop_f1_score": f1_score,
                "crop_f1_score_se": f1_se,
                "non_crop_ua": self.calculate_user_accuracy(nc_mean),
                "non_crop_pa": self.calculate_producer_accuracy(nc_mean),
                "non_crop_ua_se": self._calculate_standard_error(
                    nc_var, nc_mean, nc_cov_ua, "ua_yu", "ua_xu"
                ),
                "non_crop_pa_se": self._calculate_standard_error(
                    nc_var, nc_mean, nc_cov_pa, "pa_yu", "pa_xu"
                ),
                "crop_area_proportion": area_prop,
                "crop_area_proportion_se": se_dict["area_se"],
            },
            index=[0],
        )
