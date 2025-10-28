"""
Statistical Distribution Validation

Validates that generated data follows specified statistical distributions
using hypothesis tests (KS, Chi-Squared, Anderson-Darling).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class DistributionType(Enum):
    """Types of probability distributions."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    ZIPF = "zipf"
    DISCRETE = "discrete"
    EXPONENTIAL = "exponential"
    POISSON = "poisson"


class TestType(Enum):
    """Statistical hypothesis tests."""
    KOLMOGOROV_SMIRNOV = "ks_test"
    CHI_SQUARED = "chi_squared"
    ANDERSON_DARLING = "anderson_darling"


@dataclass
class DistributionSpec:
    """
    Specification for a probability distribution.
    
    Attributes:
        name: Human-readable name
        dist_type: Type of distribution
        parameters: Distribution parameters
        bounds: Optional (min, max) bounds for clipping
    """
    name: str
    dist_type: DistributionType
    parameters: Dict[str, Any]
    bounds: Optional[Tuple[float, float]] = None


@dataclass
class StatisticalTestResult:
    """
    Result of a statistical hypothesis test.
    
    Attributes:
        test_type: Type of test performed
        statistic: Test statistic value
        p_value: P-value from test
        passed: Whether test passed (p_value >= alpha)
        alpha: Significance level used
        sample_size: Size of sample tested
        message: Human-readable result message
    """
    test_type: TestType
    statistic: float
    p_value: float
    passed: bool
    alpha: float
    sample_size: int
    message: str


@dataclass
class DistributionReport:
    """
    Comprehensive distribution validation report.
    
    Attributes:
        distribution_name: Name of distribution tested
        distribution_spec: Distribution specification
        test_results: List of test results
        sample_statistics: Descriptive statistics of sample
        passed: Whether all tests passed
    """
    distribution_name: str
    distribution_spec: DistributionSpec
    test_results: List[StatisticalTestResult]
    sample_statistics: Dict[str, float]
    passed: bool


class StatisticalValidator:
    """
    Statistical distribution validator.
    
    Validates that generated data follows expected probability
    distributions using rigorous hypothesis testing.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical validator.
        
        Args:
            alpha: Significance level for hypothesis tests (default 0.05)
        """
        self.alpha = alpha
        self.distribution_specs: Dict[str, DistributionSpec] = {}
        logger.info(f"StatisticalValidator initialized (alpha={alpha})")
    
    def register_distribution(
        self,
        name: str,
        dist_type: DistributionType,
        parameters: Dict[str, Any],
        bounds: Optional[Tuple[float, float]] = None
    ) -> DistributionSpec:
        """
        Register a distribution specification.
        
        Args:
            name: Distribution name
            dist_type: Type of distribution
            parameters: Distribution parameters
            bounds: Optional bounds for clipping
            
        Returns:
            DistributionSpec object
            
        Example:
            >>> validator = StatisticalValidator()
            >>> spec = validator.register_distribution(
            ...     "student_workload",
            ...     DistributionType.NORMAL,
            ...     {"mean": 18, "std": 3},
            ...     bounds=(12, 27)
            ... )
        """
        spec = DistributionSpec(
            name=name,
            dist_type=dist_type,
            parameters=parameters,
            bounds=bounds
        )
        self.distribution_specs[name] = spec
        logger.debug(f"Registered distribution: {name} ({dist_type.value})")
        return spec
    
    def kolmogorov_smirnov_test(
        self,
        data: np.ndarray,
        dist_spec: DistributionSpec
    ) -> StatisticalTestResult:
        """
        Perform Kolmogorov-Smirnov test for continuous distributions.
        
        Args:
            data: Sample data
            dist_spec: Distribution specification
            
        Returns:
            StatisticalTestResult
        """
        try:
            # Get appropriate scipy distribution
            if dist_spec.dist_type == DistributionType.NORMAL:
                mean = dist_spec.parameters.get("mean", 0)
                std = dist_spec.parameters.get("std", 1)
                statistic, p_value = stats.kstest(
                    data, 
                    lambda x: stats.norm.cdf(x, loc=mean, scale=std)
                )
            elif dist_spec.dist_type == DistributionType.UNIFORM:
                low = dist_spec.parameters.get("low", 0)
                high = dist_spec.parameters.get("high", 1)
                statistic, p_value = stats.kstest(
                    data,
                    lambda x: stats.uniform.cdf(x, loc=low, scale=high-low)
                )
            elif dist_spec.dist_type == DistributionType.EXPONENTIAL:
                scale = dist_spec.parameters.get("scale", 1)
                statistic, p_value = stats.kstest(
                    data,
                    lambda x: stats.expon.cdf(x, scale=scale)
                )
            else:
                raise ValueError(f"KS test not applicable for {dist_spec.dist_type.value}")
            
            passed = p_value >= self.alpha
            message = (
                f"KS test: statistic={statistic:.4f}, p-value={p_value:.4f} "
                f"({'PASS' if passed else 'FAIL'} at α={self.alpha})"
            )
            
            return StatisticalTestResult(
                test_type=TestType.KOLMOGOROV_SMIRNOV,
                statistic=statistic,
                p_value=p_value,
                passed=passed,
                alpha=self.alpha,
                sample_size=len(data),
                message=message
            )
            
        except Exception as e:
            logger.error(f"KS test failed: {e}")
            return StatisticalTestResult(
                test_type=TestType.KOLMOGOROV_SMIRNOV,
                statistic=0.0,
                p_value=0.0,
                passed=False,
                alpha=self.alpha,
                sample_size=len(data),
                message=f"KS test error: {str(e)}"
            )
    
    def chi_squared_test(
        self,
        observed: np.ndarray,
        expected: np.ndarray
    ) -> StatisticalTestResult:
        """
        Perform Chi-Squared test for categorical distributions.
        
        Args:
            observed: Observed frequencies
            expected: Expected frequencies
            
        Returns:
            StatisticalTestResult
        """
        try:
            statistic, p_value = stats.chisquare(observed, expected)
            passed = p_value >= self.alpha
            
            message = (
                f"Chi-Squared test: statistic={statistic:.4f}, p-value={p_value:.4f} "
                f"({'PASS' if passed else 'FAIL'} at α={self.alpha})"
            )
            
            return StatisticalTestResult(
                test_type=TestType.CHI_SQUARED,
                statistic=statistic,
                p_value=p_value,
                passed=passed,
                alpha=self.alpha,
                sample_size=int(np.sum(observed)),
                message=message
            )
            
        except Exception as e:
            logger.error(f"Chi-Squared test failed: {e}")
            return StatisticalTestResult(
                test_type=TestType.CHI_SQUARED,
                statistic=0.0,
                p_value=0.0,
                passed=False,
                alpha=self.alpha,
                sample_size=int(np.sum(observed)) if len(observed) > 0 else 0,
                message=f"Chi-Squared test error: {str(e)}"
            )
    
    def anderson_darling_test(
        self,
        data: np.ndarray,
        dist_spec: DistributionSpec
    ) -> StatisticalTestResult:
        """
        Perform Anderson-Darling test for normal distribution.
        
        More sensitive to distribution tails than KS test.
        
        Args:
            data: Sample data
            dist_spec: Distribution specification (must be normal)
            
        Returns:
            StatisticalTestResult
        """
        try:
            if dist_spec.dist_type != DistributionType.NORMAL:
                raise ValueError("Anderson-Darling test only for normal distribution")
            
            # Standardize data
            mean = dist_spec.parameters.get("mean", 0)
            std = dist_spec.parameters.get("std", 1)
            standardized = (data - mean) / std
            
            result = stats.anderson(standardized, dist='norm')
            statistic = result.statistic
            
            # Use 5% significance level (index 2 in critical_values)
            critical_value = result.critical_values[2]
            passed = statistic < critical_value
            
            # Approximate p-value (not directly provided by scipy)
            p_value = 0.05 if not passed else 0.10
            
            message = (
                f"Anderson-Darling test: statistic={statistic:.4f}, "
                f"critical_value={critical_value:.4f} at 5% "
                f"({'PASS' if passed else 'FAIL'})"
            )
            
            return StatisticalTestResult(
                test_type=TestType.ANDERSON_DARLING,
                statistic=statistic,
                p_value=p_value,
                passed=passed,
                alpha=0.05,
                sample_size=len(data),
                message=message
            )
            
        except Exception as e:
            logger.error(f"Anderson-Darling test failed: {e}")
            return StatisticalTestResult(
                test_type=TestType.ANDERSON_DARLING,
                statistic=0.0,
                p_value=0.0,
                passed=False,
                alpha=self.alpha,
                sample_size=len(data),
                message=f"Anderson-Darling test error: {str(e)}"
            )
    
    def validate_continuous_distribution(
        self,
        data: List[float],
        distribution_name: str
    ) -> DistributionReport:
        """
        Validate continuous distribution with multiple tests.
        
        Args:
            data: Sample data
            distribution_name: Name of registered distribution
            
        Returns:
            DistributionReport
        """
        if distribution_name not in self.distribution_specs:
            raise ValueError(f"Unknown distribution: {distribution_name}")
        
        spec = self.distribution_specs[distribution_name]
        data_array = np.array(data)
        
        # Calculate descriptive statistics
        sample_stats = {
            "mean": float(np.mean(data_array)),
            "std": float(np.std(data_array)),
            "median": float(np.median(data_array)),
            "min": float(np.min(data_array)),
            "max": float(np.max(data_array)),
            "count": len(data_array)
        }
        
        # Run tests
        test_results = []
        
        # KS test
        ks_result = self.kolmogorov_smirnov_test(data_array, spec)
        test_results.append(ks_result)
        
        # Anderson-Darling test (only for normal)
        if spec.dist_type == DistributionType.NORMAL:
            ad_result = self.anderson_darling_test(data_array, spec)
            test_results.append(ad_result)
        
        # Overall pass/fail
        passed = all(result.passed for result in test_results)
        
        report = DistributionReport(
            distribution_name=distribution_name,
            distribution_spec=spec,
            test_results=test_results,
            sample_statistics=sample_stats,
            passed=passed
        )
        
        if not passed:
            logger.warning(
                f"Distribution validation failed for {distribution_name}: "
                f"{sum(not r.passed for r in test_results)}/{len(test_results)} tests failed"
            )
        else:
            logger.info(f"Distribution validation passed for {distribution_name}")
        
        return report
    
    def validate_discrete_distribution(
        self,
        data: List[Any],
        distribution_name: str,
        expected_frequencies: Dict[Any, float]
    ) -> DistributionReport:
        """
        Validate discrete/categorical distribution.
        
        Args:
            data: Sample data
            distribution_name: Name of distribution
            expected_frequencies: Expected frequency for each category
            
        Returns:
            DistributionReport
        """
        if distribution_name not in self.distribution_specs:
            raise ValueError(f"Unknown distribution: {distribution_name}")
        
        spec = self.distribution_specs[distribution_name]
        
        # Count observed frequencies
        unique, counts = np.unique(data, return_counts=True)
        observed_dict = dict(zip(unique, counts))
        
        # Build observed and expected arrays
        categories = sorted(expected_frequencies.keys())
        observed = np.array([observed_dict.get(cat, 0) for cat in categories])
        total = len(data)
        expected = np.array([expected_frequencies[cat] * total for cat in categories])
        
        # Calculate descriptive statistics
        sample_stats = {
            "total_count": total,
            "unique_categories": len(unique),
            "most_common": str(unique[np.argmax(counts)]),
            "most_common_count": int(np.max(counts))
        }
        
        # Run Chi-Squared test
        chi2_result = self.chi_squared_test(observed, expected)
        test_results = [chi2_result]
        
        passed = chi2_result.passed
        
        report = DistributionReport(
            distribution_name=distribution_name,
            distribution_spec=spec,
            test_results=test_results,
            sample_statistics=sample_stats,
            passed=passed
        )
        
        if not passed:
            logger.warning(f"Discrete distribution validation failed for {distribution_name}")
        else:
            logger.info(f"Discrete distribution validation passed for {distribution_name}")
        
        return report
    
    def register_predefined_distributions(self) -> None:
        """
        Register predefined distributions from design specifications.
        """
        # Student workload distribution
        self.register_distribution(
            "student_workload",
            DistributionType.NORMAL,
            {"mean": 18, "std": 3},
            bounds=(12, 27)
        )
        
        # Course popularity (Zipf)
        self.register_distribution(
            "course_popularity",
            DistributionType.ZIPF,
            {"alpha": 1.5},
            bounds=None
        )
        
        # Faculty workload (discrete)
        self.register_distribution(
            "faculty_workload",
            DistributionType.DISCRETE,
            {"values": [2, 3, 4], "probabilities": [0.3, 0.5, 0.2]},
            bounds=None
        )
        
        logger.info("Registered predefined distributions from design specifications")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get validation statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "registered_distributions": len(self.distribution_specs),
            "alpha": self.alpha,
            "distribution_types": list(set(
                spec.dist_type.value 
                for spec in self.distribution_specs.values()
            ))
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"StatisticalValidator("
            f"distributions={stats['registered_distributions']}, "
            f"alpha={stats['alpha']})"
        )
