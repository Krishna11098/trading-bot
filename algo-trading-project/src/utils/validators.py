"""
Data validation utilities for trading system.

Provides validation for OHLCV data, features, and configuration.
"""
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class DataValidator:
    """
    Validates trading data quality and integrity.
    
    Checks for:
    - Missing columns
    - Invalid values (NaN, inf)
    - Data type consistency
    - OHLCV constraints (High >= Low, etc.)
    """
    
    REQUIRED_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
    
    @classmethod
    def validate_ohlcv(
        cls,
        df: pd.DataFrame,
        raise_error: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Validate OHLCV data format and constraints.
        
        Args:
            df: DataFrame to validate
            raise_error: Whether to raise error on validation failure
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check required columns
        missing = [c for c in cls.REQUIRED_OHLCV_COLUMNS if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
        
        if missing:
            if raise_error:
                raise ValidationError("; ".join(errors))
            return False, errors
        
        # Check for empty data
        if len(df) == 0:
            errors.append("DataFrame is empty")
        
        # Check data types
        for col in ["Open", "High", "Low", "Close"]:
            if not np.issubdtype(df[col].dtype, np.number):
                errors.append(f"Column {col} is not numeric")
        
        # Check for NaN values
        nan_cols = df[cls.REQUIRED_OHLCV_COLUMNS].isna().sum()
        nan_cols = nan_cols[nan_cols > 0]
        if len(nan_cols) > 0:
            errors.append(f"NaN values found: {nan_cols.to_dict()}")
        
        # Check for infinite values
        inf_mask = np.isinf(df[["Open", "High", "Low", "Close"]].values)
        if inf_mask.any():
            errors.append("Infinite values found in price columns")
        
        # Check OHLCV constraints
        constraint_errors = cls._check_ohlcv_constraints(df)
        errors.extend(constraint_errors)
        
        # Check for negative prices
        if (df[["Open", "High", "Low", "Close"]] < 0).any().any():
            errors.append("Negative prices found")
        
        # Check for zero volume
        zero_volume_pct = (df["Volume"] == 0).mean() * 100
        if zero_volume_pct > 50:
            errors.append(f"High percentage of zero volume: {zero_volume_pct:.1f}%")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            for error in errors:
                logger.warning(f"Validation error: {error}")
            
            if raise_error:
                raise ValidationError("; ".join(errors))
        
        return is_valid, errors
    
    @staticmethod
    def _check_ohlcv_constraints(df: pd.DataFrame) -> List[str]:
        """Check OHLCV logical constraints."""
        errors = []
        
        # High should be >= Low
        invalid_hl = (df["High"] < df["Low"]).sum()
        if invalid_hl > 0:
            errors.append(f"High < Low in {invalid_hl} rows")
        
        # High should be >= Open and Close
        invalid_high = ((df["High"] < df["Open"]) | (df["High"] < df["Close"])).sum()
        if invalid_high > 0:
            errors.append(f"High < Open or Close in {invalid_high} rows")
        
        # Low should be <= Open and Close
        invalid_low = ((df["Low"] > df["Open"]) | (df["Low"] > df["Close"])).sum()
        if invalid_low > 0:
            errors.append(f"Low > Open or Close in {invalid_low} rows")
        
        return errors
    
    @classmethod
    def validate_datetime_index(
        cls,
        df: pd.DataFrame,
        raise_error: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame has proper datetime index.
        
        Args:
            df: DataFrame to validate
            raise_error: Whether to raise error on failure
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("Index is not DatetimeIndex")
        else:
            # Check for duplicates
            duplicates = df.index.duplicated().sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate timestamps")
            
            # Check if sorted
            if not df.index.is_monotonic_increasing:
                errors.append("Index is not sorted chronologically")
        
        is_valid = len(errors) == 0
        
        if not is_valid and raise_error:
            raise ValidationError("; ".join(errors))
        
        return is_valid, errors
    
    @classmethod
    def validate_features(
        cls,
        df: pd.DataFrame,
        required_features: Optional[List[str]] = None,
        raise_error: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Validate feature data for ML training/prediction.
        
        Args:
            df: DataFrame with features
            required_features: List of required feature columns
            raise_error: Whether to raise error on failure
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if required_features:
            missing = [f for f in required_features if f not in df.columns]
            if missing:
                errors.append(f"Missing features: {missing}")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                errors.append(f"Infinite values in feature: {col}")
        
        # Check NaN percentage
        nan_pct = df.isna().mean() * 100
        high_nan = nan_pct[nan_pct > 50]
        if len(high_nan) > 0:
            errors.append(f"High NaN percentage in columns: {high_nan.to_dict()}")
        
        is_valid = len(errors) == 0
        
        if not is_valid and raise_error:
            raise ValidationError("; ".join(errors))
        
        return is_valid, errors


def validate_signal_column(
    df: pd.DataFrame,
    signal_col: str = "combined_signal",
) -> bool:
    """
    Validate signal column values.
    
    Args:
        df: DataFrame with signal column
        signal_col: Name of signal column
    
    Returns:
        True if valid
    """
    if signal_col not in df.columns:
        logger.warning(f"Signal column '{signal_col}' not found")
        return False
    
    signals = df[signal_col].unique()
    valid_signals = {-1, 0, 1}
    
    invalid = set(signals) - valid_signals
    if invalid:
        logger.warning(f"Invalid signal values found: {invalid}")
        return False
    
    return True


def clean_and_validate(
    df: pd.DataFrame,
    drop_na: bool = True,
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    Clean and validate OHLCV data in one step.
    
    Args:
        df: Raw DataFrame
        drop_na: Whether to drop remaining NaN rows
        fill_method: Method for filling NaN values
    
    Returns:
        Cleaned and validated DataFrame
    """
    df = df.copy()
    
    # Forward fill, then backward fill
    df = df.fillna(method=fill_method)
    df = df.fillna(method="bfill")
    
    # Drop any remaining NaN
    if drop_na:
        df = df.dropna()
    
    # Sort by index
    df = df.sort_index()
    
    # Validate
    DataValidator.validate_ohlcv(df, raise_error=False)
    
    return df
