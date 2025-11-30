"""
Data loading module for OHLCV data.

Handles CSV loading with validation and memory-efficient processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Generator
import logging

logger = logging.getLogger(__name__)

# Required columns for OHLCV data (with aliases)
REQUIRED_COLUMNS = ['open', 'high', 'low', 'close']
COLUMN_ALIASES = {
    'timestamp': 'datetime',
    'time': 'datetime',
    'date': 'datetime',
    'tick_volume': 'volume',
    'vol': 'volume'
}


def validate_ohlcv(df: pd.DataFrame) -> bool:
    """
    Validate OHLCV DataFrame has required columns and valid data.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns.str.lower())
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for NaN in critical columns
    critical_cols = ['open', 'high', 'low', 'close']
    for col in critical_cols:
        col_lower = col.lower()
        if col_lower in df.columns:
            nan_count = df[col_lower].isna().sum()
            if nan_count > 0:
                logger.warning(f"Column '{col}' has {nan_count} NaN values")

    # Validate OHLC relationships
    if 'high' in df.columns and 'low' in df.columns:
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            logger.warning(f"Found {invalid_hl} rows where high < low")

    return True


def load_ohlcv(
    path: str | Path,
    datetime_format: str = "%Y-%m-%d %H:%M:%S",
    chunk_size: Optional[int] = None,
    validate: bool = True
) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file.

    Args:
        path: Path to CSV file
        datetime_format: Format string for datetime parsing
        chunk_size: If provided, process in chunks for memory efficiency
        validate: Whether to validate the data

    Returns:
        DataFrame with datetime index and float32 OHLCV columns
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info(f"Loading OHLCV data from {path}")

    if chunk_size:
        # Memory-efficient chunked loading
        df = _load_chunked(path, datetime_format, chunk_size)
    else:
        # Standard loading
        df = pd.read_csv(path)

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    # Apply column aliases (timestamp -> datetime, tick_volume -> volume)
    df.rename(columns=COLUMN_ALIASES, inplace=True)

    # Parse datetime and set as index
    if 'datetime' in df.columns:
        # Handle timezone-aware timestamps
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df['datetime'] = df['datetime'].dt.tz_localize(None)  # Remove timezone for simplicity
        df.set_index('datetime', inplace=True)
    elif df.index.name == 'datetime' or isinstance(df.index, pd.DatetimeIndex):
        pass  # Already has datetime index
    else:
        # Try first column as datetime
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], utc=True)
        df[first_col] = df[first_col].dt.tz_localize(None)
        df.set_index(first_col, inplace=True)
        df.index.name = 'datetime'

    # Sort by datetime
    df.sort_index(inplace=True)

    # Convert to float32 for memory efficiency (CRITICAL for M2)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    # Validate if requested
    if validate:
        validate_ohlcv(df.reset_index())

    logger.info(f"Loaded {len(df):,} rows from {df.index.min()} to {df.index.max()}")

    return df


def _load_chunked(
    path: Path,
    datetime_format: str,
    chunk_size: int
) -> pd.DataFrame:
    """Load CSV in chunks and concatenate."""
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        chunk.columns = chunk.columns.str.lower()
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def load_ohlcv_generator(
    path: str | Path,
    chunk_size: int = 100_000
) -> Generator[pd.DataFrame, None, None]:
    """
    Generator for memory-efficient streaming of large OHLCV files.

    Yields:
        DataFrame chunks
    """
    path = Path(path)
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        chunk.columns = chunk.columns.str.lower()
        if 'datetime' in chunk.columns:
            chunk['datetime'] = pd.to_datetime(chunk['datetime'])
            chunk.set_index('datetime', inplace=True)

        # Convert to float32
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype(np.float32)

        yield chunk


def get_data_info(path: str | Path) -> dict:
    """
    Get information about data file without loading it entirely.

    Returns:
        Dictionary with file info
    """
    path = Path(path)
    info = {
        'path': str(path),
        'size_mb': path.stat().st_size / (1024 * 1024),
    }

    # Read first and last few rows
    sample_head = pd.read_csv(path, nrows=5)
    info['columns'] = list(sample_head.columns)

    # Count total rows (memory-efficient)
    with open(path, 'r') as f:
        info['total_rows'] = sum(1 for _ in f) - 1  # Subtract header

    return info
