"""
data_generation.py

Main module for generating synthetic server capacity metrics for the AWS-CapacityForecaster project.
This module creates enterprise-realistic time-series data with banking-specific patterns (quarterly peaks,
holiday effects) using the enhanced server archetype system.

Features:
- Multiple server archetypes (web, database, application, batch)
- Correlated metrics (CPU, memory, disk, network)
- Realistic seasonality (weekly, quarterly, annual)
- Business metadata (business unit, criticality, region)
- Configurable scale (50-200 servers, 1-5 years)

Usage:
    python src/data_generation.py --output data/synthetic/server_metrics.csv

    # Or import as module
    from src.data_generation import generate_full_dataset
    df = generate_full_dataset()
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.utils.config import get_data_config, get_feature_engineering_config
from src.utils.server_archetypes import get_archetype, assign_archetypes_to_fleet
from src.utils.data_utils import generate_server_metadata, add_calendar_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_full_dataset(
    num_servers: Optional[int] = None,
    years_of_data: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    granularity: str = 'daily',
    include_metadata: bool = True,
    include_calendar_features: bool = True
) -> pd.DataFrame:
    """
    Generate complete synthetic dataset for all servers across time range.

    Args:
        num_servers: Number of servers (default from config)
        years_of_data: Years of historical data (default from config)
        start_date: Start date 'YYYY-MM-DD' (default from config)
        end_date: End date 'YYYY-MM-DD' (default from config)
        granularity: 'daily' or 'hourly'
        include_metadata: Add business metadata columns
        include_calendar_features: Add calendar-based features

    Returns:
        DataFrame with columns: timestamp, server_id, cpu_p95, mem_p95, disk_p95,
                               net_in_p95, net_out_p95, [metadata], [calendar features]
    """
    logger.info("="*70)
    logger.info("Starting Synthetic Data Generation for AWS-CapacityForecaster")
    logger.info("="*70)

    # Load configuration
    data_config = get_data_config()

    num_servers = num_servers or data_config.get('num_servers', 120)
    start_date = start_date or data_config.get('start_date', '2022-01-01')
    end_date = end_date or data_config.get('end_date', '2025-12-31')

    logger.info(f"Configuration:")
    logger.info(f"  Servers: {num_servers}")
    logger.info(f"  Date Range: {start_date} to {end_date}")
    logger.info(f"  Granularity: {granularity}")

    # Generate timestamp range
    if granularity == 'daily':
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    elif granularity == 'hourly':
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    logger.info(f"  Total timestamps: {len(date_range):,}")

    # Assign archetypes to servers
    logger.info("\nAssigning server archetypes...")
    archetype_assignments = assign_archetypes_to_fleet(num_servers)

    archetype_counts = {}
    for archetype in archetype_assignments.values():
        archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1

    logger.info("  Archetype distribution:")
    for archetype, count in sorted(archetype_counts.items()):
        logger.info(f"    {archetype}: {count} servers ({count/num_servers*100:.1f}%)")

    # Generate metrics for each server
    logger.info("\nGenerating time-series metrics...")
    all_data = []

    for server_id, archetype_type in archetype_assignments.items():
        if int(server_id.split('_')[1]) % 20 == 0:  # Progress update every 20 servers
            logger.info(f"  Processing {server_id}...")

        # Create archetype instance
        archetype = get_archetype(archetype_type, server_id)

        # Generate metrics for each timestamp
        for idx, timestamp in enumerate(date_range):
            # Calculate time factor (business hours, weekends)
            time_factor = archetype.get_time_factor(timestamp)

            # Calculate trend factor (gradual growth over time)
            trend_factor = idx / len(date_range)

            # Add quarterly peaks (banking-specific)
            qtr_factor = _get_quarterly_peak_factor(timestamp, data_config)

            # Add holiday effects
            holiday_factor = _get_holiday_factor(timestamp, data_config)

            # Combine all time-based factors
            combined_factor = time_factor * qtr_factor * holiday_factor

            # Generate correlated metrics
            metrics = archetype.generate_correlated_metrics(
                timestamp=timestamp,
                time_factor=combined_factor,
                trend_factor=trend_factor
            )

            # Create record
            record = {
                'timestamp': timestamp,
                'server_id': server_id,
                **metrics
            }

            all_data.append(record)

    # Create DataFrame
    logger.info("\nBuilding DataFrame...")
    df = pd.DataFrame(all_data)

    # Set timestamp as index
    df.set_index('timestamp', inplace=True)

    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Add metadata if requested
    if include_metadata:
        logger.info("\nAdding business metadata...")
        metadata_df = generate_server_metadata(
            n_servers=num_servers
        )

        # Add archetype info to metadata
        metadata_df['server_type'] = metadata_df['server_id'].map(archetype_assignments)

        # Merge with metrics
        df = df.reset_index().merge(metadata_df, on='server_id', how='left').set_index('timestamp')
        logger.info(f"  Added columns: {list(metadata_df.columns)}")

    # Add calendar features if requested
    if include_calendar_features:
        logger.info("\nAdding calendar features...")
        df = add_calendar_features(df.reset_index(), date_col='timestamp').set_index('timestamp')

    logger.info("\n" + "="*70)
    logger.info("Data Generation Complete")
    logger.info("="*70)
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Servers: {df['server_id'].nunique()}")
    logger.info(f"\nSample statistics:")
    logger.info(f"\n{df[['cpu_p95', 'mem_p95', 'disk_p95', 'net_in_p95', 'net_out_p95']].describe()}")

    return df


def _get_quarterly_peak_factor(timestamp: datetime, config: Dict) -> float:
    """
    Calculate quarterly peak factor for banking workloads.

    Banking/financial services see significant load increases at end of quarter
    (Mar 31, Jun 30, Sep 30, Dec 31) due to reporting and reconciliation.

    Args:
        timestamp: Current timestamp
        config: Data configuration

    Returns:
        Multiplier (1.0 = normal, >1.0 = peak)
    """
    if not config.get('seasonality', {}).get('quarterly_peaks', False):
        return 1.0

    # Check if we're in the last 5 days of a quarter
    month = timestamp.month
    day = timestamp.day

    # Quarter end months: March (3), June (6), September (9), December (12)
    if month in [3, 6, 9, 12]:
        days_in_month = pd.Timestamp(timestamp.year, month, 1).days_in_month

        # Last 5 days of quarter
        if day >= days_in_month - 4:
            # Peak intensity increases as we get closer to quarter end
            days_from_end = days_in_month - day
            peak_intensity = 1.0 + (0.3 * (5 - days_from_end) / 5)  # Up to 1.3x on last day
            return peak_intensity

    return 1.0


def _get_holiday_factor(timestamp: datetime, config: Dict) -> float:
    """
    Calculate holiday effect factor (reduced load).

    Args:
        timestamp: Current timestamp
        config: Data configuration

    Returns:
        Multiplier (1.0 = normal, <1.0 = reduced load)
    """
    if not config.get('seasonality', {}).get('holiday_effect', False):
        return 1.0

    # US Federal Holidays (major ones affecting financial services)
    year = timestamp.year
    month = timestamp.month
    day = timestamp.day

    # New Year's Day (Jan 1)
    if month == 1 and day == 1:
        return 0.5

    # Week between Christmas and New Year (Dec 25-31)
    if month == 12 and day >= 25:
        return 0.6

    # Independence Day (Jul 4)
    if month == 7 and day == 4:
        return 0.7

    # Thanksgiving week (4th Thursday of November, approximate)
    if month == 11 and 22 <= day <= 28:
        return 0.7

    # Christmas Eve (Dec 24)
    if month == 12 and day == 24:
        return 0.7

    return 1.0


def save_dataset(
    df: pd.DataFrame,
    output_path: str,
    format: str = 'csv',
    compress: bool = False
) -> None:
    """
    Save generated dataset to file.

    Args:
        df: DataFrame to save
        output_path: Output file path
        format: 'csv' or 'parquet'
        compress: Whether to compress output
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving dataset to {output_path}...")

    if format == 'csv':
        if compress:
            output_path = output_path.with_suffix('.csv.gz')
            df.to_csv(output_path, compression='gzip')
        else:
            df.to_csv(output_path)
    elif format == 'parquet':
        df.to_parquet(output_path, compression='snappy' if compress else None)
    else:
        raise ValueError(f"Unknown format: {format}")

    file_size_mb = output_path.stat().st_size / 1024**2
    logger.info(f"  Saved: {output_path}")
    logger.info(f"  Size: {file_size_mb:.2f} MB")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic server capacity metrics"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/synthetic/server_metrics.csv',
        help='Output file path'
    )
    parser.add_argument(
        '--servers',
        type=int,
        help='Number of servers (default from config)'
    )
    parser.add_argument(
        '--years',
        type=int,
        help='Years of data (default from config)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date YYYY-MM-DD (default from config)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date YYYY-MM-DD (default from config)'
    )
    parser.add_argument(
        '--granularity',
        type=str,
        choices=['daily', 'hourly'],
        default='daily',
        help='Time granularity'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'parquet'],
        default='csv',
        help='Output format'
    )
    parser.add_argument(
        '--compress',
        action='store_true',
        help='Compress output file'
    )
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Exclude business metadata'
    )
    parser.add_argument(
        '--no-calendar',
        action='store_true',
        help='Exclude calendar features'
    )

    args = parser.parse_args()

    # Generate dataset
    df = generate_full_dataset(
        num_servers=args.servers,
        start_date=args.start_date,
        end_date=args.end_date,
        granularity=args.granularity,
        include_metadata=not args.no_metadata,
        include_calendar_features=not args.no_calendar
    )

    # Save dataset
    save_dataset(
        df=df,
        output_path=args.output,
        format=args.format,
        compress=args.compress
    )

    logger.info("\n[OK] Data generation complete!")


if __name__ == '__main__':
    main()
