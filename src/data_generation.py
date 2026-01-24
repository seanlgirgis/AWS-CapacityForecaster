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

Log files:
    Logs are written to: logs/data_generation_YYYYMMDD_HHMMSS.log
"""

import pandas as pd
import numpy as np
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.utils.config import get_data_config, get_feature_engineering_config
from src.utils.server_archetypes import get_archetype, assign_archetypes_to_fleet
from src.utils.data_utils import generate_server_metadata, add_calendar_features
from src.utils.logging_config import setup_logger, LogSection, get_log_file_path

# Setup logging with file output
logger = setup_logger(__name__, 'data_generation')


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
    generation_start_time = time.time()

    logger.info("=" * 70)
    logger.info("SYNTHETIC DATA GENERATION - AWS-CapacityForecaster")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {get_log_file_path(logger)}")

    # Load configuration
    logger.info("\n[PHASE 1/6] Loading configuration...")
    config_start = time.time()
    data_config = get_data_config()

    num_servers = num_servers or data_config.get('num_servers', 120)
    start_date = start_date or data_config.get('start_date', '2022-01-01')
    end_date = end_date or data_config.get('end_date', '2025-12-31')

    logger.info(f"  Configuration loaded in {time.time() - config_start:.2f}s")
    logger.info(f"  Parameters:")
    logger.info(f"    - num_servers: {num_servers}")
    logger.info(f"    - start_date: {start_date}")
    logger.info(f"    - end_date: {end_date}")
    logger.info(f"    - granularity: {granularity}")
    logger.info(f"    - include_metadata: {include_metadata}")
    logger.info(f"    - include_calendar_features: {include_calendar_features}")

    # Generate timestamp range
    logger.info("\n[PHASE 2/6] Generating timestamp range...")
    timestamp_start = time.time()

    if granularity == 'daily':
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    elif granularity == 'hourly':
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    else:
        logger.error(f"Unknown granularity: {granularity}")
        raise ValueError(f"Unknown granularity: {granularity}")

    logger.info(f"  Generated {len(date_range):,} timestamps in {time.time() - timestamp_start:.2f}s")
    logger.info(f"  First timestamp: {date_range[0]}")
    logger.info(f"  Last timestamp: {date_range[-1]}")

    # Assign archetypes to servers
    logger.info("\n[PHASE 3/6] Assigning server archetypes...")
    archetype_start = time.time()
    archetype_assignments = assign_archetypes_to_fleet(num_servers)

    archetype_counts = {}
    for archetype in archetype_assignments.values():
        archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1

    logger.info(f"  Assigned archetypes to {num_servers} servers in {time.time() - archetype_start:.2f}s")
    logger.info("  Archetype distribution:")
    for archetype, count in sorted(archetype_counts.items()):
        logger.info(f"    - {archetype}: {count} servers ({count/num_servers*100:.1f}%)")

    # Generate metrics for each server
    logger.info("\n[PHASE 4/6] Generating time-series metrics...")
    metrics_start = time.time()
    all_data = []
    total_records_expected = num_servers * len(date_range)
    logger.info(f"  Expected records: {total_records_expected:,}")
    logger.info(f"  Processing {num_servers} servers x {len(date_range):,} timestamps...")

    servers_processed = 0
    for server_id, archetype_type in archetype_assignments.items():
        servers_processed += 1
        if servers_processed % 20 == 0 or servers_processed == 1:  # Progress update every 20 servers
            elapsed = time.time() - metrics_start
            pct_complete = (servers_processed / num_servers) * 100
            records_so_far = len(all_data)
            rate = records_so_far / elapsed if elapsed > 0 else 0
            logger.info(f"  [{pct_complete:5.1f}%] Processing {server_id} ({archetype_type}) | {records_so_far:,} records | {rate:,.0f} rec/s")

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

    # Log metrics generation completion
    metrics_elapsed = time.time() - metrics_start
    logger.info(f"  [100.0%] Metrics generation complete")
    logger.info(f"  Total records generated: {len(all_data):,}")
    logger.info(f"  Generation time: {metrics_elapsed:.2f}s")
    logger.info(f"  Average rate: {len(all_data)/metrics_elapsed:,.0f} records/second")

    # Create DataFrame
    logger.info("\n[PHASE 5/6] Building DataFrame...")
    df_start = time.time()
    df = pd.DataFrame(all_data)

    # Set timestamp as index
    df.set_index('timestamp', inplace=True)

    logger.info(f"  DataFrame created in {time.time() - df_start:.2f}s")
    logger.info(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"  Columns: {list(df.columns)}")

    # Add metadata if requested
    if include_metadata:
        logger.info("\n  Adding business metadata...")
        metadata_start = time.time()
        metadata_df = generate_server_metadata(
            n_servers=num_servers
        )

        # Add archetype info to metadata
        metadata_df['server_type'] = metadata_df['server_id'].map(archetype_assignments)

        # Merge with metrics
        df = df.reset_index().merge(metadata_df, on='server_id', how='left').set_index('timestamp')
        logger.info(f"  Metadata added in {time.time() - metadata_start:.2f}s")
        logger.info(f"  New columns: {list(metadata_df.columns)}")

    # Add calendar features if requested
    if include_calendar_features:
        logger.info("\n  Adding calendar features...")
        calendar_start = time.time()
        df = add_calendar_features(df.reset_index(), date_col='timestamp').set_index('timestamp')
        logger.info(f"  Calendar features added in {time.time() - calendar_start:.2f}s")

    # Final summary
    total_elapsed = time.time() - generation_start_time

    logger.info("\n" + "=" * 70)
    logger.info("[PHASE 6/6] DATA GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nFinal Dataset Summary:")
    logger.info(f"  Total records: {df.shape[0]:,}")
    logger.info(f"  Total columns: {df.shape[1]}")
    logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"  Unique servers: {df['server_id'].nunique()}")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    logger.info(f"\nMetric Statistics:")
    metrics_cols = ['cpu_p95', 'mem_p95', 'disk_p95', 'net_in_p95', 'net_out_p95']
    for col in metrics_cols:
        stats = df[col].describe()
        logger.info(f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")

    logger.info(f"\nPerformance:")
    logger.info(f"  Total time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)")
    logger.info(f"  Records/second: {df.shape[0]/total_elapsed:,.0f}")
    logger.info(f"\nLog file: {get_log_file_path(logger)}")

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
    save_start = time.time()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "-" * 70)
    logger.info("SAVING DATASET")
    logger.info("-" * 70)
    logger.info(f"  Output path: {output_path}")
    logger.info(f"  Format: {format}")
    logger.info(f"  Compression: {compress}")
    logger.info(f"  Records to save: {len(df):,}")

    if format == 'csv':
        if compress:
            output_path = output_path.with_suffix('.csv.gz')
            logger.info(f"  Writing compressed CSV to: {output_path}")
            df.to_csv(output_path, compression='gzip')
        else:
            logger.info(f"  Writing CSV to: {output_path}")
            df.to_csv(output_path)
    elif format == 'parquet':
        logger.info(f"  Writing Parquet to: {output_path}")
        df.to_parquet(output_path, compression='snappy' if compress else None)
    else:
        logger.error(f"Unknown format: {format}")
        raise ValueError(f"Unknown format: {format}")

    file_size_mb = output_path.stat().st_size / 1024**2
    save_elapsed = time.time() - save_start

    logger.info(f"\n  [OK] File saved successfully")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    logger.info(f"  Write time: {save_elapsed:.2f}s")
    logger.info(f"  Write rate: {file_size_mb/save_elapsed:.2f} MB/s")


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

    logger.info("\n" + "=" * 70)
    logger.info("[OK] DATA GENERATION PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output file: {args.output}")
    logger.info(f"Log file: {get_log_file_path(logger)}")


if __name__ == '__main__':
    main()
