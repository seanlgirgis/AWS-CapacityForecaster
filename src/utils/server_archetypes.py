"""
server_archetypes.py

This module defines server archetypes (web, database, application, batch) with realistic
resource utilization patterns and correlations. This enhances synthetic data generation
to create heterogeneous infrastructure that mirrors real enterprise environments.

Archetypes are based on 8 years of production infrastructure observation at Citi Financial
and industry-standard server workload patterns.

Usage:
    from src.utils.server_archetypes import ServerArchetype, get_archetype

    web_server = get_archetype('web')
    metrics = web_server.generate_baseline_metrics()

Log files:
    When used as standalone, logs to: logs/server_archetypes_YYYYMMDD_HHMMSS.log
    When imported, uses parent logger configuration.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Setup module-level logger (uses parent's file handler if configured)
logger = logging.getLogger(__name__)


class ServerType(Enum):
    """Server archetype types."""
    WEB = "web"
    DATABASE = "database"
    APPLICATION = "application"
    BATCH = "batch"


@dataclass
class ArchetypeProfile:
    """
    Defines the characteristics of a server archetype.

    Attributes:
        name: Archetype name
        base_cpu: Base CPU utilization (mean %)
        base_memory: Base memory utilization (mean %)
        base_disk: Base disk utilization (mean %)
        base_network: Base network utilization (mean Mbps)

        cpu_variance: Standard deviation for CPU
        memory_variance: Standard deviation for memory
        disk_variance: Standard deviation for disk
        network_variance: Standard deviation for network

        cpu_memory_correlation: Correlation coefficient between CPU and memory
        cpu_network_correlation: Correlation coefficient between CPU and network
        memory_disk_correlation: Correlation coefficient between memory and disk

        business_hours_factor: Multiplier during business hours (9 AM - 5 PM)
        weekend_factor: Multiplier during weekends
        spike_probability: Probability of random spike per time period
        spike_magnitude: Magnitude of spikes (multiplier)
    """
    name: str

    # Base utilization (mean)
    base_cpu: float
    base_memory: float
    base_disk: float
    base_network: float

    # Variance (standard deviation)
    cpu_variance: float
    memory_variance: float
    disk_variance: float
    network_variance: float

    # Correlations
    cpu_memory_correlation: float
    cpu_network_correlation: float
    memory_disk_correlation: float

    # Time-based factors
    business_hours_factor: float
    weekend_factor: float

    # Spike characteristics
    spike_probability: float
    spike_magnitude: float

    # Growth trend
    monthly_growth_rate: float  # % per month


# Define archetype profiles based on industry patterns
ARCHETYPE_PROFILES = {
    ServerType.WEB: ArchetypeProfile(
        name="Web Server",
        # Web servers: High CPU during requests, moderate memory, low disk
        base_cpu=45.0,
        base_memory=35.0,
        base_disk=20.0,
        base_network=150.0,

        cpu_variance=15.0,
        memory_variance=8.0,
        disk_variance=5.0,
        network_variance=50.0,

        # Strong CPU-network correlation (requests drive both)
        cpu_memory_correlation=0.5,
        cpu_network_correlation=0.8,
        memory_disk_correlation=0.2,

        # High sensitivity to business hours
        business_hours_factor=1.6,
        weekend_factor=0.5,

        # Moderate spike frequency (traffic bursts)
        spike_probability=0.03,
        spike_magnitude=1.8,

        monthly_growth_rate=0.5,
    ),

    ServerType.DATABASE: ArchetypeProfile(
        name="Database Server",
        # Database: High memory (caching), high disk, moderate CPU
        base_cpu=35.0,
        base_memory=70.0,
        base_disk=55.0,
        base_network=100.0,

        cpu_variance=12.0,
        memory_variance=10.0,
        disk_variance=15.0,
        network_variance=30.0,

        # Memory pressure drives disk I/O
        cpu_memory_correlation=0.6,
        cpu_network_correlation=0.4,
        memory_disk_correlation=0.7,

        # Moderate business hours impact
        business_hours_factor=1.3,
        weekend_factor=0.7,

        # Low spike probability (steady state)
        spike_probability=0.01,
        spike_magnitude=1.4,

        monthly_growth_rate=1.0,  # Data grows steadily
    ),

    ServerType.APPLICATION: ArchetypeProfile(
        name="Application Server",
        # App servers: Balanced utilization
        base_cpu=50.0,
        base_memory=55.0,
        base_disk=30.0,
        base_network=120.0,

        cpu_variance=18.0,
        memory_variance=15.0,
        disk_variance=10.0,
        network_variance=40.0,

        # Moderate correlations across the board
        cpu_memory_correlation=0.7,
        cpu_network_correlation=0.6,
        memory_disk_correlation=0.4,

        # Strong business hours pattern
        business_hours_factor=1.5,
        weekend_factor=0.6,

        # Moderate spikes (batch processes)
        spike_probability=0.02,
        spike_magnitude=1.6,

        monthly_growth_rate=0.8,
    ),

    ServerType.BATCH: ArchetypeProfile(
        name="Batch Processing Server",
        # Batch: Spiky CPU, moderate memory, high disk I/O
        base_cpu=30.0,
        base_memory=45.0,
        base_disk=40.0,
        base_network=80.0,

        cpu_variance=25.0,  # Very variable
        memory_variance=12.0,
        disk_variance=20.0,
        network_variance=35.0,

        # CPU spikes during batch jobs
        cpu_memory_correlation=0.4,
        cpu_network_correlation=0.3,
        memory_disk_correlation=0.5,

        # Off-hours processing pattern
        business_hours_factor=0.8,  # Lower during day
        weekend_factor=1.2,  # Higher on weekends (batch windows)

        # High spike probability (scheduled jobs)
        spike_probability=0.08,
        spike_magnitude=2.5,

        monthly_growth_rate=0.3,
    ),
}


class ServerArchetype:
    """
    Generates realistic metrics for a specific server archetype.
    """

    def __init__(self, archetype_type: ServerType, server_id: str):
        """
        Initialize a server archetype.

        Args:
            archetype_type: Type of server (web, database, etc.)
            server_id: Unique server identifier
        """
        self.type = archetype_type
        self.profile = ARCHETYPE_PROFILES[archetype_type]
        self.server_id = server_id

        # Create deterministic but varied seed per server
        self.seed = hash(server_id) % (2**32)
        self.rng = np.random.RandomState(self.seed)

        logger.debug(f"Initialized {archetype_type.value} archetype for {server_id} (seed={self.seed})")



    def generate_correlated_metrics(
        self,
        timestamp,
        time_factor: float = 1.0,
        trend_factor: float = 0.0
    ) -> Dict[str, float]:
        """
        Generate correlated CPU, memory, disk, and network metrics.

        Uses Cholesky decomposition to create correlated Gaussian variables.

        Args:
            timestamp: Datetime timestamp for the metric
            time_factor: Multiplier for time-based patterns (business hours, etc.)
            trend_factor: Linear growth factor (0-1 scale for time progression)

        Returns:
            Dictionary with cpu_p95, mem_p95, disk_p95, net_in_p95, net_out_p95
        """
        # Correlation matrix (4x4: CPU, Memory, Disk, Network)
        corr_matrix = np.array([
            [1.0, self.profile.cpu_memory_correlation, 0.1, self.profile.cpu_network_correlation],
            [self.profile.cpu_memory_correlation, 1.0, self.profile.memory_disk_correlation, 0.2],
            [0.1, self.profile.memory_disk_correlation, 1.0, 0.3],
            [self.profile.cpu_network_correlation, 0.2, 0.3, 1.0]
        ])

        # Cholesky decomposition for correlated random variables
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            # If matrix is not positive definite, use uncorrelated
            L = np.eye(4)

        # Generate independent standard normal variables
        z = self.rng.randn(4)

        # Transform to correlated variables
        correlated = L @ z

        # Scale by variance and add to base with time factor and trend
        cpu = (
            self.profile.base_cpu * time_factor * (1 + trend_factor * self.profile.monthly_growth_rate / 100)
            + correlated[0] * self.profile.cpu_variance
        )

        memory = (
            self.profile.base_memory * time_factor * (1 + trend_factor * self.profile.monthly_growth_rate / 100)
            + correlated[1] * self.profile.memory_variance
        )

        disk = (
            self.profile.base_disk * (1 + trend_factor * self.profile.monthly_growth_rate / 100)
            + correlated[2] * self.profile.disk_variance
        )

        network_base = (
            self.profile.base_network * time_factor
            + correlated[3] * self.profile.network_variance
        )

        # Add spike if probability triggers
        if self.rng.rand() < self.profile.spike_probability:
            spike_mult = self.profile.spike_magnitude
            cpu *= spike_mult
            memory *= (spike_mult * 0.7)  # Memory spikes less than CPU
            network_base *= (spike_mult * 0.8)

        # Clip to valid ranges
        cpu = np.clip(cpu, 0, 100)
        memory = np.clip(memory, 0, 100)
        disk = np.clip(disk, 0, 100)
        network_in = np.clip(network_base, 0, 1000)
        network_out = np.clip(network_base * 0.6, 0, 600)  # Outbound typically less

        return {
            'cpu_p95': round(cpu, 2),
            'mem_p95': round(memory, 2),
            'disk_p95': round(disk, 2),
            'net_in_p95': round(network_in, 2),
            'net_out_p95': round(network_out, 2),
        }




    def get_time_factor(self, timestamp) -> float:
        """
        Calculate time-based adjustment factor.

        Args:
            timestamp: Datetime object

        Returns:
            Multiplier for base metrics based on time patterns
        """
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek

        # Business hours (9 AM - 5 PM)
        if 9 <= hour <= 17:
            bh_factor = self.profile.business_hours_factor
        else:
            bh_factor = 1.0

        # Weekend factor
        if day_of_week >= 5:  # Saturday, Sunday
            weekend_factor = self.profile.weekend_factor
        else:
            weekend_factor = 1.0

        # Combine factors
        return bh_factor * weekend_factor


def get_archetype(server_type: str, server_id: str) -> ServerArchetype:
    """
    Factory function to get a server archetype.

    Args:
        server_type: String type ('web', 'database', 'application', 'batch')
        server_id: Unique server identifier

    Returns:
        ServerArchetype instance

    Raises:
        ValueError: If server_type is not recognized
    """
    type_map = {
        'web': ServerType.WEB,
        'database': ServerType.DATABASE,
        'db': ServerType.DATABASE,
        'application': ServerType.APPLICATION,
        'app': ServerType.APPLICATION,
        'batch': ServerType.BATCH,
    }

    server_type_lower = server_type.lower()
    if server_type_lower not in type_map:
        logger.error(f"Unknown server type '{server_type}'. Valid types: {list(type_map.keys())}")
        raise ValueError(
            f"Unknown server type '{server_type}'. "
            f"Valid types: {list(type_map.keys())}"
        )

    logger.debug(f"Creating {server_type} archetype for {server_id}")
    return ServerArchetype(type_map[server_type_lower], server_id)


def assign_archetypes_to_fleet(num_servers: int, distribution: Dict[str, float] = None) -> Dict[str, str]:
    """
    Assign archetypes to a fleet of servers based on distribution.

    Args:
        num_servers: Total number of servers
        distribution: Dictionary of archetype -> proportion (e.g., {'web': 0.4, 'database': 0.3, ...})
                     If None, uses default enterprise distribution

    Returns:
        Dictionary mapping server_id -> archetype_type
    """
    logger.info(f"Assigning archetypes to fleet of {num_servers} servers")

    if distribution is None:
        # Default enterprise distribution (based on typical infrastructure)
        distribution = {
            'web': 0.35,
            'application': 0.40,
            'database': 0.15,
            'batch': 0.10,
        }
        logger.debug(f"Using default distribution: {distribution}")

    # Validate distribution sums to 1.0
    total = sum(distribution.values())
    if not np.isclose(total, 1.0):
        logger.error(f"Distribution must sum to 1.0, got {total}")
        raise ValueError(f"Distribution must sum to 1.0, got {total}")

    # Calculate counts per archetype
    assignments = {}
    server_idx = 0

    for archetype, proportion in distribution.items():
        count = int(num_servers * proportion)
        for _ in range(count):
            server_id = f"server_{server_idx:03d}"
            assignments[server_id] = archetype
            server_idx += 1

    # Assign any remaining servers to the most common archetype
    remainder = num_servers - server_idx
    while server_idx < num_servers:
        server_id = f"server_{server_idx:03d}"
        most_common = max(distribution, key=distribution.get)
        assignments[server_id] = most_common
        server_idx += 1

    if remainder > 0:
        logger.debug(f"Assigned {remainder} remaining servers to most common archetype")

    # Log summary
    archetype_counts = {}
    for archetype in assignments.values():
        archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1

    logger.info(f"Fleet assignment complete: {archetype_counts}")

    return assignments


def get_archetype_metadata() -> List[Dict[str, str]]:
    """
    Get metadata about all available archetypes for documentation.

    Returns:
        List of dictionaries with archetype information
    """
    metadata = []
    for server_type, profile in ARCHETYPE_PROFILES.items():
        metadata.append({
            'type': server_type.value,
            'name': profile.name,
            'base_cpu': f"{profile.base_cpu}%",
            'base_memory': f"{profile.base_memory}%",
            'base_disk': f"{profile.base_disk}%",
            'cpu_memory_corr': f"{profile.cpu_memory_correlation:.2f}",
            'spike_probability': f"{profile.spike_probability*100:.1f}%",
            'description': _get_archetype_description(server_type)
        })
    return metadata


def _get_archetype_description(server_type: ServerType) -> str:
    """Get human-readable description of archetype."""
    descriptions = {
        ServerType.WEB: "High CPU/network correlation, business hours sensitive, request-driven",
        ServerType.DATABASE: "High memory/disk, steady state, data growth trend",
        ServerType.APPLICATION: "Balanced resources, strong business hours pattern, moderate spikes",
        ServerType.BATCH: "Spiky CPU, off-hours processing, scheduled job patterns"
    }
    return descriptions.get(server_type, "")
