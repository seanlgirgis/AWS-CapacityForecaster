# Complete Code Walkthrough with Inline Comments

**Part 2 of 5** | [◀ Back to Index](README.md) | [Next: Function Catalog ▶](03_Function_Catalog.md)

---

## Table of Contents

1. [server_archetypes.py - Complete Walkthrough](#1-server_archetypespy---complete-walkthrough)
2. [data_generation.py - Complete Walkthrough](#2-data_generationpy---complete-walkthrough)
3. [Configuration System - Complete Walkthrough](#3-configuration-system---complete-walkthrough)
4. [Data Utilities - Key Functions](#4-data-utilities---key-functions)
5. [Design Decisions Explained](#5-design-decisions-explained)

---

## 1. server_archetypes.py - Complete Walkthrough

**File:** `src/utils/server_archetypes.py` (356 lines)
**Purpose:** Define server archetypes with realistic resource utilization patterns

### 1.1 Imports and Setup

```python
"""
server_archetypes.py

This module defines server archetypes (web, database, application, batch) with realistic
resource utilization patterns and correlations. This enhances synthetic data generation
to create heterogeneous infrastructure that mirrors real enterprise environments.

WHY: In real datacenters, different server types have unique behaviors:
- Web servers: High CPU-network correlation (requests drive both)
- Database servers: High memory-disk correlation (caching → I/O)
- Application servers: Balanced resource usage
- Batch servers: Spiky CPU usage (scheduled jobs)

This heterogeneity is critical for realistic ML model training.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class ServerType(Enum):
    """
    Server archetype types.

    WHY USE ENUM: Type safety, prevents typos like 'wed' instead of 'web'
    """
    WEB = "web"
    DATABASE = "database"
    APPLICATION = "application"
    BATCH = "batch"


@dataclass
class ArchetypeProfile:
    """
    Defines the characteristics of a server archetype.

    WHY DATACLASS:
    - Clean definition of structured data
    - Automatic __init__, __repr__, __eq__
    - Type hints for all attributes
    - Immutable by default (can add frozen=True)

    DESIGN PHILOSOPHY:
    Each archetype is defined by:
    1. Base metrics (mean values)
    2. Variance (spread around mean)
    3. Correlations (how metrics relate)
    4. Time patterns (business hours, weekends)
    5. Spike behavior (frequency and magnitude)
    """
    name: str

    # Base utilization (mean) - WHERE THE METRIC CENTERS
    # Example: base_cpu=45 means CPU will average around 45%
    base_cpu: float
    base_memory: float
    base_disk: float
    base_network: float

    # Variance (standard deviation) - HOW MUCH METRICS VARY
    # Example: cpu_variance=15 means CPU will typically be ±15% from base
    # So if base=45, expect range of 30-60% most of the time (1 std dev)
    cpu_variance: float
    memory_variance: float
    disk_variance: float
    network_variance: float

    # Correlations - HOW METRICS MOVE TOGETHER
    # Range: -1.0 to +1.0
    # +1.0 = perfect positive correlation (both increase together)
    #  0.0 = no correlation (independent)
    # -1.0 = perfect negative correlation (one up, other down)
    #
    # Example: cpu_memory_correlation=0.7 means when CPU is high,
    # memory is LIKELY to be high too (but not always)
    cpu_memory_correlation: float
    cpu_network_correlation: float
    memory_disk_correlation: float

    # Time-based factors - HOW TIME OF DAY/WEEK AFFECTS LOAD
    # These are MULTIPLIERS applied to base metrics
    # Example: business_hours_factor=1.6 means 60% increase during 9-5
    # Example: weekend_factor=0.5 means 50% reduction on weekends
    business_hours_factor: float
    weekend_factor: float

    # Spike characteristics - RANDOM LOAD SPIKES
    # spike_probability: Chance per timestamp (0.03 = 3% = ~1 per day)
    # spike_magnitude: Multiplier during spike (1.8 = 80% increase)
    spike_probability: float
    spike_magnitude: float

    # Growth trend - LINEAR GROWTH OVER TIME
    # Example: monthly_growth_rate=0.5 means 0.5% growth per month
    # Over 4 years: 0.5% * 48 months = 24% total growth
    monthly_growth_rate: float
```

### 1.2 Archetype Profiles Definition

```python
# Define archetype profiles based on industry patterns
# SOURCE: 8 years of production infrastructure observation at Citi Financial
# plus industry benchmarks from DataDog, New Relic reports

ARCHETYPE_PROFILES = {

    # ============================================================
    # WEB SERVER PROFILE
    # ============================================================
    # CHARACTERISTICS:
    # - Request-driven workload (HTTP/HTTPS traffic)
    # - Stateless (low memory, no caching)
    # - High CPU during request processing
    # - High network (receive requests, send responses)
    # - Minimal disk I/O (no database on same server)
    # - STRONG business hours pattern (user-driven)
    # ============================================================

    ServerType.WEB: ArchetypeProfile(
        name="Web Server",

        # BASE METRICS - Where metrics center around
        base_cpu=45.0,           # MODERATE CPU - request processing
        base_memory=35.0,        # LOW MEMORY - stateless, no caching
        base_disk=20.0,          # MINIMAL DISK - static content only
        base_network=150.0,      # HIGH NETWORK - HTTP traffic

        # VARIANCE - How much metrics fluctuate
        # WHY HIGHER CPU VARIANCE: Burst traffic patterns
        # WHY LOWER MEMORY VARIANCE: Stateless, predictable
        cpu_variance=15.0,       # Moderate variability
        memory_variance=8.0,     # Low variability (stateless)
        disk_variance=5.0,       # Very low (minimal I/O)
        network_variance=50.0,   # High (traffic bursts)

        # CORRELATIONS - How metrics relate
        # WHY 0.5 CPU-MEMORY: Some correlation but not strong (stateless)
        # WHY 0.8 CPU-NETWORK: STRONG - requests drive both! This is key!
        # WHY 0.2 MEMORY-DISK: Weak - no caching to disk relationship
        cpu_memory_correlation=0.5,
        cpu_network_correlation=0.8,   # KEY CHARACTERISTIC OF WEB SERVERS
        memory_disk_correlation=0.2,

        # TIME PATTERNS
        # WHY 1.6 BUSINESS HOURS: User traffic peaks during workday
        # WHY 0.5 WEEKEND: 50% drop in user traffic on weekends
        business_hours_factor=1.6,     # HIGH sensitivity to business hours
        weekend_factor=0.5,            # Significant weekend drop

        # SPIKE BEHAVIOR
        # WHY 0.03 PROBABILITY: Traffic surges (viral content, campaigns)
        # WHY 1.8 MAGNITUDE: 80% spike is realistic for web traffic
        spike_probability=0.03,        # 3% chance = ~1 spike per day
        spike_magnitude=1.8,           # 80% increase

        # GROWTH
        # WHY 0.5% MONTHLY: Steady user growth typical
        monthly_growth_rate=0.5,
    ),

    # ============================================================
    # DATABASE SERVER PROFILE
    # ============================================================
    # CHARACTERISTICS:
    # - Query-driven workload (SQL operations)
    # - HIGH memory for buffer pools, query cache
    # - HIGH disk I/O (reads/writes)
    # - CPU optimized by query optimization
    # - STEADY STATE operation (always active)
    # - MODERATE business hours impact (some background jobs)
    # ============================================================

    ServerType.DATABASE: ArchetypeProfile(
        name="Database Server",

        # BASE METRICS
        base_cpu=35.0,           # LOWER CPU - optimized queries, indexing
        base_memory=70.0,        # HIGH MEMORY - buffer pools, caching
        base_disk=55.0,          # HIGH DISK - constant I/O
        base_network=100.0,      # MODERATE NETWORK - query results

        # VARIANCE
        # WHY LOWER CPU VARIANCE: Steady query patterns
        # WHY HIGHER DISK VARIANCE: Checkpoint writes, vacuum operations
        cpu_variance=12.0,       # Lower (optimized, predictable)
        memory_variance=10.0,    # Lower (stable buffer pools)
        disk_variance=15.0,      # Moderate (periodic writes)
        network_variance=30.0,   # Lower (stable query patterns)

        # CORRELATIONS
        # WHY 0.6 CPU-MEMORY: Memory pressure affects query performance
        # WHY 0.7 MEMORY-DISK: KEY - When memory fills, DB swaps to disk!
        cpu_memory_correlation=0.6,
        cpu_network_correlation=0.4,
        memory_disk_correlation=0.7,   # KEY DATABASE CHARACTERISTIC

        # TIME PATTERNS
        # WHY 1.3 BUSINESS HOURS: Moderate increase (queries + background)
        # WHY 0.7 WEEKEND: Still active (maintenance, batch jobs)
        business_hours_factor=1.3,
        weekend_factor=0.7,            # DB always somewhat active

        # SPIKE BEHAVIOR
        # WHY 0.01 PROBABILITY: Rare (databases are steady-state)
        # WHY 1.4 MAGNITUDE: Smaller spikes (long-running queries)
        spike_probability=0.01,        # Low - steady operation
        spike_magnitude=1.4,

        # GROWTH
        # WHY 1.0% MONTHLY: Data grows steadily (inserts accumulate)
        monthly_growth_rate=1.0,       # Data grows over time
    ),

    # ============================================================
    # APPLICATION SERVER PROFILE
    # ============================================================
    # CHARACTERISTICS:
    # - Business logic processing (stateful)
    # - BALANCED resource usage across all metrics
    # - Mix of compute, memory, I/O
    # - STRONG business hours correlation
    # - Moderate spike frequency (batch processes)
    # ============================================================

    ServerType.APPLICATION: ArchetypeProfile(
        name="Application Server",

        # BASE METRICS - ALL BALANCED
        base_cpu=50.0,           # BALANCED
        base_memory=55.0,        # BALANCED (stateful apps)
        base_disk=30.0,          # BALANCED (some caching)
        base_network=120.0,      # BALANCED

        # VARIANCE
        # WHY HIGHER VARIANCE: Diverse workloads (API calls, batch, etc.)
        cpu_variance=18.0,       # Higher variability
        memory_variance=15.0,    # Higher (stateful operations)
        disk_variance=10.0,
        network_variance=40.0,

        # CORRELATIONS
        # WHY 0.7 CPU-MEMORY: STRONG - Stateful apps use both together
        # All correlations moderate (diverse workload)
        cpu_memory_correlation=0.7,    # Strong for stateful apps
        cpu_network_correlation=0.6,
        memory_disk_correlation=0.4,

        # TIME PATTERNS
        # WHY 1.5 BUSINESS HOURS: Strong business hours pattern
        # WHY 0.6 WEEKEND: 40% reduction (business-driven)
        business_hours_factor=1.5,
        weekend_factor=0.6,

        # SPIKE BEHAVIOR
        # WHY 0.02 PROBABILITY: Moderate (batch processes, reports)
        spike_probability=0.02,
        spike_magnitude=1.6,

        # GROWTH
        monthly_growth_rate=0.8,
    ),

    # ============================================================
    # BATCH PROCESSING SERVER PROFILE
    # ============================================================
    # CHARACTERISTICS:
    # - Scheduled job execution (cron, ETL)
    # - VERY SPIKY CPU (idle → 100% → idle)
    # - INVERSE time pattern (runs off-hours!)
    # - HIGH variance (unpredictable job patterns)
    # - WEAKER correlations (diverse job types)
    # ============================================================

    ServerType.BATCH: ArchetypeProfile(
        name="Batch Processing Server",

        # BASE METRICS
        base_cpu=30.0,           # LOW baseline - idle between jobs
        base_memory=45.0,        # MODERATE - job-dependent
        base_disk=40.0,          # MODERATE - I/O during processing
        base_network=80.0,       # LOWER - mostly local processing

        # VARIANCE
        # WHY VERY HIGH CPU VARIANCE: Spiky workload (0% → 100% → 0%)
        cpu_variance=25.0,       # VERY HIGH - spiky jobs
        memory_variance=12.0,
        disk_variance=20.0,
        network_variance=35.0,

        # CORRELATIONS
        # WHY WEAKER CORRELATIONS: Diverse job types
        # Different jobs stress different resources
        cpu_memory_correlation=0.4,
        cpu_network_correlation=0.3,
        memory_disk_correlation=0.5,

        # TIME PATTERNS - INVERSE PATTERN!
        # WHY 0.8 BUSINESS HOURS: LOWER during day (avoid production impact)
        # WHY 1.2 WEEKEND: HIGHER on weekends (batch windows)
        business_hours_factor=0.8,     # INVERSE - lower during day
        weekend_factor=1.2,            # INVERSE - higher on weekends

        # SPIKE BEHAVIOR
        # WHY 0.08 PROBABILITY: VERY HIGH - scheduled jobs trigger spikes
        # WHY 2.5 MAGNITUDE: LARGE - batch jobs consume full CPU
        spike_probability=0.08,        # 8% = multiple spikes per day
        spike_magnitude=2.5,           # 150% increase (can go to 100%)

        # GROWTH
        monthly_growth_rate=0.3,       # Slower - job count grows slowly
    ),
}
```

### 1.3 ServerArchetype Class

```python
class ServerArchetype:
    """
    Generates realistic metrics for a specific server archetype.

    DESIGN PATTERN: Factory + Strategy
    - Factory: get_archetype() creates instances
    - Strategy: Each archetype has different behavior via profile

    KEY INSIGHT: Use server_id for deterministic random seed
    WHY: Each server gets unique but reproducible behavior
    """

    def __init__(self, archetype_type: ServerType, server_id: str):
        """
        Initialize a server archetype.

        IMPLEMENTATION DETAIL:
        - Uses hash(server_id) for seed → deterministic randomness
        - Each server_id produces different but reproducible values
        - Same server_id always generates same sequence

        WHY DETERMINISTIC: Reproducible experiments, debugging
        """
        self.type = archetype_type
        self.profile = ARCHETYPE_PROFILES[archetype_type]
        self.server_id = server_id

        # Create deterministic but varied seed per server
        # hash('server_001') → large int → mod 2^32 → valid seed
        self.seed = hash(server_id) % (2**32)

        # Create dedicated random number generator for this server
        # WHY SEPARATE RNG: Don't affect global random state
        self.rng = np.random.RandomState(self.seed)

    def generate_correlated_metrics(
        self,
        timestamp,
        time_factor: float = 1.0,
        trend_factor: float = 0.0
    ) -> Dict[str, float]:
        """
        Generate correlated CPU, memory, disk, and network metrics.

        Uses Cholesky decomposition to create correlated Gaussian variables.

        MATHEMATICAL FOUNDATION:
        ========================

        Goal: Generate 4 random variables with specified correlations

        Problem: Standard random generators produce INDEPENDENT variables
        - np.random.randn() gives independent N(0,1) values
        - We need CORRELATED values

        Solution: Cholesky Decomposition

        STEP 1: Define desired correlation matrix
        ┌                                      ┐
        │ 1.0   cpu_mem   0.1      cpu_net    │
        │ cpu_mem  1.0    mem_disk   0.2      │
        │ 0.1   mem_disk  1.0       0.3       │
        │ cpu_net  0.2    0.3       1.0       │
        └                                      ┘

        STEP 2: Cholesky decomposition
        corr_matrix = L @ L.T  (where L is lower triangular)

        STEP 3: Generate independent random vector
        z = [z1, z2, z3, z4] where each zi ~ N(0, 1) independent

        STEP 4: Transform to correlated
        correlated = L @ z

        RESULT: 'correlated' has the desired correlation structure!

        PROOF (simplified):
        Cov(L@z) = L @ Cov(z) @ L.T = L @ I @ L.T = L @ L.T = corr_matrix

        IMPLEMENTATION:
        ================

        Args:
            timestamp: Current timestamp (for spike logic)
            time_factor: Multiplier from business hours/weekend (0.5 - 2.0)
            trend_factor: Growth over time (0.0 = start, 1.0 = end)

        Returns:
            Dictionary with cpu_p95, mem_p95, disk_p95, net_in_p95, net_out_p95
        """

        # ====================================================================
        # STEP 1: BUILD CORRELATION MATRIX
        # ====================================================================
        # 4x4 matrix for [CPU, Memory, Disk, Network]
        # Diagonal = 1.0 (each metric correlates perfectly with itself)
        # Off-diagonal = correlation coefficients from profile

        corr_matrix = np.array([
            # CPU row
            [1.0, self.profile.cpu_memory_correlation, 0.1, self.profile.cpu_network_correlation],
            # Memory row (symmetric, so cpu_mem appears again)
            [self.profile.cpu_memory_correlation, 1.0, self.profile.memory_disk_correlation, 0.2],
            # Disk row
            [0.1, self.profile.memory_disk_correlation, 1.0, 0.3],
            # Network row
            [self.profile.cpu_network_correlation, 0.2, 0.3, 1.0]
        ])

        # WHY HARD-CODED VALUES (0.1, 0.2, 0.3)?
        # These are weak correlations for pairs we don't explicitly model
        # e.g., CPU-Disk correlation is generally weak (0.1)
        # Could be made configurable if needed, but these are reasonable defaults

        # ====================================================================
        # STEP 2: CHOLESKY DECOMPOSITION
        # ====================================================================
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            # FAILURE CASE: Matrix not positive definite
            # WHY HAPPENS: Invalid correlation values (impossible combinations)
            # FALLBACK: Use identity matrix (no correlation)
            L = np.eye(4)
            # NOTE: This shouldn't happen with our profiles, but defensive coding

        # ====================================================================
        # STEP 3: GENERATE INDEPENDENT RANDOM VARIABLES
        # ====================================================================
        # Generate 4 independent standard normal variables
        # Each: mean=0, std=1, independent
        z = self.rng.randn(4)

        # ====================================================================
        # STEP 4: TRANSFORM TO CORRELATED
        # ====================================================================
        # Matrix multiplication: L @ z
        # Result: 4 values with desired correlation structure
        correlated = L @ z

        # ====================================================================
        # STEP 5: SCALE AND ADD TO BASE
        # ====================================================================
        # Formula: metric = base * time_factor * (1 + trend) + random * variance
        #
        # COMPONENTS:
        # - base: Mean value from profile (e.g., 45 for CPU)
        # - time_factor: Business hours/weekend adjustment (0.5 - 2.0)
        # - trend: Linear growth over time (0.0 - 1.0)
        # - growth_rate: Monthly growth rate from profile (e.g., 0.5%)
        # - random: Correlated random value from step 4
        # - variance: Standard deviation from profile (e.g., 15 for CPU)
        #
        # EXAMPLE:
        # CPU = 45 * 1.6 * (1 + 0.5 * 0.005) + correlated[0] * 15
        #     = 72 * 1.0025 + (random value) * 15
        #     = ~72.2 + random variation

        cpu = (
            self.profile.base_cpu * time_factor *
            (1 + trend_factor * self.profile.monthly_growth_rate / 100)
            + correlated[0] * self.profile.cpu_variance
        )

        memory = (
            self.profile.base_memory * time_factor *
            (1 + trend_factor * self.profile.monthly_growth_rate / 100)
            + correlated[1] * self.profile.memory_variance
        )

        # NOTE: Disk doesn't use time_factor (not affected by business hours)
        # WHY: Disk usage grows with data, not with request volume
        disk = (
            self.profile.base_disk *
            (1 + trend_factor * self.profile.monthly_growth_rate / 100)
            + correlated[2] * self.profile.disk_variance
        )

        network_base = (
            self.profile.base_network * time_factor
            + correlated[3] * self.profile.network_variance
        )

        # ====================================================================
        # STEP 6: APPLY SPIKES (PROBABILISTIC)
        # ====================================================================
        # Random spike with specified probability
        # IMPLEMENTATION: Roll dice, if < probability, apply spike

        if self.rng.rand() < self.profile.spike_probability:
            # SPIKE TRIGGERED!
            spike_mult = self.profile.spike_magnitude

            # Apply spike to CPU (full magnitude)
            cpu *= spike_mult

            # Apply spike to memory (70% of magnitude)
            # WHY 70%: Memory spikes less than CPU (doesn't grow as fast)
            memory *= (spike_mult * 0.7)

            # Apply spike to network (80% of magnitude)
            network_base *= (spike_mult * 0.8)

            # NOTE: Disk not spiked (disk usage doesn't spike suddenly)

        # ====================================================================
        # STEP 7: CLIP TO VALID RANGES
        # ====================================================================
        # Ensure values are within realistic bounds
        # CPU/Memory/Disk: 0-100%
        # Network: 0-1000 Mbps (arbitrary but realistic max)

        cpu = np.clip(cpu, 0, 100)
        memory = np.clip(memory, 0, 100)
        disk = np.clip(disk, 0, 100)
        network_in = np.clip(network_base, 0, 1000)

        # Network out is typically 60% of network in
        # WHY: Asymmetric - servers receive requests, send responses
        # Responses are often smaller than requests (compressed, aggregated)
        network_out = np.clip(network_base * 0.6, 0, 600)

        # ====================================================================
        # RETURN RESULTS
        # ====================================================================
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

        COMBINES:
        1. Business hours effect (9 AM - 5 PM)
        2. Weekend effect (Sat-Sun)

        RETURNS: Combined multiplier

        EXAMPLES:
        - Weekday 10 AM for web server: 1.6 * 1.0 = 1.6 (60% increase)
        - Weekday 10 PM for web server: 1.0 * 1.0 = 1.0 (normal)
        - Saturday 2 PM for web server: 1.6 * 0.5 = 0.8 (20% decrease overall)
        - Sunday 2 PM for batch server: 0.8 * 1.2 = 0.96 (net decrease)

        WHY MULTIPLY: Effects compound (busy time + busy day = very busy)
        """
        hour = timestamp.hour          # 0-23
        day_of_week = timestamp.dayofweek  # 0 (Mon) - 6 (Sun)

        # Business hours factor (9 AM - 5 PM)
        if 9 <= hour <= 17:
            bh_factor = self.profile.business_hours_factor
        else:
            bh_factor = 1.0

        # Weekend factor (Saturday, Sunday)
        if day_of_week >= 5:
            weekend_factor = self.profile.weekend_factor
        else:
            weekend_factor = 1.0

        # Combine factors (multiplicative)
        return bh_factor * weekend_factor
```

### 1.4 Factory Functions

```python
def get_archetype(server_type: str, server_id: str) -> ServerArchetype:
    """
    Factory function to get a server archetype.

    DESIGN PATTERN: Factory Method
    WHY: Centralizes object creation, allows for easy extension

    IMPLEMENTATION:
    - Maps string names to ServerType enum
    - Creates and returns ServerArchetype instance
    - Supports aliases (e.g., 'db' → DATABASE)

    Args:
        server_type: String type ('web', 'database', 'application', 'batch')
        server_id: Unique server identifier

    Returns:
        ServerArchetype instance

    Raises:
        ValueError: If server_type is not recognized
    """
    # Mapping of string names to enum values
    # Includes aliases for convenience
    type_map = {
        'web': ServerType.WEB,
        'database': ServerType.DATABASE,
        'db': ServerType.DATABASE,        # Alias
        'application': ServerType.APPLICATION,
        'app': ServerType.APPLICATION,    # Alias
        'batch': ServerType.BATCH,
    }

    server_type_lower = server_type.lower()

    if server_type_lower not in type_map:
        raise ValueError(
            f"Unknown server type '{server_type}'. "
            f"Valid types: {list(type_map.keys())}"
        )

    return ServerArchetype(type_map[server_type_lower], server_id)


def assign_archetypes_to_fleet(
    num_servers: int,
    distribution: Dict[str, float] = None
) -> Dict[str, str]:
    """
    Assign archetypes to a fleet of servers based on distribution.

    ALGORITHM:
    1. For each archetype, calculate count = num_servers * proportion
    2. Assign that many servers to the archetype
    3. Handle rounding (any remaining servers go to most common archetype)

    EXAMPLE:
    num_servers=120, distribution={'web': 0.35, 'app': 0.40, 'db': 0.15, 'batch': 0.10}

    Counts:
    - web: 120 * 0.35 = 42
    - app: 120 * 0.40 = 48
    - db: 120 * 0.15 = 18
    - batch: 120 * 0.10 = 12
    Total: 42 + 48 + 18 + 12 = 120 ✓

    Args:
        num_servers: Total number of servers
        distribution: Dictionary of archetype → proportion
                     If None, uses default enterprise distribution

    Returns:
        Dictionary mapping server_id → archetype_type
    """
    if distribution is None:
        # Default enterprise distribution
        # SOURCE: Typical datacenter composition based on workload analysis
        distribution = {
            'web': 0.35,          # Frontend tier
            'application': 0.40,  # Business logic tier (largest)
            'database': 0.15,     # Data tier (fewer but larger instances)
            'batch': 0.10,        # Background processing
        }

    # Validate distribution sums to 1.0
    total = sum(distribution.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Distribution must sum to 1.0, got {total}")

    # Calculate counts per archetype
    assignments = {}
    server_idx = 0

    for archetype, proportion in distribution.items():
        # Calculate count for this archetype
        count = int(num_servers * proportion)

        # Assign servers
        for _ in range(count):
            server_id = f"server_{server_idx:03d}"  # Format: server_000, server_001, ...
            assignments[server_id] = archetype
            server_idx += 1

    # Assign any remaining servers to the most common archetype
    # WHY NEEDED: Rounding can leave a few servers unassigned
    # Example: 121 servers, 0.35 proportion → 42.35 → 42 assigned, 1 remaining
    while server_idx < num_servers:
        server_id = f"server_{server_idx:03d}"
        most_common = max(distribution, key=distribution.get)
        assignments[server_id] = most_common
        server_idx += 1

    return assignments
```

---

## 2. data_generation.py - Complete Walkthrough

**File:** `src/data_generation.py` (390 lines)
**Purpose:** Main orchestration for synthetic data generation

### 2.1 Main Generation Function

```python
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

    ORCHESTRATION PATTERN:
    This is the "conductor" that coordinates all components:
    1. Load configuration
    2. Generate timestamp range
    3. Assign archetypes
    4. Generate metrics (nested loops: servers × timestamps)
    5. Enrich with metadata
    6. Add calendar features
    7. Validate and return

    PERFORMANCE CONSIDERATIONS:
    - Nested loops: O(servers × timestamps)
    - 120 servers × 1,461 timestamps = 175,320 iterations
    - Each iteration: ~0.1ms → Total: ~17 seconds
    - Memory: 18 MB for DataFrame
    - No parallelization (CPU-bound, overhead would dominate)

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

    # ====================================================================
    # PHASE 1: CONFIGURATION AND SETUP
    # ====================================================================
    logger.info("="*70)
    logger.info("Starting Synthetic Data Generation for AWS-CapacityForecaster")
    logger.info("="*70)

    # Load configuration from config.yaml
    data_config = get_data_config()

    # Use provided values or fall back to config
    # WHY OPTIONAL PARAMETERS: Flexibility (CLI, code, config all work)
    num_servers = num_servers or data_config.get('num_servers', 120)
    start_date = start_date or data_config.get('start_date', '2022-01-01')
    end_date = end_date or data_config.get('end_date', '2025-12-31')

    logger.info(f"Configuration:")
    logger.info(f"  Servers: {num_servers}")
    logger.info(f"  Date Range: {start_date} to {end_date}")
    logger.info(f"  Granularity: {granularity}")

    # ====================================================================
    # PHASE 2: GENERATE TIMESTAMP RANGE
    # ====================================================================
    # Create pandas DatetimeIndex with specified frequency
    # WHY PANDAS: Excellent datetime handling, timezone support, easy manipulation

    if granularity == 'daily':
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        # 'D' = calendar day frequency
        # Example: 2022-01-01, 2022-01-02, 2022-01-03, ...

    elif granularity == 'hourly':
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')
        # 'h' = hourly frequency (note: lowercase 'h' for pandas 2.0+)
        # Example: 2022-01-01 00:00, 2022-01-01 01:00, ...
        # WARNING: Hourly data is 24× larger! Memory/time impact

    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    logger.info(f"  Total timestamps: {len(date_range):,}")
    # Example: 1,461 days for 4 years

    # ====================================================================
    # PHASE 3: ASSIGN ARCHETYPES TO SERVERS
    # ====================================================================
    logger.info("\nAssigning server archetypes...")

    # This creates the mapping: server_id → archetype
    # Example: {'server_000': 'web', 'server_001': 'web', ...}
    archetype_assignments = assign_archetypes_to_fleet(num_servers)

    # Count servers per archetype (for logging)
    archetype_counts = {}
    for archetype in archetype_assignments.values():
        archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1

    logger.info("  Archetype distribution:")
    for archetype, count in sorted(archetype_counts.items()):
        logger.info(f"    {archetype}: {count} servers ({count/num_servers*100:.1f}%)")

    # ====================================================================
    # PHASE 4: GENERATE METRICS (MAIN LOOP)
    # ====================================================================
    # This is the heart of the generation process
    # NESTED LOOPS: For each server, for each timestamp, generate metrics

    logger.info("\nGenerating time-series metrics...")
    all_data = []  # Will accumulate all records

    # OUTER LOOP: Iterate over servers
    for server_id, archetype_type in archetype_assignments.items():

        # Progress logging every 20 servers
        # WHY: User feedback for long-running process
        if int(server_id.split('_')[1]) % 20 == 0:
            logger.info(f"  Processing {server_id}...")

        # Create archetype instance for this server
        # Each server gets its own archetype with deterministic random seed
        archetype = get_archetype(archetype_type, server_id)

        # INNER LOOP: Iterate over timestamps
        for idx, timestamp in enumerate(date_range):

            # ----------------------------------------------------------
            # CALCULATE TIME-BASED FACTORS
            # ----------------------------------------------------------

            # 1. Business hours / weekend factor
            time_factor = archetype.get_time_factor(timestamp)
            # Returns: 0.5 - 2.0 (typically)
            # Example: Friday 2 PM for web server → 1.6

            # 2. Trend factor (linear growth over time)
            trend_factor = idx / len(date_range)
            # Returns: 0.0 at start, 1.0 at end
            # Example: At 25% through dataset → 0.25

            # 3. Quarterly peak factor (banking-specific)
            qtr_factor = _get_quarterly_peak_factor(timestamp, data_config)
            # Returns: 1.0 (normal) to 1.3 (30% increase at quarter-end)
            # Example: March 31 → 1.3

            # 4. Holiday effect (reduced load)
            holiday_factor = _get_holiday_factor(timestamp, data_config)
            # Returns: 0.5 (50% reduction) to 1.0 (normal)
            # Example: December 25 → 0.6

            # ----------------------------------------------------------
            # COMBINE FACTORS
            # ----------------------------------------------------------
            # Multiplicative combination
            # WHY MULTIPLY: Effects compound
            # Example: Business hours + quarter-end = very busy
            combined_factor = time_factor * qtr_factor * holiday_factor
            # Range: ~0.3 (holiday weekend) to ~2.0 (peak business hours at quarter-end)

            # ----------------------------------------------------------
            # GENERATE CORRELATED METRICS
            # ----------------------------------------------------------
            metrics = archetype.generate_correlated_metrics(
                timestamp=timestamp,
                time_factor=combined_factor,
                trend_factor=trend_factor
            )
            # Returns: {cpu_p95, mem_p95, disk_p95, net_in_p95, net_out_p95}

            # ----------------------------------------------------------
            # CREATE RECORD
            # ----------------------------------------------------------
            record = {
                'timestamp': timestamp,
                'server_id': server_id,
                **metrics  # Unpack metrics dict into record
            }

            all_data.append(record)

    # ====================================================================
    # PHASE 5: BUILD DATAFRAME
    # ====================================================================
    logger.info("\nBuilding DataFrame...")

    # Convert list of dicts to DataFrame
    # PERFORMANCE: This is fast (C-level pandas operation)
    df = pd.DataFrame(all_data)

    # Set timestamp as index
    # WHY: Time-series operations are easier with datetime index
    df.set_index('timestamp', inplace=True)

    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # ====================================================================
    # PHASE 6: ADD BUSINESS METADATA
    # ====================================================================
    if include_metadata:
        logger.info("\nAdding business metadata...")

        # Generate metadata for all servers
        metadata_df = generate_server_metadata(n_servers=num_servers)
        # Returns: DataFrame with server_id, app_name, business_unit, criticality, region

        # Add archetype info to metadata
        # Map: server_id → archetype
        metadata_df['server_type'] = metadata_df['server_id'].map(archetype_assignments)

        # Merge with metrics
        # HOW: Join on server_id
        # RESULT: Each metric record gets metadata columns
        df = df.reset_index().merge(metadata_df, on='server_id', how='left').set_index('timestamp')

        logger.info(f"  Added columns: {list(metadata_df.columns)}")

    # ====================================================================
    # PHASE 7: ADD CALENDAR FEATURES
    # ====================================================================
    if include_calendar_features:
        logger.info("\nAdding calendar features...")

        # Add year, month, quarter, dayofweek, is_weekend, is_eoq, is_holiday
        df = add_calendar_features(df.reset_index(), date_col='timestamp').set_index('timestamp')

    # ====================================================================
    # PHASE 8: FINAL LOGGING AND RETURN
    # ====================================================================
    logger.info("\n" + "="*70)
    logger.info("Data Generation Complete")
    logger.info("="*70)
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Servers: {df['server_id'].nunique()}")
    logger.info(f"\nSample statistics:")
    logger.info(f"\n{df[['cpu_p95', 'mem_p95', 'disk_p95', 'net_in_p95', 'net_out_p95']].describe()}")

    return df
```

### 2.2 Helper Functions

```python
def _get_quarterly_peak_factor(timestamp: datetime, config: Dict) -> float:
    """
    Calculate quarterly peak factor for banking workloads.

    BUSINESS CONTEXT:
    Banking/financial services see significant load increases at end of quarter
    due to:
    - Regulatory reporting
    - Financial close processes
    - Reconciliation activities
    - Statement generation

    IMPLEMENTATION:
    - Last 5 days of quarter: Gradual ramp up
    - Last day (e.g., March 31): 30% increase
    - 5 days before: 6% increase
    - Linearly interpolated between

    WHY LAST 5 DAYS: Matches real enterprise patterns
    WHY 30% MAX: Realistic based on Citi observations

    Args:
        timestamp: Current timestamp
        config: Data configuration

    Returns:
        Multiplier (1.0 = normal, 1.3 = 30% increase)
    """
    # Check if feature is enabled
    if not config.get('seasonality', {}).get('quarterly_peaks', False):
        return 1.0

    # Extract date components
    month = timestamp.month
    day = timestamp.day

    # Check if we're in a quarter-end month (March, June, September, December)
    if month in [3, 6, 9, 12]:
        # Get days in this month (handles leap years)
        days_in_month = pd.Timestamp(timestamp.year, month, 1).days_in_month

        # Check if we're in the last 5 days
        if day >= days_in_month - 4:
            # Calculate proximity to quarter end
            days_from_end = days_in_month - day
            # days_from_end: 4 (5 days before) → 0 (last day)

            # Calculate intensity
            # Formula: 1.0 + (0.3 * (5 - days_from_end) / 5)
            #
            # Examples:
            # - 5 days before (day 27 in March): days_from_end=4
            #   intensity = 1.0 + (0.3 * (5-4) / 5) = 1.0 + 0.06 = 1.06
            # - 1 day before (day 30): days_from_end=1
            #   intensity = 1.0 + (0.3 * (5-1) / 5) = 1.0 + 0.24 = 1.24
            # - Last day (day 31): days_from_end=0
            #   intensity = 1.0 + (0.3 * (5-0) / 5) = 1.0 + 0.30 = 1.30

            peak_intensity = 1.0 + (0.3 * (5 - days_from_end) / 5)
            return peak_intensity

    # Not in quarter-end period
    return 1.0


def _get_holiday_factor(timestamp: datetime, config: Dict) -> float:
    """
    Calculate holiday effect factor (reduced load).

    BUSINESS CONTEXT:
    Banking/financial services see reduced activity during holidays:
    - Fewer transactions (markets closed)
    - Reduced staff (offices closed)
    - Maintenance windows (good time for upgrades)

    HOLIDAYS MODELED:
    - New Year's Day: 50% reduction (major holiday)
    - Christmas week: 40% reduction (extended holiday period)
    - July 4th: 30% reduction
    - Thanksgiving week: 30% reduction
    - Christmas Eve: 30% reduction

    WHY THESE REDUCTIONS: Based on historical traffic analysis

    Args:
        timestamp: Current timestamp
        config: Data configuration

    Returns:
        Multiplier (0.5 = 50% reduction, 1.0 = no effect)
    """
    # Check if feature is enabled
    if not config.get('seasonality', {}).get('holiday_effect', False):
        return 1.0

    year = timestamp.year
    month = timestamp.month
    day = timestamp.day

    # New Year's Day (Jan 1)
    if month == 1 and day == 1:
        return 0.5  # 50% reduction

    # Week between Christmas and New Year (Dec 25-31)
    if month == 12 and day >= 25:
        return 0.6  # 40% reduction

    # Independence Day (Jul 4)
    if month == 7 and day == 4:
        return 0.7  # 30% reduction

    # Thanksgiving week (approximate: Nov 22-28)
    # NOTE: Actual Thanksgiving is 4th Thursday, this is simplified
    if month == 11 and 22 <= day <= 28:
        return 0.7  # 30% reduction

    # Christmas Eve (Dec 24)
    if month == 12 and day == 24:
        return 0.7  # 30% reduction

    # Not a holiday
    return 1.0
```

---

## 5. Design Decisions Explained

### 5.1 Why Cholesky Decomposition?

**Problem:** Need to generate correlated random variables

**Alternative Approaches:**
1. **Rejection sampling** - Generate pairs, reject if correlation wrong (SLOW)
2. **Copulas** - Complex, overkill for Gaussian case
3. **Cholesky** - Fast, mathematically elegant, standard approach ✅

**Why Cholesky Wins:**
- O(n³) decomposition (one-time cost)
- O(n²) transformation (per sample)
- Mathematically guaranteed to preserve correlations
- Standard practice in quantitative finance, ML

### 5.2 Why Separate Archetypes?

**Alternative:** Single model with parameters

**Why Archetypes:**
- **Clearer code** - Each archetype is self-contained
- **Easier to extend** - Add new archetype without changing existing
- **Better documentation** - Each archetype documents real-world behavior
- **Easier testing** - Test each archetype independently

### 5.3 Why Deterministic Random Seeds?

**Alternative:** Truly random generation

**Why Deterministic:**
- **Reproducibility** - Same input always gives same output
- **Debugging** - Can replay exact scenario
- **Testing** - Predictable behavior for unit tests
- **Auditing** - Can verify data generation process

---

## Navigation

📄 [◀ Back to Index](README.md)
📄 [Next: Function Catalog ▶](03_Function_Catalog.md)

---

**Document End - Part 2 of 5**
**Status:** ✅ Complete
