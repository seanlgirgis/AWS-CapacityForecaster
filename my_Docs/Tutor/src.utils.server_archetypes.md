`src/utils/server_archetypes.py`

It's a really well-structured, realistic module — exactly the kind of thoughtful synthetic data helper that makes your capacity forecasting project feel enterprise-grade and Citi-inspired.

Since you want to go **slow and tutorial-style** (like I'm sitting next to you explaining a new codebase), we'll break it down in **small, bite-sized chunks** — 10–20 lines at a time.

I'll explain:
- What the code is doing
- Why it's useful for **AWS-CapacityForecaster** (tying back to ML forecasting, capacity risk, seasonal patterns)
- Key Python concepts used (assuming beginner-to-intermediate level)
- Any Citi/real-world connection I see

Then you tell me:
- "next chunk" (or "explain more about X")
- or ask questions about anything confusing

Let's start right at the top.

### Chunk 1: Docstring + Imports + Logger (lines ~1–15)

```python
""" server_archetypes.py
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
```

**Explanation step-by-step:**

1. **Triple-quoted docstring** at the very top  
   → This is the module-level documentation. Anyone who does `help(server_archetypes)` or reads the file sees this immediately.  
   → It clearly states the **purpose**: realistic synthetic server behaviors (not just random numbers — correlated, time-sensitive, growing over months).  
   → Explicitly mentions **your Citi experience** — great for portfolio/resume storytelling.

2. **Usage example**  
   → Shows exactly how someone (or future-you) should use this file. Very professional.

3. **Imports**
   - `logging` → standard way to log messages (debug, info, warning, error). Later code uses `logger.debug(...)` etc.
   - `numpy as np` → core for random numbers, correlations, math. Almost every serious data/ML project uses it.
   - `typing` → `Dict`, `List`, `Tuple` → modern type hints (helps IDEs, readability, mypy if you use it).
   - `dataclasses.dataclass` → clean way to create simple classes that are mostly just data containers (less boilerplate than regular class).
   - `enum.Enum` → creates nice named constants (e.g. `ServerType.WEB` instead of magic string "web").

4. **logger = logging.getLogger(__name__)**  
   → `__name__` is automatically `"src.utils.server_archetypes"` when imported, or `"__main__"` if run directly.  
   → This is the **recommended** way to get a logger per module — allows hierarchical control (e.g. turn on debug only for this module).  
   → Later you'll see `logger.debug(...)` — very useful for tracing without cluttering print().

**Why this matters for the project**  
This file is the foundation of **realistic synthetic data**. In real Citi work (and now here), servers aren't identical — web servers spike with traffic, databases are memory-heavy, batch jobs run off-hours with huge spikes.  
By modeling different **archetypes**, your time-series data becomes heterogeneous → forecasting models learn more realistic patterns → better risk flagging (e.g. "this database is seasonally at risk in Q4").

Great — thanks for the "next" signal.  
Now that we have the full current code (pulled fresh from your public repo), we'll continue the tutorial exactly where we left off.

We finished **Chunk 1**: docstring, imports, logger setup.

### Chunk 2: ServerType Enum (lines ~17–22)

```python
class ServerType(Enum):
    """Server archetype types."""
    WEB = "web"
    DATABASE = "database"
    APPLICATION = "application"
    BATCH = "batch"
```

**Line-by-line breakdown (beginner-friendly):**

- `class ServerType(Enum):`  
  → We're defining a new class that **inherits** from `Enum` (imported earlier from `enum`).  
  → Enums are perfect when you have a fixed set of named options that shouldn't change (like server categories).  
  → Instead of using raw strings everywhere ("web", "database"), you use `ServerType.WEB` — safer, auto-complete friendly in IDEs, and typos become obvious.

- `"""Server archetype types."""`  
  → Short docstring for the enum itself (good habit).

- The four lines:  
  ```python
  WEB = "web"
  DATABASE = "database"
  APPLICATION = "application"
  BATCH = "batch"
  ```
  → These create enum members.  
  → `ServerType.WEB` is an enum object whose `.value` is the string `"web"`.  
  → You can compare them safely: `if server.type == ServerType.DATABASE:`  
  → Or get the string when needed: `server.type.value` → `"database"`

**Why this is smart for your project (Citi / capacity planning connection)**  
In real enterprise environments (like Citi's thousands of servers), you group machines into workload types because each behaves differently:
- Web servers → traffic-driven spikes during business hours
- Databases → memory & disk heavy, steadier but grow over time
- App servers → balanced, business-hour sensitive
- Batch → night/weekend spikes, very bursty

Using an `Enum` makes the code self-documenting and prevents bugs like someone accidentally typing `'webb'` instead of `'web'`.

**Quick test you can run mentally:**
```python
print(ServerType.WEB)          # → ServerType.WEB
print(ServerType.WEB.value)    # → 'web'
print(ServerType.DATABASE == ServerType.DATABASE)  # True
```

Great — we're moving at a nice steady pace.  

We just covered **Chunk 2**: the `ServerType` Enum.  

Now let's go to the next logical small piece: the `@dataclass` that defines the **profile** for each archetype. This is where the real "personality" of each server type starts to come to life.

### Chunk 3: ArchetypeProfile dataclass (lines ~24–50ish in current version)

```python
from dataclasses import dataclass, field

@dataclass
class ArchetypeProfile:
    """Configuration and behavioral profile for a specific server archetype."""
    
    # Base utilization levels (mean daily P95 % over long term)
    cpu_base: float = 25.0
    memory_base: float = 40.0
    disk_io_base: float = 15.0    # in MB/s, normalized
    network_base: float = 20.0    # in % of capacity
    
    # Daily/weekly variance (standard deviation as % of base)
    cpu_std: float = 8.0
    memory_std: float = 6.0
    disk_io_std: float = 12.0
    network_std: float = 10.0
    
    # Seasonal / business-cycle multipliers (peak factors)
    seasonal_peak_factor: float = 1.4     # e.g. end-of-quarter / black-friday style
    seasonal_trough_factor: float = 0.65  # quiet periods
    
    # Correlation structure (how metrics move together)
    # Values between -1 and 1; positive = move together, negative = trade-off
    correlation_cpu_memory: float = 0.75
    correlation_cpu_network: float = 0.60
    correlation_memory_disk: float = 0.40
    
    # Growth trend over years (annual % increase in baseline)
    annual_growth_rate_cpu: float = 0.12   # 12% YoY typical enterprise
    annual_growth_rate_memory: float = 0.18
    annual_growth_rate_disk_io: float = 0.09
    annual_growth_rate_network: float = 0.14
    
    # Workload timing patterns
    diurnal_peak_hour_start: int = 9
    diurnal_peak_hour_end: int = 17
    batch_night_window: bool = False      # true for batch jobs
```

**Step-by-step explanation (keeping it beginner-friendly but realistic):**

1. `@dataclass` decorator  
   → This is a Python 3.7+ feature (from `dataclasses` module we imported earlier).  
   → It automatically creates `__init__`, `__repr__`, `__eq__`, etc. for you — so you don’t have to write a long class manually.  
   → Perfect for "data containers" like this — lots of attributes, little behavior.

2. All the fields (e.g. `cpu_base: float = 25.0`)  
   → These are the **default values** for a generic server of this type.  
   → You can override them later when you create a specific profile.  
   → Type hints (`: float`) are optional but excellent for clarity + IDE support.

3. **Base utilization levels** (`cpu_base`, `memory_base`, etc.)  
   → Realistic starting points for **P95 daily** metrics (95th percentile — common in enterprise monitoring like BMC TrueSight / AppDynamics).  
   → Citi-style: most production servers run 20–50% average, but P95 captures the busy moments.

4. **Variance** (`*_std`)  
   → How much the metric normally fluctuates day-to-day.  
   → Higher std → spikier / less predictable workload (batch = high disk/network variance).

5. **Seasonal factors**  
   → `seasonal_peak_factor = 1.4` → during peak periods (end-of-quarter, holidays), baseline × 1.4.  
   → `seasonal_trough_factor = 0.65` → quiet times drop to 65% of normal.  
   → Very Citi-relevant: banking has strong quarterly/annual cycles.

6. **Correlations**  
   → This is gold for realism.  
   → `correlation_cpu_memory = 0.75` → when CPU is high, memory usually is too (common in web/app servers).  
   → Negative values (not used here yet) could model trade-offs (e.g. high network → lower disk).  
   → Later code will use these to generate **correlated random numbers** (via multivariate normal distribution — very common in synthetic enterprise data).

7. **Annual growth rates**  
   → Simulates organic business growth → apps get more users, databases grow, etc.  
   → Memory grows fastest (18% YoY) — very realistic in virtualized/cloud environments.

8. **Diurnal / batch patterns**  
   → `diurnal_peak_hour_start/end` → business hours (9–17) see higher load.  
   → `batch_night_window` → flag for overnight heavy jobs (will invert the diurnal pattern).

**Why this chunk is so powerful for your project goals**

- **#1 Data Science/ML**: These parameters create rich, **non-stationary**, **correlated**, **seasonal** time series — perfect challenge for Prophet, XGBoost, feature engineering (lags, rolling stats, holiday flags, growth trend).  
- **#2 Capacity Planning**: Realistic heterogeneity → better risk detection (some servers hit P95=95% during peaks), seasonal flagging, underutilized clustering.  
- Citi connection: mirrors the kinds of workload patterns you observed across thousands of endpoints.



