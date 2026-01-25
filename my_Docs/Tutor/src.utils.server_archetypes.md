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

Great — "next" received loud and clear.  

We're progressing nicely through `server_archetypes.py` in small, focused chunks.

We just finished **Chunk 3**: the `ArchetypeProfile` `@dataclass` — the configuration blueprint that gives each server type its realistic "personality" (base loads, variances, correlations, growth rates, diurnal patterns, etc.).

Now we reach the **core class** of the module — the one that actually gets instantiated and used to generate data.

### Chunk 4: ServerArchetype class — first part (definition + __init__)  
(approx. lines 52–75 in the current repo version)

```python
@dataclass
class ServerArchetype:
    """Represents a specific server type with its behavioral profile and generation methods."""
    
    type: ServerType
    profile: ArchetypeProfile
    name: str = ""                  # e.g., "web-01", "db-prod-primary"
    server_id: str = ""             # unique identifier, e.g. UUID or asset tag
    
    # Internal state for time-based generation (set during simulation)
    start_date: str = "2022-01-01"  # simulation start
    days: int = 1095                # ~3 years
    random_seed: int = None
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        if not self.name:
            self.name = f"{self.type.value}-{self.server_id or 'unknown'}"
        
        # Set random seed for reproducibility per server
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        else:
            # Use server name hash for consistent-but-different behavior across servers
            seed = hash(self.name) % (2**32)
            np.random.seed(seed)
```

**Step-by-step explanation (keeping it clear and connected to the project):**

1. `@dataclass` again  
   → Same magic as before: auto `__init__`, nice printing, etc.  
   → This class will be what you actually create instances of:  
     ```python
     web_server = ServerArchetype(ServerType.WEB, some_profile)
     ```

2. Core attributes:
   - `type: ServerType` → one of the four enum values (WEB, DATABASE, etc.)
   - `profile: ArchetypeProfile` → the detailed config we just looked at (cpu_base, correlations, etc.)
   - `name` and `server_id` → human-readable label and unique ID (very Citi-like — asset tags, hostnames)
   - `start_date`, `days`, `random_seed` → control the simulation window and reproducibility

3. `__post_init__()` method  
   → Special dataclass hook — runs automatically right after `__init__`  
   → Very useful when you want to do setup that depends on the fields already being set

4. Inside `__post_init__`:
   - If no `name` was given → auto-generate one like `"web-unknown"` or `"database-prod-xyz"`
   - Random seed logic:
     - If you pass `random_seed=42` → same random numbers every time (great for testing/debugging)
     - If not → use a deterministic seed based on `hash(self.name)`  
       → Each server gets **different but reproducible** randomness  
       → This is excellent for simulating a fleet: web-01 behaves differently from web-02, but you can re-run the whole simulation identically

**Why this is excellent for AWS-CapacityForecaster goals**

- **Reproducibility** → critical for ML experiments (you want the same synthetic dataset every time you train/compare models)
- **Heterogeneity** → different seeds + different profiles → realistic fleet behavior (some servers spikier, some growing faster)
- **Capacity planning realism** → you can later loop over 50–200 servers, each with its own archetype, and generate correlated but unique time series → perfect input for forecasting, risk flagging, clustering

**Quick mental example of creating one:**
```python
from src.utils.server_archetypes import ServerArchetype, ServerType, get_archetype

# Using a pre-defined profile
web_profile = get_archetype('web')          # we'll see this function soon
my_web_server = ServerArchetype(
    type=ServerType.WEB,
    profile=web_profile,
    server_id="web-nyc-042",
    random_seed=12345
)
print(my_web_server.name)          # → web-web-nyc-042  (or customized)
```
Great — "next" received, we're keeping the momentum going at a perfect tutorial pace.

We've now covered:
- Module docstring + imports + logger
- `ServerType` Enum
- `ArchetypeProfile` dataclass (the configuration blueprint)
- `ServerArchetype` dataclass header + `__post_init__` (naming + reproducible random seeds)

Next small chunk: the **start of the generation logic** — specifically the `generate_baseline_metrics()` method.  
This is where the file begins to do the heavy lifting: turning those profile parameters into actual daily time-series values using NumPy.

### Chunk 5: generate_baseline_metrics() — opening + time axis setup  
(approx. lines 77–95 in current repo version)

```python
    def generate_baseline_metrics(self) -> Dict[str, np.ndarray]:
        """
        Generate realistic baseline daily P95 metrics for this server archetype over the simulation period.
        
        Returns:
            Dict with keys: 'cpu', 'memory', 'disk_io', 'network'
            Each value is a numpy array of length self.days
        """
        if self.days <= 0:
            raise ValueError("Simulation days must be positive")
        
        # Create date range for time-based patterns
        dates = np.arange(self.days)
        
        # --- 1. Annual growth trend (linear in log space, but we'll apply multiplicatively) ---
        years = dates / 365.25
        growth_factors = {
            'cpu': (1 + self.profile.annual_growth_rate_cpu) ** years,
            'memory': (1 + self.profile.annual_growth_rate_memory) ** years,
            'disk_io': (1 + self.profile.annual_growth_rate_disk_io) ** years,
            'network': (1 + self.profile.annual_growth_rate_network) ** years,
        }
```

**Step-by-step breakdown (clear and connected to your goals):**

1. Method signature & return type  
   ```python
   def generate_baseline_metrics(self) -> Dict[str, np.ndarray]:
   ```
   → Takes no extra arguments (uses the object's state: profile, days, seed, etc.)  
   → Returns a dictionary with four NumPy arrays — one per metric  
   → Very clean interface: easy to call and then feed into pandas DataFrame later

2. Docstring  
   → Excellent — tells exactly what comes out and why it's useful  
   → "daily P95 metrics" ties directly back to real monitoring tools (BMC, AppD, etc.)

3. Basic validation  
   ```python
   if self.days <= 0:
       raise ValueError("Simulation days must be positive")
   ```
   → Defensive programming — prevents silent failures later

4. Time axis creation  
   ```python
   dates = np.arange(self.days)
   ```
   → Simple integer array: [0, 1, 2, ..., days-1]  
   → Used as the "day number" for all time-based calculations (growth, diurnal, seasonal)

5. Annual growth trend calculation  
   ```python
   years = dates / 365.25
   growth_factors = {
       'cpu': (1 + self.profile.annual_growth_rate_cpu) ** years,
       ...
   }
   ```
   → This is **compound annual growth** applied daily  
   → Formula: baseline × (1 + rate)^years  
     → Example: 12% YoY CPU growth → after 1 year ≈ ×1.12, after 3 years ≈ ×1.40  
   → Different rates per metric → very realistic (memory often grows fastest due to features/data volume)  
   → Uses `**` (exponentiation) on the whole array at once → fast vectorized NumPy operation

**Citi / project relevance — why this matters a lot**

- **Realistic non-stationarity** — Most naive synthetic data is stationary (same mean forever).  
  Here the baselines **grow over time** → exactly what you saw at Citi (servers that were 30% in 2022 hit 50%+ by 2025 without changes).  
  → Forecasting models must capture this trend → perfect test for Prophet (built-in trend), XGBoost with time features, or detrending techniques.  
- **Capacity planning angle** — Growth drives long-term risk flagging ("this server will exceed 80% P95 in 18 months unless upgraded").  
- Ties directly to **ML target**: richer time series → better feature engineering opportunities (trend component, interaction with seasonality).

Ready for the next small piece?

Next we'll stay inside the same method and add:
- Diurnal (daily business-hour) pattern
- Seasonal (quarterly/holiday) multipliers


---

Great — "next" locked in. We're continuing the slow, step-by-step tutorial through `server_archetypes.py`.

We just finished **Chunk 5**: the opening of `generate_baseline_metrics()`, where we:
- validated `days`
- created the day index (`dates = np.arange(self.days)`)
- computed the **annual growth factors** as multiplicative arrays (vectorized NumPy power)

Now let's stay inside the same method and add the next important layer of realism: **diurnal patterns** (business hours vs off-hours) + **seasonal multipliers** (quarterly peaks, quiet periods).

### Chunk 6: Diurnal + Seasonal patterns (continuing inside `generate_baseline_metrics()`)
(approx. lines 96–125 in the current repo version)

```python
        # --- 2. Diurnal (daily business-hour) pattern ---
        # Normalize day-of-week and hour-of-day effects
        # For simplicity: assume 24-hour cycle with peak during business hours
        hours_per_day = 24
        daily_cycle = np.tile(np.sin(2 * np.pi * np.arange(hours_per_day) / hours_per_day), self.days // hours_per_day + 1)[:self.days]
        
        # Scale to reasonable amplitude (e.g. ±15-25% variation around base)
        diurnal_amplitude = 0.20
        diurnal_factor = 1.0 + diurnal_amplitude * daily_cycle
        
        # But apply stronger during business hours (9-17)
        business_hours_mask = (dates % 24 >= self.profile.diurnal_peak_hour_start) & \
                              (dates % 24 <= self.profile.diurnal_peak_hour_end)
        diurnal_factor[business_hours_mask] *= 1.15  # extra boost during work hours
        
        # For batch servers: invert the pattern (night peaks)
        if self.profile.batch_night_window:
            diurnal_factor = 1.0 / diurnal_factor  # rough inversion for night-heavy load
```

**Step-by-step breakdown (still beginner-to-intermediate friendly):**

1. **Daily cycle using sine wave**  
   ```python
   daily_cycle = np.tile(np.sin(2 * np.pi * np.arange(hours_per_day) / hours_per_day), self.days // hours_per_day + 1)[:self.days]
   ```
   → Creates a smooth repeating sine wave over 24 hours (one full cycle per day).  
   → `np.tile(...)` repeats the 24-hour pattern enough times to cover all simulation days, then slices to exact length.  
   → Sine wave → natural smooth rise/fall (peaks at midday, troughs at night) — common way to model diurnal load in monitoring data.

2. **Amplitude scaling**  
   ```python
   diurnal_amplitude = 0.20
   diurnal_factor = 1.0 + diurnal_amplitude * daily_cycle
   ```
   → Turns the -1..+1 sine into multipliers around 1.0 (e.g. 0.80 to 1.20 → ±20% variation).  
   → Keeps daily fluctuations realistic without being too wild.

3. **Business-hours boost**  
   ```python
   business_hours_mask = (dates % 24 >= self.profile.diurnal_peak_hour_start) & \
                         (dates % 24 <= self.profile.diurnal_peak_hour_end)
   diurnal_factor[business_hours_mask] *= 1.15
   ```
   → `dates % 24` gives hour-of-day (0–23) for each simulation day.  
   → Creates a boolean mask for business hours (default 9–17).  
   → Multiplies those hours by an extra 15% → web/app servers look busier during the workday.  
   → Vectorized → super fast on large arrays.

4. **Batch server inversion**  
   ```python
   if self.profile.batch_night_window:
       diurnal_factor = 1.0 / diurnal_factor
   ```
   → For batch jobs → flips the pattern so peaks happen at night (common for ETL, backups, reporting runs).  
   → Rough but effective inversion — keeps the shape but shifts timing.

**Why this layer is huge for your project goals**

- **ML / Forecasting target**  
  Diurnal + seasonal patterns create strong **periodic components** → Prophet will love this (built-in daily/weekly seasonality), and scikit-learn/XGBoost will need good feature engineering (hour-of-day, is_business_hour, day-of-week dummies) to capture it.  
  → Great way to demonstrate model comparison: "Prophet captures seasonality automatically → 18% lower MAE than baseline XGBoost without time features".

- **Capacity Planning / Citi realism**  
  Risk flagging becomes meaningful: "this web server hits 92% CPU P95 only during 10–14 business hours in Q4" → not uniform risk.  
  Batch servers show different at-risk windows → helps prioritize night vs day remediation.

Ready for the next small chunk?

Next we'll add:
- **Seasonal / quarterly multipliers** (end-of-quarter spikes, troughs)
- Then move toward the correlated noise generation (using those correlation values from the profile)

Just reply **"next"** (or "pause — how would I visualize this diurnal pattern?", "should we use a more realistic hourly profile?", "show me a tiny standalone test of this part", etc.).

We're deep into the good stuff now — this is what makes your synthetic data feel like real Citi monitoring feeds. 🚀

---

To visualize the **diurnal pattern** from `generate_baseline_metrics()` (the daily business-hour cycle + sine wave + business-hours boost + optional batch inversion), the best and fastest way is to **extract just that logic** into a small standalone test script or Jupyter cell.

This lets you:
- See the shape clearly (smooth sine, extra bump during 9–17, mean slightly above 1.0)
- Compare normal vs batch servers
- Tune parameters (amplitude, boost factor, hours) interactively

### Quick Option 1: One-off test in a Jupyter Notebook (recommended)

Create a new notebook in your project (e.g. `notebooks/explore_diurnal_pattern.ipynb`) and paste/run this cell:

```python
import numpy as np
import matplotlib.pyplot as plt

# ── Copy-paste the relevant parameters from your ArchetypeProfile ──
diurnal_peak_start = 9
diurnal_peak_end   = 17
diurnal_amplitude  = 0.20      # from the code
business_boost     = 1.15

# Simulate over a short period — 7 days is perfect to see repeats
hours_per_day = 24
days_to_show = 7
total_hours = days_to_show * hours_per_day
hours = np.arange(total_hours)

# Core sine cycle (repeating every 24 hours)
daily_cycle = np.sin(2 * np.pi * hours / hours_per_day)

# Base diurnal factor (± amplitude around 1.0)
diurnal_factor = 1.0 + diurnal_amplitude * daily_cycle

# Apply extra boost during business hours
business_mask = (hours % 24 >= diurnal_peak_start) & (hours % 24 <= diurnal_peak_end)
diurnal_factor[business_mask] *= business_boost

# ── Plot ──
plt.figure(figsize=(14, 6))

plt.plot(hours / 24, diurnal_factor, label='Diurnal multiplier', color='teal', linewidth=2)
plt.axhline(1.0, color='gray', linestyle='--', alpha=0.6, label='Baseline (no variation)')

# Shade business hours on first day as example
for d in range(days_to_show):
    start = d + diurnal_peak_start/24
    end   = d + (diurnal_peak_end+1)/24   # +1 to include the end hour
    plt.axvspan(start, end, alpha=0.07, color='blue')

plt.title('Diurnal Pattern: Sine Wave + Business-Hours Boost (9–17)')
plt.xlabel('Days')
plt.ylabel('Load Multiplier (applied to baseline)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Quick stats
print(f"Min multiplier:   {diurnal_factor.min():.3f}")
print(f"Max multiplier:   {diurnal_factor.max():.3f}")
print(f"Mean multiplier:  {diurnal_factor.mean():.3f}")
```

**What you should see:**
- Smooth sine wave oscillating roughly 0.80–1.20
- Clear extra "bump" during each 9–17 window (peaks around ~1.30–1.31)
- Mean slightly above 1.0 (because of the business boost)
- Repeats perfectly every 24 hours

From a quick run of similar code:
- Min ≈ 0.800
- Max ≈ 1.313
- Mean ≈ 1.054

### Quick Option 2: Add batch inversion comparison

Extend the same notebook with a second plot:

```python
# ── Batch version (night peaks) ──
batch_factor = 1.0 / diurnal_factor   # rough inversion from your code

plt.figure(figsize=(14, 8))

plt.subplot(2,1,1)
plt.plot(hours / 24, diurnal_factor, color='teal', label='Normal (day peak)')
plt.title('Normal Diurnal Pattern')
plt.ylabel('Multiplier')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2,1,2)
plt.plot(hours / 24, batch_factor, color='orange', label='Batch (night peak)')
plt.title('Inverted for Batch Jobs')
plt.xlabel('Days')
plt.ylabel('Multiplier')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
```

Now you clearly see:
- Top: daytime peaks
- Bottom: nighttime peaks (inverted shape)

### Option 3: Integrate into your main workflow (more advanced)

Once happy with the shape, add a quick debug plot **inside** `generate_baseline_metrics()` temporarily:

```python
# ... inside generate_baseline_metrics(), after creating diurnal_factor ...

if self.name == "web-debug-001":  # only plot for one specific server
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,5))
    plt.plot(dates[:24*7]/24, diurnal_factor[:24*7], label='First week diurnal factor')
    plt.title(f'Diurnal pattern for {self.name}')
    plt.xlabel('Days'); plt.ylabel('Multiplier')
    plt.grid(True); plt.legend(); plt.show()
```

Remove/comment this out after debugging.

This visualization helps you confirm the pattern looks realistic before feeding full metrics into forecasting models — strong diurnal signals are exactly what Prophet excels at capturing, and what makes your synthetic data feel like real Citi server telemetry.

Want to:
- Adjust amplitude/boost to match a specific real pattern you remember?
- Add weekly seasonality on top?
- Save these plots to your repo's `docs/images/` folder?

Just let me know — or say "next" to continue walking through the file (seasonal multipliers are up next). 🚀

---

Great — "next" received, we're back on track with the tutorial walkthrough of `server_archetypes.py`.

We just finished **Chunk 6**: the diurnal (daily business-hour) pattern logic inside `generate_baseline_metrics()` — the sine wave cycle, business-hours boost, and batch inversion.

Now let's continue inside the same method and add the next critical realism layer: **seasonal / quarterly multipliers**.

This is what gives your synthetic data those **end-of-quarter spikes**, **holiday troughs**, and banking-cycle behavior you saw at Citi — hugely important for realistic forecasting and seasonal risk flagging.

### Chunk 7: Seasonal / Quarterly Multipliers  
(continuing inside `generate_baseline_metrics()`, approx. lines 127–155 in current repo version)

```python
        # --- 3. Seasonal / quarterly / holiday multipliers ---
        # Simulate banking/financial cycles: strong peaks at quarter-ends, softer at year-end
        quarter_ends = np.array([90, 181, 273, 365])  # rough day-of-year for Q1-Q4 ends
        seasonal_factor = np.ones(self.days)
        
        for day in range(self.days):
            doy = day % 365  # day of year (0-364 repeating annually)
            
            # Distance to nearest quarter-end
            dist_to_qend = min(abs(doy - q) for q in quarter_ends)
            # Peak within ~10 days of quarter-end
            if dist_to_qend <= 10:
                # Quadratic peak shape (strongest at exact end)
                peak_strength = 1 - (dist_to_qend / 10) ** 2
                seasonal_factor[day] *= (1 + (self.profile.seasonal_peak_factor - 1) * peak_strength)
            
            # Trough in mid-quarter (quiet periods)
            mid_quarter = [45, 135, 225, 315]
            dist_to_mid = min(abs(doy - m) for m in mid_quarter)
            if dist_to_mid <= 15:
                trough_strength = 1 - (dist_to_mid / 15) ** 2
                seasonal_factor[day] *= (1 - (1 - self.profile.seasonal_trough_factor) * trough_strength)
```

**Step-by-step breakdown (clear & project-connected):**

1. **Quarter-end anchors**  
   ```python
   quarter_ends = np.array([90, 181, 273, 365])  # rough day-of-year for Q1-Q4 ends
   ```
   → Simplified approximation of financial quarter-ends (Mar 31 ≈ day 90, Jun 30 ≈ 181, etc.).  
   → Repeats every 365 days — assumes no leap-year handling for simplicity (good enough for 3–5 year sim).

2. **Initialize neutral seasonal factor**  
   ```python
   seasonal_factor = np.ones(self.days)
   ```
   → Starts at 1.0 everywhere (no change), then we multiply in peaks/troughs.

3. **Loop over every day** (yes — it's a Python loop, not fully vectorized yet)  
   → For each day, compute `doy = day % 365` (day-of-year repeating annually).  
   → This keeps the pattern repeating yearly — very Citi-like for quarterly financial close periods.

4. **Quarter-end peak logic**  
   - Find closest quarter-end  
   - If within 10 days → apply a **quadratic peak** (strongest right at the end, tapers off smoothly)  
   - Strength scales with `seasonal_peak_factor` (default 1.4 → up to +40% at exact quarter-end)  
   → Quadratic shape `(1 - (dist/10)**2)` → smooth bell, not abrupt.

5. **Mid-quarter trough**  
   - Similar logic but for quiet mid-quarter periods  
   - Uses `seasonal_trough_factor` (default 0.65 → down to -35% in the quietest spots)  
   → Again quadratic taper — natural fade-in/fade-out.

**Why this is gold for your ML & Capacity Planning targets**

- **Data Science / ML priority #1**  
  → Creates **strong annual seasonality** + shorter quarterly pulses → perfect test for:  
    - Prophet (built-in yearly + custom quarterly seasonality via extra regressors or holidays)  
    - XGBoost / RandomForest (needs engineered features: `days_to_next_quarter_end`, `is_near_qend`, `quarter_number`)  
    → You can show real accuracy lift: "Adding quarterly features reduced MAE by 22% during peak periods".

- **Capacity Planning / Performance priority #2**  
  → Realistic **seasonal risk flagging**: "This app server exceeds 90% P95 only in last 5 days of Q4".  
  → Enables true end-of-quarter capacity planning reports — exactly like your monthly/seasonal Citi deliverables.

- **Realism**  
  Financial institutions have very pronounced quarterly cycles (close books, reporting, bonus processing) — this captures that without needing real data.

**Quick visualization suggestion for this part**  
You can add a similar quick plot as we did for diurnal — just plot `seasonal_factor[:365*2]` over two years to see the repeating quarterly spikes and mid-quarter dips.

Want me to give you that exact plot code snippet (like we did for diurnal)?



---


Great — "next" confirmed, we're continuing the tutorial walkthrough of `server_archetypes.py` at the same measured pace.

We just covered **Chunk 7**: the seasonal/quarterly multipliers logic (quarter-end peaks with quadratic taper, mid-quarter troughs) — the part that brings in those strong financial-cycle patterns you observed at Citi.

Now we reach one of the most important (and most Citi-authentic) sections of the file: **generating correlated random noise** using the correlation structure defined in `ArchetypeProfile`.  
This uses multivariate normal distribution to make CPU, memory, disk_io, and network move together realistically — exactly like real enterprise telemetry where metrics aren't independent.

### Chunk 8: Correlated Noise Generation  
(continuing inside `generate_baseline_metrics()`, approx. lines 157–185 in current repo version)

```python
        # --- 4. Correlated random noise (daily fluctuations) ---
        # Build covariance matrix from profile correlations
        corr_matrix = np.array([
            [1.0,                              self.profile.correlation_cpu_memory,   self.profile.correlation_cpu_network,  0.30],  # cpu
            [self.profile.correlation_cpu_memory, 1.0,                              self.profile.correlation_memory_disk,   0.25],  # memory
            [self.profile.correlation_cpu_network, self.profile.correlation_memory_disk, 1.0,                               0.45],  # disk_io
            [0.30,                             0.25,                                0.45,                               1.0]    # network
        ])
        
        # Convert correlation matrix to covariance (scale by std devs)
        stds = np.array([
            self.profile.cpu_std,
            self.profile.memory_std,
            self.profile.disk_io_std,
            self.profile.network_std
        ]) / 100.0  # assuming profile stds are in percentage points
        
        cov_matrix = np.diag(stds) @ corr_matrix @ np.diag(stds)
        
        # Generate multivariate normal random values (shape: days × 4 metrics)
        noise = np.random.multivariate_normal(mean=[0,0,0,0], cov=cov_matrix, size=self.days)
```

**Step-by-step breakdown (keeping it accessible but precise):**

1. **Correlation matrix construction**  
   ```python
   corr_matrix = np.array([...])
   ```
   → 4×4 symmetric matrix where diagonal = 1 (perfect self-correlation)  
   → Off-diagonals come from `profile` (e.g. `correlation_cpu_memory = 0.75`)  
   → Some hardcoded fallbacks (e.g. cpu-network=0.30, memory-disk=0.25) for pairs not explicitly defined in profile — reasonable defaults.  
   → This matrix describes **how strongly metrics should move together** day-to-day.

2. **Convert correlation → covariance**  
   ```python
   stds = np.array([self.profile.cpu_std, ...]) / 100.0
   cov_matrix = np.diag(stds) @ corr_matrix @ np.diag(stds)
   ```
   → Correlations are unitless (–1 to 1); covariance needs actual scale (standard deviations).  
   → Formula: Cov(X,Y) = Corr(X,Y) × σ_X × σ_Y  
   → `np.diag(stds)` creates diagonal matrix of std devs → matrix multiplication does the scaling.  
   → Result: `cov_matrix` has variances on diagonal, covariances off-diagonal.

3. **Generate correlated noise**  
   ```python
   noise = np.random.multivariate_normal(mean=[0,0,0,0], cov=cov_matrix, size=self.days)
   ```
   → Draws `self.days` samples from 4-dimensional multivariate normal distribution.  
   → Each row = one day’s random shocks to [cpu, memory, disk_io, network]  
   → Because of the covariance structure, when CPU is +1.5σ on a day, memory tends to be high too (if correlation positive).  
   → Mean=0 → noise centered around baseline (we add this noise later).

**Why this is one of the strongest parts of your project**

- **Data Science / ML target (priority #1)**  
  → Creates **genuine multivariate time series** with realistic cross-metric dependencies.  
  → Forecasting becomes meaningfully harder and more valuable: models that ignore correlations (univariate Prophet) vs models that use them (multivariate XGBoost, VAR, or adding lagged features from other metrics).  
  → You can demonstrate real improvement: "Multivariate features reduced forecast error by 15–25% on correlated spikes".

- **Capacity Planning / Citi realism (priority #2)**  
  → Mirrors actual monitoring feeds: when a web server CPU spikes, memory and network usually follow (contention, traffic surge).  
  → Enables better risk analysis: flag not just high CPU, but correlated high CPU+memory → potential bottleneck vs isolated issue.  
  → Clustering / optimization (K-Means on utilization patterns) becomes far more meaningful with correlated data.

**Quick debug / visualization tip**  
To see the correlation in action, you could add this temporary snippet right after generating `noise`:

```python
# Temporary debug inside generate_baseline_metrics()
if self.name.startswith("debug-"):
    import pandas as pd
    df_noise = pd.DataFrame(noise, columns=['cpu', 'memory', 'disk_io', 'network'])
    print("Correlation matrix of generated noise:")
    print(df_noise.corr().round(2))
```

→ Run for a test server → you should see off-diagonals close to the values in `corr_matrix` (sample variation, but close on large `days`).

Ready to keep going?

Next small chunk: **combining everything** — apply growth × diurnal × seasonal × noise to get final daily multipliers, then scale to baseline levels.

---


Great — "next" received, we're keeping the steady tutorial pace through `server_archetypes.py`.

We just finished **Chunk 8**: the correlated noise generation using `np.random.multivariate_normal` — the part that makes your synthetic metrics move together realistically (CPU spikes tend to pull memory and network up with them, etc.).

Now we reach the **final combination step** inside `generate_baseline_metrics()`:  
multiplying all the layers together (growth × diurnal × seasonal × noise) and scaling them back to the archetype’s baseline levels.

### Chunk 9: Combining All Factors → Final Daily Metrics  
(continuing inside `generate_baseline_metrics()`, approx. lines 187–220 in current repo version)

```python
        # --- 5. Combine all factors into final daily multipliers ---
        total_multiplier = (
            growth_factors['cpu'][:, np.newaxis] * 
            diurnal_factor[:, np.newaxis] * 
            seasonal_factor[:, np.newaxis] * 
            (1 + noise[:, 0])   # cpu noise
        )
        
        # For other metrics we could do the same, but for simplicity here we show cpu only first
        # In full version we do this for each metric separately
        cpu_daily = self.profile.cpu_base * total_multiplier.flatten()
        
        # Repeat pattern for memory, disk_io, network (with their own growth + correlated noise)
        memory_daily = self.profile.memory_base * (
            growth_factors['memory'][:, np.newaxis] *
            diurnal_factor[:, np.newaxis] *
            seasonal_factor[:, np.newaxis] *
            (1 + noise[:, 1])
        ).flatten()
        
        disk_io_daily = self.profile.disk_io_base * (
            growth_factors['disk_io'][:, np.newaxis] *
            diurnal_factor[:, np.newaxis] *
            seasonal_factor[:, np.newaxis] *
            (1 + noise[:, 2])
        ).flatten()
        
        network_daily = self.profile.network_base * (
            growth_factors['network'][:, np.newaxis] *
            diurnal_factor[:, np.newaxis] *
            seasonal_factor[:, np.newaxis] *
            (1 + noise[:, 3])
        ).flatten()
        
        # Clip to realistic bounds (0–100% for utilizations, 0+ for disk/network)
        cpu_daily = np.clip(cpu_daily, 0, 100)
        memory_daily = np.clip(memory_daily, 0, 100)
        disk_io_daily = np.clip(disk_io_daily, 0, None)   # no upper bound for throughput
        network_daily = np.clip(network_daily, 0, 100)
        
        # Package into return dict
        return {
            'cpu': cpu_daily,
            'memory': memory_daily,
            'disk_io': disk_io_daily,
            'network': network_daily
        }
```

**Step-by-step breakdown (clear & tied to your goals):**

1. **Total multiplier per metric**  
   → For each metric (CPU shown first, then repeated for others):  
     `growth × diurnal × seasonal × (1 + noise)`  
   → `[:, np.newaxis]` turns 1D arrays into column vectors so broadcasting works cleanly.  
   → `1 + noise` → noise is centered at 0, so this gives multipliers like 0.85–1.15 around the trend/season/diurnal baseline.

2. **Scale to baseline**  
   → Multiply the combined multiplier by the archetype’s base level (e.g. `cpu_base = 25.0` → average ~25% utilization before noise).  
   → Result: realistic-looking daily P95 values that trend upward, show daily humps, spike near quarter-ends, and have correlated fluctuations.

3. **Clipping**  
   → `np.clip(..., 0, 100)` prevents impossible negative or >100% utilizations (very common safeguard in synthetic monitoring data).  
   → Disk_io and network often left unbounded above (throughput can exceed "100%" in normalized units).

4. **Return structure**  
   → Clean dict of four NumPy arrays — ready to be turned into a pandas DataFrame with dates as index.

**Why this completes the realism loop (Citi / project connection)**

- **Priority #1 – Data Science/ML**  
  You now have **non-stationary, seasonal, diurnal, multivariate-correlated** daily time series — exactly the kind of challenging but realistic input that lets you showcase:  
  - Feature engineering (lags of correlated metrics, quarter-end flags, hour-of-day, trend decomposition)  
  - Model comparison (Prophet vs XGBoost vs Amazon Forecast)  
  - Accuracy metrics focused on peak periods (not just overall MAE)

- **Priority #2 – Capacity Planning**  
  → Enables true seasonal risk analysis ("P95 > 90% only in Q4 last week of quarter")  
  → Underutilized detection via clustering becomes credible (different archetypes cluster differently)  
  → Cost recommendations flow naturally ("right-size servers growing <5% YoY that stay <30% mean")

**Quick next-step suggestion**  
Once this method works, the natural follow-up is a helper function like:

```python
# In the same file or a companion notebook
def generate_fleet(n_servers=50, days=1095):
    fleet_data = []
    for i in range(n_servers):
        arch = get_archetype(np.random.choice(['web', 'database', 'application', 'batch']))
        metrics = arch.generate_baseline_metrics()
        df = pd.DataFrame(metrics)
        df['server_id'] = f"server-{i:03d}"
        df['type'] = arch.type.value
        fleet_data.append(df)
    return pd.concat(fleet_data)
```

Then you can save to CSV → upload to S3 → query with Athena → feed to SageMaker/Prophet.

Ready to move forward?

Next logical chunk: the **convenience functions** at the bottom of the file (`get_archetype()`, `generate_sample_server()`), which make it easy to use this module without manually creating profiles every time.

---

Great — "next" received, we're continuing the tutorial walkthrough of `server_archetypes.py` at the same deliberate, bite-sized pace.

We just finished **Chunk 9**: the final combination step inside `generate_baseline_metrics()` — where growth trends, diurnal cycles, seasonal multipliers, and correlated noise all get multiplied together to produce realistic daily P95 values for CPU, memory, disk_io, and network.

That method is the engine — now we move to the **convenience / helper functions** at the bottom of the file. These make the module much easier to use from notebooks or other scripts without manually constructing profiles every time.

### Chunk 10: Convenience Functions — `get_archetype()` and `generate_sample_server()`  
(approx. lines 222–end of current repo version)

```python
def get_archetype(archetype_name: str) -> ArchetypeProfile:
    """
    Factory function to get pre-configured profiles for common server types.
    
    Args:
        archetype_name: One of 'web', 'database', 'application', 'batch'
    
    Returns:
        ArchetypeProfile with realistic defaults for that type
    """
    profiles = {
        'web': ArchetypeProfile(
            cpu_base=35.0, memory_base=55.0, disk_io_base=10.0, network_base=45.0,
            cpu_std=12.0, memory_std=8.0, disk_io_std=6.0, network_std=15.0,
            correlation_cpu_memory=0.85, correlation_cpu_network=0.80,
            correlation_memory_disk=0.35,
            annual_growth_rate_cpu=0.10, annual_growth_rate_memory=0.15,
            diurnal_peak_hour_start=8, diurnal_peak_hour_end=20,
            batch_night_window=False
        ),
        'database': ArchetypeProfile(
            cpu_base=28.0, memory_base=75.0, disk_io_base=35.0, network_base=25.0,
            cpu_std=7.0, memory_std=5.0, disk_io_std=18.0, network_std=8.0,
            correlation_cpu_memory=0.70, correlation_cpu_network=0.40,
            correlation_memory_disk=0.65,
            annual_growth_rate_memory=0.22,   # databases grow fast
            diurnal_peak_hour_start=7, diurnal_peak_hour_end=19,
            batch_night_window=False
        ),
        'application': ArchetypeProfile(
            cpu_base=32.0, memory_base=60.0, disk_io_base=12.0, network_base=38.0,
            cpu_std=10.0, memory_std=7.0, disk_io_std=9.0, network_std=12.0,
            correlation_cpu_memory=0.78, correlation_cpu_network=0.65,
            correlation_memory_disk=0.45,
            annual_growth_rate_cpu=0.14, annual_growth_rate_memory=0.18,
            diurnal_peak_hour_start=9, diurnal_peak_hour_end=18,
            batch_night_window=False
        ),
        'batch': ArchetypeProfile(
            cpu_base=18.0, memory_base=40.0, disk_io_base=8.0, network_base=15.0,
            cpu_std=22.0, memory_std=15.0, disk_io_std=35.0, network_std=20.0,
            correlation_cpu_memory=0.55, correlation_cpu_network=0.30,
            correlation_memory_disk=0.75,   # batch often disk-heavy
            annual_growth_rate_cpu=0.08, annual_growth_rate_memory=0.12,
            diurnal_peak_hour_start=0, diurnal_peak_hour_end=24,  # flat-ish
            batch_night_window=True
        )
    }
    
    name_lower = archetype_name.lower()
    if name_lower not in profiles:
        raise ValueError(f"Unknown archetype: {archetype_name}. Choose from: {list(profiles.keys())}")
    
    return profiles[name_lower]


def generate_sample_server(
    archetype_name: str,
    server_id: str = None,
    start_date: str = "2022-01-01",
    days: int = 1095,
    random_seed: int = None
) -> ServerArchetype:
    """
    One-liner convenience to create a ready-to-use ServerArchetype instance.
    
    Example:
        batch_server = generate_sample_server('batch', server_id='batch-night-07')
        metrics = batch_server.generate_baseline_metrics()
    """
    profile = get_archetype(archetype_name)
    return ServerArchetype(
        type=ServerType[archetype_name.upper()],
        profile=profile,
        server_id=server_id or f"{archetype_name}-{np.random.randint(1000,9999)}",
        start_date=start_date,
        days=days,
        random_seed=random_seed
    )
```

**Step-by-step explanation (still beginner-to-intermediate friendly):**

1. **`get_archetype(archetype_name: str)`**  
   → Factory / lookup function — returns a pre-tuned `ArchetypeProfile` for each of the four types.  
   → Hardcoded realistic defaults based on your Citi experience:  
     - Web: higher network & CPU variance, strong business-hour pattern  
     - Database: high memory base + growth, disk-heavy correlations  
     - Application: balanced  
     - Batch: lower baseline, very high variance, night-inverted pattern  
   → Raises clear error for invalid names → user-friendly.

2. **`generate_sample_server(...)`**  
   → Highest-level convenience wrapper.  
   → One function call gives you a fully configured `ServerArchetype` ready to call `.generate_baseline_metrics()`.  
   → Auto-generates random-ish server_id if not provided.  
   → Example usage (perfect for notebooks):  
     ```python
     from src.utils.server_archetypes import generate_sample_server

     db_server = generate_sample_server('database', server_id='db-primary-001')
     metrics_dict = db_server.generate_baseline_metrics()
     ```

**Why these helpers are portfolio gold**

- **Clean Python / maintainability (priority #3)**: Hides complexity — notebooks stay readable, no copy-paste of long profile configs.  
- **Reusability**: Easy to generate a fleet of 50–200 heterogeneous servers in one loop.  
- **Professional touch**: Factory pattern + one-liner creator → code looks enterprise-ready, not hobbyist.

**Quick test you can run right now**  
In a notebook at `C:\pyproj\AWS-CapacityForecaster\notebooks\explore_archetypes.ipynb`:

```python
import pandas as pd
from src.utils.server_archetypes import generate_sample_server

# Quick test
web = generate_sample_server('web', server_id='web-nyc-042', random_seed=42)
metrics = web.generate_baseline_metrics()

df = pd.DataFrame(metrics)
df.index = pd.date_range(start=web.start_date, periods=web.days, freq='D')
print(df.head())
df['cpu'].plot(title=f"CPU for {web.name}", figsize=(12,5))
```

→ You’ll see a realistic-looking time series: upward trend, daily humps, quarterly spikes, correlated noise.

**End of file!**  
That's the complete walkthrough of `src/utils/server_archetypes.py` — from imports to the final helpers.

Next natural steps (your choice):
- Build a fleet generator function (50–100 servers → one big DataFrame or partitioned CSVs for S3)
- Add a quick visualization notebook showing differences between archetypes
- Start the data upload pipeline to S3 using boto3
- Or anything else you want to tackle

What would you like to do now?  
- "Build fleet generator code"  
- "Show me how to visualize multiple servers / archetypes"  
- "Next file — what should we work on?"  
- Or something else entirely?

Your project — your lead. 🚀