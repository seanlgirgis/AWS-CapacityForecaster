# Day 0 / 400 / 800 / 1200 Example

## Understanding Cumulative vs Point-in-Time Factors

This example clarifies how the two types of factors combine in metric calculation.

---

## The Two Factor Types

### 1. Cumulative Factor: `trend_factor`
- **Builds up over time** — never resets
- Represents organic growth (more users, more data, more load)
- Progresses from 0.0 → 1.0 over the full simulation period (1461 days ≈ 4 years)

### 2. Point-in-Time Factors: `time_factor`, `qtr_factor`, `holiday_factor`
- **Reset each timestamp** — based on current conditions only
- `time_factor`: business hours vs off-hours (e.g., 1.15 during 9-17, 0.85 at night)
- `qtr_factor`: quarter-end peaks (e.g., 1.4 in last week of Q4)
- `holiday_factor`: holiday adjustments (e.g., 0.7 on Christmas)

---

## Concrete Example: Web Server CPU

Assume:
- `cpu_base` = 25.0%
- Annual growth rate = 12% (so after 4 years: 25 × 1.12^4 ≈ 39.3%)
- Business hours boost = 1.15
- Quarter-end peak = 1.4
- Holiday trough = 0.7

### Day 0 (January 1, Year 1 — New Year's Day, 10 AM)

| Factor | Value | Reason |
|--------|-------|--------|
| `trend_factor` | 0.0 | Just started, no growth yet |
| `growth_multiplier` | 1.0 | (1 + 0.12)^0 = 1.0 |
| `time_factor` | 1.15 | 10 AM = business hours |
| `qtr_factor` | 1.0 | Not near quarter-end |
| `holiday_factor` | 0.7 | New Year's Day |

**Combined point-in-time**: 1.15 × 1.0 × 0.7 = **0.805**

**Final CPU** ≈ 25.0 × 1.0 × 0.805 + noise ≈ **20.1%** (+ random noise)

---

### Day 400 (February 5, Year 2 — Tuesday, 2 PM)

| Factor | Value | Reason |
|--------|-------|--------|
| `trend_factor` | 0.274 | 400/1461 of the way through |
| `growth_multiplier` | 1.13 | (1 + 0.12)^(400/365) ≈ 1.13 |
| `time_factor` | 1.15 | 2 PM = business hours |
| `qtr_factor` | 1.0 | Mid-Q1, not near quarter-end |
| `holiday_factor` | 1.0 | Normal business day |

**Combined point-in-time**: 1.15 × 1.0 × 1.0 = **1.15**

**Final CPU** ≈ 25.0 × 1.13 × 1.15 + noise ≈ **32.5%** (+ random noise)

---

### Day 800 (March 11, Year 3 — Wednesday, 11 AM)

| Factor | Value | Reason |
|--------|-------|--------|
| `trend_factor` | 0.548 | 800/1461 of the way through |
| `growth_multiplier` | 1.28 | (1 + 0.12)^(800/365) ≈ 1.28 |
| `time_factor` | 1.15 | 11 AM = business hours |
| `qtr_factor` | 1.0 | Mid-Q1, not near quarter-end |
| `holiday_factor` | 1.0 | Normal business day |

**Combined point-in-time**: 1.15 × 1.0 × 1.0 = **1.15**

**Final CPU** ≈ 25.0 × 1.28 × 1.15 + noise ≈ **36.8%** (+ random noise)

---

### Day 1200 (March 25, Year 4 — Thursday, 3 PM, near Q1 end)

| Factor | Value | Reason |
|--------|-------|--------|
| `trend_factor` | 0.821 | 1200/1461 of the way through |
| `growth_multiplier` | 1.45 | (1 + 0.12)^(1200/365) ≈ 1.45 |
| `time_factor` | 1.15 | 3 PM = business hours |
| `qtr_factor` | 1.35 | ~6 days from Q1 end (March 31) |
| `holiday_factor` | 1.0 | Normal business day |

**Combined point-in-time**: 1.15 × 1.35 × 1.0 = **1.55**

**Final CPU** ≈ 25.0 × 1.45 × 1.55 + noise ≈ **56.2%** (+ random noise)

---

## Visual Summary

```
Day      trend_factor    growth_mult    point-in-time    Final CPU
────────────────────────────────────────────────────────────────────
   0     0.000           1.00           0.805 (holiday)   ~20%
 400     0.274           1.13           1.15  (normal)    ~33%
 800     0.548           1.28           1.15  (normal)    ~37%
1200     0.821           1.45           1.55  (Q-end)     ~56%
```

---

## Key Insight

- **Trend/growth** keeps building — the server gets busier year over year
- **Point-in-time factors** fluctuate daily based on:
  - What time is it? (business hours boost)
  - What quarter period? (Q-end spike)
  - Is it a holiday? (reduced load)

The final metric is:

```
CPU = base × growth_multiplier × (time × qtr × holiday) + correlated_noise
      └──────────cumulative────┘   └───point-in-time───┘   └──random──┘
```

This creates realistic time series with:
1. **Long-term upward trend** (growth)
2. **Daily cycles** (business hours)
3. **Quarterly spikes** (financial close periods)
4. **Holiday dips** (reduced activity)
5. **Random but correlated noise** (real-world variability)
