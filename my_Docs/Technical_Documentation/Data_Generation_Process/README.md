# Data Generation Process - Complete Technical Documentation

**AWS-CapacityForecaster | Synthetic Data Generation System**
**Version:** 1.0 | **Date:** 2026-01-22 | **Status:** Production-Ready

---

## 📚 Documentation Index

This folder contains comprehensive technical documentation for the synthetic data generation system that creates enterprise-realistic server capacity metrics for the AWS-CapacityForecaster project.

### Core Documentation Files

| # | Document | Description | Pages | Status |
|---|----------|-------------|-------|--------|
| 00 | [**MASTER_TECHNICAL_GUIDE.md**](00_MASTER_TECHNICAL_GUIDE.md) | Complete overview with architecture, flow diagrams, and component deep dive | 25+ | ✅ Complete |
| 01 | [**Configuration_System.md**](01_Configuration_System.md) | YAML configuration, parameter validation, environment overrides | 12 | ✅ Complete |
| 02 | **Code_Walkthrough.md** | Line-by-line code explanation with inline comments | 30+ | 📝 In Progress |
| 03 | **Function_Catalog.md** | Complete function reference with I/O specs | 20+ | 📝 In Progress |
| 04 | **Flowcharts_Diagrams.md** | Visual process flows, sequence diagrams, state machines | 15+ | 📝 In Progress |
| 05 | **Success_Criteria.md** | Validation rules, quality metrics, testing procedures | 10+ | 📝 In Progress |

---

## 🎯 Quick Start

### For Understanding the System
1. **Start here:** [00_MASTER_TECHNICAL_GUIDE.md](00_MASTER_TECHNICAL_GUIDE.md)
   - Executive overview
   - High-level architecture
   - Data flow diagrams

2. **Then read:** [01_Configuration_System.md](01_Configuration_System.md)
   - How to configure data generation
   - Parameter meanings
   - Environment overrides

### For Code Review
3. **Detailed walkthrough:** [02_Code_Walkthrough.md](02_Code_Walkthrough.md)
   - Inline comments for every function
   - Design decisions explained
   - Edge cases documented

4. **Function reference:** [03_Function_Catalog.md](03_Function_Catalog.md)
   - All functions listed
   - Input/Output specifications
   - Usage examples

### For Visual Learners
5. **Diagrams:** [04_Flowcharts_Diagrams.md](04_Flowcharts_Diagrams.md)
   - Process flowcharts
   - Sequence diagrams
   - Architecture diagrams

### For Quality Assurance
6. **Validation:** [05_Success_Criteria.md](05_Success_Criteria.md)
   - Quality metrics
   - Test procedures
   - Acceptance criteria

---

## 📊 What This System Generates

### Output Dataset Specifications

**File:** `data/synthetic/server_metrics_full.csv.gz`

| Attribute | Value |
|-----------|-------|
| **Records** | 175,320 |
| **Servers** | 120 |
| **Time Range** | 2022-01-01 to 2025-12-31 (4 years) |
| **Granularity** | Daily |
| **File Size** | 3.04 MB (compressed) |
| **Columns** | 18 |
| **Quality** | 100% complete, 0 missing values in metrics |

### Metrics Generated

1. **cpu_p95** - CPU P95 utilization (0-100%)
2. **mem_p95** - Memory P95 utilization (0-100%)
3. **disk_p95** - Disk P95 utilization (0-100%)
4. **net_in_p95** - Network inbound P95 (0-1000 Mbps)
5. **net_out_p95** - Network outbound P95 (0-600 Mbps)

### Metadata Included

- **server_id** - Unique identifier
- **server_type** - Archetype (web, database, application, batch)
- **business_unit** - Trading, Retail, Compliance, IT
- **criticality** - High, Medium, Low
- **region** - US-East, US-West, EU, Asia
- **app_name** - Application identifier
- **Calendar features** - year, month, quarter, dayofweek, is_weekend, is_eoq, is_holiday

---

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              Configuration Layer                     │
│                 (config.yaml)                        │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│         Server Archetype Assignment                  │
│    (Web: 35%, App: 40%, DB: 15%, Batch: 10%)       │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│        Time Series Generation (Per Server)          │
│  • Correlated metrics (Cholesky decomposition)      │
│  • Time-based factors (business hours, weekends)    │
│  • Seasonal patterns (quarterly, holidays)          │
│  • Spike modeling                                   │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│           Metadata Enrichment                        │
│  • Business context                                  │
│  • Calendar features                                 │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│       Output & Validation                            │
│  • CSV.gz file (3.04 MB)                            │
│  • Visualizations (7 panels)                        │
│  • Quality reports                                  │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 Code Files Reference

### Core Modules

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| **src/utils/server_archetypes.py** | 356 | Defines 4 server archetypes with correlation models | ✅ Production |
| **src/data_generation.py** | 390 | Main generation orchestration | ✅ Production |
| **src/utils/data_utils.py** | 466 | Metadata and calendar features | ✅ Production |
| **src/utils/config.py** | 254 | Configuration management | ✅ Production |

### Supporting Scripts

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| **scripts/visualize_synthetic_data.py** | 106 | Generate 7-panel dashboard | ✅ Complete |
| **scripts/download_external_data.py** | 306 | Sample data and analysis | ✅ Complete |

### Configuration

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| **config/config.yaml** | 107 | Central configuration | ✅ Production |

**Total Code:** ~1,985 lines across 7 files

---

## 📈 Key Features

### 1. Server Archetypes

Four distinct server types with unique behaviors:

- **Web Servers (35%)** - High CPU-Network correlation, business hours sensitive
- **Application Servers (40%)** - Balanced metrics, strong daily patterns
- **Database Servers (15%)** - High memory/disk, steady state
- **Batch Servers (10%)** - Spiky CPU, off-hours processing

### 2. Realistic Correlations

- **CPU ↔ Memory:** 0.4 - 0.7 (varies by archetype)
- **CPU ↔ Network:** 0.3 - 0.8 (strongest for web servers)
- **Memory ↔ Disk:** 0.2 - 0.7 (strongest for databases)

Generated using **Cholesky decomposition** for mathematically correct correlations.

### 3. Banking-Specific Seasonality

- **Quarterly Peaks** - Up to 30% increase at end-of-quarter (Mar 31, Jun 30, Sep 30, Dec 31)
- **Holiday Effects** - 30-50% reduction during holidays
- **Weekly Patterns** - Higher weekdays, lower weekends
- **Business Hours** - 0.8x to 1.6x multiplier (varies by archetype)

### 4. Quality Assurance

- ✅ 100% data completeness
- ✅ Valid ranges (0-100% for utilization)
- ✅ Realistic correlations
- ✅ Detectable seasonality
- ✅ Archetype differentiation

---

## 🚀 Usage Examples

### Basic Usage

```bash
# Activate environment
. env_setter.ps1

# Generate with defaults (120 servers, 4 years)
python -m src.data_generation

# Output: data/synthetic/server_metrics_full.csv
```

### Custom Parameters

```bash
# Generate 200 servers for 5 years
python -m src.data_generation --servers 200 --years 5

# Generate with custom date range
python -m src.data_generation \
  --start-date "2020-01-01" \
  --end-date "2025-12-31"

# Generate compressed output
python -m src.data_generation \
  --output data/custom/metrics.csv \
  --compress

# Generate hourly data
python -m src.data_generation \
  --granularity hourly \
  --servers 50 \
  --years 1
```

### Environment Overrides

```bash
# Override via environment variables
export CF_DATA_NUM_SERVERS=150
export CF_DATA_START_DATE="2023-01-01"
python -m src.data_generation
```

### Visualization

```bash
# Generate quality dashboard
python scripts/visualize_synthetic_data.py

# Output: reports/data_quality/synthetic_data_overview.png
```

---

## 📋 Success Criteria

### Data Quality Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Completeness | >95% | 100% | ✅ |
| Valid ranges | 100% | 100% | ✅ |
| Correlations | ±0.15 of target | Within range | ✅ |
| Seasonal patterns | Detectable | Visible | ✅ |
| Missing values | <5% | 0% | ✅ |
| Duplicate records | 0 | 0 | ✅ |

### Performance Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Generation time | <60s | ~20s | ✅ |
| Memory usage | <100 MB | 18 MB | ✅ |
| File size | <10 MB | 3.04 MB | ✅ |
| Compression ratio | >3:1 | 6:1 | ✅ |

---

## 🔍 Troubleshooting

### Common Issues

**Issue:** "Config file not found"
- **Solution:** Ensure `config/config.yaml` exists or set `CAPACITY_FORECASTER_CONFIG` env var

**Issue:** "Import errors"
- **Solution:** Activate virtual environment with `env_setter.ps1`

**Issue:** "Memory error with hourly data"
- **Solution:** Reduce `--servers` or `--years` parameters

**Issue:** "Slow generation"
- **Solution:** Use daily granularity, or reduce server count for testing

---

## 📞 Support & Contact

**Project Repository:** https://github.com/seanlgirgis/AWS-CapacityForecaster
**Documentation Issues:** Create issue on GitHub
**Author:** Sean L. Girgis

---

## 📝 Document Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| 00_MASTER_TECHNICAL_GUIDE | ✅ Complete | 2026-01-22 |
| 01_Configuration_System | ✅ Complete | 2026-01-22 |
| 02_Code_Walkthrough | 📝 In Progress | - |
| 03_Function_Catalog | 📝 In Progress | - |
| 04_Flowcharts_Diagrams | 📝 In Progress | - |
| 05_Success_Criteria | 📝 In Progress | - |

---

**Last Updated:** 2026-01-22
**Version:** 1.0
**Status:** Production-Ready
