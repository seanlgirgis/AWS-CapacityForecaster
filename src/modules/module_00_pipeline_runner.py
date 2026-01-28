# src/module_00_pipeline_runner.py
import argparse
import logging
from pathlib import Path
import time
from typing import List, Optional

from src.utils.config import load_config, validate_config

# Import your module mains (adjust paths if needed)
from src.modules.module_01_data_generation import main_process as run_data_gen
from src.modules.module_02_data_load import main_process as run_data_load
from src.modules.module_03_etl_feature_eng import main_process as run_etl
from src.modules.module_04_model_training import main_process as run_training
from src.modules.module_05_risk_capacity_analysis import main_process as run_risk

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s')

MODULES = [
    ("01 - Data Generation", run_data_gen),
    ("02 - Data Load & Validation", run_data_load),
    ("03 - ETL & Feature Engineering", run_etl),
    ("04 - Model Training & Forecasting", run_training),
    ("05 - Risk & Capacity Analysis", run_risk),
]

def run_pipeline(config, selected: Optional[List[str]] = None):
    start_total = time.time()
    successes = []
    
    for name, func in MODULES:
        module_num = name.split(" - ")[0]
        if selected and module_num not in selected:
            logger.info(f"Skipping {name}")
            continue
            
        logger.info(f"Starting {name}...")
        module_start = time.time()
        
        try:
            func(config)
            duration = time.time() - module_start
            logger.info(f"✔ {name} completed in {duration:.1f}s")
            successes.append(module_num)
        except Exception as e:
            logger.error(f"✘ {name} failed: {str(e)}")
            raise  # or continue with --force flag later
    
    total_duration = time.time() - start_total
    logger.info(f"Pipeline finished. Success: {len(successes)}/{len(MODULES)} modules | Total: {total_duration:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AWS-CapacityForecaster Pipeline Runner")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--env", default="local", nargs="?", const="local", choices=["local", "sagemaker"])
    parser.add_argument("--only", nargs="+", help="Run only these modules e.g. --only 03 05")
    parser.add_argument("--skip", nargs="+", help="Skip these modules")
    args = parser.parse_args()
    
    config = load_config(Path(args.config))
    if args.env:
        config.setdefault('execution', {})['mode'] = args.env
    validate_config(config)
    
    selected = None
    if args.only:
        selected = args.only
    elif args.skip:
        selected = [m[0].split(" - ")[0] for m in MODULES if m[0].split(" - ")[0] not in args.skip]
    
    run_pipeline(config, selected)