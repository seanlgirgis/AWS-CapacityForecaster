
import subprocess
import sys
import os

def install_dependencies():
    """Installs required packages for Module 04 (AutoGluon/Prophet stack)"""
    print("Wrapper: Installing runtime dependencies...")
    
    required_packages = [
        "prophet",
        "holidays",
        "PyYAML",
        "matplotlib",
        "plotly",
        "python-dotenv",
        "pandas==2.1.4",      # Pin to ensure compatibility with numpy<2.0.0
        "scikit-learn<1.5.0", # Pin to ensure compatibility with legacy environment
        "numpy<2.0.0",        # Pin for stability
        "scipy<1.13.0",       # Pin for fitpack compatibility
        "pyarrow>=14.0.0",    # Modern arrow for Pandas 2.x
        "fastparquet",        # Fallback engine
        "autogluon>=1.1.0"    # Modern AutoML
    ]
    
    # Check for local environment vs SageMaker
    if not os.path.exists("/opt/ml"):
        print("Wrapper: Local environment detected. Skipping pip install (assuming env is ready).")
        return

    try:
        # Install all at once
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + required_packages + ["--quiet"])
        print("Wrapper: Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Wrapper: Dependency installation failed: {e}")
        # We don't exit here, we try to run anyway in case they are pre-installed
        
def run_inner_script():
    """Launches the actual logic script in a fresh process"""
    # Determine path based on environment
    if os.path.exists("/opt/ml/processing/input/project_root"):
        # SageMaker: Code is mounted in project_root, not current dir
        script_path = "/opt/ml/processing/input/project_root/src/modules/module_04_inner.py"
        config_path = "/opt/ml/processing/input/project_root/config/config.yaml"
    else:
        # Local: Running from project root
        script_path = "src/modules/module_04_inner.py"
        config_path = "config/config.yaml"
        
    print(f"Wrapper: Launching inner script: {script_path}")
    print(f"Wrapper: Using config: {config_path}")
    
    # Pass all arguments received by this wrapper to the inner script
    # AND explicitly set the config path
    args = [sys.executable, script_path, "--config", config_path] + sys.argv[1:]
    
    try:
        # Flush stdout before forking
        sys.stdout.flush()
        
        # Run subprocess
        subprocess.check_call(args)
        print("Wrapper: Inner script completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Wrapper: Inner script failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    install_dependencies()
    run_inner_script()