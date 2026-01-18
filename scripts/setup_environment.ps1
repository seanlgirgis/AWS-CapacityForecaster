# Setup AWS-CapacityForecaster directory structure and placeholders

$basePath = "C:\pyproj\AWS-CapacityForecaster"

# Create folders
New-Item -Path "$basePath\src" -ItemType Directory -Force
New-Item -Path "$basePath\src\utils" -ItemType Directory -Force
New-Item -Path "$basePath\notebooks" -ItemType Directory -Force
New-Item -Path "$basePath\tests" -ItemType Directory -Force
New-Item -Path "$basePath\config" -ItemType Directory -Force
New-Item -Path "$basePath\scripts" -ItemType Directory -Force
New-Item -Path "$basePath\docs\images" -ItemType Directory -Force  # Assuming docs exists

# Populate empty src files
New-Item -Path "$basePath\src\__init__.py" -ItemType File -Force
New-Item -Path "$basePath\src\data_generation.py" -ItemType File -Force
New-Item -Path "$basePath\src\etl_pipeline.py" -ItemType File -Force
New-Item -Path "$basePath\src\ml_forecasting.py" -ItemType File -Force
New-Item -Path "$basePath\src\risk_analysis.py" -ItemType File -Force
New-Item -Path "$basePath\src\visualization.py" -ItemType File -Force

# Utils
New-Item -Path "$basePath\src\utils\__init__.py" -ItemType File -Force
New-Item -Path "$basePath\src\utils\aws_utils.py" -ItemType File -Force
New-Item -Path "$basePath\src\utils\data_utils.py" -ItemType File -Force
New-Item -Path "$basePath\src\utils\ml_utils.py" -ItemType File -Force
New-Item -Path "$basePath\src\utils\config.py" -ItemType File -Force

# Notebooks (empty .ipynb placeholders - add JSON content manually or via Jupyter)
New-Item -Path "$basePath\notebooks\01_data_generation.ipynb" -ItemType File -Force
New-Item -Path "$basePath\notebooks\02_etl_pipeline.ipynb" -ItemType File -Force
New-Item -Path "$basePath\notebooks\03_ml_forecasting.ipynb" -ItemType File -Force
New-Item -Path "$basePath\notebooks\04_risk_analysis.ipynb" -ItemType File -Force
New-Item -Path "$basePath\notebooks\05_visualization.ipynb" -ItemType File -Force

# Tests
New-Item -Path "$basePath\tests\test_data_generation.py" -ItemType File -Force
New-Item -Path "$basePath\tests\test_etl_pipeline.py" -ItemType File -Force
New-Item -Path "$basePath\tests\test_ml_forecasting.py" -ItemType File -Force
New-Item -Path "$basePath\tests\test_risk_analysis.py" -ItemType File -Force
New-Item -Path "$basePath\tests\test_visualization.py" -ItemType File -Force

# Config
New-Item -Path "$basePath\config\config.yaml" -ItemType File -Force

# Scripts
New-Item -Path "$basePath\scripts\deploy_sagemaker_job.py" -ItemType File -Force
New-Item -Path "$basePath\scripts\lambda_handler.py" -ItemType File -Force

# Root files
New-Item -Path "$basePath\requirements.txt" -ItemType File -Force
New-Item -Path "$basePath\README.md" -ItemType File -Force
New-Item -Path "$basePath\.gitignore" -ItemType File -Force
New-Item -Path "$basePath\.env" -ItemType File -Force

Write-Output "Directory structure created and placeholders populated at $basePath"# Setup AWS-CapacityForecaster directory structure and placeholders

$basePath = "C:\pyproj\AWS-CapacityForecaster"

# Create folders
New-Item -Path "$basePath\src" -ItemType Directory -Force
New-Item -Path "$basePath\src\utils" -ItemType Directory -Force
New-Item -Path "$basePath\notebooks" -ItemType Directory -Force
New-Item -Path "$basePath\tests" -ItemType Directory -Force
New-Item -Path "$basePath\config" -ItemType Directory -Force
New-Item -Path "$basePath\scripts" -ItemType Directory -Force
New-Item -Path "$basePath\docs\images" -ItemType Directory -Force  # Assuming docs exists

# Populate empty src files
New-Item -Path "$basePath\src\__init__.py" -ItemType File -Force
New-Item -Path "$basePath\src\data_generation.py" -ItemType File -Force
New-Item -Path "$basePath\src\etl_pipeline.py" -ItemType File -Force
New-Item -Path "$basePath\src\ml_forecasting.py" -ItemType File -Force
New-Item -Path "$basePath\src\risk_analysis.py" -ItemType File -Force
New-Item -Path "$basePath\src\visualization.py" -ItemType File -Force

# Utils
New-Item -Path "$basePath\src\utils\__init__.py" -ItemType File -Force
New-Item -Path "$basePath\src\utils\aws_utils.py" -ItemType File -Force
New-Item -Path "$basePath\src\utils\data_utils.py" -ItemType File -Force
New-Item -Path "$basePath\src\utils\ml_utils.py" -ItemType File -Force
New-Item -Path "$basePath\src\utils\config.py" -ItemType File -Force

# Notebooks (empty .ipynb placeholders - add JSON content manually or via Jupyter)
New-Item -Path "$basePath\notebooks\01_data_generation.ipynb" -ItemType File -Force
New-Item -Path "$basePath\notebooks\02_etl_pipeline.ipynb" -ItemType File -Force
New-Item -Path "$basePath\notebooks\03_ml_forecasting.ipynb" -ItemType File -Force
New-Item -Path "$basePath\notebooks\04_risk_analysis.ipynb" -ItemType File -Force
New-Item -Path "$basePath\notebooks\05_visualization.ipynb" -ItemType File -Force

# Tests
New-Item -Path "$basePath\tests\test_data_generation.py" -ItemType File -Force
New-Item -Path "$basePath\tests\test_etl_pipeline.py" -ItemType File -Force
New-Item -Path "$basePath\tests\test_ml_forecasting.py" -ItemType File -Force
New-Item -Path "$basePath\tests\test_risk_analysis.py" -ItemType File -Force
New-Item -Path "$basePath\tests\test_visualization.py" -ItemType File -Force

# Config
New-Item -Path "$basePath\config\config.yaml" -ItemType File -Force

# Scripts
New-Item -Path "$basePath\scripts\deploy_sagemaker_job.py" -ItemType File -Force
New-Item -Path "$basePath\scripts\lambda_handler.py" -ItemType File -Force

# Root files
New-Item -Path "$basePath\requirements.txt" -ItemType File -Force
New-Item -Path "$basePath\README.md" -ItemType File -Force
New-Item -Path "$basePath\.gitignore" -ItemType File -Force
New-Item -Path "$basePath\.env" -ItemType File -Force

Write-Output "Directory structure created and placeholders populated at $basePath"