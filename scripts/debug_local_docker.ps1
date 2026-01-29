# Utility to run SageMaker container locally for debugging
$REGION = "us-east-1"
$ACCOUNT = "683313688378" # AWS SageMaker Image Account
$IMAGE_TAG = "1.2-1-cpu-py3" # The header image we are using for AutoGluon
$FULL_IMAGE = "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/sagemaker-scikit-learn:$IMAGE_TAG"

Write-Host "1. Logging into ECR..."
aws ecr get-login-password --region $REGION --profile study | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

if ($LASTEXITCODE -eq 0) {
    Write-Host "2. Pulling Image $FULL_IMAGE..."
    docker pull $FULL_IMAGE
    
    Write-Host "3. Running Container (Simulated SageMaker Job)..."
    
    # Run the wrapper script directly to mimic the job
    docker run -it --rm `
        -v ${PWD}:/opt/ml/processing/input/project_root `
        -v ${PWD}/src/modules/module_04_model_training.py:/opt/ml/processing/input/code/module_04_model_training.py `
        -w /opt/ml/processing/input/code `
        -e PYTHONPATH="/opt/ml/processing/input/project_root" `
        $FULL_IMAGE `
        python3 module_04_model_training.py --env local
}
else {
    Write-Error "Failed to login to ECR."
}
