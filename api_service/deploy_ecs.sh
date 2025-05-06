#!/bin/bash
# Deploy the SIFT API Service to AWS ECS Fargate
# Optimized for cost-efficiency and security

set -e

# Configuration
AWS_REGION=${AWS_REGION:-"us-east-1"}
ECR_REPOSITORY_NAME=${ECR_REPOSITORY_NAME:-"sift-api-service"}
ECS_CLUSTER_NAME=${ECS_CLUSTER_NAME:-"sift-cluster"}
ECS_SERVICE_NAME=${ECS_SERVICE_NAME:-"sift-api-service"}
ECS_TASK_FAMILY=${ECS_TASK_FAMILY:-"sift-api-service"}
ECS_CONTAINER_NAME=${ECS_CONTAINER_NAME:-"api-service"}

# Load settings from .env file if it exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env"
    export $(grep -v '^#' .env | xargs)
fi

# Check for required environment variables
required_vars=(
    "AWS_ACCESS_KEY_ID"
    "AWS_SECRET_ACCESS_KEY"
    "AWS_REGION"
    "DATABASE_URL"
    "JWT_SECRET_KEY"
    "SECRET_KEY"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var environment variable is required"
        exit 1
    fi
done

# Set up AWS CLI
echo "Setting up AWS CLI configuration..."
aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set region $AWS_REGION
aws configure set output json

# Create CloudWatch log group with cost-optimized retention period (3 days)
echo "Setting up CloudWatch log group..."
LOG_GROUP_NAME="/ecs/$ECS_TASK_FAMILY"

# Check if log group exists
log_group_exists=$(aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP_NAME" --query "length(logGroups)" --output text)

# Create log group if it doesn't exist
if [ "$log_group_exists" == "0" ]; then
    echo "Creating log group $LOG_GROUP_NAME..."
    aws logs create-log-group --log-group-name "$LOG_GROUP_NAME"
fi

# Set retention policy
echo "Setting log retention policy to 3 days..."
aws logs put-retention-policy --log-group-name "$LOG_GROUP_NAME" --retention-in-days 3

# Get AWS account ID
echo "Getting AWS account ID..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Error: Failed to get AWS account ID"
    exit 1
fi

# Create or verify required IAM roles
echo "Setting up required IAM roles..."

# Check if ecsTaskExecutionRole exists
EXECUTION_ROLE_EXISTS=$(aws iam get-role --role-name ecsTaskExecutionRole 2>&1 || echo "")
if [[ $EXECUTION_ROLE_EXISTS == *"NoSuchEntity"* ]]; then
    echo "Creating ecsTaskExecutionRole..."
    EXECUTION_TRUST_POLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
    aws iam create-role --role-name ecsTaskExecutionRole --assume-role-policy-document "$EXECUTION_TRUST_POLICY"
    aws iam attach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
    # Wait for role propagation
    echo "Waiting for role propagation (15 seconds)..."
    sleep 15
else
    echo "Using existing ecsTaskExecutionRole"
fi

# Check if ecsTaskRole exists
TASK_ROLE_EXISTS=$(aws iam get-role --role-name ecsTaskRole 2>&1 || echo "")
if [[ $TASK_ROLE_EXISTS == *"NoSuchEntity"* ]]; then
    echo "Creating ecsTaskRole..."
    TASK_TRUST_POLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
    aws iam create-role --role-name ecsTaskRole --assume-role-policy-document "$TASK_TRUST_POLICY"
    
    # Add required permissions for the SIFT application
    aws iam attach-role-policy --role-name ecsTaskRole --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
    aws iam attach-role-policy --role-name ecsTaskRole --policy-arn arn:aws:iam::aws:policy/AmazonSQSFullAccess
    
    # Wait for role propagation
    echo "Waiting for role propagation (15 seconds)..."
    sleep 15
else
    echo "Using existing ecsTaskRole"
fi

# Check if ECR repository exists, create if it doesn't
echo "Checking ECR repository..."
ECR_REPO_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME"
aws ecr describe-repositories --repository-names "$ECR_REPOSITORY_NAME" > /dev/null 2>&1 || \
    aws ecr create-repository --repository-name "$ECR_REPOSITORY_NAME" --image-scanning-configuration scanOnPush=true

# Log in to ECR
echo "Logging in to ECR..."
aws ecr get-login-password | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# Build Docker image
echo "Building Docker image..."
IMAGE_TAG=$(date +%Y%m%d%H%M%S)
docker build -t "$ECR_REPOSITORY_NAME:$IMAGE_TAG" .

# Tag and push image to ECR
echo "Tagging and pushing image to ECR..."
docker tag "$ECR_REPOSITORY_NAME:$IMAGE_TAG" "$ECR_REPO_URI:$IMAGE_TAG"
docker tag "$ECR_REPOSITORY_NAME:$IMAGE_TAG" "$ECR_REPO_URI:latest"
docker push "$ECR_REPO_URI:$IMAGE_TAG"
docker push "$ECR_REPO_URI:latest"

# Register task definition
echo "Registering ECS task definition..."
TASK_DEFINITION_JSON=$(cat <<EOF
{
    "family": "$ECS_TASK_FAMILY",
    "executionRoleArn": "arn:aws:iam::$AWS_ACCOUNT_ID:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::$AWS_ACCOUNT_ID:role/ecsTaskRole",
    "networkMode": "awsvpc",
    "containerDefinitions": [
        {
            "name": "$ECS_CONTAINER_NAME",
            "image": "$ECR_REPO_URI:$IMAGE_TAG",
            "cpu": 256,
            "memory": 512,
            "essential": true,
            "portMappings": [
                {
                    "containerPort": 8000,
                    "hostPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {
                    "name": "DATABASE_URL",
                    "value": "$DATABASE_URL"
                },
                {
                    "name": "JWT_SECRET_KEY",
                    "value": "$JWT_SECRET_KEY"
                },
                {
                    "name": "SECRET_KEY",
                    "value": "$SECRET_KEY"
                },
                {
                    "name": "SERVER_HOST",
                    "value": "0.0.0.0"
                },
                {
                    "name": "SERVER_PORT",
                    "value": "8000"
                },
                {
                    "name": "AWS_DEFAULT_REGION",
                    "value": "$AWS_REGION"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "$LOG_GROUP_NAME",
                    "awslogs-region": "$AWS_REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8000/api/v1/health || exit 1"],
                "interval": 30,
                "timeout": 5,
                "retries": 3,
                "startPeriod": 30
            }
        }
    ],
    "requiresCompatibilities": [
        "FARGATE"
    ],
    "cpu": "256",
    "memory": "512"
}
EOF
)

TASK_DEFINITION=$(aws ecs register-task-definition \
    --cli-input-json "$TASK_DEFINITION_JSON" \
    --query "taskDefinition.taskDefinitionArn" \
    --output text)

echo "Task definition registered: $TASK_DEFINITION"

# Check if ECS cluster exists, create if it doesn't
echo "Checking ECS cluster..."
aws ecs describe-clusters --clusters "$ECS_CLUSTER_NAME" --query "clusters[0].clusterName" --output text || \
    aws ecs create-cluster --cluster-name "$ECS_CLUSTER_NAME"

# Get or create VPC and subnet information
echo "Setting up network configuration..."

# Get default VPC and subnets
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)
if [ -z "$VPC_ID" ] || [ "$VPC_ID" == "None" ]; then
    echo "Error: No default VPC found. Please create a VPC or specify an existing VPC ID."
    exit 1
fi

SUBNET_IDS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query "Subnets[?MapPublicIpOnLaunch==\`true\`].SubnetId" \
    --output text | tr '\t' ',')
if [ -z "$SUBNET_IDS" ] || [ "$SUBNET_IDS" == "None" ]; then
    echo "Error: No public subnets found in VPC $VPC_ID"
    exit 1
fi

# Create security group if it doesn't exist
SECURITY_GROUP_NAME="sift-api-service-sg"
SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query "SecurityGroups[0].GroupId" \
    --output text)

if [ -z "$SECURITY_GROUP_ID" ] || [ "$SECURITY_GROUP_ID" == "None" ]; then
    echo "Creating security group $SECURITY_GROUP_NAME..."
    SECURITY_GROUP_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP_NAME" \
        --description "Security group for SIFT API Service" \
        --vpc-id "$VPC_ID" \
        --query "GroupId" \
        --output text)
    
    # Allow inbound traffic to the API port
    aws ec2 authorize-security-group-ingress \
        --group-id "$SECURITY_GROUP_ID" \
        --protocol tcp \
        --port 8000 \
        --cidr 0.0.0.0/0
else
    echo "Using existing security group $SECURITY_GROUP_NAME: $SECURITY_GROUP_ID"
fi

# Check if the ECS service exists
SERVICE_EXISTS=$(aws ecs list-services --cluster "$ECS_CLUSTER_NAME" --query "serviceArns[?contains(@,'$ECS_SERVICE_NAME')]" --output text)

if [ -z "$SERVICE_EXISTS" ]; then
    # Create the ECS service
    echo "Creating ECS service..."
    aws ecs create-service \
        --cluster "$ECS_CLUSTER_NAME" \
        --service-name "$ECS_SERVICE_NAME" \
        --task-definition "$TASK_DEFINITION" \
        --desired-count 1 \
        --launch-type "FARGATE" \
        --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
        --health-check-grace-period-seconds 60
else
    # Update the ECS service
    echo "Updating ECS service..."
    aws ecs update-service \
        --cluster "$ECS_CLUSTER_NAME" \
        --service "$ECS_SERVICE_NAME" \
        --task-definition "$TASK_DEFINITION" \
        --force-new-deployment
fi

echo "Deployment initiated! The ECS service will be updated shortly."
echo "You can check the status with: aws ecs describe-services --cluster $ECS_CLUSTER_NAME --services $ECS_SERVICE_NAME"

# Wait for service to stabilize
echo "Waiting for service to stabilize (this may take a few minutes)..."
aws ecs wait services-stable --cluster "$ECS_CLUSTER_NAME" --services "$ECS_SERVICE_NAME"

# Get the public IP of the task
echo "Getting task information..."
TASK_ARN=$(aws ecs list-tasks --cluster "$ECS_CLUSTER_NAME" --service-name "$ECS_SERVICE_NAME" --query "taskArns[0]" --output text)
if [ -n "$TASK_ARN" ] && [ "$TASK_ARN" != "None" ]; then
    echo "Task ARN: $TASK_ARN"
    NETWORK_INTERFACE=$(aws ecs describe-tasks --cluster "$ECS_CLUSTER_NAME" --tasks "$TASK_ARN" --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text)
    
    if [ -n "$NETWORK_INTERFACE" ] && [ "$NETWORK_INTERFACE" != "None" ]; then
        PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids "$NETWORK_INTERFACE" --query "NetworkInterfaces[0].Association.PublicIp" --output text)
        echo "Service is now accessible at: http://$PUBLIC_IP:8000"
        echo "API documentation: http://$PUBLIC_IP:8000/docs"
    fi
fi

echo "Deployment completed successfully!"
