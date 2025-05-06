#!/bin/bash
# Deployment script for SIFT PPE Detection Service on ECS Fargate

set -e

# Configuration
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="529710251669"
ECR_REPO_NAME="sift-model-service"
ECS_CLUSTER_NAME="sift-cluster"
ECS_SERVICE_NAME="sift-ppe-detector"
ECS_TASK_FAMILY="sift-ppe-detector-task"
EXECUTION_ROLE_NAME="ecsTaskExecutionRole"
TASK_ROLE_NAME="sift-ppe-task-role"
LOG_GROUP_NAME="/ecs/sift-ppe-detector"
S3_BUCKET_NAME="osha-mvp-bucket"
SQS_QUEUE_URL="https://sqs.us-east-1.amazonaws.com/529710251669/OshaFrameQueue"

echo "=== Deploying SIFT PPE Detection Service to ECS Fargate ==="

# Step 1: Create ECR repository if it doesn't exist
echo "Creating ECR repository if it doesn't exist..."
aws ecr describe-repositories --repository-names $ECR_REPO_NAME > /dev/null 2>&1 || \
  aws ecr create-repository --repository-name $ECR_REPO_NAME

# Step 2: Authenticate Docker with ECR
echo "Authenticating Docker with ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Step 3: Build, tag and push Docker image to ECR
echo "Building Docker image..."
docker build -t $ECR_REPO_NAME:latest .

echo "Tagging and pushing image to ECR..."
docker tag $ECR_REPO_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest

# Step 4: Create CloudWatch log group if it doesn't exist
echo "Setting up CloudWatch log group..."
aws logs create-log-group --log-group-name $LOG_GROUP_NAME > /dev/null 2>&1 || echo "Log group already exists"

# Set log retention to reduce costs
echo "Setting log retention period to 3 days..."
aws logs put-retention-policy --log-group-name $LOG_GROUP_NAME --retention-in-days 3

# Step 5: Check if task execution role exists, create if it doesn't
echo "Checking ECS task execution role..."
aws iam get-role --role-name $EXECUTION_ROLE_NAME > /dev/null 2>&1 || {
  echo "Creating ECS task execution role..."
  
  # Create trust policy
  cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

  # Create role
  aws iam create-role --role-name $EXECUTION_ROLE_NAME --assume-role-policy-document file://trust-policy.json
  
  # Attach required policies
  aws iam attach-role-policy --role-name $EXECUTION_ROLE_NAME --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
}

# Step 6: Check if task role exists, create if it doesn't
echo "Checking ECS task role..."
aws iam get-role --role-name $TASK_ROLE_NAME > /dev/null 2>&1 || {
  echo "Creating ECS task role..."
  
  # Create trust policy (reusing the one created above)
  
  # Create role
  aws iam create-role --role-name $TASK_ROLE_NAME --assume-role-policy-document file://trust-policy.json
  
  # Attach required policies
  aws iam attach-role-policy --role-name $TASK_ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonSQSFullAccess
  aws iam attach-role-policy --role-name $TASK_ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
  
  # Create inline policy for RDS access
  cat > rds-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "rds-db:connect"
            ],
            "Resource": "*"
        }
    ]
}
EOF
    
  aws iam put-role-policy --role-name $TASK_ROLE_NAME --policy-name RDSAccess --policy-document file://rds-policy.json
}

# Step 7: Create ECS task definition
echo "Creating ECS task definition..."

# Create task definition JSON
cat > task-definition.json << EOF
{
    "family": "${ECS_TASK_FAMILY}",
    "networkMode": "awsvpc",
    "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/${EXECUTION_ROLE_NAME}",
    "taskRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/${TASK_ROLE_NAME}",
    "containerDefinitions": [
        {
            "name": "sift-ppe-detector",
            "image": "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:latest",
            "essential": true,
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "${LOG_GROUP_NAME}",
                    "awslogs-region": "${AWS_REGION}",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "environment": [
                {
                    "name": "AWS_REGION",
                    "value": "${AWS_REGION}"
                },
                {
                    "name": "SQS_QUEUE_URL",
                    "value": "${SQS_QUEUE_URL}"
                },
                {
                    "name": "S3_BUCKET_NAME",
                    "value": "${S3_BUCKET_NAME}"
                },
                {
                    "name": "LOG_LEVEL",
                    "value": "INFO"
                }
            ],
            "secrets": [
                {
                    "name": "DATABASE_URL",
                    "valueFrom": "arn:aws:ssm:${AWS_REGION}:${AWS_ACCOUNT_ID}:parameter/sift/database_url"
                }
            ],
            "portMappings": [
                {
                    "containerPort": 8000,
                    "hostPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "healthCheck": {
                "command": [
                    "CMD-SHELL",
                    "curl -f http://localhost:8000/health || exit 1"
                ],
                "interval": 30,
                "timeout": 5,
                "retries": 3,
                "startPeriod": 60
            },
            "memory": 4096,
            "cpu": 1024
        }
    ],
    "requiresCompatibilities": [
        "FARGATE"
    ],
    "cpu": "1024",
    "memory": "4096"
}
EOF

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Step 8: Create ECS cluster if it doesn't exist
echo "Creating ECS cluster if it doesn't exist..."
aws ecs create-cluster --cluster-name $ECS_CLUSTER_NAME > /dev/null 2>&1 || echo "Cluster may already exist"

# Step 9: Create or update ECS service
echo "Checking if ECS service exists..."
SERVICE_EXISTS=$(aws ecs describe-services --cluster $ECS_CLUSTER_NAME --services $ECS_SERVICE_NAME --query "services[?status!='INACTIVE'].status" --output text 2>/dev/null || echo "")

if [ -z "$SERVICE_EXISTS" ]; then
    echo "Creating new ECS service..."
    # Get default VPC ID
    VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)
    echo "Using default VPC: $VPC_ID"
    
    # Get subnet IDs from default VPC
    SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[0:2].SubnetId" --output text | tr '\t' ',')
    echo "Using subnets: $SUBNET_IDS"
    
    # Create security group if it doesn't exist
    echo "Creating security group..."
    SG_EXISTS=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=sift-ppe-service-sg" "Name=vpc-id,Values=$VPC_ID" --query "length(SecurityGroups)" --output text)
    
    if [ "$SG_EXISTS" = "0" ]; then
        echo "Creating new security group..."
        SG_ID=$(aws ec2 create-security-group --group-name sift-ppe-service-sg --description "Security group for SIFT PPE Detection Service" --vpc-id $VPC_ID --query "GroupId" --output text)
        
        # Add inbound rules
        aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8000 --cidr 0.0.0.0/0
    else
        echo "Security group already exists, retrieving ID..."
        SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=sift-ppe-service-sg" "Name=vpc-id,Values=$VPC_ID" --query "SecurityGroups[0].GroupId" --output text)
    fi
    
    echo "Using security group: $SG_ID"
    
    aws ecs create-service \
        --cluster $ECS_CLUSTER_NAME \
        --service-name $ECS_SERVICE_NAME \
        --task-definition $ECS_TASK_FAMILY \
        --desired-count 1 \
        --launch-type FARGATE \
        --platform-version LATEST \
        --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
        --scheduling-strategy REPLICA
else
    echo "Updating existing ECS service..."
    aws ecs update-service \
        --cluster $ECS_CLUSTER_NAME \
        --service $ECS_SERVICE_NAME \
        --task-definition $ECS_TASK_FAMILY \
        --desired-count 1
fi

echo "=== Deployment complete ==="
echo "The SIFT PPE Detection Service is now running on ECS Fargate"
echo "Monitor the service at: https://console.aws.amazon.com/ecs/home?region=${AWS_REGION}#/clusters/${ECS_CLUSTER_NAME}/services/${ECS_SERVICE_NAME}"
