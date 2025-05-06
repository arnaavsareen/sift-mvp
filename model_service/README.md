# SIFT PPE Detection Service

The SIFT PPE Detection Service processes images from an AWS SQS queue, performs PPE (Personal Protective Equipment) detection using YOLOv8, and stores results in a PostgreSQL database.

## Features

- **Event-driven architecture** using AWS SQS for message queuing
- **YOLOv8 integration** for state-of-the-art PPE violation detection
- **PostgreSQL storage** for detection results and analytics
- **FastAPI endpoints** for controlling and monitoring the processing pipeline
- **Docker containerization** for easy deployment
- **Comprehensive error handling** and monitoring
- **ECS Fargate deployment** for scalable, managed container execution

## Prerequisites

- Python 3.9+
- PostgreSQL database
- AWS account with access to:
  - S3 bucket for storing images
  - SQS queue for message handling
  - ECS Fargate for container deployment
  - ECR for container registry
- Docker (for containerized deployment)

## Environment Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and update with your settings:

   ```bash
   cp .env.example .env
   ```

3. Edit `.env` with your AWS credentials, database URL, and other configuration

## Local Development

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run database migrations:

   ```bash
   alembic upgrade head
   ```

4. Start the API server:

   ```bash
   python -m uvicorn app.api.api:app --reload
   ```

5. Access the API documentation:

   ```plaintext
   http://localhost:8000/docs
   ```

## Docker Deployment

1. Build the Docker image:

   ```bash
   docker build -t sift-model-service .
   ```

2. Run the container with environment variables:

   ```bash
   docker run -d -p 8000:8000 \
     -e AWS_ACCESS_KEY_ID=your_key \
     -e AWS_SECRET_ACCESS_KEY=your_secret \
     -e AWS_REGION=us-east-1 \
     -e DATABASE_URL=postgresql://user:password@host:port/sift_db \
     -e SQS_QUEUE_URL=your_queue_url \
     -e S3_BUCKET_NAME=your_bucket \
     --name sift-model-service \
     sift-model-service
   ```

## AWS ECS Fargate Deployment

For production deployment on AWS ECS Fargate:

1. Use the provided deployment script:

   ```bash
   ./deploy_ecs.sh
   ```

   This script will:
   - Build and push the Docker image to Amazon ECR
   - Create necessary IAM roles and policies
   - Set up CloudWatch logs with 3-day retention
   - Create ECS task definition and service
   - Configure networking and security

2. Alternatively, use Terraform for infrastructure as code:

   ```bash
   cd terraform
   terraform init
   terraform plan
   terraform apply
   ```

## API Endpoints

- **GET /health** - Health check endpoint
- **GET /api/v1/processor/status** - Get processor status
- **GET /api/v1/processor/health** - Comprehensive health check
- **POST /api/v1/processor/start** - Start SQS processing
- **POST /api/v1/processor/stop** - Stop SQS processing
- **POST /api/v1/processor/run-once** - Process a batch once

## Architecture

The service follows a modular architecture:

- **API Layer**: FastAPI application with endpoints for control
- **Service Layer**: Core business logic with SQS processor
- **Model Layer**: YOLOv8 integration for PPE detection
- **Database Layer**: PostgreSQL integration with SQLAlchemy

## License

Copyright (c) 2025 SIFT
