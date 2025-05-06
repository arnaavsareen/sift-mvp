# SIFT API Service

A comprehensive API service layer for the SIFT PPE detection platform, providing endpoints for detections, cameras, violations, and analytics.

## Features

- **Complete REST API**: Endpoints for detections, cameras, violations, and analytics
- **JWT Authentication**: Secure endpoints with role-based permissions
- **Caching Layer**: Redis-backed caching with local memory fallback
- **Rate Limiting**: Token bucket algorithm to prevent API abuse
- **Database Integration**: SQLAlchemy ORM with PostgreSQL
- **Background Tasks**: Efficient handling of expensive operations
- **Real-time Notifications**: WebSocket support for new violations
- **Comprehensive Documentation**: OpenAPI/Swagger documentation
- **Cost-optimized Deployment**: Configured for AWS ECS Fargate
- **Security Best Practices**: Non-root users, minimal dependencies, secret management

## Architecture

The API service is designed to work alongside the existing SIFT PPE Detection Service. While the detection service processes images from SQS and performs object detection, this API service provides a RESTful interface for accessing and analyzing the detection results.

Key components:
- **FastAPI Framework**: High-performance, modern Python web framework
- **SQLAlchemy ORM**: Database access layer with query optimization
- **Redis Cache**: Optional caching layer for frequently accessed endpoints
- **JWT Authentication**: Token-based authentication with role-based access control
- **WebSockets**: Real-time notifications for new violations
- **Background Task Manager**: Thread pool for expensive operations

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL database
- Redis server (optional, for caching)
- Docker (for containerized deployment)

### Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file based on `.env.example`
5. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

## API Endpoints

The API is organized into the following groups:

### Authentication
- `POST /api/v1/auth/login`: Authenticate and get access token
- `POST /api/v1/auth/login/json`: JSON-based login endpoint

### Detections
- `GET /api/v1/detections`: List detection events with filtering
- `GET /api/v1/detections/{detection_id}`: Get detection details
- `POST /api/v1/detections`: Create a new detection record
- `GET /api/v1/detections/statistics`: Get detection statistics
- `GET /api/v1/detections/export`: Export detection data (CSV/JSON)

### Cameras
- `GET /api/v1/cameras`: List cameras with filtering
- `GET /api/v1/cameras/{camera_id}`: Get camera details
- `POST /api/v1/cameras`: Add a new camera
- `PUT /api/v1/cameras/{camera_id}`: Update camera details
- `PATCH /api/v1/cameras/{camera_id}/status`: Update camera status
- `DELETE /api/v1/cameras/{camera_id}`: Remove a camera
- `GET /api/v1/cameras/{camera_id}/timeline`: Get violation timeline for camera

### Violations
- `GET /api/v1/violations`: List safety violations with filtering
- `GET /api/v1/violations/{violation_id}`: Get violation details
- `POST /api/v1/violations`: Create a new violation record
- `PATCH /api/v1/violations/{violation_id}/status`: Update violation status
- `GET /api/v1/violations/trends`: Get violation trends over time
- `GET /api/v1/violations/hotspots`: Get violation hotspots by location
- `GET /api/v1/violations/export`: Export violation data (CSV/JSON)

### Analytics
- `GET /api/v1/analytics/dashboard`: Get KPI dashboard data
- `GET /api/v1/analytics/charts/compliance`: Get compliance chart data
- `GET /api/v1/analytics/comparison/areas`: Compare safety metrics by area
- `GET /api/v1/analytics/comparison/cameras`: Compare safety metrics by camera
- `POST /api/v1/analytics/export`: Export analytics data (CSV/JSON)

### Users
- `GET /api/v1/users`: List users (admin only)
- `POST /api/v1/users`: Create new user (admin only)
- `GET /api/v1/users/me`: Get current user profile
- `PUT /api/v1/users/me`: Update current user profile
- `PUT /api/v1/users/me/password`: Change current user password
- `GET /api/v1/users/{user_id}`: Get user details (admin only)
- `PUT /api/v1/users/{user_id}`: Update user (admin only)
- `DELETE /api/v1/users/{user_id}`: Delete user (admin only)

## Deployment

### Docker

Build and run with Docker:

```bash
docker build -t sift-api-service .
docker run -p 8000:8000 --env-file .env sift-api-service
```

### AWS ECS Fargate

This service is designed to be deployed to AWS ECS Fargate for cost-efficient operation. The included `deploy_ecs.sh` script handles the deployment process:

1. Set up required AWS resources (ECR repository, ECS cluster, security groups)
2. Build and push Docker image to ECR
3. Register ECS task definition with cost-optimized resource settings
4. Deploy or update ECS service
5. Configure appropriate CloudWatch logs retention (3 days by default)

To deploy:

```bash
# Make sure AWS credentials are set in .env
# AWS_ACCESS_KEY_ID=xxx
# AWS_SECRET_ACCESS_KEY=xxx
# AWS_REGION=us-east-1

chmod +x deploy_ecs.sh
./deploy_ecs.sh
```

## Cost Optimization

The service includes several cost optimization strategies:

1. **Efficient Resource Allocation**: Minimal CPU/memory settings (256CPU/512MB)
2. **Reduced CloudWatch Log Retention**: 3-day retention period instead of default 30 days
3. **Multi-stage Docker Build**: Smaller production image for faster deployment
4. **Caching Layer**: Reduces database load and improves response times
5. **Background Task Manager**: Handles expensive operations without blocking

## Security

Security measures implemented in this service include:

1. **JWT Authentication**: Secure token-based authentication
2. **Role-based Access Control**: Fine-grained permissions
3. **Rate Limiting**: Prevents API abuse
4. **Non-root Container**: Runs as non-privileged user
5. **Minimal Dependencies**: Reduces attack surface
6. **Input Validation**: Comprehensive request validation
7. **Custom Exception Handling**: Prevents information leakage

## License

[MIT License](LICENSE)
