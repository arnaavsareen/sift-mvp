# SIFT: Real-time AI Compliance Monitoring for Manufacturing

sift-mvp

SIFT is an AWS-native, microservice-based MVP for real-time compliance monitoring in manufacturing. The architecture consists of four main services: data ingestion (from cameras and sensors), AI inference (for real-time analysis), storage (persisting results and metadata), and a web dashboard (for visualization and alerts). Each service is containerized and designed for scalability and independent deployment.

## Service Directories
- `ingestion_service/`: Handles data ingestion from manufacturing cameras/sensors and pushes to AWS S3/SQS.
- `model_service/`: Runs AI inference (e.g., YOLOv5) on incoming data and returns compliance results.
- `api_service/`: Provides REST APIs for data access, storage, and integration with RDS.
- `web_dashboard/`: React-based dashboard for real-time monitoring and visualization.

## Next Steps
- **Day 2:** Implement the ingestion pipeline to capture and push camera data to AWS S3/SQS.

