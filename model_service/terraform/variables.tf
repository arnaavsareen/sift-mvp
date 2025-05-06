variable "aws_region" {
  description = "The AWS region to deploy to"
  type        = string
  default     = "us-east-1"
}

variable "ecr_repo_name" {
  description = "Name of the ECR repository"
  type        = string
  default     = "sift-model-service"
}

variable "service_name" {
  description = "Name of the ECS service"
  type        = string
  default     = "sift-ppe-detector"
}

variable "cluster_name" {
  description = "Name of the ECS cluster"
  type        = string
  default     = "sift-cluster"
}

variable "cpu" {
  description = "CPU units for the ECS task"
  type        = string
  default     = "1024"
}

variable "memory" {
  description = "Memory for the ECS task in MB"
  type        = string
  default     = "4096"
}

variable "service_count" {
  description = "Number of instances of the service to run"
  type        = number
  default     = 1
}

variable "database_url" {
  description = "PostgreSQL database connection URL"
  type        = string
  sensitive   = true
  default     = "postgresql://arnaavsareen@host.docker.internal:5432/sift_db"
}

variable "sqs_queue_url" {
  description = "URL of the SQS queue for frame messages"
  type        = string
  default     = "https://sqs.us-east-1.amazonaws.com/529710251669/OshaFrameQueue"
}

variable "s3_bucket_name" {
  description = "S3 bucket name containing frames to process"
  type        = string
  default     = "osha-mvp-bucket"
}
