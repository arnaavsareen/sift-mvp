output "ecr_repository_url" {
  description = "The URL of the ECR repository"
  value       = aws_ecr_repository.model_service.repository_url
}

output "ecs_cluster_name" {
  description = "The name of the ECS cluster"
  value       = aws_ecs_cluster.sift_cluster.name
}

output "ecs_service_name" {
  description = "The name of the ECS service"
  value       = aws_ecs_service.model_service.name
}

output "cloudwatch_log_group" {
  description = "The CloudWatch log group for the ECS service"
  value       = aws_cloudwatch_log_group.model_service.name
}

output "task_definition_arn" {
  description = "The ARN of the task definition"
  value       = aws_ecs_task_definition.model_service.arn
}

output "deployment_instructions" {
  description = "Instructions for deploying updates"
  value       = "To deploy updates: (1) Build and push a new Docker image to ECR, (2) Update the ECS service to use the new image."
}
