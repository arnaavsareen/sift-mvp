provider "aws" {
  region = var.aws_region
}

# ECR Repository
resource "aws_ecr_repository" "model_service" {
  name                 = var.ecr_repo_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "model_service" {
  name              = "/ecs/${var.service_name}"
  retention_in_days = 3
}

# IAM Roles
resource "aws_iam_role" "ecs_task_execution_role" {
  name = "ecs-task-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task_role" {
  name = "sift-ppe-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "task_s3" {
  role       = aws_iam_role.ecs_task_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}

resource "aws_iam_role_policy_attachment" "task_sqs" {
  role       = aws_iam_role.ecs_task_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSQSFullAccess"
}

resource "aws_iam_policy" "rds_access" {
  name        = "RDSAccessPolicy"
  description = "Allow RDS database connection"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action   = ["rds-db:connect"]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "task_rds" {
  role       = aws_iam_role.ecs_task_role.name
  policy_arn = aws_iam_policy.rds_access.arn
}

# SSM Parameter for DB Connection
resource "aws_ssm_parameter" "database_url" {
  name        = "/sift/database_url"
  description = "PostgreSQL connection string"
  type        = "SecureString"
  value       = var.database_url
}

# ECS Cluster
resource "aws_ecs_cluster" "sift_cluster" {
  name = var.cluster_name
}

# ECS Task Definition
resource "aws_ecs_task_definition" "model_service" {
  family                   = "${var.service_name}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name      = var.service_name
      image     = "${aws_ecr_repository.model_service.repository_url}:latest"
      essential = true
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.model_service.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "AWS_REGION"
          value = var.aws_region
        },
        {
          name  = "SQS_QUEUE_URL"
          value = var.sqs_queue_url
        },
        {
          name  = "S3_BUCKET_NAME"
          value = var.s3_bucket_name
        },
        {
          name  = "LOG_LEVEL"
          value = "INFO"
        }
      ]
      
      secrets = [
        {
          name      = "DATABASE_URL"
          valueFrom = aws_ssm_parameter.database_url.arn
        }
      ]
      
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

# VPC Configuration (simplified - in practice, use existing VPC or create properly configured one)
resource "aws_default_vpc" "default" {
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [aws_default_vpc.default.id]
  }
}

resource "aws_security_group" "ecs_tasks" {
  name        = "${var.service_name}-sg"
  description = "Allow inbound traffic for SIFT PPE service"
  vpc_id      = aws_default_vpc.default.id

  ingress {
    protocol    = "tcp"
    from_port   = 8000
    to_port     = 8000
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ECS Service
resource "aws_ecs_service" "model_service" {
  name            = var.service_name
  cluster         = aws_ecs_cluster.sift_cluster.id
  task_definition = aws_ecs_task_definition.model_service.arn
  desired_count   = var.service_count
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets          = data.aws_subnets.default.ids
    assign_public_ip = true
  }
}
