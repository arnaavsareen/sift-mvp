terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 4.0"
    }
  }
  required_version = ">= 1.0.0"
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region to deploy resources."
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "S3 bucket name for frames."
  type        = string
  default     = "osha-mvp-bucket"
}

variable "queue_name" {
  description = "SQS queue name for frame messages."
  type        = string
  default     = "OshaFrameQueue"
}

resource "aws_s3_bucket" "frames" {
  bucket = var.bucket_name
  acl    = "private"

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  lifecycle_rule {
    id      = "expire_frames"
    enabled = true
    prefix  = "frames/"
    expiration {
      days = 7
    }
  }
}

resource "aws_sqs_queue" "frame_queue" {
  name                       = var.queue_name
  visibility_timeout_seconds = 30
  message_retention_seconds  = 86400
}

data "aws_iam_policy_document" "ingest_policy_doc" {
  statement {
    effect = "Allow"
    actions = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.frames.arn}/frames/*"]
  }
  statement {
    effect = "Allow"
    actions = ["sqs:SendMessage"]
    resources = [aws_sqs_queue.frame_queue.arn]
  }
}

resource "aws_iam_policy" "ingest_policy" {
  name   = "IngestPolicy"
  policy = data.aws_iam_policy_document.ingest_policy_doc.json
}

data "aws_iam_policy_document" "assume_task" {
  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "ingest_role" {
  name               = "IngestRole"
  assume_role_policy = data.aws_iam_policy_document.assume_task.json
}

resource "aws_iam_role_policy_attachment" "ingest_role_attach" {
  role       = aws_iam_role.ingest_role.name
  policy_arn = aws_iam_policy.ingest_policy.arn
}

resource "aws_iam_user" "ingest_user" {
  name = "IngestUser"
}

resource "aws_iam_user_policy_attachment" "ingest_user_attach" {
  user       = aws_iam_user.ingest_user.name
  policy_arn = aws_iam_policy.ingest_policy.arn
}

resource "aws_iam_access_key" "ingest_user_key" {
  user = aws_iam_user.ingest_user.name
}

output "s3_bucket_name" {
  value = aws_s3_bucket.frames.bucket
}

output "sqs_queue_url" {
  value = aws_sqs_queue.frame_queue.url
}

output "ingest_role_arn" {
  value = aws_iam_role.ingest_role.arn
}

output "dev_access_key_id" {
  value     = aws_iam_access_key.ingest_user_key.id
  sensitive = true
}

output "dev_secret_access_key" {
  value     = aws_iam_access_key.ingest_user_key.secret
  sensitive = true
}
