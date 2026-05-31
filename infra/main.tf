# ── Provider ──────────────────────────────────────────────────────────────────
terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Variables ─────────────────────────────────────────────────────────────────
variable "aws_region"    { default = "us-east-1" }
variable "project_name"  { default = "adobe-mlops" }
variable "environment"   { default = "dev" }

locals {
  prefix = "${var.project_name}-${var.environment}"
  tags   = { Project = var.project_name, Env = var.environment, ManagedBy = "terraform" }
}

# ── ECR Repository ────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "api" {
  name                 = "${local.prefix}-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration { scan_on_push = true }
  encryption_configuration     { encryption_type = "AES256" }

  tags = local.tags
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ── S3 Buckets ────────────────────────────────────────────────────────────────
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket        = "${local.prefix}-mlflow-artifacts-${random_id.suffix.hex}"
  force_destroy = true
  tags          = local.tags
}

resource "aws_s3_bucket" "data" {
  bucket        = "${local.prefix}-data-${random_id.suffix.hex}"
  force_destroy = true
  tags          = local.tags
}

resource "random_id" "suffix" { byte_length = 4 }

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration { status = "Enabled" }
}

# ── IAM Role for SageMaker ────────────────────────────────────────────────────
resource "aws_iam_role" "sagemaker" {
  name = "${local.prefix}-sagemaker-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "sagemaker_s3" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# ── SageMaker Endpoint (blue/green ready) ─────────────────────────────────────
resource "aws_sagemaker_model" "stable" {
  name               = "${local.prefix}-stable"
  execution_role_arn = aws_iam_role.sagemaker.arn

  primary_container {
    image          = "${aws_ecr_repository.api.repository_url}:stable"
    environment    = { MLFLOW_TRACKING_URI = "http://mlflow.internal:5000" }
  }
  tags = local.tags
}

resource "aws_sagemaker_endpoint_configuration" "blue_green" {
  name = "${local.prefix}-ep-config"

  production_variants {
    variant_name           = "stable"
    model_name             = aws_sagemaker_model.stable.name
    initial_instance_count = 1
    instance_type          = "ml.t3.medium"
    initial_variant_weight = 0.8    # 80 % traffic
  }

  tags = local.tags
}

resource "aws_sagemaker_endpoint" "api" {
  name                 = "${local.prefix}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.blue_green.name
  tags                 = local.tags
}

# ── Outputs ───────────────────────────────────────────────────────────────────
output "ecr_repo_url"          { value = aws_ecr_repository.api.repository_url }
output "mlflow_bucket"         { value = aws_s3_bucket.mlflow_artifacts.bucket }
output "sagemaker_endpoint"    { value = aws_sagemaker_endpoint.api.name }
output "sagemaker_role_arn"    { value = aws_iam_role.sagemaker.arn }
