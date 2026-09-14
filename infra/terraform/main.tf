locals {
  name      = "${var.project_name}-${var.environment}"
  s3_prefix = "document-qa"
  services = {
    api    = { cpu = 1024, memory = 2048, count = var.api_desired_count, command = ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] }
    worker = { cpu = 2048, memory = 4096, count = var.worker_desired_count, command = ["python", "-m", "app.workers.document_worker"] }
  }
}

resource "aws_s3_bucket" "documents" {
  bucket_prefix = "${local.name}-"
  force_destroy = false
}

resource "aws_s3_bucket_public_access_block" "documents" {
  bucket                  = aws_s3_bucket.documents.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_sqs_queue" "dead_letter" {
  name                      = "${local.name}-processing-dlq"
  message_retention_seconds = 1209600
  sqs_managed_sse_enabled   = true
}

resource "aws_sqs_queue" "processing" {
  name                       = "${local.name}-processing"
  receive_wait_time_seconds  = 20
  visibility_timeout_seconds = var.queue_visibility_timeout
  sqs_managed_sse_enabled    = true
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dead_letter.arn
    maxReceiveCount     = 3
  })
}

resource "aws_security_group" "tasks" {
  name_prefix = "${local.name}-tasks-"
  description = "ECS tasks: private API access and outbound dependency access"
  vpc_id      = var.vpc_id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_vpc_security_group_ingress_rule" "api" {
  for_each          = var.api_client_cidrs
  security_group_id = aws_security_group.tasks.id
  cidr_ipv4         = each.value
  from_port         = 8000
  to_port           = 8000
  ip_protocol       = "tcp"
}

resource "aws_security_group" "database" {
  name_prefix = "${local.name}-database-"
  description = "PostgreSQL access only from ECS tasks"
  vpc_id      = var.vpc_id
}

resource "aws_vpc_security_group_ingress_rule" "database" {
  security_group_id            = aws_security_group.database.id
  referenced_security_group_id = aws_security_group.tasks.id
  from_port                    = 5432
  to_port                      = 5432
  ip_protocol                  = "tcp"
}

resource "aws_db_subnet_group" "database" {
  name       = local.name
  subnet_ids = var.private_subnet_ids
}

resource "aws_db_instance" "database" {
  identifier                  = local.name
  engine                      = "postgres"
  engine_version              = "16"
  instance_class              = var.db_instance_class
  allocated_storage           = 20
  storage_type                = "gp3"
  storage_encrypted           = true
  db_name                     = "document_qa"
  username                    = "document_qa_admin"
  manage_master_user_password = true
  db_subnet_group_name        = aws_db_subnet_group.database.name
  vpc_security_group_ids      = [aws_security_group.database.id]
  publicly_accessible         = false
  backup_retention_period     = 7
  deletion_protection         = true
  skip_final_snapshot         = false
  final_snapshot_identifier   = "${local.name}-final"
}

# Secret containers only. Values are deliberately not handled in Terraform state.
resource "aws_secretsmanager_secret" "google_api_key" {
  name                    = "${local.name}/google-api-key"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret" "database_url" {
  name                    = "${local.name}/database-url"
  recovery_window_in_days = 7
}
