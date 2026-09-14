resource "aws_ecr_repository" "application" {
  name                 = local.name
  image_tag_mutability = "IMMUTABLE"
  force_delete         = false
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecs_cluster" "application" {
  name = local.name
}

resource "aws_cloudwatch_log_group" "application" {
  for_each          = local.services
  name              = "/ecs/${local.name}/${each.key}"
  retention_in_days = 14
}

locals {
  application_environment = {
    STORAGE_BACKEND             = "s3"
    S3_BUCKET_NAME              = aws_s3_bucket.documents.id
    S3_PREFIX                   = local.s3_prefix
    AWS_REGION                  = var.aws_region
    DOCUMENT_PROCESSING_MODE    = "sqs"
    SQS_QUEUE_URL               = aws_sqs_queue.processing.url
    SQS_VISIBILITY_TIMEOUT      = tostring(var.queue_visibility_timeout)
    SQS_MAX_RECEIVE_COUNT       = "3"
    DATABASE_BACKEND            = "postgres"
    QUERY_ENGINE                = "agent"
    LLM_MODEL                   = var.llm_model
    HUGGINGFACE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE                  = "1000"
    CHUNK_OVERLAP               = "200"
    VECTOR_STORE_PATH           = "app/data/vectorstores/faiss_index"
    VECTOR_STORE_DIR            = "app/data/vectorstores"
  }
}

resource "aws_ecs_task_definition" "application" {
  for_each                 = local.services
  family                   = "${local.name}-${each.key}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(each.value.cpu)
  memory                   = tostring(each.value.memory)
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task[each.key].arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }
  container_definitions = jsonencode([{
    name        = each.key
    image       = var.container_image
    essential   = true
    command     = each.value.command
    environment = [for name, value in local.application_environment : { name = name, value = value }]
    secrets = [
      { name = "GOOGLE_API_KEY", valueFrom = aws_secretsmanager_secret.google_api_key.arn },
      { name = "DATABASE_URL", valueFrom = aws_secretsmanager_secret.database_url.arn }
    ]
    portMappings = each.key == "api" ? [{ containerPort = 8000, protocol = "tcp" }] : []
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.application[each.key].name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "ecs"
      }
    }
  }])
}

resource "aws_ecs_service" "application" {
  for_each        = local.services
  name            = "${local.name}-${each.key}"
  cluster         = aws_ecs_cluster.application.id
  task_definition = aws_ecs_task_definition.application[each.key].arn
  desired_count   = each.value.count
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.tasks.id]
    assign_public_ip = false
  }
  depends_on = [aws_iam_role_policy.execution, aws_iam_role_policy.task, aws_db_instance.database]
}
