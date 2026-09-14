locals {
  task_trust_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" }, Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role" "execution" {
  name               = "${local.name}-execution"
  assume_role_policy = local.task_trust_policy
}

resource "aws_iam_role_policy" "execution" {
  role = aws_iam_role.execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*" },
      { Effect = "Allow", Action = ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"], Resource = aws_ecr_repository.application.arn },
      { Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = [for group in aws_cloudwatch_log_group.application : "${group.arn}:*"] },
      { Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = [aws_secretsmanager_secret.google_api_key.arn, aws_secretsmanager_secret.database_url.arn] }
    ]
  })
}

resource "aws_iam_role" "task" {
  for_each           = local.services
  name               = "${local.name}-${each.key}"
  assume_role_policy = local.task_trust_policy
}

resource "aws_iam_role_policy" "task" {
  for_each = local.services
  role     = aws_iam_role.task[each.key].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:ListBucket"], Resource = aws_s3_bucket.documents.arn,
      Condition = { StringLike = { "s3:prefix" = ["${local.s3_prefix}/*"] } } },
      { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject"], Resource = "${aws_s3_bucket.documents.arn}/${local.s3_prefix}/*" },
      { Effect = "Allow", Action = each.key == "api" ? ["sqs:SendMessage"] : ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"], Resource = aws_sqs_queue.processing.arn }
    ]
  })
}
