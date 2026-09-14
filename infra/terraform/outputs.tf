output "document_bucket" {
  value = aws_s3_bucket.documents.id
}
output "processing_queue_url" {
  value = aws_sqs_queue.processing.url
}
output "dead_letter_queue_url" {
  value = aws_sqs_queue.dead_letter.url
}
output "ecr_repository_url" {
  value = aws_ecr_repository.application.repository_url
}
output "ecs_cluster_name" {
  value = aws_ecs_cluster.application.name
}
output "database_endpoint" {
  value = aws_db_instance.database.endpoint
}
output "database_master_secret_arn" {
  description = "Reference only; never outputs the password. Not a DATABASE_URL value."
  value       = aws_db_instance.database.master_user_secret[0].secret_arn
}
output "application_secret_arns" {
  description = "Empty application secret containers; values must be populated outside Terraform before starting tasks."
  value = {
    google_api_key = aws_secretsmanager_secret.google_api_key.arn
    database_url   = aws_secretsmanager_secret.database_url.arn
  }
}
