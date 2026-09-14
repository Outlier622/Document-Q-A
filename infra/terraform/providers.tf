provider "aws" {
  region = var.aws_region

  # No credentials, profiles, account lookups, or AWS data sources in this draft.
  # fmt/init -backend=false/validate do not configure a live AWS session.
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}
