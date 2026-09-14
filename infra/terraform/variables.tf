variable "aws_region" {
  type    = string
  default = "us-east-2"
}

variable "project_name" {
  type    = string
  default = "document-qa"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,23}$", var.project_name))
    error_message = "Use 3-24 lowercase letters, digits or hyphens, beginning with a letter."
  }
}

variable "environment" {
  type    = string
  default = "dev"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging or prod."
  }
}

variable "vpc_id" {
  description = "Existing VPC; no VPC is queried or created during local validation."
  type        = string
}

variable "private_subnet_ids" {
  description = "Existing private subnets in the VPC, spanning at least two AZs. Must provide outbound connectivity for ECS."
  type        = list(string)
  validation {
    condition     = length(distinct(var.private_subnet_ids)) >= 2
    error_message = "Provide at least two distinct private subnet IDs."
  }
}

variable "api_client_cidrs" {
  description = "Private network CIDRs allowed to reach API port 8000. Empty means no inbound API access."
  type        = set(string)
  default     = []
  validation {
    condition     = alltrue([for cidr in var.api_client_cidrs : can(cidrnetmask(cidr))])
    error_message = "Every entry must be a valid IPv4 CIDR."
  }
}

variable "container_image" {
  description = "Prebuilt Linux/amd64 image URI for the current Agent stack. No image is built or pushed by Terraform."
  type        = string
}

variable "api_desired_count" {
  description = "Keep zero until image, secret values, and networking are verified in a future deployment."
  type        = number
  default     = 0
  validation {
    condition     = var.api_desired_count >= 0 && floor(var.api_desired_count) == var.api_desired_count
    error_message = "Desired count must be a nonnegative integer."
  }
}

variable "worker_desired_count" {
  type    = number
  default = 0
  validation {
    condition     = var.worker_desired_count >= 0 && floor(var.worker_desired_count) == var.worker_desired_count
    error_message = "Desired count must be a nonnegative integer."
  }
}

variable "llm_model" {
  type    = string
  default = "gemini-3.5-flash"
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "queue_visibility_timeout" {
  description = "Worker has no visibility renewal yet; adjust to measured document processing time before deployment."
  type        = number
  default     = 900
  validation {
    condition     = var.queue_visibility_timeout >= 30 && var.queue_visibility_timeout <= 43200 && floor(var.queue_visibility_timeout) == var.queue_visibility_timeout
    error_message = "Visibility timeout must be an integer between 30 and 43200 seconds."
  }
}
