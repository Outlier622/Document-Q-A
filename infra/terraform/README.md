# Terraform infrastructure draft — local validation only

This directory describes a **future AWS deployment** of Document Q&A. It does
not change the running local application or its `.env`. No AWS resources have
been created or deployment-tested by this addition.

## Scope

| File | Resources / responsibility |
|---|---|
| `versions.tf`, `providers.tf` | Terraform and AWS provider constraints; no credentials or remote backend |
| `variables.tf`, `terraform.tfvars.example` | Region, names, external network IDs, image URI, task counts |
| `main.tf` | Private S3 bucket, encrypted SQS queue and DLQ, security groups, private PostgreSQL RDS, empty application secret containers |
| `iam.tf` | ECS execution role and separate API/worker task roles |
| `ecs.tf` | ECR repository, ECS cluster, API/worker task definitions and services, CloudWatch logs |
| `outputs.tf` | Resource names, endpoints and secret references; no secret values |

The infrastructure describes S3 storage, SQS processing and shared PostgreSQL for
separate API/worker containers. Those environment values exist only in task
definitions; the current local application stays on local files, synchronous
processing and SQLite.

## Permitted local checks

Run from this directory with Terraform installed:

```powershell
terraform fmt -check -recursive
terraform init -backend=false -input=false
terraform validate
```

`init` downloads the provider from HashiCorp/Terraform Registry and writes local
metadata and a dependency lock file. This requires internet access, **not an AWS
account connection**. `validate` checks configuration consistency with the
installed provider schema; it does not evaluate a deployment plan or verify AWS
permissions, quotas, network IDs, image availability or secret values.

You do not need `.env`, AWS credentials, or `terraform.tfvars` for these checks.
The checked-in `terraform.tfvars.example` contains placeholders only.

Do not run `plan`, `apply`, `destroy`, `import`, or remote-state commands as part
of this local-only exercise. A zero ECS desired count is **not** a dry-run mode:
applying this configuration would still create billable resources such as RDS.
There is no mechanism here that makes an eventual apply automatically safe or free.

If using the project-local CLI downloaded for validation, from the repository root:

```powershell
& .\.tools\terraform\terraform.exe -chdir=infra/terraform fmt -check -recursive
& .\.tools\terraform\terraform.exe -chdir=infra/terraform init -backend=false -input=false
& .\.tools\terraform\terraform.exe -chdir=infra/terraform validate
```

`.terraform/`, CLI binaries, state files, plan files and real `.tfvars` files are
ignored by Git. Keep `.terraform.lock.hcl` in version control to record the
validated provider selection. Existing AWS files are retained unchanged.

## Work still required before any future deployment

Local validation completed on 2026-09-14 with Terraform **1.13.5** and the
HashiCorp AWS provider **6.64.0** (recorded in `.terraform.lock.hcl`):

- `terraform fmt -check -recursive`: passed.
- `terraform init -backend=false -input=false`: passed; downloaded the signed provider.
- `terraform validate`: passed (`Success! The configuration is valid.`).
- No AWS plan, apply, destroy, import, or account lookup was executed.

The checks above do not complete the deployment prerequisites below.

1. Supply actual VPC/subnet IDs. The subnets must belong to that VPC and span at
   least two availability zones for RDS. VPC, routes, NAT and endpoints are not
   created here. ECS tasks need outbound access to image storage, logs, Secrets
   Manager, S3/SQS, model downloads and Gemini; SG egress alone does not provide it.
2. Build and test a Linux/amd64 image with the **current Agent dependencies**,
   then publish it to this ECR repository. `container_image` must identify that
   image. The preserved `Dockerfile.ecs` / `requirements.ecs.txt` still represent
   the older stack and must be updated/tested separately before deployment.
3. Populate the application secret containers outside Terraform. The Google
   secret contains the API key. The database secret contains a complete URL such
   as `postgresql+psycopg://USER:URL_ENCODED_PASSWORD@HOST:5432/document_qa?sslmode=require`.
   RDS manages its master password in a separate secret; that secret is JSON and
   cannot be injected directly as `DATABASE_URL`. Provision a suitable application
   DB user and decide how password rotation updates the application's URL secret.
4. Review private API access. There is no ALB, public IP, TLS termination, DNS,
   VPN, or frontend deployment here. A private client route and allowed CIDR are
   required; a Streamlit frontend on a laptop cannot reach private tasks by default.
5. Keep both service counts at zero until prerequisites are ready. Review IAM,
   secrets, costs, observability, startup coordination, and database migration.
6. Measure upload processing time before setting SQS visibility timeout. The
   worker still has no heartbeat/visibility renewal and may process duplicates.
   Choose a deletion/backup policy and a unique final snapshot name; RDS deletion
   protection is enabled and S3/ECR force deletion is disabled in this draft.

## Accurate project / resume wording

“Authored Terraform configurations for an AWS document-processing architecture
and validated their formatting and provider-schema consistency locally.”

Do not claim that Terraform deployed the application, proved reproducible cloud
deployment, or passed AWS end-to-end tests based on these local checks.

## References

- [Terraform validate](https://developer.hashicorp.com/terraform/cli/commands/validate)
- [Terraform init](https://developer.hashicorp.com/terraform/cli/commands/init)
- [AWS provider documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
