import os

from dotenv import load_dotenv


load_dotenv()


class Config:
    def __init__(self):
        # Existing application settings.
        self.CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
        self.HUGGINGFACE_EMBEDDING_MODEL = self.get_required_env(
            "HUGGINGFACE_EMBEDDING_MODEL"
        )

        # Keep the old attribute name for compatibility with existing code.
        self.GOOGLE_API_KEY = self.get_required_env("GOOGLE_API_KEY")
        self.GROQ_API_KEY = self.GOOGLE_API_KEY

        self.LLM_MODEL = self.get_required_env("LLM_MODEL")
        self.VECTOR_STORE_PATH = self.get_required_env("VECTOR_STORE_PATH")
        self.VECTOR_STORE_DIR = os.getenv(
            "VECTOR_STORE_DIR",
            "app/data/vectorstores",
        )
        self.CHUNK_OVERLAP = int(self.get_required_env("CHUNK_OVERLAP"))
        self.CHUNK_SIZE = int(self.get_required_env("CHUNK_SIZE"))

        # Storage abstraction. Local remains the default so the current
        # application behavior does not change until STORAGE_BACKEND=s3.
        self.STORAGE_BACKEND = os.getenv(
            "STORAGE_BACKEND",
            "local",
        ).strip().lower()

        if self.STORAGE_BACKEND not in {"local", "s3"}:
            raise ValueError(
                "STORAGE_BACKEND must be either 'local' or 's3'"
            )

        self.AWS_REGION = os.getenv("AWS_REGION", "").strip()
        self.AWS_PROFILE = os.getenv("AWS_PROFILE", "").strip()
        self.S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "").strip()
        self.S3_PREFIX = os.getenv("S3_PREFIX", "").strip().strip("/")

        if self.STORAGE_BACKEND == "s3" and not self.S3_BUCKET_NAME:
            raise ValueError(
                "S3_BUCKET_NAME is required when STORAGE_BACKEND=s3"
            )

        # Document processing can remain synchronous for regression testing or
        # use SQS for asynchronous worker-based processing.
        self.DOCUMENT_PROCESSING_MODE = os.getenv(
            "DOCUMENT_PROCESSING_MODE",
            "sync",
        ).strip().lower()
        if self.DOCUMENT_PROCESSING_MODE not in {"sync", "sqs"}:
            raise ValueError(
                "DOCUMENT_PROCESSING_MODE must be either 'sync' or 'sqs'"
            )

        self.SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL", "").strip()
        self.SQS_QUEUE_NAME = os.getenv(
            "SQS_QUEUE_NAME",
            "document-qa-processing",
        ).strip()
        self.SQS_DLQ_NAME = os.getenv(
            "SQS_DLQ_NAME",
            "document-qa-processing-dlq",
        ).strip()
        self.SQS_WAIT_TIME_SECONDS = int(
            os.getenv("SQS_WAIT_TIME_SECONDS", "20")
        )
        self.SQS_VISIBILITY_TIMEOUT = int(
            os.getenv("SQS_VISIBILITY_TIMEOUT", "120")
        )
        self.SQS_MAX_RECEIVE_COUNT = int(
            os.getenv("SQS_MAX_RECEIVE_COUNT", "3")
        )


        # Shared application persistence.
        self.DATABASE_BACKEND = os.getenv(
            "DATABASE_BACKEND",
            "sqlite",
        ).strip().lower()
        if self.DATABASE_BACKEND not in {"sqlite", "postgres"}:
            raise ValueError(
                "DATABASE_BACKEND must be either 'sqlite' or 'postgres'"
            )

        self.SQLITE_DATABASE_PATH = os.getenv(
            "SQLITE_DATABASE_PATH",
            "app/data/app.db",
        ).strip()

        self.DATABASE_URL = os.getenv(
            "DATABASE_URL",
            "",
        ).strip()

        if self.DATABASE_BACKEND == "postgres" and not self.DATABASE_URL:
            raise ValueError(
                "DATABASE_URL is required when DATABASE_BACKEND=postgres"
            )

    def get_required_env(self, env_variable):
        value = os.getenv(env_variable)
        if value is None or not value.strip():
            raise ValueError(
                f"Invalid or missing '{env_variable}' in the environment variables"
            )
        return value