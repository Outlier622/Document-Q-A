import os

from dotenv import load_dotenv


load_dotenv()


class Config:
    def __init__(self):
        self.QUERY_ENGINE = os.getenv("QUERY_ENGINE", "agent").strip().lower()
        if self.QUERY_ENGINE not in {"agent", "legacy"}:
            raise ValueError("QUERY_ENGINE must be agent or legacy")
        self.AGENT_MAX_ROUNDS = int(os.getenv("AGENT_MAX_ROUNDS", "4"))
        self.AGENT_MAX_TOOL_CALLS = int(os.getenv("AGENT_MAX_TOOL_CALLS", "6"))
        self.AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "180"))
        self.AGENT_TOP_K = int(os.getenv("AGENT_TOP_K", "5"))
        for name, upper in (("AGENT_MAX_ROUNDS", 12), ("AGENT_MAX_TOOL_CALLS", 20),
                            ("AGENT_TIMEOUT_SECONDS", 240), ("AGENT_TOP_K", 20)):
            if not 1 <= getattr(self, name) <= upper:
                raise ValueError(f"{name} must be between 1 and {upper}")
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
        # use Kafka or SQS for asynchronous worker-based processing.
        self.DOCUMENT_PROCESSING_MODE = os.getenv(
            "DOCUMENT_PROCESSING_MODE",
            "sync",
        ).strip().lower()
        if self.DOCUMENT_PROCESSING_MODE not in {"sync", "sqs", "kafka"}:
            raise ValueError(
                "DOCUMENT_PROCESSING_MODE must be sync, sqs, or kafka"
            )

        self.KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "document-processing")
        self.KAFKA_DLQ_TOPIC = os.getenv("KAFKA_DLQ_TOPIC", "document-processing-dlq")
        self.KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "document-workers")
        self.KAFKA_MAX_ATTEMPTS = int(os.getenv("KAFKA_MAX_ATTEMPTS", "3"))
        if not 1 <= self.KAFKA_MAX_ATTEMPTS <= 10:
            raise ValueError("KAFKA_MAX_ATTEMPTS must be between 1 and 10")
        if self.KAFKA_TOPIC == self.KAFKA_DLQ_TOPIC:
            raise ValueError("Kafka input and dead-letter topics must be different")

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
