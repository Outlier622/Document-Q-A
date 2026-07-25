import json
from typing import Any

import boto3

from app.config.configuration import Config
from app.core.logger import configure_logging


logger = configure_logging("QUEUE_SERVICE")


class QueueService:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.queue_url = self.config.SQS_QUEUE_URL
        self._client = None

    def _get_client(self):
        if self._client is None:
            session_kwargs = {}
            if self.config.AWS_PROFILE:
                session_kwargs["profile_name"] = self.config.AWS_PROFILE

            session = boto3.Session(**session_kwargs)
            client_kwargs = {}
            if self.config.AWS_REGION:
                client_kwargs["region_name"] = self.config.AWS_REGION

            self._client = session.client("sqs", **client_kwargs)
        return self._client

    def _require_queue_url(self) -> str:
        if not self.queue_url:
            raise RuntimeError(
                "SQS_QUEUE_URL is required for asynchronous document processing"
            )
        return self.queue_url

    def send_document_job(
        self,
        job_id: str,
        session_id: str,
        document_id: str,
        uploaded_filename: str,
    ) -> str:
        body = {
            "job_id": job_id,
            "session_id": session_id,
            "document_id": document_id,
            "uploaded_filename": uploaded_filename,
        }
        response = self._get_client().send_message(
            QueueUrl=self._require_queue_url(),
            MessageBody=json.dumps(body),
        )
        message_id = response.get("MessageId", "")
        logger.info("Queued document job. job_id=%s message_id=%s", job_id, message_id)
        return message_id

    def receive_document_jobs(self, max_messages: int = 1) -> list[dict[str, Any]]:
        response = self._get_client().receive_message(
            QueueUrl=self._require_queue_url(),
            MaxNumberOfMessages=max(1, min(max_messages, 10)),
            WaitTimeSeconds=self.config.SQS_WAIT_TIME_SECONDS,
            VisibilityTimeout=self.config.SQS_VISIBILITY_TIMEOUT,
            AttributeNames=["ApproximateReceiveCount"],
        )
        return response.get("Messages", [])

    def delete_message(self, receipt_handle: str) -> None:
        self._get_client().delete_message(
            QueueUrl=self._require_queue_url(),
            ReceiptHandle=receipt_handle,
        )