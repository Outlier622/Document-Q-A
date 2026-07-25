import json
import time

from app.config.configuration import Config
from app.core.logger import configure_logging
from app.services.document_processing_service import process_document_job
from app.services.queue_service import QueueService
from app.services.session_service import (
    get_processing_job,
    is_session_active,
    mark_processing_job_cancelled,
    mark_processing_job_completed,
    mark_processing_job_failed,
    mark_processing_job_processing,
    mark_processing_job_retrying,
)


logger = configure_logging("DOCUMENT_WORKER")
config = Config()
queue_service = QueueService(config)


def _parse_message(message: dict) -> dict:
    body = json.loads(message.get("Body", "{}"))
    required = ("job_id", "session_id", "document_id", "uploaded_filename")
    missing = [key for key in required if not body.get(key)]
    if missing:
        raise ValueError(f"SQS message missing required fields: {missing}")
    return body


def process_sqs_message(message: dict) -> None:
    receipt_handle = message["ReceiptHandle"]
    attributes = message.get("Attributes", {})
    receive_count = int(attributes.get("ApproximateReceiveCount", "1"))

    try:
        body = _parse_message(message)
    except Exception:
        logger.exception("Discarding malformed SQS message")
        queue_service.delete_message(receipt_handle)
        return

    job_id = body["job_id"]
    job = get_processing_job(job_id)

    if job is None:
        logger.warning("Discarding unknown job message. job_id=%s", job_id)
        queue_service.delete_message(receipt_handle)
        return

    if job["status"] in {"COMPLETED", "CANCELLED"}:
        logger.info(
            "Skipping already-finalized job. job_id=%s status=%s",
            job_id,
            job["status"],
        )
        queue_service.delete_message(receipt_handle)
        return

    if not is_session_active(body["session_id"]):
        mark_processing_job_cancelled(job_id)
        queue_service.delete_message(receipt_handle)
        return

    mark_processing_job_processing(job_id, receive_count)

    try:
        process_document_job(
            session_id=body["session_id"],
            document_id=body["document_id"],
            uploaded_filename=body["uploaded_filename"],
        )
        mark_processing_job_completed(job_id)
        queue_service.delete_message(receipt_handle)
        logger.info("Completed and acknowledged job. job_id=%s", job_id)

    except Exception as error:
        error_message = str(error)
        logger.exception(
            "Document job failed. job_id=%s attempt=%s",
            job_id,
            receive_count,
        )

        if receive_count >= config.SQS_MAX_RECEIVE_COUNT:
            mark_processing_job_failed(
                job_id=job_id,
                attempt_count=receive_count,
                error_message=error_message,
            )
            # Do not delete the message. The queue redrive policy can move it
            # to the DLQ after the configured maxReceiveCount is reached.
        else:
            mark_processing_job_retrying(
                job_id=job_id,
                attempt_count=receive_count,
                error_message=error_message,
            )


def main() -> None:
    if config.DOCUMENT_PROCESSING_MODE != "sqs":
        raise RuntimeError(
            "Set DOCUMENT_PROCESSING_MODE=sqs before starting the worker."
        )
    if config.STORAGE_BACKEND != "s3":
        raise RuntimeError(
            "SQS processing requires STORAGE_BACKEND=s3 so the API and worker "
            "share durable document artifacts."
        )

    logger.info("Document worker started. queue=%s", config.SQS_QUEUE_URL)

    while True:
        try:
            messages = queue_service.receive_document_jobs(max_messages=1)
            for message in messages:
                process_sqs_message(message)
        except KeyboardInterrupt:
            logger.info("Document worker stopped by user")
            break
        except Exception:
            logger.exception("Worker polling error; retrying shortly")
            time.sleep(2)


if __name__ == "__main__":
    main()