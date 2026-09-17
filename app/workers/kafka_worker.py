"""Single-consumer local PDF worker with explicit offset commits and a DLQ."""
import json
import time

from app.config.configuration import Config
from app.core.logger import configure_logging
from app.services.document_processing_service import process_document_job
from app.services.kafka_queue_service import KafkaQueueService
from app.services import session_service as sessions

logger = configure_logging("KAFKA_WORKER")


def process_message(raw_value: bytes, queue: KafkaQueueService, config: Config):
    """Return only when this record can be committed. DLQ errors propagate."""
    try:
        body = json.loads(raw_value)
        required = ("job_id", "session_id", "document_id", "uploaded_filename")
        if not isinstance(body, dict) or any(
            not isinstance(body.get(key), str) or not body[key] for key in required
        ):
            raise ValueError("Missing or invalid document job fields")
    except (ValueError, UnicodeDecodeError) as error:
        queue.send_dead_letter(raw_value, str(error))
        return

    job = sessions.get_processing_job(body["job_id"])
    if job is None or any(body[key] != job[key] for key in required):
        queue.send_dead_letter(raw_value, "Unknown job or payload does not match persisted job")
        return
    job_id = job["job_id"]
    if job["status"] in {"COMPLETED", "CANCELLED"}:
        return
    if job["status"] == "FAILED":
        queue.send_dead_letter(raw_value, job["last_error"] or "Job already failed")
        return

    # Recover the crash window after attachment but before job completion.
    # Do not rewrite a live index that the API may already be querying.
    session = sessions.get_active_session(job["session_id"])
    if session and session["document_id"] == job["document_id"]:
        sessions.mark_processing_job_completed(job_id)
        return

    attempt = job["attempt_count"]
    error_message = job["last_error"] or "Attempt limit reached after worker restart"
    while attempt < config.KAFKA_MAX_ATTEMPTS:
        if not sessions.is_session_active(job["session_id"]):
            sessions.mark_processing_job_cancelled(job_id)
            return
        attempt += 1
        sessions.mark_processing_job_processing(job_id, attempt)
        try:
            process_document_job(
                session_id=job["session_id"], document_id=job["document_id"],
                uploaded_filename=job["uploaded_filename"],
            )
        except Exception as error:
            error_message = str(error)
            logger.exception("Processing failed: job=%s attempt=%s", job_id, attempt)
            if not sessions.is_session_active(job["session_id"]):
                sessions.mark_processing_job_cancelled(job_id)
                return
            sessions.mark_processing_job_retrying(job_id, attempt, error_message)
            if attempt < config.KAFKA_MAX_ATTEMPTS:
                time.sleep(2)
        else:
            sessions.mark_processing_job_completed(job_id)
            return

    # Publish first: a broker failure must never silently discard the input.
    queue.send_dead_letter(raw_value, error_message)
    sessions.mark_processing_job_failed(job_id, attempt, error_message)


def main():
    from kafka import KafkaConsumer
    config = Config()
    if config.DOCUMENT_PROCESSING_MODE != "kafka":
        raise RuntimeError("Set DOCUMENT_PROCESSING_MODE=kafka before starting this worker")
    queue = KafkaQueueService(config)
    consumer = KafkaConsumer(
        config.KAFKA_TOPIC,
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS.split(","),
        group_id=config.KAFKA_GROUP_ID,
        enable_auto_commit=False, auto_offset_reset="earliest",
        max_poll_records=1, max_poll_interval_ms=3600000,
    )
    logger.info("Kafka worker started: topic=%s", config.KAFKA_TOPIC)
    try:
        while True:
            records = consumer.poll(timeout_ms=1000, max_records=1)
            for batch in records.values():
                for record in batch:
                    process_message(record.value, queue, config)
                    consumer.commit()
    except KeyboardInterrupt:
        logger.info("Kafka worker stopped")
    finally:
        # Any processing/DLQ/commit error exits without advancing to later records.
        # Kubernetes restarts the worker; Kafka redelivers the uncommitted record.
        consumer.close(autocommit=False)
        queue.close()


if __name__ == "__main__":
    main()
