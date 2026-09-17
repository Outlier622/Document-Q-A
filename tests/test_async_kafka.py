"""Offline queue tests share the existing disposable database fixture."""
import asyncio
import io
import json
import uuid
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_foundation_interfaces  # isolates the DB before importing services
from fastapi import UploadFile
from app.services import session_service as sessions, rag_service
from app.services.kafka_queue_service import KafkaQueueService
from app.services.queue_service import create_queue_service, QueueService
from app.workers import kafka_worker as worker


class KafkaTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(KAFKA_MAX_ATTEMPTS=3)
        self.queue = Mock()
        self.session = sessions.start_or_resume_session(str(uuid.uuid4()))["session_id"]
        self.body = dict(job_id=str(uuid.uuid4()), session_id=self.session,
                         document_id=str(uuid.uuid4()), uploaded_filename="test.pdf")
        sessions.create_processing_job(**self.body)
        self.raw = json.dumps(self.body).encode()
        sleeper = patch.object(worker.time, "sleep")
        sleeper.start()
        self.addCleanup(sleeper.stop)
        log_patch = patch.object(worker.logger, "exception")
        log_patch.start()
        self.addCleanup(log_patch.stop)

    def job(self):
        return sessions.get_processing_job(self.body["job_id"])

    def process(self):
        worker.process_message(self.raw, self.queue, self.config)

    def test_success_and_duplicate_delivery(self):
        with patch.object(worker, "process_document_job") as process:
            self.process()
            self.process()
        process.assert_called_once()
        self.assertEqual(self.job()["status"], "COMPLETED")
        self.queue.send_dead_letter.assert_not_called()

    def test_retries_then_success(self):
        with patch.object(worker, "process_document_job", side_effect=[RuntimeError("temporary"), None]):
            self.process()
        self.assertEqual(self.job()["status"], "COMPLETED")
        self.assertEqual(self.job()["attempt_count"], 2)

    def test_retry_exhaustion_publishes_dead_letter(self):
        with patch.object(worker, "process_document_job", side_effect=RuntimeError("bad PDF")) as process:
            self.process()
        self.assertEqual(process.call_count, 3)
        self.assertEqual(self.job()["status"], "FAILED")
        self.queue.send_dead_letter.assert_called_once_with(self.raw, "bad PDF")

    def test_dead_letter_failure_propagates_and_resume_does_not_reprocess(self):
        self.queue.send_dead_letter.side_effect = RuntimeError("broker unavailable")
        with patch.object(worker, "process_document_job", side_effect=RuntimeError("bad PDF")) as process:
            with self.assertRaisesRegex(RuntimeError, "broker unavailable"):
                self.process()
            self.assertEqual(self.job()["status"], "RETRYING")
            self.queue.send_dead_letter.side_effect = None
            self.process()
        self.assertEqual(process.call_count, 3)
        self.assertEqual(self.job()["status"], "FAILED")

    def test_malformed_and_forged_jobs_never_process(self):
        with patch.object(worker, "process_document_job") as process:
            for payload in [b"not-json", b"[]", b'{}', b'\xff',
                            json.dumps({**self.body, "session_id": "foreign"}).encode()]:
                worker.process_message(payload, self.queue, self.config)
        process.assert_not_called()
        self.assertEqual(self.queue.send_dead_letter.call_count, 5)
        self.assertEqual(self.job()["status"], "PENDING")

    def test_ended_session_cancels_job(self):
        with patch.object(sessions, "is_session_active", return_value=False), \
                patch.object(worker, "process_document_job") as process:
            self.process()
        process.assert_not_called()
        self.assertEqual(self.job()["status"], "CANCELLED")

    def test_duplicate_attachment_preserves_history(self):
        sessions.set_session_document(self.session, self.body["document_id"], "test.pdf")
        sessions.save_message(self.session, self.body["document_id"], "Question", "Answer")
        sessions.set_session_document(self.session, self.body["document_id"], "test.pdf")
        self.assertEqual(len(sessions.get_active_session(self.session)["chat_history"]), 1)

    def test_recovery_after_attachment_does_not_rewrite_live_index(self):
        sessions.set_session_document(self.session, self.body["document_id"], "test.pdf")
        sessions.mark_processing_job_processing(self.body["job_id"], 3)
        with patch.object(worker, "process_document_job") as process:
            self.process()
        process.assert_not_called()
        self.assertEqual(self.job()["status"], "COMPLETED")

    def test_local_upload_is_queued_without_s3(self):
        sessions.mark_processing_job_cancelled(self.body["job_id"])
        body = {**self.body, "job_id": str(uuid.uuid4())}
        with patch.object(rag_service.config, "DOCUMENT_PROCESSING_MODE", "kafka"), \
                patch.object(rag_service, "storage_service") as storage, \
                patch.object(rag_service, "queue_service") as queue:
            queue.send_document_job.return_value = "topic:0:42"
            response = asyncio.run(rag_service.enqueue_pdf_processing(
                pdf_file=UploadFile(filename="test.pdf", file=io.BytesIO(b"PDF")),
                session_id=self.session, document_id=body["document_id"],
                job_id=body["job_id"], saved_pdf_path="unused.pdf",
            ))
        self.assertEqual(response.status_code, 202)
        self.assertEqual(json.loads(response.body)["queue_message_id"], "topic:0:42")
        storage.save_uploaded_pdf.assert_called_once()
        queue.send_document_job.assert_called_once()

    def test_producer_waits_for_ack_and_keys_by_session(self):
        config = SimpleNamespace(KAFKA_TOPIC="jobs")
        queue = KafkaQueueService(config)
        producer = Mock()
        producer.send.return_value.get.return_value = SimpleNamespace(topic="jobs", partition=0, offset=2)
        with patch.object(queue, "_get_producer", return_value=producer):
            self.assertEqual(queue.send_document_job(**self.body), "jobs:0:2")
        self.assertEqual(producer.send.call_args.kwargs["key"], self.session.encode())
        producer.send.return_value.get.assert_called_once_with(timeout=30)

    def test_factory_keeps_sync_independent_of_kafka(self):
        config = SimpleNamespace(DOCUMENT_PROCESSING_MODE="sync", SQS_QUEUE_URL="")
        self.assertIsInstance(create_queue_service(config), QueueService)
        config.DOCUMENT_PROCESSING_MODE = "kafka"
        self.assertIsInstance(create_queue_service(config), KafkaQueueService)

    def test_consumer_commits_only_after_processing_returns(self):
        self.config.DOCUMENT_PROCESSING_MODE = "kafka"
        self.config.KAFKA_TOPIC = "jobs"
        self.config.KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
        self.config.KAFKA_GROUP_ID = "tests"
        consumer = Mock()
        consumer.poll.side_effect = [{0: [SimpleNamespace(value=self.raw)]}, KeyboardInterrupt()]
        with patch.object(worker, "Config", return_value=self.config), \
                patch("kafka.KafkaConsumer", return_value=consumer), \
                patch.object(worker, "KafkaQueueService", return_value=self.queue), \
                patch.object(worker, "process_message") as process:
            consumer.commit.side_effect = lambda: self.assertEqual(process.call_count, 1)
            worker.main()
        consumer.commit.assert_called_once()
        consumer.close.assert_called_once_with(autocommit=False)

    def test_consumer_does_not_commit_failed_record(self):
        self.config.DOCUMENT_PROCESSING_MODE = "kafka"
        self.config.KAFKA_TOPIC = "jobs"
        self.config.KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
        self.config.KAFKA_GROUP_ID = "tests"
        consumer = Mock()
        consumer.poll.return_value = {0: [SimpleNamespace(value=self.raw)]}
        with patch.object(worker, "Config", return_value=self.config), \
                patch("kafka.KafkaConsumer", return_value=consumer), \
                patch.object(worker, "KafkaQueueService", return_value=self.queue), \
                patch.object(worker, "process_message", side_effect=RuntimeError("DLQ offline")):
            with self.assertRaisesRegex(RuntimeError, "DLQ offline"):
                worker.main()
        consumer.commit.assert_not_called()
        consumer.close.assert_called_once_with(autocommit=False)


if __name__ == "__main__":
    unittest.main()
