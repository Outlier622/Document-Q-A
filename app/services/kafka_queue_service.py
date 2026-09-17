"""Lazy Kafka producer: sync mode never needs a broker or Kafka imports."""
import json
import threading

from app.config.configuration import Config


class KafkaQueueService:
    def __init__(self, config: Config):
        self.config = config
        self._producer = None
        self._lock = threading.Lock()

    def _get_producer(self):
        with self._lock:
            if self._producer is None:
                from kafka import KafkaProducer
                self._producer = KafkaProducer(
                    bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS.split(","),
                    acks="all", retries=5, max_in_flight_requests_per_connection=1,
                    max_block_ms=10000, request_timeout_ms=10000,
                    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
                )
            return self._producer

    def send_document_job(self, job_id, session_id, document_id, uploaded_filename):
        result = self._get_producer().send(
            self.config.KAFKA_TOPIC,
            key=session_id.encode("utf-8"),
            value=dict(job_id=job_id, session_id=session_id,
                       document_id=document_id, uploaded_filename=uploaded_filename),
        ).get(timeout=30)
        return f"{result.topic}:{result.partition}:{result.offset}"

    def send_dead_letter(self, raw_value: bytes, reason: str):
        # Keep the original bytes losslessly for inspection; PDFs are never in Kafka.
        import base64
        self._get_producer().send(
            self.config.KAFKA_DLQ_TOPIC,
            value={"payload_base64": base64.b64encode(raw_value).decode("ascii"),
                   "error": reason},
        ).get(timeout=30)

    def close(self):
        if self._producer is not None:
            self._producer.close(timeout=10)
