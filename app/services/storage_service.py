import shutil
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from app.config.configuration import Config
from app.core.logger import configure_logging


logger = configure_logging("STORAGE_SERVICE")


class StorageService:
    """Storage abstraction for local development and S3-backed persistence.

    Processing still uses local working files because the PDF parser and FAISS
    APIs operate on filesystem paths. In S3 mode, durable artifacts are synced
    to S3 and the local filesystem acts as a cache/work directory.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.backend = self.config.STORAGE_BACKEND
        self.bucket_name = self.config.S3_BUCKET_NAME
        self.prefix = self.config.S3_PREFIX
        self._s3_client = None

    @property
    def is_s3(self) -> bool:
        return self.backend == "s3"

    def _get_s3_client(self):
        if not self.is_s3:
            raise RuntimeError("S3 client requested while STORAGE_BACKEND=local")

        if self._s3_client is None:
            session_kwargs = {}
            if self.config.AWS_PROFILE:
                session_kwargs["profile_name"] = self.config.AWS_PROFILE

            session = boto3.Session(**session_kwargs)
            client_kwargs = {}
            if self.config.AWS_REGION:
                client_kwargs["region_name"] = self.config.AWS_REGION

            self._s3_client = session.client("s3", **client_kwargs)

        return self._s3_client

    def make_key(self, *parts: str) -> str:
        cleaned_parts = [
            str(part).strip("/")
            for part in parts
            if part is not None and str(part).strip("/")
        ]
        if self.prefix:
            cleaned_parts.insert(0, self.prefix)
        return "/".join(cleaned_parts)

    def _document_base_key(self, session_id: str, document_id: str) -> str:
        return self.make_key(
            "sessions",
            session_id,
            "documents",
            document_id,
        )

    def pdf_key(self, session_id: str, document_id: str) -> str:
        return f"{self._document_base_key(session_id, document_id)}/original.pdf"

    def text_key(self, session_id: str, document_id: str) -> str:
        return f"{self._document_base_key(session_id, document_id)}/extracted.txt"

    def vector_prefix(self, session_id: str, document_id: str) -> str:
        return f"{self._document_base_key(session_id, document_id)}/faiss"

    def save_uploaded_pdf(
        self,
        upload_file,
        local_path: str,
        session_id: str,
        document_id: str,
    ) -> str | None:
        """Save an uploaded PDF locally and, in S3 mode, persist it to S3."""
        local_file = Path(local_path)
        local_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            upload_file.file.seek(0)
        except Exception:
            pass

        with local_file.open("wb") as output_file:
            shutil.copyfileobj(upload_file.file, output_file)

        if not self.is_s3:
            return None

        key = self.pdf_key(session_id, document_id)
        self.upload_file_to_key(local_file, key)
        logger.info(
            "Uploaded PDF to S3. bucket=%s key=%s",
            self.bucket_name,
            key,
        )
        return key

    def sync_processed_artifacts(
        self,
        session_id: str,
        document_id: str,
        text_path: str,
        vector_store_path: str,
    ) -> dict:
        """Upload extracted text and all FAISS artifacts when using S3."""
        if not self.is_s3:
            return {
                "storage_backend": "local",
                "text_key": None,
                "vector_keys": [],
            }

        text_file = Path(text_path)
        if not text_file.exists():
            raise FileNotFoundError(f"Extracted text not found: {text_file}")

        text_key = self.text_key(session_id, document_id)
        self.upload_file_to_key(text_file, text_key)

        vector_dir = Path(vector_store_path)
        if not vector_dir.exists():
            raise FileNotFoundError(
                f"FAISS vector store not found: {vector_store_path}"
            )

        vector_keys = []
        vector_prefix = self.vector_prefix(session_id, document_id)

        for local_file in sorted(vector_dir.rglob("*")):
            if not local_file.is_file():
                continue
            relative_path = local_file.relative_to(vector_dir).as_posix()
            key = f"{vector_prefix}/{relative_path}"
            self.upload_file_to_key(local_file, key)
            vector_keys.append(key)

        if not vector_keys:
            raise FileNotFoundError(
                f"No FAISS artifacts found in: {vector_store_path}"
            )

        logger.info(
            "Synced processed document artifacts to S3. "
            "session_id=%s document_id=%s vector_files=%s",
            session_id,
            document_id,
            len(vector_keys),
        )

        return {
            "storage_backend": "s3",
            "text_key": text_key,
            "vector_keys": vector_keys,
        }

    def ensure_vector_store_local(
        self,
        session_id: str,
        document_id: str,
        local_vector_store_path: str,
    ) -> bool:
        """Ensure a FAISS directory exists locally.

        Returns True when artifacts had to be downloaded from S3 and False when
        a complete local cache was already available.
        """
        vector_dir = Path(local_vector_store_path)
        if self._is_vector_store_ready(vector_dir):
            return False

        if not self.is_s3:
            raise FileNotFoundError(
                f"FAISS index not found at: {local_vector_store_path}"
            )

        prefix = self.vector_prefix(session_id, document_id).rstrip("/") + "/"
        client = self._get_s3_client()
        paginator = client.get_paginator("list_objects_v2")

        objects = []
        for page in paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=prefix,
        ):
            objects.extend(page.get("Contents", []))

        if not objects:
            raise FileNotFoundError(
                f"No FAISS artifacts found in S3 under: {prefix}"
            )

        vector_dir.mkdir(parents=True, exist_ok=True)

        for item in objects:
            key = item["Key"]
            relative_path = key[len(prefix):]
            if not relative_path:
                continue

            destination = vector_dir / Path(relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.download_file_from_key(key, destination)

        if not self._is_vector_store_ready(vector_dir):
            raise FileNotFoundError(
                "FAISS artifacts were downloaded from S3 but the local "
                f"vector store is incomplete: {local_vector_store_path}"
            )

        logger.info(
            "Restored FAISS artifacts from S3 cache. "
            "session_id=%s document_id=%s",
            session_id,
            document_id,
        )
        return True

    def _is_vector_store_ready(self, vector_dir: Path) -> bool:
        if not vector_dir.exists() or not vector_dir.is_dir():
            return False

        return (
            (vector_dir / "index.faiss").exists()
            and (vector_dir / "index.pkl").exists()
        )

    def upload_file_to_key(self, local_path: str | Path, key: str) -> None:
        if not self.is_s3:
            raise RuntimeError("upload_file_to_key requires STORAGE_BACKEND=s3")

        client = self._get_s3_client()
        client.upload_file(
            str(local_path),
            self.bucket_name,
            key,
        )

    def download_file_from_key(
        self,
        key: str,
        local_path: str | Path,
    ) -> None:
        if not self.is_s3:
            raise RuntimeError(
                "download_file_from_key requires STORAGE_BACKEND=s3"
            )

        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        client = self._get_s3_client()
        client.download_file(
            self.bucket_name,
            key,
            str(destination),
        )

    def delete_key(self, key: str) -> None:
        if not self.is_s3:
            raise RuntimeError("delete_key requires STORAGE_BACKEND=s3")

        client = self._get_s3_client()
        client.delete_object(Bucket=self.bucket_name, Key=key)

    def validate_s3_access(self) -> None:
        if not self.is_s3:
            raise RuntimeError(
                "Set STORAGE_BACKEND=s3 before validating S3 access"
            )

        try:
            self._get_s3_client().head_bucket(Bucket=self.bucket_name)
        except ClientError:
            logger.exception(
                "Unable to access S3 bucket: %s",
                self.bucket_name,
            )
            raise
    def ensure_pdf_local(
        self,
        session_id: str,
        document_id: str,
        local_pdf_path: str,
    ) -> bool:
        local_file = Path(local_pdf_path)

        if local_file.exists() and local_file.is_file():
            return False

        if not self.is_s3:
            raise FileNotFoundError(
                f"PDF not found at: {local_pdf_path}"
            )

        key = self.pdf_key(session_id, document_id)
        self.download_file_from_key(key, local_file)

        if not local_file.exists() or not local_file.is_file():
            raise FileNotFoundError(
                f"PDF could not be restored from S3: {key}"
            )

        logger.info(
            "Restored source PDF from S3. "
            "session_id=%s document_id=%s key=%s",
            session_id,
            document_id,
            key,
        )
        return True