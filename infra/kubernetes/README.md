# Local Kubernetes and Kafka

This is a local development deployment for Docker Desktop Kubernetes. It uses
no AWS resources and does not run Terraform. The original sync/SQS paths and
AWS files remain available. The root `.env` is not changed by this setup.

## What runs

```text
Browser -> Streamlit -> FastAPI -> Kafka document-processing topic
                          |                   |
                          |               Kafka worker
                          |                   |
                          +-- shared PVC -----+
                              SQLite, PDFs, text, FAISS, model cache

Worker -> document-processing-dlq (invalid or exhausted jobs)
```

The application Deployment has one pod containing API, worker, and frontend
containers. This keeps SQLite and local files on the same node and shared volume.
An init container creates the database before API/worker startup. Kafka runs in
a separate single-broker KRaft StatefulSet with its own persistent volume.
ConfigMap, Secret, Services, resource limits and startup/readiness probes support
the deployment. Services are cluster-local; access uses port forwarding.

Kafka handles **PDF ingestion**, not LLM conversations: upload returns HTTP 202
with a job ID, the worker extracts text and builds FAISS, and the frontend polls
the existing job endpoint. LangGraph continues to handle questions after the
document becomes ready. The initial embedding-model download requires internet;
Gemini calls also require internet and your API key.

## Option A: test Kafka with the existing host application

Start Docker Desktop with Linux containers. From the repository root, activate
the existing Python 3.12 environment and install the updated requirements:

```powershell
conda activate rag_llm
python -m pip install -r requirements.txt
docker compose -f docker-compose.kafka.yml up -d --wait
```

In **both** API and worker terminals, from the same repository directory, set:

```powershell
conda activate rag_llm
$env:DOCUMENT_PROCESSING_MODE = "kafka"
$env:STORAGE_BACKEND = "local"
$env:DATABASE_BACKEND = "sqlite"
$env:SQLITE_DATABASE_PATH = "app/data/app.db"
$env:KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
```

Start the API in one terminal and the worker in another:

```powershell
# Terminal 1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# Terminal 2
python -m app.workers.kafka_worker
```

Start the frontend in a third activated terminal:

```powershell
python -m streamlit run frontend.py
```

Open http://localhost:8501, upload a small text PDF and watch it progress from
PENDING to PROCESSING to COMPLETED. Once attached, use Strict Document mode to
ask a question. A corrupt PDF should retry up to three times and finish FAILED.
To see PENDING clearly, upload while the worker is stopped, then start it.
Only run one worker for this local setup.

Inspect failed messages:

```powershell
docker compose -f docker-compose.kafka.yml exec kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic document-processing-dlq --from-beginning --max-messages 1 --timeout-ms 10000
```

The dead-letter payload contains base64-encoded original job JSON and an error;
the PDF itself stays on disk. Stop host processes with Ctrl+C and stop the broker
with `docker compose -f docker-compose.kafka.yml down` (keeps its data volume).
To return to synchronous uploads, set `$env:DOCUMENT_PROCESSING_MODE = "sync"`
and restart the API; the worker is then unnecessary.

## Option B: run everything in local Kubernetes

1. Enable Kubernetes in Docker Desktop settings and wait until it is running.
   Use Linux containers and allocate sufficient memory for Python ML packages
   and Kafka (8 GiB is a reasonable starting point). Check that the local context
   and default storage class exist:

   ```powershell
   kubectl --context docker-desktop get nodes
   kubectl --context docker-desktop get storageclass
   ```

2. Build the current application image from the repository root. This uses the
   Agent-compatible `requirements.txt`, not the retained ECS dependency file.

   ```powershell
   docker build -f Dockerfile.local -t document-qa:local .
   ```

   The manifests use `imagePullPolicy: Never`. Docker Desktop's Kubernetes must
   have this image available. If its provisioner uses a separate image store,
   load the image into that cluster's nodes before deploying; an
   `ErrImageNeverPull` event means the image is missing in that store.

3. Create the namespace and copy **only** the Google API key from the existing
   root `.env` into a Secret. The helper sends the key through stdin, without
   printing it or writing a secret manifest. Other `.env` settings do not override
   Kubernetes configuration.

   ```powershell
   kubectl --context docker-desktop apply -f infra/kubernetes/namespace.yaml
   python infra/kubernetes/create_secret.py
   ```

4. Render, apply and wait:

   ```powershell
   kubectl kustomize infra/kubernetes
   kubectl --context docker-desktop apply -k infra/kubernetes
   kubectl --context docker-desktop -n document-qa rollout status statefulset/kafka --timeout=300s
   kubectl --context docker-desktop -n document-qa rollout status deployment/document-qa --timeout=600s
   ```

5. Forward ports and leave the terminal open:

   ```powershell
   kubectl --context docker-desktop -n document-qa port-forward service/document-qa 8501:8501 8000:8000
   ```

   Stop the host API/frontend first if they occupy these ports. Open
   http://localhost:8501 (UI) or http://localhost:8000/docs (API), then perform the
   same upload and question checks as Option A. Kubernetes uses a new PVC-backed
   database; it does not import or overwrite the host's existing conversations.

## Inspection and updates

```powershell
kubectl --context docker-desktop -n document-qa get pods,pvc,services
kubectl --context docker-desktop -n document-qa logs deployment/document-qa -c worker --tail=100
kubectl --context docker-desktop -n document-qa logs deployment/document-qa -c api --tail=100
kubectl --context docker-desktop -n document-qa describe pods
kubectl --context docker-desktop -n document-qa exec kafka-0 -- /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group document-workers
```

After code changes, rebuild the local image and restart the application:

```powershell
docker build -f Dockerfile.local -t document-qa:local .
kubectl --context docker-desktop -n document-qa rollout restart deployment/document-qa
```

After changing the ConfigMap or Secret, also restart the application because
container environment variables are loaded at startup. Worker startup can briefly
fail while Kafka starts; Kubernetes restarts it automatically. A host worker exits
on broker/commit failures and must be restarted manually.

To stop workloads while retaining data:

```powershell
kubectl --context docker-desktop -n document-qa scale deployment/document-qa --replicas=0
kubectl --context docker-desktop -n document-qa scale statefulset/kafka --replicas=0
```

Reapplying the manifests restores replicas to one. Deleting the namespace or
PVCs can delete persisted documents, conversations and Kafka records.

## Delivery behavior and scope

- Producer waits for broker acknowledgement; records contain job metadata only.
- Consumer disables auto-commit and processes one record at a time. It commits
  after processing succeeds, cancellation, or successful dead-letter handling.
- Attempts are persisted in SQLite, so restarting does not reset the retry limit.
  Unknown/malformed/mismatched jobs go to the DLQ without touching another job.
- Completed/cancelled jobs are skipped on redelivery. Reattaching the same PDF
  preserves chat history. Delivery is **at least once**, not exactly once;
  a crash between external effects and commits may repeat processing or DLQ entries.
- Database updates and Kafka publication are not one transaction. A process crash
  before publication can leave a PENDING job without a Kafka record; an ambiguous
  publish timeout may leave a FAILED job with a delivered record. Automatic outbox
  recovery and a retry UI are not implemented. Inspect job state and worker/DLQ
  logs; do not advertise this as a production-grade job system.
- Keep one app replica, one worker and one input partition. Do not scale this
  SQLite/local-volume design horizontally. Production scaling requires shared
  object storage, a shared database, job claiming, and stronger idempotency.
- Kafka is local PLAINTEXT, replication factor one, without broker redundancy or
  authentication. It is not exposed publicly. Processing across all attempts must
  fit within the consumer's one-hour poll interval; very large jobs are unsupported.
- The worker has no application-level readiness endpoint. Pod readiness does not
  prove end-to-end Kafka consumption; verify an actual upload and its job status.

## Validation

```powershell
python -B -m unittest discover -s tests -v
docker compose -f docker-compose.kafka.yml config --quiet
kubectl kustomize infra/kubernetes
```

Offline tests cover queue dispatch, success/redelivery, retries, cancellation,
payload validation, failed DLQ publication and offset commit behavior. They mock
Kafka and PDF processing; they do not prove live broker or cluster operation.
At implementation time Docker Desktop's engine was not running and no Kubernetes
context was configured, so image builds and live deployment were not verified.

References: [Apache Kafka Docker guide](https://kafka.apache.org/41/getting-started/docker/),
[Kafka consumer configuration](https://kafka-python.readthedocs.io/en/stable/apidoc/KafkaConsumer.html),
[Kubernetes multi-container Pods](https://kubernetes.io/docs/concepts/workloads/pods/).
