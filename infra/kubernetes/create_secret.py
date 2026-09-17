"""Copy only GOOGLE_API_KEY from the ignored root .env to local Kubernetes."""
import base64
import json
from pathlib import Path
import subprocess

from dotenv import dotenv_values


def main():
    values = dotenv_values(Path(__file__).resolve().parents[2] / ".env")
    key = values.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Set GOOGLE_API_KEY in the project's .env first.")
    secret = {
        "apiVersion": "v1", "kind": "Secret", "type": "Opaque",
        "metadata": {"name": "document-qa-secrets", "namespace": "document-qa"},
        "data": {"GOOGLE_API_KEY": base64.b64encode(key.encode()).decode()},
    }
    subprocess.run(
        ["kubectl", "--context", "docker-desktop", "apply", "-f", "-"],
        input=json.dumps(secret), text=True, check=True,
    )


if __name__ == "__main__":
    main()
