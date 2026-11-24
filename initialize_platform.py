
from pathlib import Path
import subprocess
import sys
import os

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")


def run_module(module_name):

    print(f"Initializing: {module_name}")

    subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=ROOT,
        check=True
    )


def initialize_platform():

    if not os.getenv("OPENAI_API_KEY"):

        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    dataset = ROOT / "data" / "insurance_claims.csv"

    risk_model = ROOT / "models" / "risk_model.joblib"

    anomaly_model = ROOT / "models" / "anomaly_model.joblib"

    vector_db = ROOT / "chroma_db" / "chroma.sqlite3"

    if not dataset.exists():

        run_module("data.generate_data")

    if not risk_model.exists():

        run_module("models.train_risk_model")

    if not anomaly_model.exists():

        run_module("models.train_anomaly_model")

    if not vector_db.exists():

        run_module("knowledge_base.ingest")

    print("Platform initialization completed.")

    return True


if __name__ == "__main__":

    initialize_platform()
