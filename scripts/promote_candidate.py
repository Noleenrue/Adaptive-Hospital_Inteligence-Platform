"""Promote a validated candidate bundle into models/ with backup and rollback.

Steps: verify candidate -> back up deployed files -> atomic replace ->
validate (tests, /health, /predict) -> restore the backup on any failure.

Usage: python scripts/promote_candidate.py <candidate-version>
"""

import hashlib
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
PROMOTED = [
    "readmission_xgb_pipeline.joblib",
    "platt_calibrator.joblib",
    "manifest.json",
]
# Existing files that are backed up too (legacy files are never modified).
BACKED_UP = PROMOTED + [
    "readmission_xgboost_v1.joblib",
    "readmission_xgboost_v1_metadata.json",
]
# These tests write to and delete from the real PostgreSQL database.
DB_TESTS = ["tests/test_repository.py", "tests/test_database_connection.py"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(directory, names):
    return {
        n: sha256(directory / n) for n in names if (directory / n).exists()
    }


def restore(backup_dir, before):
    """Put models/ back exactly as it was before promotion."""

    for name in PROMOTED:
        target = MODELS_DIR / name

        if name in before:
            shutil.copy2(backup_dir / name, target)
        elif target.exists():
            target.unlink()  # file did not exist before promotion

    after = snapshot(MODELS_DIR, BACKED_UP)

    if after != before:
        raise RuntimeError("Rollback verification FAILED: hashes differ.")


def validate(manifest, run_predict_smoke=False):
    from fastapi.testclient import TestClient

    env = {k: v for k, v in os.environ.items() if k != "READMISSION_MODEL_DIR"}
    cmd = [sys.executable, "-m", "pytest", "tests", "-q"]

    for t in DB_TESTS:
        cmd += ["--deselect", t]

    tests = subprocess.run(
        cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True
    )
    summary = (tests.stdout.strip().splitlines() or ["(no output)"])[-1]
    print("pytest:", summary)

    if tests.returncode != 0:
        print(tests.stdout[-3000:])
        raise RuntimeError("Automated tests failed after promotion.")

    os.environ.pop("READMISSION_MODEL_DIR", None)

    import api.main as api_main

    saved = []
    api_main.save_prediction = lambda **kw: saved.append(kw) or len(saved)

    client = TestClient(api_main.app)
    health = client.get("/health").json()
    print("health:", health)

    expected_id = api_main.predictor.artifact_id

    if health["model_version"] != manifest["version"]:
        raise RuntimeError("/health reports the wrong model version.")

    if health["artifact_id"] != expected_id:
        raise RuntimeError("/health reports the wrong artifact id.")

    body = None
    stored_version = None

    if run_predict_smoke:
        patient = json.loads(
            (PROJECT_ROOT / "scripts" / "_sample_patient.json").read_text()
        )
        response = client.post("/predict", json=patient)

        if response.status_code != 200:
            raise RuntimeError(f"/predict failed: {response.status_code}")

        body = response.json()
        print("predict:", body)

        if body["model_version"] != manifest["version"]:
            raise RuntimeError("/predict reports the wrong model version.")

        if body["artifact_id"] != expected_id:
            raise RuntimeError("/predict reports the wrong artifact id.")

        if saved[0]["model_version"] != api_main.predictor.db_model_version:
            raise RuntimeError("Stored model version is not the loaded version.")

        stored_version = saved[0]["model_version"]

    return summary, health, body, stored_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument(
        "--skip-predict-smoke",
        action="store_true",
        help="Do not create an in-memory sample prediction during validation.",
    )
    args = parser.parse_args()
    version = args.version
    candidate = MODELS_DIR / "candidates" / version

    # 3. Candidate files and manifest must belong together.
    manifest = json.loads((candidate / "manifest.json").read_text("utf-8"))
    candidate_hashes = snapshot(candidate, PROMOTED)
    candidate_artifact_hashes = {
        name: candidate_hashes[name]
        for name in ("readmission_xgb_pipeline.joblib", "platt_calibrator.joblib")
    }

    for name in ("readmission_xgb_pipeline.joblib", "platt_calibrator.joblib"):
        if manifest["files"][name] != candidate_artifact_hashes[name]:
            sys.exit(f"Candidate {name} does not match its manifest hash.")

    from src.models.predictor import ReadmissionPredictor

    loaded = ReadmissionPredictor(model_dir=candidate)  # schema + policy checks
    print("candidate verified:", loaded.model_version, loaded.artifact_id)

    # 1 & 2. Back up and record hashes before touching anything.
    before = snapshot(MODELS_DIR, BACKED_UP)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = MODELS_DIR / "backups" / stamp
    backup_dir.mkdir(parents=True)

    for name in before:
        shutil.copy2(MODELS_DIR / name, backup_dir / name)

    if snapshot(backup_dir, BACKED_UP) != before:
        sys.exit("Backup verification failed; nothing was changed.")

    from src.models.predictor import ReadmissionPredictor as RP

    old = RP()
    record = {
        "created": stamp,
        "previous_model_version": old.model_version,
        "previous_artifact_id": old.artifact_id,
        "previous_sha256": before,
        "candidate_version": manifest["version"],
        "candidate_artifact_id": loaded.artifact_id,
        "candidate_sha256": candidate_hashes,
    }
    (backup_dir / "backup_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print("backup:", backup_dir)

    # 4. Stage everything next to the target, then swap atomically.
    staged = []

    try:
        for name in PROMOTED:
            tmp = MODELS_DIR / f".{name}.promoting"
            if name == "manifest.json":
                promoted_manifest = dict(manifest)
                promoted_manifest["status"] = "promoted"
                tmp.write_text(
                    json.dumps(promoted_manifest, indent=2) + "\n",
                    encoding="utf-8",
                )
            else:
                shutil.copy2(candidate / name, tmp)
            staged.append((tmp, MODELS_DIR / name))

            if name in candidate_artifact_hashes and (
                sha256(tmp) != candidate_artifact_hashes[name]
            ):
                raise RuntimeError(f"Staged copy of {name} is corrupt.")

        for tmp, target in staged:
            os.replace(tmp, target)

        promoted_hashes = snapshot(MODELS_DIR, PROMOTED)
        for name, expected_hash in candidate_artifact_hashes.items():
            if promoted_hashes[name] != expected_hash:
                raise RuntimeError(f"Promoted artifact {name} hash mismatch.")

        deployed_manifest = json.loads(
            (MODELS_DIR / "manifest.json").read_text(encoding="utf-8")
        )
        if deployed_manifest.get("version") != manifest["version"]:
            raise RuntimeError("Promoted manifest has the wrong version.")
        if deployed_manifest.get("status") != "promoted":
            raise RuntimeError("Promoted manifest status is not 'promoted'.")
        if deployed_manifest.get("files") != manifest.get("files"):
            raise RuntimeError("Promoted manifest artifact hashes changed.")

        summary, health, body, stored = validate(
            manifest, run_predict_smoke=not args.skip_predict_smoke
        )
    except Exception as exc:
        print("PROMOTION FAILED:", exc)

        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)

        restore(backup_dir, before)
        print("Rolled back; models/ restored and verified from", backup_dir)
        sys.exit(1)

    record.update(
        promoted=True, pytest=summary, health=health, predict_response=body,
        stored_db_model_version=stored,
        promoted_manifest_sha256=sha256(MODELS_DIR / "manifest.json"),
        predict_smoke_skipped=args.skip_predict_smoke,
    )
    (backup_dir / "backup_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print("PROMOTION SUCCEEDED")


if __name__ == "__main__":
    main()
