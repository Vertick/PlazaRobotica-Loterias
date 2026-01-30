# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import json
import sys
import argparse
import hashlib
import os
from datetime import datetime, timezone
from jsonschema import Draft202012Validator


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_against_schema(instance: dict, schema: dict, label: str) -> list[str]:
    errors = []
    validator = Draft202012Validator(schema)
    for err in sorted(validator.iter_errors(instance), key=str):
        errors.append(f"{label}: {err.message}")
    return errors


def extract_profile_signature(profile: dict) -> dict:
    return {
        "profile_id": profile["profile_id"],
        "reference": profile["reference"]["type"],
        "conditions": [
            {
                "metric": c["metric"],
                "rule": c["rule"],
                "params": c["params"]
            }
            for c in profile["conditions"]
        ]
    }


def extract_run_signature(run: dict) -> dict:
    cleaned_conditions = []
    for c in run["conditions"]:
        params = {
            k: v for k, v in c["params"].items()
            if k in ("lower", "upper", "max_bin_count", "min_nonempty_bins")
        }
        cleaned_conditions.append({
            "metric": c["metric"],
            "rule": c["rule"],
            "params": params
        })

    return {
        "profile_id": run["config"]["name"],
        "reference": run["reference"]["type"],
        "conditions": cleaned_conditions
    }


def cross_check(profile_sig: dict, run_sig: dict) -> list[str]:
    errors = []

    if profile_sig["profile_id"] != run_sig["profile_id"]:
        errors.append("Profile ID mismatch")

    if profile_sig["reference"] != run_sig["reference"]:
        errors.append("Reference mismatch")

    for pc in profile_sig["conditions"]:
        matches = [
            rc for rc in run_sig["conditions"]
            if rc["metric"] == pc["metric"] and rc["rule"] == pc["rule"]
        ]

        if not matches:
            errors.append(f"Missing condition for metric '{pc['metric']}'")
            continue

        rc = matches[0]
        for k, v in pc["params"].items():
            if k in rc["params"] and rc["params"][k] != v:
                errors.append(
                    f"Param mismatch in '{pc['metric']}.{k}': "
                    f"profile={v}, run={rc['params'][k]}"
                )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Monte Carlo run")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profile-schema", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--run-schema", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    report = {
        "validated_at": utc_now(),
        "status": "PASSED",
        "checks": {},
        "errors": []
    }

    try:
        profile = load_json(args.profile)
        run = load_json(args.run)
        profile_schema = load_json(args.profile_schema)
        run_schema = load_json(args.run_schema)

        prof_err = validate_against_schema(profile, profile_schema, "profile.schema")
        run_err = validate_against_schema(run, run_schema, "run.schema")

        report["checks"]["profile_schema"] = "OK" if not prof_err else "FAILED"
        report["checks"]["run_schema"] = "OK" if not run_err else "FAILED"
        report["errors"].extend(prof_err + run_err)

        cross_err = cross_check(
            extract_profile_signature(profile),
            extract_run_signature(run)
        )

        report["checks"]["cross_check"] = "OK" if not cross_err else "FAILED"
        report["errors"].extend(cross_err)

        if report["errors"]:
            report["status"] = "FAILED"

    except Exception as e:
        report["status"] = "FAILED"
        report["errors"].append(f"Validator exception: {e}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if report["status"] != "PASSED":
        sys.exit(1)


if __name__ == "__main__":
    main()