"""Unattended phase entry point; every phase is safe to resume."""
import argparse
import traceback
from .config import SweepConfig, manifest
from .storage import atomic_json, event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "validate", "preflight", "tune", "run", "report"))
    parser.add_argument("--output")
    arguments = parser.parse_args()
    config = SweepConfig(**({"output": arguments.output} if arguments.output else {}))
    atomic_json(config.root / "manifest.json", manifest(config))
    try:
        if arguments.phase == "prepare":
            from .dataset import prepare
            prepare(config)
        elif arguments.phase == "validate":
            from .validation import validate
            validate(config)
        elif arguments.phase=="tune":
            from .runner import early_tuning
            early_tuning(config)
        else:
            from .runner import run
            run(config, phase=arguments.phase)
    except Exception as exception:
        event(config, "failed", requested_phase=arguments.phase, error=str(exception), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
