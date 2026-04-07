#!/usr/bin/env python
"""
Unified regression test runner for volume-segmantics.

Thin wrapper around real_data_harness.run_harness() that provides named
shorthand for the bundled smoke-test configs.

Usage:
    # Smoke tests (fast, 1 epoch)
    python tests/scripts/run_smoke.py basic
    python tests/scripts/run_smoke.py losses
    python tests/scripts/run_smoke.py multitask
    python tests/scripts/run_smoke.py 2.5d
    python tests/scripts/run_smoke.py semisup

    # QA tests (longer, 8+5 epochs, full res ? overnight)
    python tests/scripts/run_smoke.py qa_basic
    python tests/scripts/run_smoke.py qa_multitask
    python tests/scripts/run_smoke.py qa_semisup
    python tests/scripts/run_smoke.py qa_2.5d
    python tests/scripts/run_smoke.py qa_losses

    # DINO variants
    python tests/scripts/run_smoke.py dino_multitask
    python tests/scripts/run_smoke.py qa_dino_multitask

    # Run with a different dataset (multiclass)
    python tests/scripts/run_smoke.py basic --data-profile multiclass
    python tests/scripts/run_smoke.py qa_multitask --data-profile multiclass
    python tests/scripts/run_smoke.py all_smoke --data-profile multiclass

    # Run any config by path
    python tests/scripts/run_smoke.py --config path/to/config.yaml

    # Dry-run (print resolved commands, skip execution)
    python tests/scripts/run_smoke.py basic --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from real_data_harness import run_harness

# Named shorthand -> config path (relative to repo root)
NAMED_CONFIGS = {
    # Smoke tests (fast, 1 epoch, image_size=64)
    "basic": "tests/scripts/configs/basic_smoke_tests.yaml",
    "losses": "tests/scripts/configs/loss_function_smoke_tests.yaml",
    "multitask": "tests/scripts/configs/multitask_smoke_tests.yaml",
    "2.5d": "tests/scripts/configs/two_5d_slicing_smoke_tests.yaml",
    "semisup": "tests/scripts/configs/semisup_smoke_tests.yaml",
    # DINO smoke tests
    "dino_basic": "tests/scripts/configs/dino_basic_smoke_tests.yaml",
    "dino_multitask": "tests/scripts/configs/dino_multitask_smoke_tests.yaml",
    "dino_semisup": "tests/scripts/configs/dino_semisup_smoke_tests.yaml",
    "dino_2.5d": "tests/scripts/configs/dino_two_5d_smoke_tests.yaml",
    # QA tests (longer, 8+5 epochs, full resolution ? overnight runs)
    "qa_basic": "tests/scripts/configs/basic_qa_tests.yaml",
    "qa_losses": "tests/scripts/configs/loss_function_qa_tests.yaml",
    "qa_multitask": "tests/scripts/configs/multitask_qa_tests.yaml",
    "qa_semisup": "tests/scripts/configs/semisup_qa_tests.yaml",
    "qa_2.5d": "tests/scripts/configs/two_5d_qa_tests.yaml",
    # DINO QA tests
    "qa_dino_basic": "tests/scripts/configs/dino_basic_qa_tests.yaml",
    "qa_dino_multitask": "tests/scripts/configs/dino_multitask_qa_tests.yaml",
    "qa_dino_semisup": "tests/scripts/configs/dino_semisup_qa_tests.yaml",
    "qa_dino_2.5d": "tests/scripts/configs/dino_two_5d_qa_tests.yaml",
}

# Groups that expand to multiple configs run in sequence
NAMED_GROUPS = {
    "all_smoke": ["basic", "losses", "multitask", "2.5d", "semisup"],
    "all_qa": ["qa_basic", "qa_losses", "qa_multitask", "qa_semisup", "qa_2.5d"],
    "all_dino_smoke": ["dino_basic", "dino_multitask", "dino_semisup", "dino_2.5d"],
    "all_dino_qa": ["qa_dino_basic", "qa_dino_multitask", "qa_dino_semisup", "qa_dino_2.5d"],
}

# Data profiles: named sets of path overrides that can be applied to any config.
# Usage: --data-profile multiclass
# This overrides the paths section of any config, keeping all other settings.
DATA_PROFILES = {
    "default": {
        "image": "training_data/vessels_256cube_DATA.h5",
        "label": "training_data/vessels_256cube_LABELS.h5",
        "unlabeled": ["training_data/vessels_256cube_DATA.h5"],
    },
    "multiclass": {
        "image": "training_data/sample_123761_IMAGE.tiff",
        "label": "training_data/sample_123761_MULTICLASS_LABEL.tiff",
        "unlabeled": ["training_data/sample_123761_IMAGE.tiff"],
    },
}


def _parse_args() -> argparse.Namespace:
    names = ", ".join(sorted(NAMED_CONFIGS))
    profiles = ", ".join(sorted(DATA_PROFILES))
    parser = argparse.ArgumentParser(
        description="Run volume-segmantics smoke tests.",
        epilog=f"Named configs: {names}",
    )
    parser.add_argument(
        "name",
        nargs="?",
        default=None,
        help=f"Named smoke test to run ({names}). Ignored if --config is given.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Explicit config path (absolute or relative to repository root).",
    )
    parser.add_argument(
        "--data-profile",
        default=None,
        dest="data_profile",
        help=f"Data profile to use ({profiles}). Overrides image/label/unlabeled paths.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print commands without executing them.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_configs",
        help="List available named configs and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_configs:
        print("Available smoke-test configs:\n")
        for name, path in sorted(NAMED_CONFIGS.items()):
            print(f"  {name:25s} {path}")
        print("\nGroups (run multiple configs in sequence):\n")
        for name, members in sorted(NAMED_GROUPS.items()):
            print(f"  {name:25s} {', '.join(members)}")
        print("\nData profiles (--data-profile <name>):\n")
        for name, paths in sorted(DATA_PROFILES.items()):
            print(f"  {name:25s} image={paths['image']}")
        return

    # Resolve data profile
    path_overrides = None
    output_suffix = None
    if args.data_profile:
        if args.data_profile not in DATA_PROFILES:
            print(
                f"Unknown data profile '{args.data_profile}'.",
                file=sys.stderr,
            )
            print(
                f"Available: {', '.join(sorted(DATA_PROFILES))}",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.data_profile != "default":
            path_overrides = DATA_PROFILES[args.data_profile]
            output_suffix = args.data_profile

    if args.config:
        configs_to_run = [Path(args.config)]
    elif args.name:
        if args.name in NAMED_GROUPS:
            configs_to_run = [Path(NAMED_CONFIGS[n]) for n in NAMED_GROUPS[args.name]]
        elif args.name in NAMED_CONFIGS:
            configs_to_run = [Path(NAMED_CONFIGS[args.name])]
        else:
            all_names = sorted(set(NAMED_CONFIGS) | set(NAMED_GROUPS))
            print(f"Unknown smoke test '{args.name}'.", file=sys.stderr)
            print(f"Available: {', '.join(all_names)}", file=sys.stderr)
            sys.exit(1)
    else:
        print(
            "Provide a named config or --config path. Use --list to see options.",
            file=sys.stderr,
        )
        sys.exit(1)

    for config_path in configs_to_run:
        run_harness(
            config_path,
            dry_run=args.dry_run,
            path_overrides=path_overrides,
            output_suffix=output_suffix,
        )


if __name__ == "__main__":
    main()
