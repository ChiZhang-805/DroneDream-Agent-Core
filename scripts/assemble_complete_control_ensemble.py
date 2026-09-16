#!/usr/bin/env python3
"""Assemble ten explicit experts without relying on a historical base package."""

import argparse
import json
from pathlib import Path

from dronedream_agent_core.training.artifact_assembly import (
    CompleteEnsembleRecipe,
    assemble_complete_ensemble,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipe = CompleteEnsembleRecipe.model_validate_json(args.recipe.read_bytes())
    package = assemble_complete_ensemble(
        recipe=recipe, source_root=args.recipe.resolve().parent, output_root=args.output
    )
    print(
        json.dumps(
            {
                "package_sha256": package.package_sha256,
                "expert_count": len(package.artifact_paths),
                "qualified_for_flight": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
