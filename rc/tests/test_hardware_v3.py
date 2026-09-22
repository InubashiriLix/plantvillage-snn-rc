import json
from pathlib import Path
import subprocess
import sys

from plantvillage_rc.hardware_refinement import build_refined_model

ROOT = Path(__file__).resolve().parents[2]


def test_v3_candidates_and_mlp_controls_are_explicit():
    configs = json.loads((ROOT / "rc/configs/hardware_refinement_v3.json").read_text())
    assert len(configs) == 42
    assert len({item["id"] for item in configs}) == len(configs)
    assert all(item["family"] in {"MLP", "RF", "ET", "SVM"} for item in configs)
    model = build_refined_model(next(item for item in configs if item.get("hidden") == [64, 32]))
    estimator = model.named_steps["model"]
    assert estimator.hidden_layer_sizes == (64, 32)
    assert estimator.max_iter == 1200
    assert estimator.max_fun == 50000


def test_v3_cli_exposes_resumeable_actions():
    result = subprocess.run(
        [sys.executable, str(ROOT / "rc/scripts/12_optimize_hardware_v3.py"), "--help"],
        capture_output=True, text=True, check=True,
    )
    for action in ("audit", "search", "evaluate", "fit", "predict", "report"):
        assert action in result.stdout
