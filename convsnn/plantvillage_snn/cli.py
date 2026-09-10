from __future__ import annotations

import argparse
import json

from .artifacts import build_report, select_top10
from .config import ExperimentConfig
from .search import focused_search
from .train import evaluate_saved_run, run_experiment


def _add_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--model", choices=["conv_snn", "compact_snn"])
    parser.add_argument("--stage", choices=["all38", "top10"])
    parser.add_argument("--phase", choices=["ideal", "hw_finetune"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--selection-fraction", type=float)
    parser.add_argument("--split-manifest-path")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--time-steps", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--channels", type=lambda value: _csv(value, int))
    parser.add_argument("--layer-betas", type=lambda value: _csv(value, float))
    parser.add_argument("--layer-thresholds", type=lambda value: _csv(value, float))
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--augmentation", choices=["none", "basic", "strong"])
    parser.add_argument("--class-balance", choices=["none", "class_weights", "weighted_sampler"])
    parser.add_argument("--top10-path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--membrane-loss-weight", type=float)
    parser.add_argument("--distill-weight", type=float)
    parser.add_argument("--distill-temperature", type=float)
    parser.add_argument("--firing-regularization-weight", type=float)
    parser.add_argument("--firing-rate-min", type=float)
    parser.add_argument("--firing-rate-max", type=float)
    parser.add_argument("--evaluate-final", action="store_const", const=True, default=None)
    parser.add_argument("--device-csv")
    parser.add_argument("--device-v-bias", type=float)
    parser.add_argument("--device-g-bins", type=int)
    parser.add_argument("--device-max-pulses", type=int)
    parser.add_argument("--max-pulses-per-update", type=int)
    parser.add_argument("--hw-scale-margin", type=float)
    parser.add_argument("--accum-decay", type=float)
    parser.add_argument("--accum-decay-interval", type=int)
    parser.add_argument("--signed-mapping", choices=["differential_pair"])
    parser.add_argument("--device-preprocess", choices=["raw", "robust"])
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-selection-samples", type=int)
    parser.add_argument("--max-final-samples", type=int)
    parser.add_argument("--device")
    parser.add_argument("--recording-enabled", type=_boolean)
    parser.add_argument("--record-internal-summaries", type=_boolean)
    parser.add_argument("--record-sample-predictions", type=_boolean)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--dataset-hash-mode", choices=["sha256", "none"])
    parser.add_argument("--recording-dir")


def _configuration(arguments: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.from_json(arguments.config) if arguments.config else ExperimentConfig()
    fields = set(config.to_dict())
    overrides = {
        key: value for key, value in vars(arguments).items()
        if key in fields and value is not None
    }
    return config.with_overrides(overrides)


def _csv(value: str, conversion):
    return [conversion(item.strip()) for item in value.split(",") if item.strip()]


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plantvillage-snn")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train an ideal or hardware-aware experiment")
    _add_experiment_arguments(train)
    search = sub.add_parser("search", help="selection-validation focused grid search")
    _add_experiment_arguments(search)
    search.add_argument("--learning-rates", default="0.001")
    search.add_argument("--time-step-grid", default="4,8")
    search.add_argument("--beta-grid", default="0.85,0.9")
    search.add_argument("--augmentation-grid", default="none,basic")
    search.add_argument("--balance-grid", default="none,class_weights")
    choose = sub.add_parser("select-top10", help="rank classes from official hardware metrics")
    choose.add_argument("--metrics", required=True)
    choose.add_argument("--output", default="artifacts/top10_classes.json")
    report = sub.add_parser("report", help="build JSON and Markdown comparison reports")
    report.add_argument("--metrics", nargs="+", required=True)
    report.add_argument("--output-dir", default="artifacts/report")
    frozen = sub.add_parser("evaluate", help="final-test evaluation of a frozen selected checkpoint")
    frozen.add_argument("--metrics", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "train":
        result = run_experiment(_configuration(args))
    elif args.command == "search":
        config = _configuration(args)
        result = focused_search(
            config,
            _csv(args.learning_rates, float),
            _csv(args.time_step_grid, int),
            _csv(args.beta_grid, float),
            _csv(args.augmentation_grid, str),
            _csv(args.balance_grid, str),
        )
    elif args.command == "select-top10":
        result = select_top10(args.metrics, args.output)
    elif args.command == "report":
        result = build_report(args.metrics, args.output_dir)
    else:
        result = evaluate_saved_run(args.metrics)
    print(json.dumps(result, indent=2, default=str))
    return 0
