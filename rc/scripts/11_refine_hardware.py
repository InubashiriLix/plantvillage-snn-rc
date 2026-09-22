#!/usr/bin/env python3
"""Second-round hardware-only development, nested evaluation and delivery fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rc'))

from plantvillage_rc.hardware_data import load_dataset, write_json
from plantvillage_rc.hardware_refinement_learning import evaluate, fit_final, freeze, run_inner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('screen', 'evaluate', 'fit'))
    parser.add_argument('--responses', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--workbook', required=True, type=Path)
    parser.add_argument('--configs', type=Path, default=ROOT / 'rc/configs/hardware_refinement_v2.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts_hardware300_v2/run')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--hours', type=float, default=4)
    args = parser.parse_args()
    try:
        if args.workers < 1 or args.hours <= 0:
            raise ValueError('workers and hours must be positive')
        dataset = load_dataset(args.responses, args.inputs, args.workbook)
        configs = json.loads(args.configs.read_text())
        if args.action == 'screen':
            fingerprint = freeze(dataset, configs, args.output_dir)
            mask = dataset.folds[20260909] != 0
            rows = run_inner(configs, dataset.x[mask], dataset.y[mask], 20260909,
                             args.output_dir / 'development_inner', fingerprint, time.time() + args.hours * 3600, args.workers)
            write_json(args.output_dir / 'screen_scores.json', [
                {'config': row['config'], 'accuracy': row['scores']['accuracy'],
                 'seconds': row['seconds'], 'warnings': row['warnings']} for row in rows])
            print('Development only: 240 samples; excluded historical outer fold 1.')
        else:
            function = evaluate if args.action == 'evaluate' else fit_final
            result = function(dataset, configs, args.output_dir, workers=args.workers, hours=args.hours)
            print(json.dumps({k: v for k, v in result.items() if k != 'paired_predictions'}, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TimeoutError) as error:
        parser.exit(2, f'Refinement stopped: {error}\n')


if __name__ == '__main__':
    main()
