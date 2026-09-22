#!/usr/bin/env python3
"""Export aggregate comparisons for the adaptive hardware follow-up."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rc'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plantvillage_rc.hardware_data import write_csv, write_json


def report(run, previous, development, output, seed_run=None):
    latest = json.loads((run / 'evaluation_summary.json').read_text())
    old = json.loads((previous / 'evaluation_summary.json').read_text())
    selection = json.loads((run / 'final_selection.json').read_text())
    protocol = json.loads((run / 'frozen_run.json').read_text())
    audit = json.loads((previous / 'audit.json').read_text())
    if latest['status'] != 'complete' or latest['prediction_count'] != 900 or len(latest['repeats']) != 3:
        raise ValueError('requires all 15 outer folds and 900 repeated predictions')
    output.mkdir(parents=True, exist_ok=True)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    summary = {k: v for k, v in latest.items() if k != 'paired_predictions'}
    summary.update(previous_mean_accuracy=old['mean_accuracy'],
                   delta_vs_previous=latest['mean_accuracy'] - old['mean_accuracy'],
                   development_sample_count=240, development_candidates=89,
                   evaluation_scope='Adaptive exploratory follow-up; feature/candidate development was not fully nested.')
    write_json(output / 'summary.json', summary)
    write_json(output / 'search_protocol.json', protocol)
    write_json(output / 'delivery_model.json', selection)
    developments = []
    for stage, filename in ((1, 'stage1/screen_scores.json'), (2, 'stage2/screen_scores.json')):
        for row in json.loads((development / filename).read_text()):
            developments.append({'stage': stage, 'candidate': row['config']['id'], 'accuracy': row['accuracy'],
                                 'configuration': json.dumps(row['config'], sort_keys=True), 'seconds': row['seconds']})
    write_csv(output / 'development_scores.csv', developments)
    variants = ('historical', 'previous', 'refined')
    metrics, pair_rows, class_rows, fold_rows = [], [], [], []
    matrices = {variant: np.zeros((10, 10), int) for variant in variants}
    old_predictions = {(r['seed'], r['sample_index']): r for r in old['paired_predictions']}
    for row in latest['repeats']:
        prior = next(r for r in old['repeats'] if r['seed'] == row['seed'])
        for variant, item in (('historical', row['historical']), ('previous', prior), ('refined', row)):
            metrics.append({'variant': variant, 'seed': row['seed'],
                            **{k: item[k] for k in ('accuracy', 'macro_f1', 'macro_recall', 'correct', 'sample_count')}})
            matrix = row['historical_confusion_matrix'] if variant == 'historical' else (prior if variant == 'previous' else row)['confusion_matrix']
            matrices[variant] += np.array(matrix)
        preds = [r for r in latest['paired_predictions'] if r['seed'] == row['seed']]
        corrected, regressed = 0, 0
        for prediction in preds:
            prior_prediction = old_predictions[(row['seed'], prediction['sample_index'])]
            assert prior_prediction['label'] == prediction['label']
            correct = prediction['prediction'] == prediction['label']
            was_correct = prior_prediction['prediction'] == prediction['label']
            corrected += correct and not was_correct
            regressed += was_correct and not correct
        pair_rows.append({'seed': row['seed'], 'corrected_vs_previous': corrected, 'regressed_vs_previous': regressed,
                          'corrected_vs_historical': row['corrected'], 'regressed_vs_historical': row['regressed']})
    write_csv(output / 'repeat_metrics.csv', metrics)
    write_csv(output / 'paired_changes.csv', pair_rows)
    for variant, matrix in matrices.items():
        for i, name in enumerate(audit['classes']):
            class_rows.append({'variant': variant, 'label': i, 'class_name': name, 'unique_samples': 30,
                               'prediction_count': 90, 'correct': int(matrix[i, i]), 'recall': float(matrix[i, i] / 90)})
    write_csv(output / 'per_class_recall.csv', class_rows)
    write_json(output / 'confusion_matrices.json', {k: v.tolist() for k, v in matrices.items()})
    for path in sorted((run / 'evaluation').glob('*/selection.json')):
        choice = json.loads(path.read_text())['selected']
        fold_rows.append({'fold': path.parent.name, 'members': '+'.join(choice['members']),
                          'weights': json.dumps(choice['weights']), 'inner_accuracy': choice['scores']['accuracy']})
    if len(fold_rows) != 15:
        raise ValueError('missing fold selections')
    write_csv(output / 'fold_selections.csv', fold_rows)
    write_csv(run / 'paired_predictions.csv', latest['paired_predictions'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'svg.hashsalt': 'hardware-refinement-v2'})

    def save(fig, name):
        fig.savefig(figures / f'{name}.png', dpi=180, bbox_inches='tight')
        svg = figures / f'{name}.svg'
        fig.savefig(svg, bbox_inches='tight', metadata={'Date': None})
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
        plt.close(fig)

    colors = ('#a3aab4', '#d19a66', '#247d70')
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for i, (variant, color) in enumerate(zip(variants, colors)):
        values = [r['accuracy'] * 100 for r in metrics if r['variant'] == variant]
        bars = ax.bar(np.arange(3) + (i - 1) * .26, values, width=.25, color=color, label=variant)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    ax.set_xticks(range(3), [str(r['seed']) for r in latest['repeats']])
    ax.set_ylim(0, 105); ax.set_ylabel('Accuracy (%)'); ax.legend(loc='lower right')
    ax.set_title('300 measured samples: exploratory repeated evaluation')
    save(fig, 'accuracy_comparison')
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    for i, (variant, color) in enumerate(zip(variants, colors)):
        ax.barh(np.arange(10) + (i - 1) * .25, np.diag(matrices[variant]) / 90 * 100,
                height=.24, color=color, label=variant)
    ax.set_yticks(range(10), [f'{i}: {name}' for i, name in enumerate(audit['classes'])], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 105); ax.set_xlabel('Mean recall over 3 repeats (%)')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1)); ax.set_title('30 unique samples per class; 90 correlated predictions')
    save(fig, 'per_class_recall')
    lines = [
        '# 九路硬件第二轮：分布特征与神经网络／树模型融合', '',
        f"三次五折平均准确率 **{latest['mean_accuracy']:.2%}**，重复间样本标准差 **{100 * latest['std_accuracy']:.2f} 个百分点**，宏 F1 **{latest['mean_macro_f1']:.2%}**。相比第一轮 {old['mean_accuracy']:.2%}，变化 **{100 * (latest['mean_accuracy'] - old['mean_accuracy']):+.2f} 个百分点**；相比历史探索 {latest['historical_mean_accuracy']:.2%}，变化 **{100 * latest['mean_accuracy_delta']:+.2f} 个百分点**。", '',
        '**性质：同一批 300 张上的后续探索，不是独立测试，也不是完全嵌套的无偏性能估计。** 特征和候选先在原种子 20260909 第一外层折以外的 240 张上进行了两阶段三折筛选（51＋38＝89 配置），再冻结 30 个候选。该 240 张会出现在其它外层验证折；虽然每个折内部重新拟合和选择，开发阶段仍已间接使用部分外层标签。所有开发尝试一并保留，不只列出成功配置。', '',
        '三个重复均使用全部 300 张，每类 30 张，未删难样本、重筛类别或换种子。900 是相关预测次数。输入始终只有九路实测电流；原始 RGB/ON/OFF 文件只用于对齐审计。', '',
        '| 种子 | 历史探索 | 第一轮 | 第二轮 | 第二轮宏 F1 |', '|---|---:|---:|---:|---:|',
    ]
    for row in latest['repeats']:
        prior = next(r for r in old['repeats'] if r['seed'] == row['seed'])
        lines.append(f"| {row['seed']} | {row['historical']['accuracy']:.2%} | {prior['accuracy']:.2%} | {row['accuracy']:.2%} | {row['macro_f1']:.2%} |")
    lines += ['', '![准确率对比](figures/accuracy_comparison.png)', '![逐类召回](figures/per_class_recall.png)', '',
              '## 改了什么', '',
              '- 增加每个通道、每行器件的 18 项响应分布统计；补偿只作用于前 64 点，tail 单独保留。',
              '- 神经网络采用 StandardScaler → PCA → L-BFGS；在训练折内部拟合，并比较正则化、PCA 白化和激活函数。',
              '- 多尺度表示结合原响应统计、补偿响应统计、相对通道响应及蛇形 4×4 池化；它们都是电流的后处理。',
              '- 在每个外层训练集内，按三折预测选取神经网络和树模型前 1/3/5 名，并选择融合比例。外层标签不进入本折拟合或权重选择。', '',
              '## 文件', '',
              '- [重复指标](repeat_metrics.csv)、[配对变化](paired_changes.csv)、[逐类召回](per_class_recall.csv)、[混淆矩阵](confusion_matrices.json)。',
              '- [所有开发尝试](development_scores.csv)、[冻结协议](search_protocol.json)、[外层折模型选择](fold_selections.csv)、[完整汇总](summary.json)。',
              '- [全数据模型配置](delivery_model.json) 在全部 300 张上再次做内层选择；其内层分数不是独立测试成绩。',
              '- 原始文件、完整预测与模型只留在本地 `artifacts_hardware300_v2/`。',
              '- [复现命令及特征定义](../../rc/HARDWARE_REFINEMENT.md)；运行环境完整版本在冻结协议中。', '',
              '要确认真实泛化提升，需固定该模型后评估新采集、带批次标识的硬件样本。本轮只能支持对当前已探索数据的比较。', '']
    if seed_run is not None:
        seed_summary = json.loads((seed_run / 'evaluation_summary.json').read_text())
        if seed_summary['status'] != 'complete' or seed_summary['prediction_count'] != 900:
            raise ValueError('seed expansion requires complete repeated predictions')
        write_json(output / 'seed_expansion_summary.json', {k: v for k, v in seed_summary.items() if k != 'paired_predictions'})
        write_csv(output / 'seed_repeat_metrics.csv', [
            {k: row[k] for k in ('seed', 'accuracy', 'macro_f1', 'macro_recall', 'correct', 'sample_count', 'corrected', 'regressed')}
            for row in seed_summary['repeats']])
        seed_matrix = np.sum([row['confusion_matrix'] for row in seed_summary['repeats']], axis=0)
        write_csv(output / 'seed_per_class_recall.csv', [
            {'label': i, 'class_name': name, 'unique_samples': 30, 'prediction_count': 90,
             'correct': int(seed_matrix[i, i]), 'recall': float(seed_matrix[i, i] / 90)}
            for i, name in enumerate(audit['classes'])])
        write_json(output / 'seed_expansion_protocol.json', json.loads((seed_run / 'frozen_run.json').read_text()))
        development_rows = json.loads((development / 'seed_screen/screen_scores.json').read_text())
        write_json(output / 'seed_development_scores.json', development_rows)
        if (seed_run / 'final_selection.json').exists():
            write_json(output / 'seed_delivery_model.json', json.loads((seed_run / 'final_selection.json').read_text()))
        delta = seed_summary['mean_accuracy'] - latest['mean_accuracy']
        lines[4:4] = [f"补充的 54 候选初始化扩展平均为 **{seed_summary['mean_accuracy']:.2%}**，完整结果见下方初始化扩展实验；两套结果均保留。", '']
        lines += ['## 初始化扩展实验（单独保留）', '',
                  '在第二轮结果出来后，又对六个神经网络配置各增加四个初始化种子，在同一个 240 张开发子集上比较。随后冻结 30＋24＝54 个候选，重新完成全部三次五折。它是进一步自适应探索，不能视为对第二轮的独立复核；扩大搜索空间也可能增加选择偏差。', '',
                  f"扩展后均值 **{seed_summary['mean_accuracy']:.2%}**，重复间标准差 **{100 * seed_summary['std_accuracy']:.2f} 个百分点**；相比 30 候选版本变化 **{100 * delta:+.2f} 个百分点**。", '',
                  '| 种子 | 30 候选版本 | 54 候选初始化扩展 |', '|---|---:|---:|']
        for base_row, seed_row in zip(latest['repeats'], seed_summary['repeats']):
            assert base_row['seed'] == seed_row['seed']
            lines.append(f"| {seed_row['seed']} | {base_row['accuracy']:.2%} | {seed_row['accuracy']:.2%} |")
        lines += ['', '[完整扩展指标](seed_expansion_summary.json)、[重复指标 CSV](seed_repeat_metrics.csv)、[逐类召回](seed_per_class_recall.csv)、[冻结协议](seed_expansion_protocol.json)、[24 个开发配置成绩](seed_development_scores.json)。', '',
                  '![初始化扩展比较](figures/seed_expansion.png)', '']
        if seed_summary['std_accuracy'] >= latest['std_accuracy']:
            lines += ['初始化扩展没有降低三个划分之间的准确率波动；不能据此声称稳定性改善。与历史方案的微小均值差异也不能证明独立泛化提升。', '']
        if (output / 'seed_delivery_model.json').exists():
            lines += ['[初始化扩展的全数据模型配置](seed_delivery_model.json)；本地模型在 `artifacts_hardware300_v2/seed_run/model.joblib`，同样没有独立测试成绩。', '']
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for offset, source, label, color in ((-.18, latest, '30 candidates', '#247d70'), (.18, seed_summary, '54 candidates', '#4d7198')):
            bars = ax.bar(np.arange(3) + offset, [r['accuracy'] * 100 for r in source['repeats']], .34, color=color, label=label)
            ax.bar_label(bars, fmt='%.2f', padding=3)
        ax.set_xticks(range(3), [str(r['seed']) for r in latest['repeats']])
        ax.set_ylim(0, 105); ax.set_ylabel('Accuracy (%)'); ax.legend(loc='lower right')
        ax.set_title('Initialization expansion: same previously explored data')
        save(fig, 'seed_expansion')
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(output / 'files_sha256.json', {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'files_sha256.json'})
    print(f"Aggregate report: {output}; accuracy {latest['mean_accuracy']:.6%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'artifacts_hardware300_v2/run')
    parser.add_argument('--previous', type=Path, default=ROOT / 'artifacts_hardware300')
    parser.add_argument('--development', type=Path, default=ROOT / 'artifacts_hardware300_v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/rc10_hardware300_v2')
    parser.add_argument('--seed-run', type=Path, help='Optional complete initialization-expansion run')
    args = parser.parse_args()
    report(args.run, args.previous, args.development, args.output, args.seed_run)
