"""Repeated nested selection for the separately recorded refinement experiment."""
from __future__ import annotations

from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
from joblib import Parallel, delayed, parallel_config
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .hardware_data import REPEAT_SEEDS, scores, sha256, write_csv, write_json
from .hardware_learning import summarize
from .hardware_models import FittedReadout, probabilities
from .hardware_refinement import build_refined_model


def freeze(dataset, configs, output):
    if not configs or len({c['id'] for c in configs}) != len(configs):
        raise ValueError('candidate identifiers must be nonempty and unique')
    root = Path(__file__).parent
    definition = {
        'data_hashes': dataset.hashes, 'candidates': configs,
        'code_hashes': {name: sha256(root / name) for name in (
            'hardware_data.py', 'hardware_models.py', 'hardware_learning.py',
            'hardware_refinement.py', 'hardware_refinement_learning.py')},
        'versions': {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl')},
        'python': platform.python_version(), 'repeat_seeds': list(REPEAT_SEEDS),
        'inner_folds': 3, 'family_top_k': [1, 3, 5], 'tree_weights': [0, .2, .4, .6, .8, 1],
        'scope': 'Exploratory follow-up on previously examined samples; no independent test.',
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'frozen_run.json'
    if path.exists() and json.loads(path.read_text()) != definition:
        raise ValueError('frozen run changed; use a new output directory')
    write_json(path, definition)
    return hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()


def inner_candidate(config, x, y, seed, output, fingerprint, deadline):
    path = Path(output) / f"{config['id']}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f'{fingerprint}:{seed}'.encode() + x.tobytes() + y.tobytes()).hexdigest()
    if path.exists():
        result = joblib.load(path)
        if result['cache_key'] != key:
            raise ValueError('inner cache does not match training data')
        return result
    probability = np.full((len(y), len(np.unique(y))), np.nan)
    caught = set()
    started = time.monotonic()
    for train, val in StratifiedKFold(3, shuffle=True, random_state=seed).split(x, y):
        if time.time() >= deadline:
            raise TimeoutError('budget reached; resume with the same configuration')
        with threadpool_limits(1), warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            model = build_refined_model(config).fit(x[train], y[train])
            probability[val] = probabilities(model, x[val], probability.shape[1])
        caught.update(str(item.message) for item in records)
    result = {'config': config, 'probabilities': probability, 'scores': scores(y, probability.argmax(1)),
              'cache_key': key, 'warnings': sorted(caught), 'seconds': time.monotonic() - started}
    temp = path.with_suffix('.tmp')
    joblib.dump(result, temp)
    temp.replace(path)
    return result


def rank(row):
    return (-row['scores']['accuracy'], -row['scores']['macro_f1'], row['config']['id'])


def select(results, y):
    """All ensemble composition and weights come from inner OOF predictions."""
    results = sorted(results, key=rank)
    options, seen = [], set()

    def add(members, weights):
        active = sorted((row['config']['id'], float(w)) for row, w in zip(members, weights) if w > 0)
        signature = tuple(active)
        if signature in seen:
            return
        seen.add(signature)
        lookup = {r['config']['id']: r for r in members}
        p = sum(w * lookup[name]['probabilities'] for name, w in active)
        options.append({'members': [a[0] for a in active], 'weights': [a[1] for a in active],
                        'scores': scores(y, p.argmax(1))})

    add([results[0]], [1.0])
    neural = [r for r in results if r['config']['family'] == 'MLP']
    trees = [r for r in results if r['config']['family'] in ('RF', 'ET')]
    for family in (neural, trees):
        for k in (1, 3, 5):
            group = family[:k]
            if group:
                add(group, [1 / len(group)] * len(group))
    if neural and trees:
        for nk in (1, 3, 5):
            for tk in (1, 3, 5):
                n, t = neural[:nk], trees[:tk]
                for w in (.2, .4, .6, .8):
                    add(n + t, [(1 - w) / len(n)] * len(n) + [w / len(t)] * len(t))
    selected = min(options, key=lambda r: (-r['scores']['accuracy'], -r['scores']['macro_f1'], len(r['members']), r['members']))
    return selected, options


def run_inner(configs, x, y, seed, output, fingerprint, deadline, workers):
    with parallel_config(backend='loky', n_jobs=workers, inner_max_num_threads=1):
        return Parallel(pre_dispatch=workers)(delayed(inner_candidate)(c, x, y, seed, output, fingerprint, deadline) for c in configs)


def fit_selection(selection, configs, x, y, classes):
    lookup = {c['id']: c for c in configs}
    with threadpool_limits(1):
        models = [build_refined_model(lookup[name]).fit(x, y) for name in selection['members']]
    return FittedReadout(models, selection['weights'], classes)


def evaluate(dataset, configs, output, *, workers=6, hours=4):
    output = Path(output)
    fingerprint = freeze(dataset, configs, output)
    deadline, start = time.time() + hours * 3600, time.monotonic()
    completed = []
    try:
        for seed in REPEAT_SEEDS:
            for fold in range(5):
                directory = output / 'evaluation' / f'seed{seed}_fold{fold + 1}'
                directory.mkdir(parents=True, exist_ok=True)
                cached = directory / 'completed.joblib'
                if cached.exists():
                    row = joblib.load(cached)
                    if row['fingerprint'] != fingerprint:
                        raise ValueError('outer cache fingerprint mismatch')
                else:
                    if time.time() >= deadline:
                        raise TimeoutError('outer evaluation budget reached')
                    train = np.flatnonzero(dataset.folds[seed] != fold)
                    val = np.flatnonzero(dataset.folds[seed] == fold)
                    inner = run_inner(configs, dataset.x[train], dataset.y[train], seed + fold,
                                      directory / 'inner', fingerprint, deadline, workers)
                    selection, options = select(inner, dataset.y[train])
                    write_json(directory / 'selection.json', {'selected': selection, 'options': options})
                    model = fit_selection(selection, configs, dataset.x[train], dataset.y[train], dataset.classes)
                    with threadpool_limits(1):
                        probability = model.predict_proba(dataset.x[val])
                    row = {'seed': seed, 'fold': fold + 1, 'indices': val, 'probabilities': probability,
                           'selected': selection, 'scores': scores(dataset.y[val], probability.argmax(1)),
                           'fingerprint': fingerprint}
                    temp = cached.with_suffix('.tmp')
                    joblib.dump(row, temp)
                    temp.replace(cached)
                    write_csv(directory / 'inner_scores.csv', [
                        {'candidate': r['config']['id'], 'family': r['config']['family'], **r['scores'],
                         'seconds': r['seconds'], 'warnings': ' | '.join(r['warnings'])} for r in sorted(inner, key=rank)])
                completed.append(row)
                write_json(output / 'progress.json', {'status': 'running', 'completed_outer_folds': len(completed),
                                                     'required_outer_folds': 15, 'seconds': time.monotonic() - start})
                print(json.dumps({'seed': seed, 'fold': fold + 1, **row['scores'], 'members': row['selected']['members']}), flush=True)
    except TimeoutError:
        write_json(output / 'progress.json', {'status': 'partial', 'completed_outer_folds': len(completed), 'required_outer_folds': 15})
        raise
    summary = summarize(dataset, completed)
    write_json(output / 'evaluation_summary.json', summary)
    write_json(output / 'progress.json', {'status': 'complete', 'completed_outer_folds': 15, 'required_outer_folds': 15,
                                         'seconds': time.monotonic() - start})
    return summary


def fit_final(dataset, configs, output, *, workers=6, hours=1):
    output = Path(output)
    fingerprint = freeze(dataset, configs, output)
    summary = json.loads((output / 'evaluation_summary.json').read_text())
    if summary['status'] != 'complete' or summary['prediction_count'] != 3 * len(dataset.y):
        raise ValueError('complete the outer evaluation first')
    inner = run_inner(configs, dataset.x, dataset.y, 20260923, output / 'final_inner', fingerprint,
                      time.time() + hours * 3600, workers)
    selected, options = select(inner, dataset.y)
    write_json(output / 'final_selection.json', {'selected': selected, 'options': options,
                                               'training_samples': len(dataset.y), 'independent_test_available': False})
    model = fit_selection(selected, configs, dataset.x, dataset.y, dataset.classes)
    joblib.dump(model, output / 'model.joblib')
    reloaded = joblib.load(output / 'model.joblib')
    with threadpool_limits(1):
        np.testing.assert_array_equal(model.predict_proba(dataset.x), reloaded.predict_proba(dataset.x))
    return selected
