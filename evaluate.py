"""Frozen numerical gates; run train.py in-process and independently recount output."""
from pathlib import Path
import hashlib
import json
import runpy
import sys
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
ns = runpy.run_path(str(root / 'train.py'))
X, y, times = (ns[k] for k in ('X', 'y', 'times'))
tr, va, te = (ns[k] for k in ('train', 'valid', 'test'))
checks = []
def check(name, value):
    assert bool(value), name
    checks.append(name)
check('50 physical lines or fewer', len((root/'train.py').read_text().splitlines()) <= 50)
check('original UCI shape', X.shape == (1567, 590))
check('original fail count', y.sum() == 104)
check('strict chronological boundaries', times[tr].max() < times[va].min() < times[te].min() and times[va].max() < times[te].min())
check('partitions exhaustive and disjoint', np.array_equal(np.r_[tr,va,te], np.arange(len(y))))
check('both labels in each split', all(len(np.unique(y[z])) == 2 for z in (tr,va,te)))
check('scores finite and bounded', np.isfinite(ns['scores']).all() and ((ns['scores'] >= 0) & (ns['scores'] <= 1)).all())
imputer = ns['model'].steps[0][1]
for col in range(X.shape[1]):
    finite = X[tr,col][~np.isnan(X[tr,col])]
    expected = np.median(finite) if len(finite) else 0.
    check(f'training-only median sensor {col}', np.isclose(imputer.statistics_[col], expected))
check('validation FPR budget', ns['summary']['validation_fp'] / (y[va] == 0).sum() <= .10)
records = pd.read_csv(root/'results/predictions.csv')
check('saved outcomes align with source', np.array_equal(records.fail.to_numpy(),y[te]))
check('saved original row alignment', np.array_equal(records.original_row.to_numpy(),ns['order'][te]))
for row in ns['rows']:
    pred = ns['policies'][row['policy']]
    truth = y[te].astype(bool)
    independent = [int((~pred & ~truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum()), int((pred & truth).sum())]
    check('confusion counts '+row['policy'], independent == [row[k] for k in ('tn','fp','fn','tp')])
    check('counts exhaustive '+row['policy'], sum(independent) == len(te))
result = {'score': ns['rows'][-1]['recall'], 'checks_passed': True, 'checks':len(checks),
          'prediction_sha256': hashlib.sha256((root/'results/predictions.csv').read_bytes()).hexdigest(),
          'scope':'Numerical verification only; no independent human or agent review yet.'}
(root/'results/verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if len(sys.argv)>1:
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2),encoding='utf-8')
print('VERIFIED',json.dumps(result))
