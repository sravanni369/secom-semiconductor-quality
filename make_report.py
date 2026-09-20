"""Build plots and a browser report from verified saved predictions, never fit a model."""
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
s=json.loads((root/'results/summary.json').read_text())
v=json.loads((root/'results/verification.json').read_text())
assert v['checks_passed']
m=pd.read_csv(root/'results/metrics.csv')
def wilson(k,n):
    z=1.959963984540054
    p=k/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    width=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-width,center+width]
uncertainty={'recall_95_wilson':wilson(4,17),'fpr_95_wilson':wilson(77,297),
             'note':'Binomial intervals assume independent observations; production temporal dependence is not modeled.'}
(root/'results/uncertainty.json').write_text(json.dumps(uncertainty,indent=2))
(root/'assets').mkdir(exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
names=['Always pass','Forest @ 0.5','Validation gate']
axs[0].bar(names,m.tp,color=['#8293a8','#5450dd','#17a897'])
axs[0].set(ylim=(0,18),ylabel='Failures caught (out of 17)',title='Accuracy alone hides missed failures')
for i,r in m.iterrows():axs[0].text(i,r.tp+.4,f'{r.tp}/17',ha='center',weight='bold')
axs[1].bar(['Validation','Later test'],[22/302*100,77/297*100],color=['#5450dd','#ef715e'])
axs[1].axhline(10,color='#333',linestyle='--',label='Assumed validation cap: 10%')
axs[1].set(ylabel='False-positive rate (%)',ylim=(0,35),title='The chosen threshold does not transfer')
axs[1].legend(fontsize=9)
for i,x in enumerate([22/302*100,77/297*100]):axs[1].text(i,x+.6,f'{x:.1f}%',ha='center',weight='bold')
fig.savefig(root/'assets/results.png',dpi=180)
code=(root/'train.py').read_text()
rows=''.join(f'<tr><td>{r.policy}</td><td>{r.tp}</td><td>{r.fp}</td><td>{r.fn}</td><td>{100*r.accuracy:.1f}%</td><td>{100*r.recall:.1f}%</td></tr>' for _,r in m.iterrows())
page='''<!doctype html><meta charset="utf-8"><title>SECOM verified experiment</title><style>
body{font:18px Segoe UI,Arial;background:#edf2fa;color:#17233b;margin:40px auto;max-width:1100px}h1{font-size:42px}header{background:#162342;color:white;padding:28px;border-radius:18px}.cards{display:flex;gap:16px;margin:20px 0}.cards div{background:white;padding:22px;border-radius:12px;flex:1}b{font-size:26px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:13px;text-align:left;border-bottom:1px solid #ccd5e4}img{width:100%}pre{background:#182238;color:#e3edff;padding:24px;overflow:auto;font:14px Consolas,monospace}section{background:white;padding:24px;margin:20px 0;border-radius:14px}p{line-height:1.55}</style>
<header><small>BOOK-TO-BUSINESS • VERIFIED NEGATIVE RESULT</small><h1>94.6% accuracy. Zero failures caught.</h1><p>Semiconductor quality triage with a fixed Random Forest recipe and a chronological holdout.</p></header>
<div class="cards"><div><b>1,567</b><br>production records</div><div><b>590</b><br>sensor columns</div><div><b>50 lines</b><br>executed Python core</div><div><b>606 checks</b><br>numerical gates passed</div></div>
<section><h2>Latest 314 records: 17 failures, 297 passes</h2><table><tr><th>Policy</th><th>Caught</th><th>False alarms</th><th>Missed</th><th>Accuracy</th><th>Recall</th></tr>ROWS</table><p>The gate flags 81 records, of which only 4 fail. Balanced accuracy is 48.8%, versus 50% for always-pass. This is not a deployable solution.</p></section>
<img src="assets/results.png" alt="Verified confusion-count and false-alarm comparison">
<section><h2>What the experiment did</h2><p>Train on the earliest 940 records, select a threshold on the next 313, test on the latest 314. The training-only median imputer precedes 500 Random Forest trees (maximum 16 leaves, seed 42). No test-driven tuning.</p><p>Validation: 1/11 failures caught and 22/302 passes flagged (7.3%). Test: 4/17 caught and 77/297 passes flagged (25.9%). Threshold: 0.102863.</p><p>Recall 95% Wilson interval: 9.6–47.3%. False-positive rate: 21.3–31.2%. Intervals assume independent records. No demonstrated causal explanation, advance-warning horizon, or financial savings.</p></section>
<section><h2>Book and data credit</h2><p>Aurélien Géron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd edition: p.199 Random Forest, pp.96–97 thresholds, p.67 median imputation. Original adaptation to UCI SECOM, McCann &amp; Johnston (2008), CC BY 4.0.</p><p>Local data contains 590 sensor columns; UCI's prose describes 591 features. Timestamp is stored separately. We report the actual parsed shape.</p></section>
<section><h2>Exact executed core</h2><pre>CODE</pre></section><p>github.com/sravanni369 • linkedin.com/in/lakshmi-sravani-p-212899272</p>'''.replace('ROWS',rows).replace('CODE',html.escape(code))
(root/'report.html').write_text(page,encoding='utf-8')
print('Report and chart generated from verified outputs.')

if __name__=='__main__':
    print(json.dumps(uncertainty,indent=2))
