import subprocess

class list(list):
    def map(self, f):
        return list(map(f, self))

import numpy as np
import pandas as pd
from numpy import array


pairs = {}

files1 = ['ivy', 'lucene', 'poi', 'synapse', 'velocity', 'camel', 'jedit', 'log4j', 'xalan', 'xerces']
files2 = ['ivy1', 'lucene1', 'lucene2', 'poi1', 'poi2', 'synapse1', 'synapse2', 'camel1', 'camel2', 'xerces1', 'jedit1', 'jedit2', 'log4j1', 'xalan1']

_ = subprocess.Popen(f'ls results/', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode('utf-8').rstrip()
_ = _.split('\n')
try:
    _ = list(_)
except ValueError:
    raise
for file in _:
    filename = f'./results/{file}'
    
    _ = subprocess.Popen(f'grep "^Running" {filename}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode('utf-8').rstrip()
    _ = _.split('\n')
    try:
        _ = list(_)
    except ValueError:
        raise
    treatments = _.map(lambda x: eval(x.split("Running")[1]))
    regex = "^[[{]"
    _ = subprocess.Popen(f'grep "{regex}" {filename}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode('utf-8').rstrip()
    _ = _.split('\n')
    try:
        _ = list(_)
    except ValueError:
        raise
    results = _.map(lambda x: eval(x))
    results = [x[0] if type(x).__name__ == 'list' else [x['f1'], x['d2h'], x['pd'], x['pf'], x['prec']] for x in results]

    for i, (t, r) in enumerate(zip(treatments, results)):
        if i < 320:
            t['dataset'] = files1[i // 32]
        else:
            t['dataset'] = files2[(i - 320) // 32]

        s = frozenset(t.items())
        if s not in pairs:
            pairs[s] = []
        pairs[s].append(r)

df = pd.DataFrame(columns=['dataset', 'smote', 'smooth', 'dodge', 'ultrasample', 'wfo', 'f1', 'pd', 'pf', 'prec'])
for s, r in pairs.items():
    t = dict(s)
    perf = np.median(r, axis=0).squeeze()

    t = {**t, 'f1': perf[0], 'pd': perf[2], 'pf': perf[3], 'prec': perf[4]}
    df = df.append(t, ignore_index=True)

max_value = df.groupby('dataset')['f1'].max()
df[df['f1'].isin(max_value)].to_csv('results.csv')
