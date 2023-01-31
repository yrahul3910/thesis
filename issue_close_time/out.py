import subprocess

class list(list):
    def map(self, f):
        return list(map(f, self))

import sys
import numpy as np
import pandas as pd
from numpy import array


if len(sys.argv) < 3:
    print(f'Usage: {sys.argv[0]} DATASET N_CLASSES')
    sys.exit(1)

pairs = {}

files1 = ['ivy', 'lucene', 'poi', 'synapse', 'velocity', 'camel', 'jedit', 'log4j', 'xalan', 'xerces']
files2 = ['ivy1', 'lucene1', 'lucene2', 'poi1', 'poi2', 'synapse1', 'synapse2', 'camel1', 'camel2', 'xerces1', 'jedit1', 'jedit2', 'log4j1', 'xalan1']

_ = subprocess.Popen(f'ls results/{sys.argv[2]}-class/{sys.argv[1]}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode('utf-8').rstrip()
_ = _.split('\n')
try:
    _ = list(_)
except ValueError:
    raise
for file in _:
    filename = f'./results/{sys.argv[2]}-class/{sys.argv[1]}/{file}'
    
    _ = subprocess.Popen(f'grep "^Running" {filename}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode('utf-8').rstrip()
    _ = _.split('\n')
    try:
        _ = list(_)
    except ValueError:
        raise
    treatments = _.map(lambda x: eval(x.split("Running")[1]))[:6]
    regex = "^[[{]"
    _ = subprocess.Popen(f'grep "{regex}" {filename}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode('utf-8').rstrip()
    _ = _.split('\n')
    try:
        _ = list(_)
    except ValueError:
        raise
    results = _.map(lambda x: eval(x))[:6]
    results = [np.array(x).squeeze() if 'list' in x.__class__.__name__ else np.array(x['accuracy']).squeeze() for x in results]

    for i, (t, r) in enumerate(zip(treatments, results)):
        (dodge, wfo, smote, ultrasample, smooth) = t
        t = {'dodge': dodge, 'wfo': wfo, 'smote': smote, 'ultrasample': ultrasample, 'smooth': smooth}

        s = frozenset(t.items())
        if s not in pairs:
            pairs[s] = []
        pairs[s].append(r)

df = pd.DataFrame(columns=['dodge', 'wfo', 'smote', 'ultrasample', 'smooth', 'accuracy'])
all_perfs = []
for s, r in pairs.items():
    t = dict(s)
    all_perfs.extend(r)
    perf = np.median(r).squeeze()

    t = {**t, 'accuracy': perf}
    df = df.append(t, ignore_index=True)

print("Cohen's d: ", 0.35 * np.std(all_perfs))
print(df)
