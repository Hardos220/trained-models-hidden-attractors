import json, glob, re, os, collections, math
import numpy as np
os.chdir(os.environ.get('ESN_DATA', 'local/esn-gpu'))
pat = re.compile(r'HE1_N300_rho0\.9_eta(1e-07|0\.0001)_s(\d)(?:_in([\d.]+))?(?:_dens([\d.]+))?\.json$')
def load(path_glob):
    out = {}
    for f in glob.glob(path_glob):
        m = pat.search(os.path.basename(f))
        if not m: continue
        r = json.load(open(f))
        out[os.path.basename(f)] = r
    return out
rows = []
for reg in ['se3', 'se11', 'le1', 'chua_I']:
    loc = load(f'node_results/N/{reg}/HE1_N300_rho0.9_*.json')
    for name, r in loc.items():
        isc = r.get('input_scale', 0.5); dn = r.get('density', 0.05)
        setting = 'base' if (isc == 0.5 and dn == 0.05) else (f'in{isc:g}' if isc != 0.5 else f'dens{dn:g}')
        for a, h in r['hidden_match'].items():
            rows.append(dict(reg=reg, a=a, setting=setting, U=r['U'], K=r['K'], sig=r['index_sum'], rep=bool(h['reproduced']),
                             hid=bool(h['reproduced']) and h.get('label') == 'not_excited', name=name, src='cpu'))
# A100 base for comparison (same names)
gpu = collections.defaultdict(dict)
for f in glob.glob('gpuq-results/*/out/N/*/HE1_N300_rho0.9_eta*_s?.json'):
    reg = f.replace(chr(92),'/').split('/')[-2]
    if reg not in ('se3', 'se11', 'le1', 'chua_I'): continue
    r = json.load(open(f)); gpu[reg][os.path.basename(f)] = r
print('settings table (hidden attractor x model):')
T = collections.defaultdict(list)
for x in rows: T[(x['reg'], x['setting'])].append(x)
for (reg, st), xs in sorted(T.items()):
    n = len(xs); rep = sum(x['rep'] for x in xs); hid = sum(x['hid'] for x in xs)
    print(f"  {reg:7s} {st:9s} n {n:3d}  U mean {np.mean([x['U'] for x in xs]):5.1f}  Σind=1 {sum(x['sig']==1 for x in xs)}/{n}  reproduced {rep:3d}  hidden {hid:3d}/{rep}")
# CPU vs GPU on base
print('CPU vs GPU, base setting:')
for reg in ['se3', 'se11', 'le1', 'chua_I']:
    loc = load(f'node_results/N/{reg}/HE1_N300_rho0.9_eta*_s?.json')
    same = diff = 0; dU = []
    for name, r in loc.items():
        if r.get('input_scale', 0.5) != 0.5 or r.get('density', 0.05) != 0.05: continue
        g = gpu[reg].get(name)
        if g is None: continue
        dU.append(r['U'] - g['U'])
        for a in r['hidden_match']:
            l1 = (r['hidden_match'][a]['reproduced'], r['hidden_match'][a].get('label')); l2 = (g['hidden_match'][a]['reproduced'], g['hidden_match'][a].get('label'))
            same += l1 == l2; diff += l1 != l2
    if dU: print(f"  {reg}: models {len(dU)}, U equal {sum(d==0 for d in dU)}/{len(dU)}, labels same {same}, differ {diff}")
# logistic: hidden | reproduced ~ log(1+U) + setting + regime
X = [x for x in rows if x['rep']]
sets = sorted({x['setting'] for x in X} - {'base'}); regs = sorted({x['reg'] for x in X})
def design(xs, with_set=True):
    cols = ['const', 'log1pU'] + [f'reg_{g}' for g in regs[1:]] + ([f'set_{s}' for s in sets] if with_set else [])
    M = []
    for x in xs:
        v = [1.0, math.log1p(x['U'])] + [float(x['reg'] == g) for g in regs[1:]] + ([float(x['setting'] == s) for s in sets] if with_set else [])
        M.append(v)
    return np.array(M), cols
def logit(M, y, it=100):
    b = np.zeros(M.shape[1])
    for _ in range(it):
        p = 1/(1+np.exp(-M@b)); W = p*(1-p)+1e-9
        H = M.T@(M*W[:, None]) + 1e-6*np.eye(len(b)); g = M.T@(y-p)
        step = np.linalg.solve(H, g); b += step
        if np.abs(step).max() < 1e-9: break
    p = 1/(1+np.exp(-M@b)); ll = float(np.sum(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12)))
    se = np.sqrt(np.diag(np.linalg.inv(M.T@(M*(p*(1-p))[:, None]) + 1e-9*np.eye(len(b)))))
    return b, se, ll
y = np.array([float(x['hid']) for x in X])
M, cols = design(X); b, se, ll1 = logit(M, y)
M0, _ = design(X, False); _, _, ll0 = logit(M0, y)
print(f'logistic on {len(X)} reproduced (attractor x model):')
for c, bb, s in zip(cols, b, se): print(f'  {c:12s} {bb:+.2f} ± {s:.2f}  z {bb/s:+.1f}')
print(f'  LR test settings beyond U: chi2 {2*(ll1-ll0):.1f}, df {len(sets)}')
