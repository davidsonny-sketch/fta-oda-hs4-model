"""원재료 모델 공통 설정 (완제품 모델과 같은 평가 틀)"""
import re, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

TARGET = 'target_hs4'
norm = lambda s: re.sub(r'[\s\-·,()\[\]./\'’]', '', str(s)).lower()

def load(path='gold_814_split.csv'):
    g = pd.read_csv(path, dtype=str).fillna('').reset_index(drop=True)
    # 같은 원료(표기만 다른 것 포함)를 한 그룹으로: 영문명 기준, 국문명이 같으면 합침
    grp = [norm(e) if e else norm(k) for e, k in zip(g.material_name_en, g.material_name)]
    first = {}
    for i, k in enumerate(g.material_name.map(norm)):
        first.setdefault(k, grp[i]); grp[i] = first[k]
    g['input_group'] = grp
    cc = next(c for c in g.columns if c.startswith('hs6_후보'))
    g['정답후보'] = [';'.join(sorted({t} | {c.replace('.', '')[:4] for c in str(h).split(';') if c.strip()})) for t, h in zip(g[TARGET], g[cc])]
    return g

def eval_frame(g, min_groups=4):
    """주 평가 대상: 서로 다른 원료 그룹이 4개 이상인 HS4 (완제품 01_baseline과 같은 기준)"""
    tr = g[g['데이터구분'] == '학습']
    ng = tr.groupby(TARGET)['input_group'].nunique()
    keep = ng[ng >= min_groups].index
    return tr[tr[TARGET].isin(keep)].reset_index(drop=True), ng

def splits(df, n=4, seed=42):
    return list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(df, df[TARGET], df['input_group']))

def metrics(y, top1, top3=None, cand=None):
    out = {'accuracy': accuracy_score(y, top1), 'macro_f1': f1_score(y, top1, average='macro', zero_division=0),
           'weighted_f1': f1_score(y, top1, average='weighted', zero_division=0)}
    if top3 is not None: out['top3_accuracy'] = float(np.mean([a in b for a, b in zip(y, top3)]))
    if cand is not None: out['lenient_accuracy'] = float(np.mean([p in c.split(';') for p, c in zip(top1, cand)]))
    return out

# 고친 점(2026-11-08): 학습 데이터에 이름이 그대로 있는 원료는 그 정답을 먼저 씀(구조 규칙보다 우선)
class Memory:
    def __init__(self, train):
        self.m = {}
        for k, e, y in zip(train.material_name, train.material_name_en, train[TARGET]):
            for key in (k, e):
                if key: self.m.setdefault(norm(key), y)
    def get(self, ko, en):
        for key in (ko, en):
            if key and norm(key) in self.m: return self.m[norm(key)]
        return None

# 구조 규칙 적용 여부 (struct_rules의 제외 + 계면활성제·폴리에터·단백질 제외)
def struct_allowed(en):
    import re
    from struct_rules import SKIP_NAME, EXCLUDE_NAME
    from hs4_bilingual import SURF_SKIP
    en = (en or '').lower()
    return not (re.search(SKIP_NAME, en) or re.search(EXCLUDE_NAME, en) or re.search(SURF_SKIP, en))
