"""최종 모델 v2 개발 틀 — 과적합을 막는 규칙
1) 시험 A·B는 여기서 절대 읽지 않음(학습 722행만 사용)
2) 학습 722행을 원료 그룹 단위로 '설계용 60%'와 '숨김 검증용 40%'로 고정 분할(seed 2026)
   - 오답을 눈으로 보고 개선안을 만드는 것은 설계용만
   - 숨김 검증용은 합계 점수만 봄(개별 오답은 보지 않음) → 설계용에서 만든 개선이 처음 보는 원료에도 통하는지 확인
3) 점수는 반복 그룹 교차검증(시드 5개 × 4겹 = 20번)의 표본외 예측으로 계산(같은 원료는 한쪽에만)
"""
import sys, os, re, pickle, warnings; warnings.filterwarnings('ignore')
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(H); sys.path.insert(0, R)
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from common import load, TARGET, Memory, struct_allowed
import hs4_bilingual as HB
from struct_rules import heading29
import hs_lexicon as LEX

g = load(f'{R}/gold_814_split.csv')
T = g[g['데이터구분'] == '학습'].reset_index(drop=True)          # 시험 A 행은 버림
pc = pd.read_csv(f'{R}/pubchem.csv', dtype=str, encoding='utf-8-sig').fillna(''); SM = dict(zip(pc.entity_id, pc.SMILES))
_rp = f'{R}/real_pubchem.csv' if os.path.exists(f'{R}/real_pubchem.csv') else f'{H}/hk_smiles_local.csv'   # Colab: 실제 파일 / 로컬: 손으로 넣은 HK 구조
_r = pd.read_csv(_rp, dtype=str, encoding='utf-8-sig').fillna(''); SM.update({k: s for k, s, t in zip(_r['키'], _r.SMILES, _r['구분']) if t == 'HK' and s})
a, b = next(GroupShuffleSplit(1, test_size=0.4, random_state=2026).split(T, groups=T.input_group))
T['구역'] = '설계'; T.loc[b, '구역'] = '숨김검증'
SEEDS = [0, 1, 2, 3, 4]

def folds():
    y = T[TARGET].where(T[TARGET].map(T[TARGET].value_counts()) >= 4, 'rare')   # 층화용(드문 코드는 한 묶음)
    for s in SEEDS:
        for k, (tr, te) in enumerate(StratifiedGroupKFold(4, shuffle=True, random_state=s).split(T, y, T.input_group)):
            yield s, k, tr, te

def base_cache(path=f'{H}/base_cache.pkl', C=20, class_weight=None, tag=''):
    """겹마다 이름 모델을 한 번만 학습해 시험 행의 확률을 저장(규칙만 바꿀 때 다시 학습하지 않도록)"""
    path = path.replace('.pkl', f'{tag}.pkl')
    if os.path.exists(path): return pickle.load(open(path, 'rb'))
    out = []
    for s, k, tr, te in folds():
        trn, tst = T.iloc[tr], T.iloc[te]
        clf = HB.HS4Classifier.__new__(HB.HS4Classifier)
        ko = [str(x) for x in trn.material_name]; en = [str(x).lower() for x in trn.material_name_en]; y = np.asarray(trn[TARGET])
        from scipy.sparse import hstack
        from sklearn.linear_model import LogisticRegression
        vk, ve = HB.vec().fit(ko), HB.vec().fit(en)
        Xtr = hstack([vk.transform(ko), ve.transform(en)]).tocsr()
        lr = LogisticRegression(C=C, max_iter=3000, class_weight=class_weight).fit(Xtr, y)
        Xte = hstack([vk.transform(tst.material_name), ve.transform(tst.material_name_en.str.lower())]).tocsr()
        P = lr.predict_proba(Xte)
        mem = Memory(trn)
        out.append(dict(seed=s, fold=k, te=te, classes=lr.classes_, P=P,
                        mem=[mem.get(k_, e_) for k_, e_ in zip(tst.material_name, tst.material_name_en)]))
    pickle.dump(out, open(path, 'wb')); return out

_ST = {}
def struct_of(i, extra_skip=None):
    """행 i의 구조 규칙 호(없으면 None). extra_skip: 구조 규칙을 쓰지 않을 이름 패턴(추가)"""
    e, en = T.entity_id[i], T.material_name_en[i]
    if i not in _ST:
        s = SM.get(e, ''); _ST[i] = heading29(s)[0] if s and struct_allowed(en) else None
    if _ST[i] and extra_skip and re.search(extra_skip, f'{T.material_name[i]} {en}', re.I): return None
    return _ST[i]

def rule_sets(ko, en, extra=()):
    """기존 규칙 + 추가 규칙군(extra: [(이름, 국문패턴, 영문패턴, 호 집합, 이 호들과 다투는 것 제거)])"""
    en = en.lower()
    rs = HB.M_rule(ko) | HB._sets(HB.RULES_EN, en); cs = HB.M_chem(ko) | HB._sets(HB.CHEM_EN, en)
    for p, drop in HB.PRIORITY:
        if re.search(p, ko, re.I) or re.search(p, en, re.I): rs -= drop; cs -= drop
    strong = set()
    for name, pk, pe, hs, excl in extra:
        if (pk and re.search(pk, ko, re.I)) or (pe and re.search(pe, en, re.I)):
            if excl: rs.clear(); cs.clear()
            strong |= hs
    return rs, cs, strong

def predict_all(cache, extra=(), extra_skip=None, use_struct=True, use_mem=True, sbonus=0.6, lex=False):
    """표본외 예측 표: 시드별로 모든 학습 행에 대해 1~3순위"""
    rows = []
    for c in cache:
        for j, i in enumerate(c['te']):
            ko, en = T.material_name[i], T.material_name_en[i]
            rs, cs, strong = rule_sets(ko, en, extra)
            sc = {h: p + (HB.BONUS if h in rs else 0) + (HB.CBONUS if h in cs else 0) + (sbonus if h in strong else 0) for h, p in zip(c['classes'], c['P'][j])}
            for h in rs | cs | strong: sc.setdefault(h, (HB.BONUS if h in rs else 0) * 0.5 + (HB.CBONUS if h in cs else 0) + (sbonus if h in strong else 0))
            t3 = sorted(sc, key=lambda h: -sc[h])[:3]; src = '이름모델'; top = sc[t3[0]]
            m = c['mem'][j] if use_mem else None
            if m: t3 = ([m] + [x for x in t3 if x != m])[:3]; src = '학습일치'
            elif lex and LEX.lookup(ko, en):
                h = LEX.lookup(ko, en)[0]; t3 = ([h] + [x for x in t3 if x != h])[:3]; src = '호용어사전'
            elif use_struct:
                s = struct_of(i, extra_skip)
                if s and not strong: t3 = ([s] + [x for x in t3 if x != s])[:3]; src = '구조'
            rows.append((c['seed'], i, t3[0], ';'.join(t3), src, round(top, 3)))
    return pd.DataFrame(rows, columns=['seed', 'i', 'p1', 'p3', 'src', 'score'])

def score(O, mask=None):
    """시드별 점수의 평균(정확도·macro F1·weighted F1·3순위 안)"""
    res = []
    for s, d in O.groupby('seed'):
        d = d.set_index('i').loc[T.index if mask is None else T.index[mask]]
        y = T[TARGET][d.index]
        res.append((accuracy_score(y, d.p1), f1_score(y, d.p1, average='macro', zero_division=0), f1_score(y, d.p1, average='weighted', zero_division=0),
                    np.mean([a in b.split(';') for a, b in zip(y, d.p3)])))
    m = np.mean(res, 0)
    return dict(n=int(len(d)), acc=round(m[0], 3), macro_f1=round(m[1], 3), w_f1=round(m[2], 3), top3=round(m[3], 3))

def report(O, label=''):
    D, Hd = (T['구역'] == '설계').values, (T['구역'] == '숨김검증').values
    print(f"{label:34s} 전체 {score(O)} | 설계 {score(O, D)} | 숨김 {score(O, Hd)}")

if __name__ == '__main__':
    print(T['구역'].value_counts().to_dict(), '코드 수', T[TARGET].nunique())
    c = base_cache(); O = predict_all(c); report(O, '기준(현재 최종 모델)')
    report(predict_all(c, use_struct=False), '  구조 규칙 없이')
    O.to_pickle(f'{H}/oof_base.pkl')
