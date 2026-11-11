"""최종 시험: 시험 A(시트에서 떼어둔 92행), 시험 B(실제 성분표 120종, 정답 2026-11-10 검증 완료 · 분류불가 10종 제외 110종 채점)
모델은 학습 722행 전체로 한 번 학습. 시험 데이터로는 아무것도 조정하지 않음"""
import sys, re; sys.path.insert(0, '..')
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from common import *
import hs4_bilingual as H
from struct_rules import heading29, SKIP_NAME, EXCLUDE_NAME
from name_norm import NameMatcher
from bom_check import OriginChecker
g = load('../gold_814_split.csv'); tr = g[g['데이터구분'] == '학습']; ta = g[g['데이터구분'] == '시험A']
pc = pd.read_csv('../pubchem.csv', dtype=str, encoding='utf-8-sig').fillna(''); SM = dict(zip(pc.entity_id, pc.SMILES))
import os
RP = pd.read_csv('../real_pubchem.csv', dtype=str, encoding='utf-8-sig').fillna('') if os.path.exists('../real_pubchem.csv') else pd.DataFrame(columns=['구분', '키', '국문명', '영문명', 'SMILES'])
SM.update({k: s for k, s, t in zip(RP['키'], RP.SMILES, RP['구분']) if t == 'HK' and s})   # 2026-11-08: 관세청 HSK 추가분 구조
clf = H.HS4Classifier(tr.material_name, tr.material_name_en, tr[TARGET]); clf.ko2en, clf.en2ko = {}, {}
MEM = Memory(tr)
import hs_lexicon as LEX
def predict(ko, en, smiles='', use_memory=True, v2=False):
    r = clf.predict(ko, en); t3 = [r['HS4_1순위'], r['HS4_2순위'], r['HS4_3순위']]; s = None
    m = MEM.get(ko, en) if use_memory else None
    if m: return m, ([m] + [x for x in t3 if x != m])[:3], '학습 데이터 일치', False
    lx = LEX.lookup(ko, en) if v2 else None                       # 2026-11-11 v2: HS 호 용어 사전
    if lx: return lx[0], ([lx[0]] + [x for x in t3 if x != lx[0]])[:3], 'HS 호 용어 사전', False
    if smiles and struct_allowed(en): s = heading29(smiles)[0]
    if s: t3 = ([s] + [x for x in t3 if x != s])[:3]
    return t3[0], t3, r['판정상태'], bool(s)
# 시험 A
PA = [predict(k, e, SM.get(i, '')) for k, e, i in zip(ta.material_name, ta.material_name_en, ta.entity_id)]
PA0 = [predict(k, e, '') for k, e in zip(ta.material_name, ta.material_name_en)]
res = [{'시험': '시험A(시트 92행)', '방법': '이름 모델', **metrics(ta[TARGET].tolist(), [p[0] for p in PA0], [p[1] for p in PA0], ta['정답후보'].tolist())},
       {'시험': '시험A(시트 92행)', '방법': '이름 모델 + 구조 규칙', **metrics(ta[TARGET].tolist(), [p[0] for p in PA], [p[1] for p in PA], ta['정답후보'].tolist())}]
PA2 = [predict(k, e, SM.get(i, ''), v2=True) for k, e, i in zip(ta.material_name, ta.material_name_en, ta.entity_id)]
res.append({'시험': '시험A(시트 92행)', '방법': 'v2: + HS 호 용어 사전', **metrics(ta[TARGET].tolist(), [p[0] for p in PA2], [p[1] for p in PA2], ta['정답후보'].tolist())})
pd.DataFrame([{'entity_id': i, 'material_name': k, 'true': y, 'pred_top1': p[0], 'pred_top3': ';'.join(p[1]), '근거': p[2]}
              for i, k, y, p in zip(ta.entity_id, ta.material_name, ta[TARGET], PA2)]).to_csv('test_a_predictions_v2.csv', index=False, encoding='utf-8-sig')
pd.DataFrame([{'entity_id': i, 'material_name': k, 'true': y, 'pred_top1': p[0], 'pred_top3': ';'.join(p[1]), '구조적용': p[3]}
              for i, k, y, p in zip(ta.entity_id, ta.material_name, ta[TARGET], PA)]).to_csv('test_a_predictions.csv', index=False, encoding='utf-8-sig')
# 시험 B: 정답 = '확정 HS4'가 있으면 그것, 없으면 '제안 HS4'(초안). 분류불가 제외
B = pd.read_csv('../test_b_v2.csv', dtype=str).fillna('')
B['정답'] = np.where(B['확정 HS4'] != '', B['확정 HS4'], B['제안 HS4']); B['정답 상태'] = np.where(B['확정 HS4'] != '', '확정', '초안')
B = B[B['정답'] != '분류불가'].reset_index(drop=True)
db = pd.read_csv('../mfds_db.csv', dtype=str).fillna(''); nm = NameMatcher(db); oc = OriginChecker(None, db, None)
gsm = {}
for _, r in g.iterrows():
    for k in (r.material_name, r.material_name_en):
        if k and SM.get(r.entity_id): gsm.setdefault(norm(k), SM[r.entity_id])
for _, r in RP[RP['구분'] == '실제성분'].iterrows():                 # 2026-11-08: 실제 성분도 PubChem 구조 사용
    for k in (r['국문명'], r['영문명']):
        if k and r.SMILES: gsm.setdefault(norm(k), r.SMILES)
out = []
for _, r in B.iterrows():
    ko0, en0, _, how0 = oc.lookup(r['입력 국문명'], r['입력 영문명'])          # 정리 전(예전 방식)
    if how0 != '식약처 DB': ko0, en0 = r['입력 국문명'], r['입력 영문명']
    m = nm.match(r['입력 국문명'], r['입력 영문명'])                          # 정리 후
    ko1, en1 = (m['국문명'], m['영문명']) if m['구분'].startswith('식약처') else (r['입력 국문명'], r['입력 영문명'])
    sm1 = gsm.get(norm(ko1)) or gsm.get(norm(en1)) or ''
    p0, p1 = predict(ko0, en0, '', use_memory=False), predict(ko1, en1, sm1)
    p2 = predict(ko1, en1, sm1, v2=True)
    if p2[2] != 'HS 호 용어 사전' and not MEM.get(ko1, en1):       # 표준명으로 못 찾았으면 입력 이름으로도 사전을 찾음
        lx = LEX.lookup(r['입력 국문명'], r['입력 영문명'])
        if lx: p2 = (lx[0], ([lx[0]] + [x for x in p2[1] if x != lx[0]])[:3], 'HS 호 용어 사전', False)
    out.append({'입력 국문명': r['입력 국문명'], '입력 영문명': r['입력 영문명'], '정답': r['정답'], '정답 상태': r['정답 상태'],
                '정리 전 찾기': how0, '정리 후 찾기': m['구분'], '정리 후 근거': m['근거'], '정리 전 1순위': p0[0], '정리 후 1순위': p1[0], '정리 후 3순위': ';'.join(p1[1]), '구조적용': p1[3], 'v2 1순위': p2[0], 'v2 3순위': ';'.join(p2[1]), 'v2 근거': p2[2]})
O = pd.DataFrame(out); O.to_csv('test_b_predictions.csv', index=False, encoding='utf-8-sig')
y = O['정답'].tolist(); cand = y
res += [{'시험': f'시험B(실제 성분 {len(O)}종, 정답 검증 완료, 초안 때 규칙 수정에 참고함)', '방법': '정리 전 이름 → 이름 모델', **metrics(y, O['정리 전 1순위'].tolist())},
        {'시험': f'시험B(실제 성분 {len(O)}종, 정답 검증 완료, 초안 때 규칙 수정에 참고함)', '방법': '성분명 정리 → 이름 모델 + 구조 규칙', **metrics(y, O['정리 후 1순위'].tolist(), O['정리 후 3순위'].str.split(';').tolist())}]
res.append({'시험': f'시험B(실제 성분 {len(O)}종, 정답 검증 완료, 초안 때 규칙 수정에 참고함)', '방법': 'v2: + HS 호 용어 사전', **metrics(y, O['v2 1순위'].tolist(), O['v2 3순위'].str.split(';').tolist())})
R = pd.DataFrame(res); R.to_csv('final_test_results.csv', index=False, encoding='utf-8-sig'); print(R.round(3).to_string())
print('시험B 식약처 찾기: 정리 전', (O['정리 전 찾기'] == '식약처 DB').mean().round(3), '→ 정리 후', O['정리 후 찾기'].str.startswith('식약처').mean().round(3), '| 정답 확정', (O['정답 상태'] == '확정').sum())
W = O[O['정리 후 1순위'] != O['정답']]
print('시험B 틀린 것', len(W), '개 (구조 적용', int((O['구조적용'] == True).sum()), '개)')
print(W[['입력 국문명', '정답', '정리 후 1순위', '정리 후 찾기', '구조적용']].to_string())
