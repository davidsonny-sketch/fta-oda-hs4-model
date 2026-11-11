"""원산지 사전점검: 완제품 HS + 원료 명세(BOM: 원료명·원산지국·금액) + FOB → 협정별 원산지 충족 여부

판정 순서 (협정마다 따로)
  1) 원료별 HS4: 사용자가 준 HS가 있으면 그것, 없으면 원재료 모델(hs_fn)로 추정
  2) 역내 원료 구분: 원산지국이 그 협정 회원국이면 역내(누적), 아니면·모르면 비원산지
  3) 세번변경기준(CTH): 비원산지 원료 중 제품과 같은 호(HS4)인 것이 없으면 충족
     - 같은 호 원료가 있어도 그 금액 합이 FOB의 10% 이하면 미소기준으로 충족
     - 소매용 포장재는 CTH 판단에서 제외
  4) 부가가치기준(RVC40): 공제법 (FOB - 비원산지 재료비) / FOB ≥ 40%
     - 원산지 모르는 원료와 소매용 포장재(비원산지면)도 비원산지 재료비에 넣음
     - 인건비·간접비·이윤을 주면 집적법도 함께 계산
  5) 품목별 기준이 'CTH 또는 RVC40'이면 둘 중 하나만 충족해도 원산지 인정
  6) 단, 국내 공정이 단순 공정(포장·병입·단순 혼합 등)뿐이면 CTH·RVC와 관계없이 불인정 (RCEP 3.6조, VKFTA 3.7조, AKFTA 규칙8)

근거 (원문 확인)
  - RCEP 제3.5조(RVC 공식, 원산지 미상 재료는 비원산지), 제3.7조(미소기준 FOB 10%), 제3.8조(소매 포장재)
  - 한-베트남 FTA 제3.3조(RVC 공식), 제3.9조(미소기준 FOB 10%), 제3.10조(소매 포장재)
  - 한-아세안 FTA 부속서3 규칙4(일반기준 RVC40 또는 CTH), 규칙10(미소기준), 규칙11(포장재)
  - 불인정 공정: RCEP 제3.6조, 한-베트남 FTA 제3.7조, 한-아세안 FTA 부속서3 규칙8 (모두 'notwithstanding' 조항)
  - 품목별 기준: 3304는 세 협정 모두 'CTH 또는 RVC40' 원문 확인. RCEP 3303·3401은 KITA 컨설팅 사례 기재. 나머지는 일반기준 가정
주의: 사전 점검용이며 최종 판정은 원산지 관리사·발급기관 확인이 필요함
"""
import re
import pandas as pd

ASEAN = {'BN', 'KH', 'ID', 'LA', 'MY', 'MM', 'PH', 'SG', 'TH', 'VN'}
AGREEMENTS = {
    'VKFTA': {'이름': '한-베트남 FTA', '회원국': {'KR', 'VN'}, '미소기준': 0.10, '근거': '제3.3조·제3.9조·제3.10조'},
    'AKFTA': {'이름': '한-아세안 FTA', '회원국': {'KR'} | ASEAN, '미소기준': 0.10, '근거': '부속서3 규칙4·10·11'},
    'RCEP': {'이름': 'RCEP', '회원국': {'KR', 'CN', 'JP', 'AU', 'NZ'} | ASEAN, '미소기준': 0.10, '근거': '제3.5조·제3.7조·제3.8조'},
    # 아래 3개는 관세청 「중소기업 100대 수출물품 FTA 원산지자료 작성가이드」 사례 검증용 (베트남 수출과는 무관)
    'EU': {'이름': '한-EU FTA', '회원국': {'KR', 'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV',
                                          'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'}, '미소기준': 0.10, '근거': '작성가이드 기재'},
    'TR': {'이름': '한-터키 FTA', '회원국': {'KR', 'TR'}, '미소기준': 0.10, '근거': '작성가이드 기재'},
    'IN': {'이름': '한-인도 CEPA', '회원국': {'KR', 'IN'}, '미소기준': 0.10, '근거': '작성가이드 기재'},
}
# 품목별 기준의 형태: 결합(또는/그리고), 세번변경 단위(4=CTH, 6=CTSH), 같은 호 허용 비율·기준가격, 부가가치 방식(BD=공제법 RVC 이상, MC=비원산지 비율 이하)
RULE_DEFAULT = {'결합': '또는', '세번': 4, '허용': 0.10, '허용기준': 'FOB', '부가': 'BD', '부가기준값': 0.40, '부가기준': 'FOB', '표기': 'CTH 또는 RVC40'}
RULE_OVERRIDE = {   # (협정, 제품 HS4): 작성가이드 화학산업 편 14~16번 사례에 적힌 기준
    ('EU', '3304'): {'결합': '또는', '세번': 4, '허용': 0.20, '허용기준': 'EXW', '부가': 'MC', '부가기준값': 0.40, '부가기준': 'EXW',
                     '표기': 'CTH(같은 호 비원산지 EXW 20%까지 허용) 또는 MC40(비원산지 ≤ EXW 40%)'},
    ('TR', '3304'): {'결합': '또는', '세번': 4, '허용': 0.20, '허용기준': 'EXW', '부가': 'MC', '부가기준값': 0.40, '부가기준': 'EXW',
                     '표기': 'CTH(같은 호 비원산지 EXW 20%까지 허용) 또는 MC40(비원산지 ≤ EXW 40%)'},
    ('IN', '3401'): {'결합': '그리고', '세번': 6, '허용': 0.0, '허용기준': 'FOB', '부가': 'BD', '부가기준값': 0.35, '부가기준': 'FOB',
                     '표기': 'CTSH 그리고 RVC35'},
}
def rule_for(ag, p4):
    return RULE_OVERRIDE.get((ag, p4), RULE_DEFAULT)
# 품목별 기준 확인 현황 {HS4: {협정: 출처}} — 기준은 모두 'CTH 또는 RVC40'
PSR_CONFIRMED = {
    '3304': {'VKFTA': '원문 확인', 'AKFTA': '원문 확인', 'RCEP': '원문 확인'},
    '3303': {'RCEP': 'KITA FTA 컨설팅 사례(G사) 기재'},
    '3401': {'RCEP': 'KITA FTA 컨설팅 사례(G사) 기재'},
}
PSR_DEFAULT = 'CTH 또는 RVC40'                       # 확인 안 된 협정·호는 일반기준 가정 → '확인 필요' 표시
RVC_MIN = 0.40
# 불인정 공정(단순 공정): CTH·RVC를 충족해도 원산지 불인정 — RCEP 제3.6조, 한-베트남 FTA 제3.7조 ("notwithstanding" 조항, 원문 확인)
MINIMAL_ARTICLE = {'VKFTA': '제3.7조', 'AKFTA': '부속서3 규칙8', 'RCEP': '제3.6조'}   # 세 협정 모두 원문 확인
MINIMAL_WORDS = r'(단순|충진만|충전만|소분|재포장|라벨|포장만|희석만|혼합만|filling only|repack)'
COUNTRY = {'한국': 'KR', '대한민국': 'KR', '국산': 'KR', '베트남': 'VN', '중국': 'CN', '일본': 'JP', '미국': 'US', '프랑스': 'FR',
           '독일': 'DE', '인도네시아': 'ID', '말레이시아': 'MY', '태국': 'TH', '필리핀': 'PH', '싱가포르': 'SG', '인도': 'IN',
           '호주': 'AU', '뉴질랜드': 'NZ', '미얀마': 'MM', '캄보디아': 'KH', '라오스': 'LA', '브루나이': 'BN', '스페인': 'ES',
           '이탈리아': 'IT', '영국': 'GB', '스위스': 'CH', '브라질': 'BR', '모름': '', '미상': '', '불명': ''}
DOCS = {
    'CTH': ['자재명세서(BOM)', '원료구입명세서', '생산공정명세서', '원료별 HS 확인 자료'],
    'RVC': ['자재명세서(BOM)', '원료구입명세서', '원료수불부', '원가산출내역서(FOB 구성)'],
    '공통': ['역내산으로 계산한 원료의 원산지(포괄)확인서 (원료 공급사 발급)'],
}


def country(x):
    """'한국', 'kr', 'Korea' 등을 국가 코드로. 비었거나 모르면 ''"""
    s = str(x or '').strip()
    if s in ('역내', '역내산'): return '*'          # 원산지확인서가 있는 역내산(협정 무관)
    if s in ('역외', '역외산', '비원산지'): return 'XX'   # 원산지 확인된 역외산
    if s.upper() in {c for a in AGREEMENTS.values() for c in a['회원국']} | {'US', 'FR', 'DE', 'IN', 'ES', 'IT', 'GB', 'CH', 'BR'}:
        return s.upper()
    if s in COUNTRY: return COUNTRY[s]
    en = {'korea': 'KR', 'vietnam': 'VN', 'viet nam': 'VN', 'china': 'CN', 'japan': 'JP', 'usa': 'US', 'france': 'FR', 'germany': 'DE'}
    return en.get(s.lower(), '')


def money(x):
    """'1,200', '1200원', 1200 → 1200.0 / 빈칸 → None"""
    if x is None or (isinstance(x, float) and pd.isna(x)): return None
    s = re.sub(r'[^\d.]', '', str(x))
    return float(s) if s else None


def hs4_of(x):
    d = re.sub(r'\D', '', str(x or ''))
    return d[:4] if len(d) >= 4 else ''


class OriginPrecheck:
    def __init__(self, hs_fn=None):
        """hs_fn(원료명) → dict(HS4=..., 후보=[...], 상태='자동 확정'|'확인 필요...') — 원재료 모델 연결용.
        없으면 BOM에 적힌 HS만 사용"""
        self.hs_fn = hs_fn

    def _material_hs(self, row):
        given = re.sub(r'\D', '', str(row.get('HS') or ''))
        if len(given) >= 4: return given, [given[:4]], '입력값'
        if self.hs_fn is None: return '', [], 'HS 없음'
        r = self.hs_fn(str(row.get('원료명', '')))
        return r['HS4'], list(r.get('후보', [r['HS4']])), r.get('상태', '모델 추정')

    def run(self, product_hs, fob, bom, packaging=(), costs=None, agreements=('VKFTA', 'AKFTA', 'RCEP'), process='', exw=None):
        """product_hs: 완제품 HS(6~10자리), fob: 제품 FOB 금액(원료 금액과 같은 단위), exw: 공장도가격(한-EU처럼 EXW 기준인 협정용, 없으면 FOB 사용)
        bom: [{'원료명', '원산지', '금액', 'HS'(선택)}], packaging: [{'포장재', '원산지', '금액', 'HS'(선택)}] (소매용)
        원산지: 국가명·코드, 또는 '역내'/'역외' / costs: 집적법용 {'인건비', '간접비', '이윤', '기타'} (선택)
        process: 국내 공정 설명 (예: '유화·혼합 제조', '벌크 수입 후 충진만'). 단순 공정이면 CTH·RVC와 관계없이 불인정"""
        minimal = bool(re.search(MINIMAL_WORDS, str(process), re.I))
        pdig, fob, exw = re.sub(r'\D', '', str(product_hs)), money(fob), money(exw)
        p4 = pdig[:4]
        mats = []
        for r in bom:
            h, cand, how = self._material_hs(r)
            mats.append({'구분': '원재료', '이름': r.get('원료명', ''), '원산지국': country(r.get('원산지')), '금액': money(r.get('금액')),
                         'HS4': h[:4], '세번': h, '후보': cand, 'HS 출처': how})
        for r in packaging:
            h = re.sub(r'\D', '', str(r.get('HS') or ''))
            mats.append({'구분': '소매용 포장재', '이름': r.get('포장재', ''), '원산지국': country(r.get('원산지')), '금액': money(r.get('금액')),
                         'HS4': h[:4], '세번': h, '후보': [], 'HS 출처': '입력값' if h else '-'})
        results, details = [], []
        for ag in agreements:
            A, RL = AGREEMENTS[ag], rule_for(ag, p4)
            lv = RL['세번']; unit = 'CTH' if lv == 4 else 'CTSH'
            pcode = pdig[:lv]
            base = {'FOB': fob, 'EXW': exw if exw else fob}
            tol_base, val_base = base[RL['허용기준']], base[RL['부가기준']]
            psr = RL['표기']
            psr_note = '작성가이드 기재' if (ag, p4) in RULE_OVERRIDE else PSR_CONFIRMED.get(p4, {}).get(ag, '일반기준 가정(품목별 기준 확인 필요)')
            rows, warn = [], []
            for m in mats:
                cc = m['원산지국']
                orig = cc == '*' or cc in A['회원국']
                mcode = m['세번'][:lv] if len(m['세번']) >= lv else ''
                same = (not orig) and m['구분'] == '원재료' and mcode != '' and mcode == pcode
                unsure = (not orig) and m['구분'] == '원재료' and not same and (
                    m['HS4'] == '' or mcode == '' and m['HS4'] == p4 or (m['HS 출처'].startswith('확인') and p4 in m['후보'])
                    or f'{p4} 가능성' in str(m['HS 출처']))   # 반제품 감지: 원재료 모델이 '제품과 같은 호 가능성'을 표시한 원료
                rows.append({**m, '협정': ag, '역내 여부': '역내' if orig else ('원산지 모름 → 비원산지' if not cc else '비원산지'),
                             'CTH': ('판단 제외(포장재)' if m['구분'] != '원재료' else '해당 없음(역내)' if orig else
                                     '세번 안 바뀜' if same else '확인 필요' if unsure else '세번 바뀜')})
            D = pd.DataFrame(rows)
            mat = D[D['구분'] == '원재료']
            # ── 세번변경(CTH/CTSH) + 같은 호 허용
            fail = mat[mat['CTH'] == '세번 안 바뀜']; uns = mat[mat['CTH'] == '확인 필요']
            if len(fail) == 0 and len(uns) == 0: cth, cth_why = '충족', f'비원산지 원료가 모두 제품과 다른 {"호" if lv == 4 else "소호"}'
            elif len(fail) == 0: cth, cth_why = '확인 필요', '원료 HS 재확인 필요: ' + ', '.join(uns['이름'])
            else:
                v = fail['금액']
                if v.isna().any() or not tol_base or RL['허용'] == 0:
                    cth, cth_why = '불충족', '같은 호 비원산지 원료' + ('(금액 없어 허용 비율 계산 불가)' if RL['허용'] else '') + ': ' + ', '.join(fail['이름'])
                elif v.sum() <= RL['허용'] * tol_base:
                    cth, cth_why = '충족(미소기준)', f"같은 호 비원산지 원료 {v.sum():,.0f} ≤ {RL['허용기준']}의 {RL['허용']:.0%}({RL['허용'] * tol_base:,.0f})"
                else:
                    cth, cth_why = '불충족', f"같은 호 비원산지 원료 {', '.join(fail['이름'])} {v.sum():,.0f} > {RL['허용기준']}의 {RL['허용']:.0%}({RL['허용'] * tol_base:,.0f})"
            # ── 부가가치 (BD: 공제법 RVC ≥ 기준, MC: 비원산지 비율 ≤ 기준)
            nonorig = D[D['역내 여부'] != '역내']
            missing = nonorig[nonorig['금액'].isna()]
            thr = RL['부가기준값']; vname = ('RVC' if RL['부가'] == 'BD' else 'MC') + f'{thr * 100:.0f}'
            if not val_base or val_base <= 0:
                rvc, rvc_v, rvc_why, room = '계산 불가', None, f"{RL['부가기준']} 없음", None
            elif len(missing):
                rvc, rvc_v, rvc_why, room = '계산 불가', None, '금액 없는 비원산지 원료: ' + ', '.join(missing['이름']), None
            else:
                vnm = nonorig['금액'].sum()
                if RL['부가'] == 'BD':
                    rvc_v = (val_base - vnm) / val_base; ok = rvc_v >= thr - 1e-12; room = (1 - thr) * val_base - vnm
                    rvc_why = f"공제법 ({val_base:,.0f} - 비원산지 {vnm:,.0f}) / {val_base:,.0f} = {rvc_v:.1%}"
                else:
                    mc = vnm / val_base; rvc_v = 1 - mc; ok = mc <= thr + 1e-12; room = thr * val_base - vnm
                    rvc_why = f"MC법 비원산지 {vnm:,.0f} / {RL['부가기준']} {val_base:,.0f} = {mc:.1%} (기준 {thr:.0%} 이하)"
                rvc = '충족' if ok else '불충족'
            buildup = None
            if costs and fob:
                vom = D[D['역내 여부'] == '역내']['금액'].fillna(0).sum() + sum(money(costs.get(k)) or 0 for k in ('인건비', '간접비', '이윤', '기타'))
                buildup = vom / fob
            # ── 협정 결론
            c_ok, v_ok = cth.startswith('충족'), rvc == '충족'
            c_tag = unit + ('(허용비율)' if '미소' in cth else '')
            if minimal: verdict, basis = '원산지 불충족', '단순 공정(불인정 공정)'
            elif RL['결합'] == '또는':
                if c_ok: verdict, basis = '원산지 충족', 'CTH' + ('(미소기준)' if '미소' in cth else '') if lv == 4 else c_tag
                elif v_ok: verdict, basis = '원산지 충족', 'RVC40' if vname == 'RVC40' else vname
                elif cth == '확인 필요' or rvc == '계산 불가': verdict, basis = '확인 필요', '-'
                else: verdict, basis = '원산지 불충족', '-'
            else:   # 그리고
                if c_ok and v_ok: verdict, basis = '원산지 충족', f'{c_tag}+{vname}'
                elif (cth == '불충족') or (rvc == '불충족'): verdict, basis = '원산지 불충족', '-'
                else: verdict, basis = '확인 필요', '-'
            if (D['원산지국'] == '').any(): warn.append('원산지 모르는 원료는 비원산지로 계산함 → 공급사 원산지확인서를 받으면 개선될 수 있음')
            if ((D['역내 여부'] == '역내') & ~D['원산지국'].isin(['KR', '*'])).any(): warn.append(f'{A["이름"]} 누적: 다른 회원국 원료는 그 나라의 원산지증명이 있어야 역내로 인정')
            if ((D['역내 여부'] == '역내') & D['원산지국'].isin(['KR', '*'])).any(): warn.append('국내 구매 원료도 공급사의 원산지(포괄)확인서가 있어야 역내산으로 인정')
            if minimal: warn.append(f'국내 공정이 단순 공정({process})이면 세번변경·부가가치와 관계없이 원산지 불인정 ({ag} {MINIMAL_ARTICLE.get(ag, "불인정 공정 조문")})')
            elif len(fail) and fob and fail['금액'].notna().all() and fail['금액'].sum() >= 0.5 * fob:
                warn.append('제품과 같은 호의 수입 원료(벌크·반제품) 비중이 커서, 국내 공정이 충진·포장뿐이면 단순 공정으로 불인정될 수 있음 → 국내 제조 공정 확인 필요')
            if psr_note.startswith('일반기준'): warn.append(f'제품 호 {p4}의 품목별 기준은 일반기준(CTH 또는 RVC40)으로 가정함 → 협정 부속서 확인 필요')
            docs = (DOCS['CTH'] if basis.startswith(('CTH', 'CTSH')) and '+' not in basis else DOCS['RVC'] if basis.startswith(('RVC', 'MC')) else DOCS['CTH'] + DOCS['RVC'][2:]) + DOCS['공통']
            results.append({'협정': ag, '협정명': A['이름'], '품목별 기준': psr, '기준 출처': psr_note, 'CTH': cth, 'CTH 근거': cth_why,
                            'RVC': rvc, 'RVC(공제법)': None if rvc_v is None else round(rvc_v, 4), 'RVC 근거': rvc_why,
                            'RVC(집적법)': None if buildup is None else round(buildup, 4),
                            '비원산지 추가 여유(RVC40 유지)': None if room is None else round(room),
                            '결론': verdict, '충족 기준': basis, '필요 서류': ' / '.join(docs), '주의': ' | '.join(warn),
                            '조문': A['근거'] + ('' if not minimal else ' / 불인정 공정 ' + MINIMAL_ARTICLE.get(ag, '조문 확인 필요'))})
            details.append(D)
        return pd.DataFrame(results), pd.concat(details, ignore_index=True)


def build_hs_fn(base='.', use_semi=True, use_lexicon=True):
    """원재료 모델 전체 파이프라인을 hs_fn으로: 성분명 정리(식약처) → 학습표 이름 일치 → HS 호 용어 사전(v2) → 분자 구조 규칙 → 이름 모델 → 반제품 감지
    base: gold_814_split.csv, mfds_db.csv, pubchem.csv, real_pubchem.csv 가 있는 폴더 (Colab: /content/rawmodel)"""
    import os
    import hs4_bilingual as H
    from common import load, TARGET, Memory, norm, struct_allowed
    from struct_rules import heading29
    from name_norm import NameMatcher
    from bom_check import semi_heading
    import hs_lexicon as LEX
    g = load(f'{base}/gold_814_split.csv'); tr = g[g['데이터구분'] == '학습']
    clf = H.HS4Classifier(tr.material_name, tr.material_name_en, tr[TARGET]); mem = Memory(tr)
    nm = NameMatcher(pd.read_csv(f'{base}/mfds_db.csv', dtype=str).fillna(''))
    pc = pd.read_csv(f'{base}/pubchem.csv', dtype=str, encoding='utf-8-sig').fillna('')
    sm_id = dict(zip(pc.entity_id, pc.SMILES)); gsm = {}
    for _, r in g.iterrows():
        for k in (r.material_name, r.material_name_en):
            if k and sm_id.get(r.entity_id): gsm.setdefault(norm(k), sm_id[r.entity_id])
    if os.path.exists(f'{base}/real_pubchem.csv'):
        RP = pd.read_csv(f'{base}/real_pubchem.csv', dtype=str, encoding='utf-8-sig').fillna('')
        for _, r in RP.iterrows():
            for k in (r['국문명'], r['영문명']):
                if k and r.SMILES: gsm.setdefault(norm(k), r.SMILES)
    def f(name):
        name = str(name)
        ko, en = (name, '') if re.search('[가-힣]', name) else ('', name)
        m = nm.match(ko or name, en)
        if m['구분'].startswith('식약처'): ko, en = m['국문명'], m['영문명']
        r = clf.predict(ko, en); top3, state = [r['HS4_1순위'], r['HS4_2순위'], r['HS4_3순위']], r['판정상태']
        hit = mem.get(ko, en) or mem.get(name, name)   # 2026-11-10: 이름을 표준명으로 바꾼 뒤에도 원래 입력 이름이 학습 데이터에 있으면 그 정답
        lex = None if hit or not use_lexicon else (LEX.lookup(ko, en) or LEX.lookup(*((name, '') if re.search('[가-힣]', name) else ('', name))))
        if hit: top3, state = ([hit] + [x for x in top3 if x != hit])[:3], '자동 확정(학습 데이터 일치)'
        elif lex: top3, state = ([lex[0]] + [x for x in top3 if x != lex[0]])[:3], f'자동 확정(HS 호 용어: {lex[1]})'   # 2026-11-11 v2
        else:
            smi = gsm.get(norm(ko)) or gsm.get(norm(en)) or ''
            s = heading29(smi)[0] if smi and struct_allowed(en) else None
            if s: top3, state = ([s] + [x for x in top3 if x != s])[:3], '자동 확정(분자 구조 규칙)'
        sh = semi_heading(name) if use_semi else None
        if sh: top3, state = ([sh] + [x for x in top3 if x != sh])[:3], '확인 필요(반제품·베이스 의심)'
        return {'HS4': top3[0], '후보': top3, '상태': state}
    return f

