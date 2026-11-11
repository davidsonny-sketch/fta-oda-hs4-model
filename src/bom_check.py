"""원산지 점검기: 완제품 세번 + 성분 목록(BOM) → 원료별 HS4 → 세번변경기준(CTH) 판정

흐름
  1) 성분 목록을 쉼표로 나누고 함량(%)·괄호 별칭 정리
  2) 식약처 원료성분 DB로 공식 국문명·영문명·CAS 찾기
  3) 원재료 판정기(이름 모델 + 분자 구조 규칙)로 원료별 HS4
  4) 제품 HS4와 원료 HS4 비교 → CTH (원료가 모두 비원산지라고 가정한 보수적 판정)
  5) 포장 설명에서 소매용 포장재를 뽑아 따로 표시 (CTH 판단 제외, RVC 계산에는 포함: VKFTA 3.10조, RCEP 3.8조)
"""
import re
import pandas as pd

PSR = {  # 확인한 품목별 원산지 기준 (VKFTA·AKFTA·RCEP 공통): 세번변경(CTH) 또는 부가가치 40%
    '3304': 'CTH 또는 RVC40',
}
PACK = [(r'플라스틱|합성수지|PET|PP|PE\b', '플라스틱 용기·포장', '3923'),
        (r'유리', '유리 용기', '7010'),
        (r'종이|지제|판지', '종이 상자', '4819'),
        (r'알루미늄|금속|캔', '금속 용기', '7612'),
        (r'부직포|시트', '부직포(시트)', '5603')]


# 2026-11-08 추가: 반제품·베이스·벌크 감지 → 완제품과 같은 호(제33류·3401) 가능성
SEMI_WORD = r'(베이스|벌크|반제품|원액|소지|bulk|\bbase\b|semi-finished|noodle)'
PRODUCT_WORD = [  # (이름에 든 말, 같은 호로 볼 HS4) — 위에서부터 먼저 걸리는 것
    (r'(샴푸|린스|컨디셔너|트리트먼트|헤어|shampoo|conditioner|hair)', '3305'),
    (r'(치약|구강|가글|toothpaste|dental|mouthwash)', '3306'),
    (r'(비누|soap)', '3401'),
    (r'(향수|퍼퓸|오드|perfume|eau de)', '3303'),
    (r'(데오드란트|입욕|면도|바스|deodorant|bath|shaving)', '3307'),
    (r'(크림|로션|에센스|세럼|앰플|토너|스킨|파운데이션|쿠션|선크림|선스크린|립|마스카라|아이섀도|파우더|팩|마스크|cream|lotion|essence|serum|ampoule|toner|foundation|cushion|sunscreen|lip|mascara|powder|mask)', '3304'),
]
def semi_heading(name):
    """반제품·베이스로 보이면 그 제품군의 HS4(제33류·3401), 아니면 None"""
    n = str(name).lower()
    if not re.search(SEMI_WORD, n, re.I): return None
    for pat, h in PRODUCT_WORD:
        if re.search(pat, n, re.I): return h
    return None


def split_ingredients(text):
    """쉼표로 나누되 괄호 안 쉼표는 유지, 함량(%) 제거, '이름(별칭)'은 (이름, 별칭)으로"""
    out, depth, cur = [], 0, ''
    text = str(text)
    for k, ch in enumerate(text):
        if ch in '([': depth += 1
        if ch in ')]': depth = max(0, depth - 1)
        # '1,2-헥산다이올'처럼 숫자 사이 쉼표는 이름의 일부 → 나누지 않음
        if ch == ',' and 0 < k < len(text) - 1 and text[k - 1].isdigit() and text[k + 1].isdigit():
            cur += ch; continue
        if ch in ',;' and depth == 0:
            out.append(cur); cur = ''
        else:
            cur += ch
    out.append(cur)
    res = []
    for s in out:
        s = re.sub(r'\d+(\.\d+)?\s*%', '', s).strip(' .')
        if not s: continue
        m = re.match(r'^(.*?)\s*\((.*)\)\s*$', s)
        main, alt = (m.group(1).strip(), m.group(2).strip()) if m else (s, '')
        res.append((main, alt))
    return res


class OriginChecker:
    def __init__(self, clf, mfds_db, predict_full, smiles_cache=None, pubchem=None):
        """clf: HS4Classifier, mfds_db: 식약처 DB(DataFrame), predict_full: 이름+구조 판정 함수,
        smiles_cache: {영문명 소문자 또는 CAS: SMILES}, pubchem: 온라인 조회 함수(없으면 캐시만)"""
        self.clf, self.pf, self.cache, self.pubchem = clf, predict_full, smiles_cache or {}, pubchem
        norm = lambda s: re.sub(r'[\s\-·,()\[\]./]', '', str(s)).lower()
        self.norm = norm
        self.by = {}
        for _, r in mfds_db.iterrows():
            for key in [r.INGR_KOR_NAME, r.INGR_ENG_NAME.split(',')[0]] + re.split(r'[;,]', r.INGR_SYNONYM):
                if key and key.strip(): self.by.setdefault(norm(key), r)

    def lookup(self, name, alt=''):
        for k in (name, alt):
            r = self.by.get(self.norm(k)) if k else None
            if r is not None:
                cas = ';'.join(c.strip() for c in re.split(r'[,\r\n\t]+', str(r.CAS_NO)) if re.search(r'\d+-\d\d-\d', c))
                return r.INGR_KOR_NAME, r.INGR_ENG_NAME.split(',')[0].strip(), cas, '식약처 DB'
        ko = name if re.search(r'[가-힣]', name) else ''
        en = '' if ko else name
        if alt and not en and not re.search(r'[가-힣]', alt): en = alt
        return ko, en, '', '입력값 그대로'

    def smiles(self, en, cas):
        for k in [c for c in cas.split(';') if c] + [en.lower()]:
            if k in self.cache: return self.cache[k]
        if self.pubchem:
            for k in [c for c in cas.split(';') if c] + ([en] if en else []):
                s = self.pubchem(k)
                if s: self.cache[k.lower()] = s; return s
        return ''

    def check(self, product_hs, ingredients, packaging=''):
        p4 = re.sub(r'\D', '', str(product_hs))[:4]
        rows = []
        for main, alt in split_ingredients(ingredients) if isinstance(ingredients, str) else ingredients:
            ko, en, cas, how = self.lookup(main, alt)
            r = self.pf(self.clf, ko, en, smiles=self.smiles(en, cas) or 'X')
            top3 = [r['HS4_1순위'], r['HS4_2순위'], r['HS4_3순위']]
            sh = semi_heading(f'{main} {alt}')
            if sh:                                            # 반제품·베이스 의심 → 그 제품군 호를 1순위로
                top3 = [sh] + [x for x in top3 if x != sh][:2]; r['HS4_1순위'] = sh; r['판정상태'] = '확인 필요(반제품·베이스 의심)'
            if r['HS4_1순위'] == p4 and sh: cth = '불충족 가능(반제품이 제품과 같은 호) → RVC40 검토'
            elif r['HS4_1순위'] == p4: cth = '불충족(같은 호)'
            elif r['판정상태'].startswith('확인') and p4 in top3: cth = '확인 필요(후보에 제품 호)'
            elif r['판정상태'].startswith('확인'): cth = '충족(원료 HS 확인 권장)'
            else: cth = '충족'
            rows.append({'성분(입력)': main + (f' ({alt})' if alt else ''), '국문명': ko, '영문명': en, 'CAS': cas, '이름 출처': how,
                         'HS4': r['HS4_1순위'], '후보': '/'.join(top3), '판정상태': r['판정상태'], '구조 근거': r.get('구조_근거', ''),
                         '구분': '원재료', 'CTH(제품 ' + p4 + ' 대비)': cth})
        for pat, nm, hs in PACK:
            if re.search(pat, str(packaging)):
                rows.append({'성분(입력)': nm, '국문명': nm, '영문명': '', 'CAS': '', '이름 출처': '포장 설명', 'HS4': hs, '후보': hs,
                             '판정상태': '포장재', '구조 근거': '', '구분': '소매용 포장재', 'CTH(제품 ' + p4 + ' 대비)': 'CTH 판단 제외(RVC에는 포함)'})
        df = pd.DataFrame(rows)
        col = 'CTH(제품 ' + p4 + ' 대비)'
        mat = df[df['구분'] == '원재료'] if len(df) else df
        if not len(mat): verdict = '성분 정보 없음'
        elif (mat[col].str.startswith('불충족')).any(): verdict = 'CTH 불충족 → RVC40(부가가치) 검토 필요'
        elif (mat[col].str.startswith('확인 필요')).any(): verdict = 'CTH 확인 필요 (일부 원료 HS 재확인)'
        else: verdict = 'CTH 충족 (모든 원료가 제품과 다른 호)'
        summary = {'제품 HS4': p4, '원산지 기준': PSR.get(p4, '협정별 품목별 기준 확인 필요(CTH 가정)'), '원재료 수': len(mat),
                   '자동 확정 원료': int(mat['판정상태'].str.startswith('자동').sum()) if len(mat) else 0,
                   '포장재 수': int((df['구분'] == '소매용 포장재').sum()) if len(df) else 0, 'CTH 판정': verdict}
        return summary, df
