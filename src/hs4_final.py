"""원재료 HS4 판정 — 최종 모델 사용용 (원료명 하나 / 성분표 하나)
파이프라인: 성분명 정리·식약처 표준명 찾기 → 학습 데이터 이름 일치 → 이름 모델(국문·영문 TF-IDF+로지스틱+HS 규칙) → 분자 구조 규칙 → 반제품·베이스 감지
사용 (Colab, 00_setup 실행 뒤 /content/rawmodel 에서):
    from hs4_final import HS4Final
    m = HS4Final('/content/rawmodel')
    m.one('나이아신아마이드')                     # 원료명 하나 (국문 또는 영문 INCI)
    m.bom('정제수, 글리세린, 1,2-헥산다이올, ...')  # 성분표 하나 (쉼표로 구분된 전성분)"""
import re, pandas as pd
from origin_check import build_hs_fn
from name_norm import NameMatcher
from bom_check import split_ingredients


class HS4Final:
    def __init__(self, base='.'):
        self.f = build_hs_fn(base)
        self.nm = NameMatcher(pd.read_csv(f'{base}/mfds_db.csv', dtype=str).fillna(''))

    def one(self, name):
        name = str(name).strip()
        ko, en = (name, '') if re.search('[가-힣]', name) else ('', name)
        m = self.nm.match(ko or name, en)
        r = self.f(name)
        c = r['후보'] + ['', '', '']
        return {'입력': name, '찾은 이름(국문)': m.get('국문명', ''), '찾은 이름(영문 INCI)': m.get('영문명', ''), '이름 찾기': m.get('구분', ''),
                'HS4 1순위': c[0], 'HS4 2순위': c[1], 'HS4 3순위': c[2], '판정상태': r['상태']}

    def bom(self, text):
        names = [n for n, _ in split_ingredients(re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', str(text))) if n.strip()]
        return pd.DataFrame([self.one(n) for n in names])
