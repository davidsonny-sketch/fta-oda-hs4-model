"""실제 성분명 → 식약처 원료명 찾기 (정리 규칙 + 확장 색인 + 비슷한 이름)
결과 구분: 식약처 일치 / 식약처 유사(확인 권장) / 포장재 / 일반명(정답 정할 수 없음) / 못 찾음
"""
import re, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PACK = r'(용기|케이스|면봉|부직포|종이|튜브|펌프|뚜껑|캡|병$|파우치|막대|스틱$|라벨|container|swab|bottle|tube|pump|cap$|pouch|paper|stick$|sheet$|cotton$|^면$)'
GENERIC = r'^(안료|색소|녹색색소|착색제|고분자|왁스|오일|계면활성제|유기\s*계면활성제|지방산|단백질|향|향료|방향료|식물\s*추출물|각종\s*식물\s*추출물|식물성\s*유지|식물성\s*오일|부형제|유기용제|천연고무라텍스|염모제.*|pigment|polymer|wax|oil|surfactant|organic surfactant|fatty acids|protein|fragrance|colorant|dye|vegetable oil|.*botanical extracts|plant extracts.*)$'
DASH = dict.fromkeys(map(ord, '\u2010\u2011\u2012\u2013\u2014\u2015\u2212'), '-')

def clean(s):
    """표기 흔들림 정리: 대시 통일, 함량(%)·'등/etc'·뒤에 붙은 분류어 제거, 괄호 속 군더더기('…ExtractWater') 제거, 띄어쓰기 정리"""
    s = str(s).translate(DASH).replace('\u3000', ' ')
    s = re.sub(r'\d+(\.\d+)?\s*%', '', s)
    s = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', s)                       # 1, 2-헥산다이올 → 1,2-헥산다이올
    s = re.sub(r'^.*?:\s*', '', s) if re.match(r'^[^:]{1,12}:\s*\S', s) else s   # '액상클리너: 알코올' → '알코올'
    s = re.sub(r'\s*(등|etc\.?)(\s+.*)?$', '', s, flags=re.I)           # '레시틴 등', '다이프로필렌글라이콜 등 유기용제'
    s = re.sub(r'\(([^()]*?)(extract)?water\)$', '', s, flags=re.I)   # 'Centella Asiatica Extract(Centella … ExtractWater)'
    s = re.sub(r'\(\s*\)', '', s)
    s = re.sub(r'\s*(일수화물|이수화물|수화물|\bmono ?hydrate|\bdi ?hydrate|\bhydrate)$', '', s, flags=re.I)   # 2026-11-08: 결정수 표기 제거
    return re.sub(r'\s+', ' ', s).strip(' .,')

norm = lambda s: re.sub(r'[\s\-·,()\[\]./\'’]', '', str(s)).lower()

# 표기법 차이만 바꾸는 안전한 변환 (물질이 바뀌지 않음)
KO_VARIANTS = [('디올', '다이올'), ('트리', '트라이'), ('디', '다이'), ('이소', '아이소'), ('글리콜', '글라이콜'), ('에칠', '에틸'), ('에틸', '에칠'),
               ('메칠', '메틸'), ('실록산', '실록세인'), ('카르', '카'), ('셀룰로스', '셀룰로오스'), ('애시드', '애씨드'), ('카모마일', '캐모마일'),
               ('추출정제수', '수'), ('스피룰리나', '스피루리나'), ('유$', '오일')]   # '유$→오일': 한국어 '○○유'는 '○○오일'과 같은 말
# 고친 점(2026-11-08): '추출정제수→추출물', '추출물 떼기'는 물질이 바뀔 수 있어 뺌 (추출물·수·원물은 서로 다른 원료)
EN_VARIANTS = [('aluminium', 'aluminum'), ('extractwater', 'water'), ('glycerine', 'glycerin'), ('sulphate', 'sulfate')]

# 업계 관용명 → 식약처 표준명 (연습 범위·일반 지식으로 만듦. 시험 B 성분명은 넣지 않음)
COMMON = {'유동파라핀': '미네랄오일', 'liquidparaffin': '미네랄오일', '폴리다이메틸실록산': '다이메티콘', 'polydimethylsiloxane': '다이메티콘',
          '실리콘오일': '다이메티콘', '에틸알코올': '에탄올', 'ethylalcohol': '에탄올', '변성에탄올': '변성알코올', 'denaturedethanol': '변성알코올',
          '소금': '소듐클로라이드', '비타민c': '아스코빅애씨드', 'vitaminc': '아스코빅애씨드', '옥틸메톡시신나메이트': '에칠헥실메톡시신나메이트',
          'octylmethoxycinnamate': '에칠헥실메톡시신나메이트', '부틸하이드록시아니솔': '비에이치에이', 'butylatedhydroxyanisole': '비에이치에이',
          '카르복시메틸셀룰로오스나트륨': '셀룰로오스검', 'carboxylmethylcellulosesodium': '셀룰로오스검', '팜오일': '오일팜오일', 'palmoil': '오일팜오일',
          '아몬드오일': '스위트아몬드오일', 'almondoil': '스위트아몬드오일', '인지질': '포스포리피드', '증류수': '정제수', 'distilledwater': '정제수', '물': '정제수'}

# 2026-11-10 추가: 화학명·무역 서류명·약칭 → 식약처 표준명 (무역 BOM은 화장품 성분명 대신 화학명을 쓰는 경우가 많음)
# 같은 물질의 다른 이름만 넣음(물질이 바뀌는 변환 없음). 시험 B 성분명은 넣지 않음
CHEM_SYNONYMS = {
    # 알코올·폴리올
    '에탄올': ['ethanol', 'ethylalcohol', '에틸알콜', '주정', 'alcohol(ethanol)'], '변성알코올': ['alcoholdenat', 'denaturedalcohol', '변성에탄올'],
    '글리세린': ['glycerol', 'glycerine', '글리세롤', '글리세린(농글리세린)', 'concentratedglycerin', '농글리세린'],
    '부틸렌글라이콜': ['1,3-butanediol', '1,3-butyleneglycol', '13bg', '1,3-bg', '1,3-부탄다이올', '1,3-부틸렌글리콜'],
    '프로필렌글라이콜': ['1,2-propanediol', 'propyleneglycol(pg)', 'monopropyleneglycol', 'mpg', '1,2-프로판다이올'],
    '1,2-헥산다이올': ['1,2-hexanediol', 'hexane-1,2-diol', '헥산-1,2-다이올'], '페녹시에탄올': ['2-phenoxyethanol', '페녹시에탄올(2-페녹시에탄올)'],
    '세테아릴알코올': ['cetostearylalcohol', '세토스테아릴알코올'], '소르비톨': ['sorbitol(d-sorbitol)', 'd-sorbitol', 'd-소르비톨'],
    # 왁스·유지·탄화수소
    '비즈왁스': ['beeswax', 'beewax', 'beeswax(밀랍)', '밀랍', '꿀벌왁스', 'whitebeeswax', 'yellowbeeswax', 'cera alba', '백랍', '황랍'],
    '카나우바왁스': ['carnaubawax', '카르나우바왁스', '카나바왁스'], '칸데릴라왁스': ['candelillawax', '칸델릴라왁스'],
    '미네랄오일': ['whitemineraloil', 'paraffinumliquidum', 'liquidpetrolatum', '백색광유', '유동파라핀'],
    '페트롤라툼': ['vaseline', 'petroleumjelly', 'whitepetrolatum', '바셀린', '백색바셀린', '와셀린'],
    '파라핀': ['paraffinwax', '파라핀왁스'], '하이드로제네이티드돌콩오일': ['hydrogenatedsoybeanoil', '대두경화유', '정제대두경화유', '경화대두유'],
    '시어버터': ['sheabutter', '시아버터'], '올리브오일': ['oliveoil'], '호호바씨오일': ['jojobaoil', '호호바오일'],
    '피마자씨오일': ['castoroil', '피마자유', '아주까리기름'], '코코넛오일': ['coconutoil', '야자유'],
    # 계면활성제·유화제
    '소듐라우레스설페이트': ['sles', 'sodiumlaurylethersulfate', 'sodiumlaurylethersulphate', '라우릴에테르황산나트륨'],
    '소듐라우릴설페이트': ['sls', 'sodiumdodecylsulfate', 'sds', '라우릴황산나트륨', '도데실황산나트륨'],
    '피이지-40하이드로제네이티드캐스터오일': ['peg-40hco', 'hco-40', 'peg-40hydrogenatedcastoroil'],
    '글리세릴스테아레이트': ['glycerylmonostearate', 'glycerolmonostearate', 'gms', '모노스테아린산글리세린'],
    '폴리솔베이트80': ['tween80', '트윈80'], '폴리솔베이트20': ['tween20', '트윈20'],
    # 산·염기·염
    '소듐하이드록사이드': ['causticsoda', 'naoh', '수산화나트륨', '가성소다'], '포타슘하이드록사이드': ['causticpotash', 'koh', '수산화칼륨', '가성칼리'],
    '트라이에탄올아민': ['tea', 'triethanolamine(tea)', '트리에탄올아민'], '시트릭애씨드': ['citricacidanhydrous', 'citricacidmonohydrate', '구연산', '무수구연산'],
    '소듐시트레이트': ['trisodiumcitrate', '구연산나트륨'], '락틱애씨드': ['lacticacid', '젖산'], '살리실릭애씨드': ['salicylicacid', '살리실산'],
    '스테아릭애씨드': ['stearicacid', '스테아린산'], '소듐클로라이드': ['nacl', '염화나트륨', '정제염'], '소듐벤조에이트': ['안식향산나트륨'],
    '포타슘솔베이트': ['소르빈산칼륨'], '소듐바이카보네이트': ['bakingsoda', 'sodiumhydrogencarbonate', '탄산수소나트륨', '중탄산나트륨'],
    '소듐카보네이트': ['sodaash', '탄산나트륨'], '칼슘카보네이트': ['탄산칼슘'], '다이소듐이디티에이': ['edta-2na', 'edta2na', 'disodiumedetate', '에데트산이나트륨'],
    # 비타민·활성 성분
    '토코페릴아세테이트': ['vitamineacetate', 'dl-alpha-tocopherylacetate', 'dl-α-tocopherylacetate', '비타민e아세테이트', '초산토코페롤'],
    '토코페롤': ['vitamine', '비타민e'], '나이아신아마이드': ['nicotinamide', 'niacinamide(vitaminb3)', '니코틴산아미드', '니코틴아마이드'],
    '판테놀': ['d-panthenol', 'provitaminb5', 'dexpanthenol(d-panthenol)'], '아스코빅애씨드': ['l-ascorbicacid', 'l-아스코르빈산', '아스코르빈산'],
    '레티놀': ['vitamina', '비타민a'], '우레아': ['urea', '요소'], '멘톨': ['l-menthol', '박하뇌'], '알란토인': ['allantoin'],
    '소듐하이알루로네이트': ['hyaluronicacidsodiumsalt', 'sodiumhyaluronate(ha)', '히알루론산나트륨', '하이알루론산나트륨'],
    # 무기 분체
    '티타늄디옥사이드': ['tio2', 'titanium(iv)oxide', 'titaniumoxide', '이산화티타늄', '산화티타늄'], '징크옥사이드': ['zno', 'zincoxide', '산화아연'],
    '탤크': ['talcum', 'talcpowder', '활석'], '마이카': ['sericite', '세리사이트', '운모'], '카올린': ['kaolinclay', '고령토'],
    '실리카': ['silicondioxide', '이산화규소', '무수규산'], '카보머': ['carbopol', '카보폴', 'carboxyvinylpolymer', '카복시비닐폴리머'],
    '다이메티콘': ['dimethylpolysiloxane', '다이메틸폴리실록산', '디메틸폴리실록산'], '정제수': ['deionizedwater', 'diwater', 'd.i.water', 'purifiedwater', 'aqua', '탈이온수'],
}
for _std, _alts in CHEM_SYNONYMS.items():
    for _a in _alts: COMMON.setdefault(norm(_a), _std)

def variants(x):
    """표기 변형 후보 (원래 이름 포함). 한 번에 하나씩 + 전부 적용한 형태"""
    out = [x]
    lo = x.lower()
    for a, b in KO_VARIANTS + EN_VARIANTS:
        src = x if re.search('[가-힣]', a) else lo
        if a.endswith('$'):
            if re.search(a, src): out.append(re.sub(a, b, src))
        elif a in src: out.append(src.replace(a, b))
    allv = x
    for a, b in KO_VARIANTS[:9]: allv = allv.replace(a, b)
    out.append(allv)
    return list(dict.fromkeys(o for o in out if o))

# 고친 점(2026-11-08): 화학 단어 목록을 넓히고(페닐·벤질·하이드록시 등), 길이가 크게 다르면(단어가 빠지거나 더해짐) 버림
CHEM_TOKENS = (r'(메틸|에틸|에칠|프로필|부틸|펜틸|헥실|옥틸|데실|라우릴|세틸|스테아릴|모노|다이|트라이|테트라|페닐|벤질|벤조|사이클로|하이드록시|아미노|'
               r'클로로|플루오로|메톡시|에톡시|아세틸|아세테이트|글리세릴|폴리|크로스|소듐|포타슘|칼슘|마그네슘|징크|하이드로제네이티드|'
               r'methyl|ethyl|propyl|butyl|pentyl|hexyl|octyl|decyl|lauryl|cetyl|stearyl|mono|di|tri|tetra|phenyl|benzyl|benzo|cyclo|hydroxy|amino|'
               r'chloro|fluoro|methoxy|ethoxy|acetyl|acetate|glyceryl|poly|cross|sodium|potassium|calcium|magnesium|zinc|hydrogenated)')
def chem_safe(a, b):
    """비슷한 이름 후보가 화학적으로 같은 물질일 수 있는지: 숫자·화학 단어가 같고 길이 차이가 작아야 함"""
    if sorted(re.findall(r'\d+', a)) != sorted(re.findall(r'\d+', b)): return False
    na, nb = norm(a), norm(b)
    if min(len(na), len(nb)) / max(len(na), len(nb), 1) < 0.8: return False
    form = lambda t: re.findall(r'(추출정제수|추출물|오일|가루|왁스|버터|수$|extractwater|extract|oil|powder|wax|butter|water$)', t.lower().replace(' ', ''))
    if form(a) != form(b): return False                              # 추출물·수·오일·가루 같은 형태가 다르면 다른 원료
    return sorted(re.findall(CHEM_TOKENS, a.lower())) == sorted(re.findall(CHEM_TOKENS, b.lower()))

class NameMatcher:
    def __init__(self, db, sim_threshold=0.81):
        self.db, self.th = db.reset_index(drop=True), sim_threshold
        keys = {}
        def add(k, i, how):
            k = k.strip()
            if len(norm(k)) >= 2: keys.setdefault(norm(k), (i, how))
        for i, r in self.db.iterrows():
            add(r.INGR_KOR_NAME, i, '국문 표준명')
            en = r.INGR_ENG_NAME if re.search(r'\d,\d', r.INGR_ENG_NAME) else r.INGR_ENG_NAME.split(',')[0]
            add(en, i, '영문 표준명')
            m = re.match(r'^(.*?)\s*\(([^)]+)\)\s*(.*)$', en)               # 'Copernicia Cerifera (Carnauba) Wax' → 'Carnauba Wax'
            if m: add(f'{m.group(2)} {m.group(3)}', i, '영문 속칭'); add(f'{m.group(1)} {m.group(3)}', i, '영문 학명')
            for s in re.split(r'[;,]', r.INGR_SYNONYM): add(s, i, '이명')
        self.keys = keys
        self.knames = list(keys); self.kidx = [keys[k][0] for k in self.knames]
        self.vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3)).fit(self.knames)
        self.M = self.vec.transform(self.knames)

    def _row(self, i):
        r = self.db.loc[i]
        en = r.INGR_ENG_NAME if re.search(r'\d,\d', r.INGR_ENG_NAME) else r.INGR_ENG_NAME.split(',')[0].strip()
        cas = ';'.join(c.strip() for c in re.split(r'[,\r\n\t]+', str(r.CAS_NO)) if re.search(r'\d+-\d\d-\d', c))
        return r.INGR_KOR_NAME, en, cas

    def match(self, kr='', en=''):
        raw = [x for x in (kr, en) if x]
        c = [clean(x) for x in raw]
        if not any(c): return dict(구분='일반명', 국문명='', 영문명='', CAS='', 근거='이름 없음', 유사도=0)
        if any(re.search(PACK, x, re.I) for x in c): return dict(구분='포장재', 국문명=c[0], 영문명='', CAS='', 근거='포장·가공 표현', 유사도=0)
        for x in c:
            k = norm(x)
            if k in self.keys:
                i, how = self.keys[k]; ko, e, cas = self._row(i)
                return dict(구분='식약처 일치', 국문명=ko, 영문명=e, CAS=cas, 근거=f'{how} 일치({x})', 유사도=1.0)
        parts = [p.strip() for x in c for m_ in [re.match(r'^(.*?)\s*\(([^()]+)\)\s*$', x)] if m_ for p in m_.groups()]
        for x in c + [p for p in parts if p]:   # 2026-11-10: '이름 (다른 이름)'이면 괄호 앞·안 이름으로도 찾음
            if norm(x) in self.keys and x not in c:
                i, how = self.keys[norm(x)]; ko, e, cas = self._row(i)
                return dict(구분='식약처 일치', 국문명=ko, 영문명=e, CAS=cas, 근거=f'괄호 속 이름 {how} 일치({x})', 유사도=1.0)
            if norm(x) in COMMON and norm(COMMON[norm(x)]) in self.keys:
                i, _ = self.keys[norm(COMMON[norm(x)])]; ko, e, cas = self._row(i)
                return dict(구분='식약처 일치', 국문명=ko, 영문명=e, CAS=cas, 근거=f'관용명 사전({x}→{ko})', 유사도=1.0)
        if all(re.search(GENERIC, x, re.I) for x in c): return dict(구분='일반명', 국문명=c[0], 영문명='', CAS='', 근거='너무 일반적인 이름', 유사도=0)
        for x in c:
            for v in variants(x)[1:]:
                if norm(v) in self.keys:
                    i, how = self.keys[norm(v)]; ko, e, cas = self._row(i)
                    return dict(구분='식약처 일치', 국문명=ko, 영문명=e, CAS=cas, 근거=f'표기 변형 일치({x}→{v})', 유사도=1.0)
        # 고친 점(2026-11-08): '○○ 오일 → ○○' 변환은 뺌. 영문명만으로는 ○○이 원래 기름인지(스쿠알란) 수지인지(유향=Olibanum) 가릴 수 없음
        best = (0, None, '')
        for x in c:
            s = (self.vec.transform([norm(x)]) @ self.M.T).toarray()[0]
            for j in np.argsort(-s)[:10]:                      # 상위 후보 중 화학적으로 안전한 첫 후보
                if s[j] <= best[0]: break
                if chem_safe(x, self.knames[j]) and chem_safe(x, self._row(self.kidx[j])[0] if re.search('[가-힣]', x) else self._row(self.kidx[j])[1]):
                    best = (float(s[j]), self.kidx[j], x); break
        if best[1] is not None and best[0] >= self.th:
            ko, e, cas = self._row(best[1])
            return dict(구분='식약처 유사', 국문명=ko, 영문명=e, CAS=cas, 근거=f'비슷한 이름({best[2]}→{ko}, {best[0]:.2f})', 유사도=round(best[0], 3))
        return dict(구분='못 찾음', 국문명=c[0] if re.search('[가-힣]', c[0]) else '', 영문명=next((x for x in c if not re.search('[가-힣]', x)), ''),
                    CAS='', 근거=f'가장 비슷한 후보 유사도 {best[0]:.2f}', 유사도=round(best[0], 3))
