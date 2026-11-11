"""원재료 material_name → HS 4자리 판정기 (국문·영문 INCI 모두 입력 가능)

입력에 따라 모델을 고름
  · 국문+영문 둘 다 있음 → 국문+영문 모델 (교차검증 관대 정답률 66.6%, 3순위 안 77.1%)
  · 영문만 있음         → 영문 모델     (66.9%, 77.4%)
  · 국문만 있음         → 국문 모델     (64.5%, 73.2%)
  한쪽만 들어와도 변환표(en_map)에 있는 원료면 반대쪽 이름을 채워서 국문+영문 모델을 씀
출력: 1~3순위 HS4, 1위 점수, 판정상태(자동 확정 / 확인 필요), 적용 규칙
"""
import re
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

AUTO_TH = 0.5   # 1위 점수 0.5 이상 → 자동 확정 (교차검증 정확도 약 87%)
BONUS, CBONUS = 0.15, 0.25
RISK_3304 = r'(베이스|벌크|반제품|크림|로션|에센스|세럼|앰플|토너|파운데이션|선크림|쿠션|립스틱|마스카라|\bbase\b|\bbulk\b|cream|lotion|essence|serum|ampoule|toner|foundation|sunscreen|cushion|lipstick|mascara)'

# ── 국문 규칙 (가이드북 5장 품목분류 가이드, 제28·29류 구조)
RULES = [  # 이름 끝말 → 가점 줄 HS4 (가이드북 5장 품목분류 가이드 기준)
    (r'추출물$|캘러스|즙$|가루$|분말$', {'1302'}),
    (r'(오일|유)$', {'1515', '1516', '1514', '1513', '1509', '3301'}),
    (r'수$', {'3301', '1302'}),
    (r'정유', {'3301'}),
    (r'왁스$|납$', {'1521', '2712', '3404'}),
    (r'(코폴리머|크로스폴리머|폴리머|중합체|셀룰로오스)$', {'3906', '3905', '3907', '3910', '3912', '3913', '3903', '3901', '3902', '3911'}),
    (r'(애씨드|산)$', {'2915', '2916', '2917', '2918', '2922', '3823'}),
    (r'(에이트|레이트|테이트|네이트)$', {'2915', '2916', '2918', '2919', '2923'}),
    (r'(클로라이드|브로마이드)$', {'2923', '2827', '3402'}),
    (r'계면활성제$', {'3402'}),
    (r'(호|안료|색소|레이크)$', {'3204', '3205', '3206', '3203'}),
    (r'(알코올|올)$', {'2905', '2906', '2907', '3823'}),
    (r'(글라이콜|다이올|디올)$', {'2905', '2909'}),
    (r'(실록세인|실록산|메티콘|실리콘)$', {'2931', '3910'}),
    (r'(옥사이드|하이드록사이드)$', {'2816', '2817', '2818', '2821', '2823', '2825', '2846', '2519'}),
    (r'(실리케이트|카보네이트|설페이트)$', {'2839', '2836', '2833', '2842'}),
    (r'(펩타이드)', {'2924', '2937', '3504'}),
    (r'(효소|아제)$', {'3507'}),
]
CHEM = [
    (r'(락테이트|락틱|시트레이트|시트릭|살리실|글루코네이트|글루코닉|타타르|타르타|말릭|말레이트|만델릭|글라이콜릭|하이드록시스테아|갈레이트|갈릭)', {'2918'}),
    (r'(벤조에이트|벤조익|신나메이트|신나믹|페닐아세|아크릴레이트|아크릴릭|메타크릴|올레에이트|올레익|리놀레|운데실레|소르베이트|소빅)', {'2916'}),
    (r'(팔미|스테아|미리스|라우레이트|라우릭|카프릴레이트|카프릴릭|카프레이트|카프릭|아세테이트|아세틱|부티레이트|프로피오네이트|발레레이트|헥사노에이트|옥타노에이트|베헤네이트|이소스테아)', {'2915'}),
    (r'(아디페이트|석시네이트|석시닉|말로네이트|세바케이트|프탈레이트|이타코네이트|옥살레이트|글루타레이트)', {'2917'}),
    (r'(나이트레이트|질산)', {'2834'}), (r'(소듐|포타슘|칼슘|마그네슘|암모늄)\s?(포스페이트|포스피네이트|폴리포스페이트)', {'2835'}),
    (r'^(소듐|포타슘|칼슘|마그네슘|징크|알루미늄|바륨|스트론튬|리튬)\s?(클로라이드|브로마이드|아이오다이드)', {'2827'}),
    (r'^(소듐|포타슘|칼슘|마그네슘|징크|알루미늄|바륨|암모늄)\s?설페이트', {'2833'}), (r'(실리케이트)$', {'2839', '2842'}),
    (r'(카보네이트)$', {'2836'}), (r'^징크\s?옥사이드|산화아연', {'2817'}), (r'티타늄\s?(디|다이)옥사이드|산화티타늄', {'2823'}),
    (r'^(아이언|철)\s?옥사이드|산화철', {'2821'}), (r'^세륨', {'2846'}), (r'^마그네슘\s?옥사이드', {'2519'}),
    (r'(아마이드|아미드)$', {'2924'}), (r'(아민|아민옥사이드)$', {'2921', '2929'}),
    (r'(글라이신|알라닌|아르지닌|라이신|류신|발린|세린|트레오닌|프롤린|글루타민|아스파|메티오닌|시스테인|타이로신|히스티딘|아미노)', {'2922'}),
    (r'(멘톨|보르네올|테르피네올|산탈올|시클로헥산올|사이클로헥산올)', {'2906'}), (r'(페놀|레조시놀|나프톨|카테콜|하이드로퀴논)', {'2907'}),
    (r'(알데하이드|날$|알$)', {'2912'}), (r'(케톤|논$|아이오논)', {'2914'}), (r'(에터|에테르)$', {'2909'}),
    (r'(락톤|쿠마린|피론)', {'2932'}), (r'(사이드|글루코사이드)$', {'2938', '3402'}),
    (r'(카페인|테오브로민|니코틴)', {'2939'}), (r'(비타민|토코페|레티놀|판테놀|바이오틴|나이아신)', {'2936'}),
    (r'(올리고펩타이드|펩타이드-\d)', {'2937', '2924'}),
    # 2026-11-08 추가: 학습 예시가 거의 없는 종류 (HS 제34·35·39·38류 호 용어 기준)
    (r'^(피이지|peg)-?\d+.*(캐스터|오일|에스터|글리세릴|스테아레이트|라우레이트|올리에이트|코코에이트|아이소스테아레이트)', {'3402'}),   # 에톡실화 유지·에스터 = 비이온 계면활성제
    (r'(폴리솔베이트|폴리소르베이트|폴리옥시에칠렌|폴리옥시에틸렌|[가-힣]레스-\d+$)', {'3402'}),
    (r'(라우릴|라우레스|미레스|코코|도데실).*(설페이트|황산)|(코카미도|암포아세테이트|암포다이아세테이트|설포석시네이트)', {'3402'}),
    (r'^(피이지|피피지|peg|ppg)-?\d+$', {'3907'}),                                     # 숫자만 붙은 폴리에틸렌·폴리프로필렌글라이콜
    (r'(단백질|콜라겐|케라틴|엘라스틴|펩톤)', {'3504'}),
    # 2026-11-08 품목표 문구에 이름이 그대로 나오는 원료 (HS 호 용어 기준)
    (r'(파라핀|바셀린|페트롤라툼|마이크로크리스탈린|세레신|오조케라이트)', {'2712'}),
    (r'(잔탄검|잔탄|히알루론|하이알루론|하이알루로|알지네이트|알긴산|키토산|키틴|풀루란|덱스트란|젤란검)', {'3913'}),
    (r'(폴리아크릴|아크릴레이트코폴리머|아크릴레이트크로스폴리머|카보머|폴리메틸메타크릴레이트)', {'3906'}),
    (r'(폴리에스테르|폴리에스터|폴리락틱|폴리카보네이트)', {'3907'}),
    (r'(파파인|브로멜라인|피신|서틸레인|리파아제|리파제|프로테아제|아밀라아제|라이소자임|효소)', {'3507'}),
    (r'(코치닐|카민|카르민|비트루트레드|비트레드|안나토|클로로필|천연색소)', {'3203'}),
    (r'(벤토나이트|풀러토|애터펄자이트)', {'2508'}), (r'(세피올라이트|파이로필라이트|규회석|질석)', {'2530'}),
    (r'^(수소|아르곤|질소|산소|헬륨)$', {'2804'}), (r'(사이클로메티콘|시클로메티콘|사이클로펜타실록산|시클로펜타실록산|사이클로헥사실록산)', {'2931'}),
    (r'(호호바|아보카도 ?오일|달맞이꽃 ?오일|로즈힙|씨 ?오일|커넬 ?오일|씨버터|시어버터|아르간)', {'1515'}),
    (r'(비즈왁스|밀랍|칸데릴라|카나우바|카르나우바)', {'1521'}),
    (r'^(이소|아이소)?(부탄|펜탄|헥산|헵탄|옥탄|데칸|도데칸|헥센|옥텐|도데케인|헥사데케인|아이소도데케인|아이소헥사데케인)$', {'2901'}),
    (r'(아디픽|세바식|옥살릭|아젤라익|트라이멜리테이트)', {'2917'}),
    (r'(페녹시에탄올|에톡시다이글라이콜|다이에틸렌글라이콜|트라이에틸렌글라이콜|다이프로필렌글라이콜|아네톨|아네솔)', {'2909'}),
    (r'(에탄올아민|탄올아민|아미노안식향|아미노벤조|글루타메이트)', {'2922'}),
    (r'사포닌', {'2938'}), (r'글리세라이즈$', {'3824'}),
]

# ── 영문(INCI) 규칙: 같은 내용을 영어 끝말로
RULES_EN = [
    (r'(extract|callus|juice|powder|flour)$', {'1302'}),
    (r'\boil$|oil leaf oil$', {'1515', '1516', '1514', '1513', '1509', '3301'}),
    (r'\bwater$', {'3301', '1302'}),
    (r'essential oil', {'3301'}),
    (r'\bwax$|^cera ', {'1521', '2712', '3404'}),
    (r'(copolymer|crosspolymer|polymer|cellulose)$', {'3906', '3905', '3907', '3910', '3912', '3913', '3903', '3901', '3902', '3911'}),
    (r'\bacid$', {'2915', '2916', '2917', '2918', '2922', '3823'}),
    (r'ate$', {'2915', '2916', '2918', '2919', '2923'}),
    (r'(chloride|bromide)$', {'2923', '2827', '3402'}),
    (r'surfactant', {'3402'}),
    (r'(^(basic|acid|pigment) \w+ \d+$|^(red|blue|green|orange|yellow) \d+$|pigment$|colorant$|\blake\b|\bdye$)', {'3204', '3205', '3206', '3203'}),
    (r'(alcohol|ol)$', {'2905', '2906', '2907', '3823'}),
    (r'(glycol|diol)$', {'2905', '2909'}),
    (r'(siloxane|methicone|silicone)$', {'2931', '3910'}),
    (r'(oxide|hydroxide)$', {'2816', '2817', '2818', '2821', '2823', '2825', '2846', '2519'}),
    (r'(silicate|carbonate|sulfate)$', {'2839', '2836', '2833', '2842'}),
    (r'peptide', {'2924', '2937', '3504'}),
    (r'(enzyme|ase)$', {'3507'}),
]
CHEM_EN = [
    (r'(lactate|lactic|citrate|citric|salicyl|gluconate|gluconic|tartar|malic|malate|mandelic|glycolic|hydroxystear|gallate|gallic|ricinoleate)', {'2918'}),
    (r'(benzoate|benzoic|cinnamate|cinnamic|phenylacet|acrylate|acrylic|methacryl|oleate|oleic|linole|undecylen|sorbate|sorbic)', {'2916'}),
    (r'(palmit|stear|myrist|laurate|lauric|caprylate|caprylic|caprate|capric|caproate|acetate|acetic|butyrate|propionate|valerate|hexanoate|heptanoate|octanoate|pelargonate|behenate|behenic|neopentanoate|hexanoin)', {'2915'}),
    (r'(adipate|succinate|succinic|malonate|sebacate|phthalate|itaconate|oxalate|glutarate|maleic)', {'2917'}),
    (r'nitrate', {'2834'}), (r'(sodium|potassium|calcium|magnesium|ammonium) (phosphate|phosphinate|polyphosphate)', {'2835'}),
    (r'^(sodium|potassium|calcium|magnesium|zinc|aluminum|barium|strontium|lithium) (chloride|bromide|iodide)', {'2827'}),
    (r'^(sodium|potassium|calcium|magnesium|zinc|aluminum|barium|ammonium) sulfate', {'2833'}), (r'silicate$', {'2839', '2842'}),
    (r'carbonate', {'2836'}), (r'^zinc oxide', {'2817'}), (r'titanium dioxide', {'2823'}),
    (r'^iron oxide', {'2821'}), (r'^cerium', {'2846'}), (r'^magnesium oxide', {'2519'}),
    (r'amide$', {'2924'}), (r'(amine|amine oxide)$', {'2921', '2929'}),
    (r'(glycine|alanine|arginine|lysine|leucine|valine|serine|threonine|proline|glutamine|glutamic|aspart|methionine|cysteine|tyrosin|histidine|amino|anthranilate)', {'2922'}),
    (r'(menthol|borneol|terpineol|santalol|cyclohexanol|bisabolol|benzyl alcohol)', {'2906'}), (r'(phenol|resorcinol|naphthol|catechol|hydroquinone)', {'2907'}),
    (r'(aldehyde|al$|anal$)', {'2912'}), (r'(ketone|(?<!id)one$|ionone)', {'2914'}), (r'(pyrrolidone|imidazolidinone|idinone)$', {'2933'}), (r'ether$', {'2909'}),
    (r'(lactone|coumarin|pyrone|flavone)', {'2932'}), (r'(oside|saponin)$', {'2938', '3402'}),
    (r'(caffeine|theobromine|nicotine|cinchon)', {'2939'}), (r'(vitamin|tocopher|retinol|panthenol|biotin|niacin|cobalamin|riboflavin)', {'2936'}),
    (r'(oligopeptide|peptide-\d)', {'2937', '2924'}),
    # 2026-11-08 추가 (국문 규칙과 같은 내용)
    (r'^peg-?\d+ .*(castor|oil|ester|glyceryl|stearate|laurate|oleate|cocoate|isostearate)', {'3402'}),
    (r'(polysorbate|polyoxyethylene|\w+eth-\d+$)', {'3402'}),
    (r'(lauryl|laureth|myreth|coco|dodecyl).*sulfate|(cocamido|amphoacetate|amphodiacetate|sulfosuccinate)', {'3402'}),
    (r'^(peg|ppg)-?\d+$', {'3907'}),
    (r'(protein|collagen|keratin|elastin|peptone)', {'3504'}),
    # 2026-11-08 품목표 문구에 이름이 그대로 나오는 원료 (HS 호 용어 기준)
    (r'(paraffin|petrolatum|vaseline|microcrystalline wax|ceresin|ozokerite)', {'2712'}),
    (r'(xanthan|hyalur|alginate|alginic|chitosan|chitin|pullulan|dextran|gellan)', {'3913'}),
    (r'(polyacryl|acrylates? (copolymer|crosspolymer)|acrylates/|carbomer|polymethyl methacrylate)', {'3906'}),
    (r'(polyester|polylactic|polycarbonate)', {'3907'}),
    (r'(papain|bromelain|ficin|sutilain|lipase|protease|amylase|lysozyme|enzyme)', {'3507'}),
    (r'(cochineal|carmine|beetroot red|beet red|annatto|chlorophyll|natural colorant)', {'3203'}),
    (r'(bentonite|fuller.?s earth|attapulgite)', {'2508'}), (r'(sepiolite|pyrophyllite|wollastonite|vermiculite)', {'2530'}),
    (r'^(hydrogen|argon|nitrogen|oxygen|helium)$', {'2804'}), (r'(cyclomethicone|cyclo\w*siloxane)', {'2931'}),
    (r'(seed oil|kernel oil|nut oil|jojoba|avocado\) oil|avocado oil|evening primrose|rosehip|rose hip|rosa canina fruit oil|seed butter|shea butter|argan)', {'1515'}),
    (r'(beeswax|bees wax|cera alba|candelilla|carnauba)', {'1521'}),
    (r'^(iso)?(butane|pentane|hexane|heptane|octane|decane|dodecane|hexene|octene|hexadecane|isododecane|isohexadecane)$', {'2901'}),
    (r'(adipic|sebacic|oxalic|azelaic|trimellit)', {'2917'}),
    (r'(phenoxyethanol|ethoxydiglycol|diethylene glycol|triethylene glycol|dipropylene glycol|anethole)', {'2909'}),
    (r'(ethanolamine|aminobenzoic|paba|glutamate)', {'2922'}),
    (r'saponins?$', {'2938'}), (r'glycerides$', {'3824'}),
]

# 이름이 이 패턴이면 (규칙 가점 중) 해당 호들을 빼서 서로 다투지 않게 함 — 고분자(39류)는 단량체(29류) 규칙보다 우선 등
PRIORITY = [
    (r'(^poly|^폴리|carbomer|카보머|xanthan|잔탄|hyalur|히알루론|하이알루론|algin|알긴|알지네이트|chitosan|키토산)', {'2915', '2916', '2917', '2918', '2924', '2905', '2909', '2932'}),
    (r'(ethanolamine|탄올아민|에탄올아민|aminobenzoic|아미노안식향|paba|glutamate|글루타메이트)', {'2921', '2916'}),
    (r'(phenoxyethanol|페녹시에탄올|ethoxydiglycol|에톡시다이글라이콜|diethylene glycol|다이에틸렌글라이콜|triethylene glycol|트라이에틸렌글라이콜|dipropylene glycol|다이프로필렌글라이콜)', {'2905', '2906'}),
]

# 구조 규칙을 쓰면 안 되는 이름(계면활성제·폴리에터): PubChem이 대표 분자 하나만 주므로 제29류로 잘못 감
SURF_SKIP = r'(caffeine|theobromine|nicotine|cochineal|carmine|annatto|chlorophyll|phenoxyethanol|ethoxydiglycol|diethylene glycol|triethylene glycol|dipropylene glycol|ethanolamine|^poly|xanthan|hyalur|algin|chitosan|polyacryl|polymethyl|carbomer|polyester|polylactic|cyclomethicone|cyclo\w*siloxane|polysorbate|peg-|ppg-|\w+eth-\d+|polyoxyethylene|sulfate$|cocamido|amphoacetate|amphodiacetate|sulfosuccinate|protein|collagen)'


def _sets(rules, n):
    s = set()
    for p, c in rules:
        if n and re.search(p, n, re.I): s |= c
    return s

def vec(): return TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), sublinear_tf=True)


class HS4Classifier:
    def __init__(self, ko, en, hs4):
        ko = [str(x) for x in ko]; en = [str(x).lower() for x in en]; y = np.asarray(hs4)
        self.vk, self.ve = vec().fit(ko), vec().fit(en)
        Xk, Xe = self.vk.transform(ko), self.ve.transform(en)
        self.m = {'ko': LogisticRegression(C=20, max_iter=3000).fit(Xk, y),
                  'en': LogisticRegression(C=20, max_iter=3000).fit(Xe, y),
                  'both': LogisticRegression(C=20, max_iter=3000).fit(hstack([Xk, Xe]).tocsr(), y)}
        self.ko2en = {}; self.en2ko = {}
        for k, e in zip(ko, en):
            self.ko2en.setdefault(k.replace(' ', ''), e); self.en2ko.setdefault(e, k)

    @classmethod
    def from_csv(cls, path):
        g = pd.read_csv(path, dtype=str).fillna('')
        return cls(g.material_name, g.material_name_en, g.target_hs4)

    def predict(self, ko='', en=''):
        ko, en = (ko or '').strip(), (en or '').strip().lower()
        if ko and not en: en = self.ko2en.get(ko.replace(' ', ''), '')
        if en and not ko: ko = self.en2ko.get(en, '')
        mode = 'both' if ko and en else ('en' if en else 'ko')
        Xk, Xe = self.vk.transform([ko]), self.ve.transform([en])
        X = {'ko': Xk, 'en': Xe, 'both': hstack([Xk, Xe]).tocsr()}[mode]
        lr = self.m[mode]
        rs = M_rule(ko) | _sets(RULES_EN, en); cs = M_chem(ko) | _sets(CHEM_EN, en)
        for p, drop in PRIORITY:
            if re.search(p, ko, re.I) or re.search(p, en, re.I): rs -= drop; cs -= drop
        sc = {h: p + (BONUS if h in rs else 0) + (CBONUS if h in cs else 0) for h, p in zip(lr.classes_, lr.predict_proba(X)[0])}
        for h in rs | cs:
            sc.setdefault(h, (BONUS if h in rs else 0) * 0.5 + (CBONUS if h in cs else 0))
        top = sorted(sc, key=lambda h: -sc[h])[:3]
        if re.search(RISK_3304, f'{ko} {en}', re.I): state = '확인 필요(3304 가능성)'
        elif sc[top[0]] >= AUTO_TH: state = '자동 확정'
        else: state = '확인 필요(점수 낮음)'
        return {'국문명': ko, '영문명': en, '사용 모델': {'ko': '국문', 'en': '영문', 'both': '국문+영문'}[mode],
                'HS4_1순위': top[0], 'HS4_2순위': top[1], 'HS4_3순위': top[2], '1위 점수': round(sc[top[0]], 3),
                '판정상태': state, '적용 규칙': ';'.join(sorted(rs | cs))}


def M_rule(n): return _sets(RULES, n)
def M_chem(n): return _sets(CHEM, n)


def cv_eval(g, folds=5, seed=42):
    """5겹 교차검증: 입력 방식별(국문/영문/국문+영문) 정답률"""
    g = g.reset_index(drop=True)
    cc = next((c for c in g.columns if c.startswith('hs6_후보')), None)   # 시트 열 이름: hs6_후보(조건부)
    cs = g[cc].astype(str) if cc else pd.Series([''] * len(g))
    cand = [({g.target_hs4[i]} | {c.replace('.', '')[:4] for c in cs[i].split(';') if c.strip()}) for i in range(len(g))]
    res = {}
    for mode, label in [('ko', '국문만'), ('en', '영문만'), ('both', '국문+영문')]:
        hit = np.zeros(len(g), bool); t3 = np.zeros(len(g), bool); sc1 = np.zeros(len(g))
        for tr, te in KFold(folds, shuffle=True, random_state=seed).split(g):
            clf = HS4Classifier(g.material_name[tr], g.material_name_en[tr], g.target_hs4[tr])
            clf.ko2en, clf.en2ko = {}, {}   # 시험 원료의 이름 채우기 금지(누출 방지)
            for i in te:
                r = clf.predict(ko=g.material_name[i] if mode != 'en' else '', en=g.material_name_en[i] if mode != 'ko' else '')
                hit[i] = r['HS4_1순위'] in cand[i]; t3[i] = g.target_hs4[i] in (r['HS4_1순위'], r['HS4_2순위'], r['HS4_3순위']); sc1[i] = r['1위 점수']
        res[label] = {'HS4 1순위 정답률(관대)': hit.mean(), 'HS4 3순위 안': t3.mean(),
                      '자동확정 비율': (sc1 >= AUTO_TH).mean(), '자동확정분 정확도': hit[sc1 >= AUTO_TH].mean()}
    return pd.DataFrame(res).T.round(3)


if __name__ == '__main__':
    import sys
    g = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else 'gold_en.csv', dtype=str).fillna('')
    print(cv_eval(g))
    clf = HS4Classifier(g.material_name, g.material_name_en, g.target_hs4)
    demo = [('글리세린', ''), ('', 'Niacinamide'), ('', 'Centella Asiatica Leaf Extract'), ('', 'Caprylic/Capric Triglyceride'),
            ('', 'Sodium Hyaluronate'), ('', 'Cetearyl Olivate'), ('', 'Lavandula Angustifolia (Lavender) Oil'),
            ('징크옥사이드', 'Zinc Oxide'), ('수분크림 베이스', '')]
    print(pd.DataFrame([clf.predict(k, e) for k, e in demo]).to_string(index=False))


# ── 분자 구조 단계 (2026-11-08 추가): PubChem SMILES → 제29류 호 (struct_rules.py)
try:
    from struct_rules import heading29, SKIP_NAME, EXCLUDE_NAME
except ImportError:
    heading29 = None

def predict_full(clf, ko='', en='', smiles=''):
    """이름 모델 + 구조 규칙. 단일 유기화합물이고 SMILES가 있으면 구조로 정한 제29류 호를 1순위로 둠"""
    r = clf.predict(ko, en)
    r['구조_호'], r['구조_근거'] = '', ''
    name_en = (r['영문명'] or '').lower()
    if heading29 and smiles and not re.search(SKIP_NAME, name_en) and not re.search(EXCLUDE_NAME, name_en):
        h, why = heading29(smiles)
        if h:
            top = [h] + [x for x in (r['HS4_1순위'], r['HS4_2순위'], r['HS4_3순위']) if x != h][:2]
            r.update({'HS4_1순위': top[0], 'HS4_2순위': top[1], 'HS4_3순위': top[2], '구조_호': h, '구조_근거': why})
            if not r['판정상태'].startswith('확인 필요(3304'): r['판정상태'] = '자동 확정(구조)'
    return r
