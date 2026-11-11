"""분자 구조(SMILES) → 관세율표 제29류 호(4단위) 추정
원칙: 제29류 주 제3호 '둘 이상의 호에 해당하면 그중 마지막 호' → 해당하는 호 후보 중 가장 큰 번호
- 에스테르·염은 산(酸) 쪽 구조를 따라감 (카복실산 에스테르는 2915~2918, 인산 에스테르 2919)
- 비타민(2936)·호르몬(2937)·배당체(2938)·알칼로이드(2939)는 구조만으로 판별하지 않음(이름 규칙·학습 모델에 맡김)
- 무기물(탄소 없음), 혼합물, 중합체, 추출물·오일 등은 판정하지 않음(None)
"""
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

P = lambda s: Chem.MolFromSmarts(s)
PAT = {
    'cooh_like': P('[CX3](=O)[OX2H1,OX1-,OX2]'),            # 카복실산·염·에스테르의 C(=O)O
    'ester': P('[CX3](=O)[OX2][#6]'),
    'amide': P('[CX3](=O)[NX3]'),
    'imide': P('[CX3](=O)[NX3][CX3](=O)'),
    'imine': P('[CX3]=[NX2;!$(N=N)]'),
    'nitrile': P('C#N'),
    'azo': P('[#6]N=N[#6]'),
    'nitro': P('[N+](=O)[O-]'),
    'sulfonamide': P('S(=O)(=O)[NX3]'),
    'sulfonic': P('[#6]S(=O)(=O)[OX2H1,OX1-]'),
    'thio': P('[#6][SX2,SX3,SX4;!$(S(=O)(=O)[OX2H1,OX1-]);!$(S(=O)(=O)N)]'),
    'phosphate_ester': P('P(=O)([OX2,OX1-])([OX2,OX1-])[OX2][#6]'),
    'p_c': P('[#6][#15]'),
    'si_c': P('[#6][Si]'),
    'quat': P('[NX4+;H0;!$([N+][O-])]'),
    'amine': P('[NX3;!$(N-C=[O,S,N]);!$(N-S(=O)=O);!$(N-[a]);!$(N=*);!$(N#*);!$([N+])]'),
    'arylamine': P('[NX3;H2,H1,H0;!$(N-C=O);!$(N-S(=O)=O)]-a'),
    'aldehyde': P('[CX3H1](=O)[#6]'),
    'ketone': P('[#6][CX3](=O)[#6]'),
    'phenol': P('[OX2H1,OX1-]c'),
    'alcohol': P('[OX2H1][CX4]'),
    'ether': P('[OD2]([#6;!$(C=O)])[#6;!$(C=O)]'),
    'epoxide': P('C1OC1'),
    'acetal': P('[CX4]([OX2][#6])[OX2][#6]'),
    'halogen': P('[#6][F,Cl,Br,I]'),
    'n_oxide': P('[NX4+][OX1-]'),
}

def _largest_organic(m):
    frags = rdmolops.GetMolFrags(m, asMols=True)
    org = [f for f in frags if any(a.GetSymbol() == 'C' for a in f.GetAtoms())]
    return max(org, key=lambda f: f.GetNumHeavyAtoms()) if org else None

def _ring_hetero(m):
    ri = m.GetRingInfo(); el = set()
    for ring in ri.AtomRings():
        if any(m.GetAtomWithIdx(i).GetSymbol() == 'Si' for i in ring): continue   # 사이클로실록세인은 유기규소(2931)
        for i in ring:
            s = m.GetAtomWithIdx(i).GetSymbol()
            if s != 'C': el.add(s)
    return el

def _acid_side_extra_O(m):
    """에스테르 결합(O–알코올 탄소)을 끊고 산 쪽 조각만 봄: (산 쪽에 추가 산소 작용기 여부, 카복실 개수, 불포화·고리 여부)"""
    CX = P('[CX3](=O)[OX2,OX1-]')
    em = Chem.RWMol(m)
    for t in m.GetSubstructMatches(PAT['ester']):
        o, r = t[2], t[3]
        if em.GetBondBetweenAtoms(o, r): em.RemoveBond(o, r)
    mm = em.GetMol()
    frags = Chem.GetMolFrags(mm, asMols=False, sanitizeFrags=False)
    hits = m.GetSubstructMatches(CX)
    acid = set()
    for t in hits:
        for fr in frags:
            if t[0] in fr: acid |= set(fr)
    if not acid: return False, 0, False
    n = 0
    for fr in frags:
        fs = set(fr); k = len({t[0] for t in hits if t[0] in fs})
        n = max(n, k)                      # 한 산(酸) 조각 안의 카복실 개수 (폴리올 에스테르는 1로 셈)
    carboxyl_o = {nb.GetIdx() for t in hits for nb in m.GetAtomWithIdx(t[0]).GetNeighbors() if nb.GetSymbol() == 'O'}
    extra = any(m.GetAtomWithIdx(i).GetSymbol() == 'O' and i not in carboxyl_o for i in acid)
    uc = any(m.GetAtomWithIdx(i).IsInRing() or m.GetAtomWithIdx(i).GetIsAromatic() for i in acid) or any(
        b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE) and b.GetBeginAtomIdx() in acid and b.GetEndAtomIdx() in acid
        and b.GetBeginAtom().GetSymbol() == 'C' and b.GetEndAtom().GetSymbol() == 'C' for b in m.GetBonds())
    return extra, n, uc

def heading29(smiles):
    """SMILES → (호, 근거) 또는 (None, 사유)"""
    if not smiles: return None, '구조 없음'
    m = Chem.MolFromSmiles(smiles)
    if m is None: return None, 'SMILES 해석 실패'
    m = _largest_organic(m)
    if m is None: return None, '무기물(탄소 없음)'
    if m.GetNumHeavyAtoms() > 120: return None, '고분자 크기'
    has = {k: m.HasSubstructMatch(p) for k, p in PAT.items()}
    H = {}
    def add(h, why): H.setdefault(h, why)
    syms = {a.GetSymbol() for a in m.GetAtoms()}
    if syms <= {'C', 'H'}:
        add(2902 if m.GetRingInfo().NumRings() else 2901, '탄화수소')
    if has['halogen'] and syms <= {'C', 'H', 'F', 'Cl', 'Br', 'I'}: add(2903, '할로겐화 탄화수소')
    if (has['sulfonic'] or has['nitro']) and syms <= {'C', 'H', 'S', 'O', 'N', 'Na', 'K'} and not (has['alcohol'] or has['phenol'] or has['cooh_like']): add(2904, '설폰화·나이트로화 탄화수소')
    if has['alcohol']: add(2906 if m.GetRingInfo().NumRings() else 2905, '알코올(고리 있으면 2906)')
    if has['phenol']: add(2907, '페놀')
    if has['ether']: add(2909, '에터')
    if has['epoxide']: add(2910, '에폭사이드')
    if has['acetal'] and not m.GetRingInfo().NumRings(): add(2911, '아세탈')
    if has['aldehyde']: add(2912, '알데하이드')
    if has['ketone']: add(2914, '케톤')
    if has['cooh_like']:
        extra, n, uc = _acid_side_extra_O(m)
        if extra: add(2918, '산소 작용기가 더 있는 카복실산(에스테르·염 포함)')
        elif n >= 2: add(2917, '폴리카복실산(에스테르·염 포함)')
        elif uc: add(2916, '불포화·고리 모노카복실산(에스테르·염 포함)')
        else: add(2915, '포화 지방족 모노카복실산(에스테르·염 포함)')
    if has['phosphate_ester']: add(2919, '인산 에스테르')
    if has['amine'] or has['arylamine']:
        o_func = has['alcohol'] or has['phenol'] or has['ether'] or has['aldehyde'] or has['ketone'] or has['cooh_like']
        add(2922 if o_func else 2921, '아민(산소 작용기 있으면 2922)')
    if has['quat']: add(2923, '제4급 암모늄')
    if has['amide']: add(2924, '카복시아마이드')
    if has['imide'] or has['imine']: add(2925, '이미드·이민')
    if has['nitrile']: add(2926, '나이트릴')
    if has['azo']: add(2927, '아조')
    if has['n_oxide']: add(2929, '그 밖의 질소 작용기')
    if has['thio'] or (has['sulfonic'] and 2904 not in H): add(2930, '유기황')
    if has['si_c'] or (has['p_c'] and not has['phosphate_ester']): add(2931, '유기-무기 화합물(Si·P)')
    rh = _ring_hetero(m)
    if rh:
        if rh == {'O'}: add(2932, '산소만 있는 헤테로고리')
        elif rh == {'N'}: add(2933, '질소만 있는 헤테로고리')
        elif rh & {'N', 'O', 'S'}: add(2934, '그 밖의 헤테로고리')
    if has['sulfonamide']: add(2935, '설폰아마이드')
    if not H: return None, '해당 작용기 없음'
    h = max(H)
    return str(h), H[h] + ' (후보 ' + ','.join(map(str, sorted(H))) + ')'

SKIP_NAME = r'(extract|oil|wax|butter|water|juice|powder|flour|polymer|copolymer|crosspolymer|gum|starch|cellulose|resin|ferment|enzyme|protein|peptide|polypeptide|glycerides|preparation|mixture|surfactant|pigment|colorant|dye|lake|fragrance|preservative|vitamin|silk|lanolin|petrolatum|paraffin|ceresin|mica|talc|clay|kaolin|bentonite|silicone|dimethicone|siloxane|polyquaternium|carbomer|pvp|alcohol denat|mineral|diamond|glass|quartz|pumice|carbon|sucrose|glucose|maltose|gel|callus|leaf|root|flower|seed|fruit|bark|peel|stem|herb|agent|^water$)'

# 제29류 주 제2호 등으로 29류에서 빠지거나, 구조만으로 정할 수 없는 군(비타민·호르몬·배당체·알칼로이드)은 구조 규칙 미적용
EXCLUDE_NAME = (r'(\bci \d+|^(basic|acid|direct|disperse|solvent|pigment) \w+ \d+|^(red|blue|green|yellow|orange|violet|black) (no\. )?\d+|fcf|allura|carmine|tartrazine|'
                r'ase$|subtilisin|bromelain|papain|sutilains|^alcohol$|ethanol|^urea$|chitosan|glycogen|dextr|inulin|'
                r'vitamin|tocopher|retinol|retinyl|panthenol|biotin|niacin|cobalamin|riboflavin|ascorb|thiamin|pyridox|folic|'
                r'oside$|saponin|glucoside|'
                r'chloride$|bromide$|methosulfate$|sulfate$)')
