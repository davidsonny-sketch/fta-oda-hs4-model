"""관세율표(HS) 호 용어 사전 — 품목표의 호 문구·류 주(註)에 이름이 직접 나오는 물질을 그 호로 연결
- 데이터의 오답을 보고 만든 규칙이 아니라, 호 문구를 류별로 차례대로 옮긴 것(제11·13·15·17·25·28·29·33·34·35·39류)
- 학습 예시가 거의 없는 드문 호(무기화합물·광물·당류 등)를 이름 모델 대신 맡음
- lookup(국문명, 영문명) → (HS4, 근거) 또는 None.  먼저 맞는 항목이 이김(특수한 것을 앞에 둠)
"""
import re

# ── 제28류 무기화합물: [양이온] + [음이온] 꼴의 이름만 대상(유기 부분이 섞인 이름은 제외)
CAT_EN = r'(?:sodium|potassium|calcium|magnesium|zinc|aluminum|aluminium|ammonium|barium|lithium|copper|cupric|iron|ferric|ferrous|manganese|strontium|tin|stannous|stannic|bismuth|cobalt|nickel|chromium|zirconium|titanium|lead)'
CAT_KO = r'(?:소듐|나트륨|포타슘|칼륨|칼슘|마그네슘|징크|아연|알루미늄|암모늄|바륨|리튬|구리|쿠퍼|철|아이언|페릭|페러스|망가니즈|망간|스트론튬|틴|주석|비스무트|코발트|니켈|크로뮴|크롬|지르코늄|티타늄|납)'
PRE_EN = r'(?:di|tri|tetra|penta|hexa|mono)?'
PRE_KO = r'(?:다이|디|트라이|트리|테트라|펜타|헥사|모노)?'
ANION = [  # (영문, 국문, 호, 호 문구)
    (r'(?:hypochlorite|chlorite|hypobromite)', r'(?:하이포클로라이트|클로라이트|차아염소산염)', '2828', '차아염소산염·아염소산염'),
    (r'(?:chlorate|perchlorate|bromate|iodate|periodate)', r'(?:클로레이트|퍼클로레이트|브로메이트|아이오데이트)', '2829', '염소산염·브로민산염·아이오딘산염'),
    (r'(?:fluoride|fluorosilicate|fluoroaluminate)', r'(?:플루오라이드|플루오로실리케이트)', '2826', '플루오르화물'),
    (r'(?:chloride|bromide|iodide|oxychloride|chlorohydrate|chlorhydroxide)', r'(?:클로라이드|브로마이드|아이오다이드|옥시클로라이드|클로로하이드레이트)', '2827', '염화물·브로민화물·아이오딘화물'),
    (r'(?:sulfide|polysulfide|hydrosulfide)', r'(?:설파이드|폴리설파이드)', '2830', '황화물'),
    (r'(?:sulfite|bisulfite|metabisulfite|thiosulfate|hydrosulfite)', r'(?:설파이트|바이설파이트|메타바이설파이트|티오설페이트|싸이오설페이트)', '2832', '아황산염·티오황산염'),
    (r'(?:sulfate|bisulfate|persulfate|alum)', r'(?:설페이트|바이설페이트|퍼설페이트|알럼|명반)', '2833', '황산염·명반·과황산염'),
    (r'(?:nitrate|nitrite)', r'(?:나이트레이트|나이트라이트)', '2834', '아질산염·질산염'),
    (r'(?:phosphate|hydrogen phosphate|dihydrogen phosphate|metaphosphate|hexametaphosphate|pyrophosphate|polyphosphate|tripolyphosphate|phosphinate|hypophosphite|phosphite)',
     r'(?:포스페이트|메타포스페이트|헥사메타포스페이트|파이로포스페이트|피로포스페이트|폴리포스페이트|트라이폴리포스페이트|하이포포스파이트)', '2835', '인산염·폴리인산염'),
    (r'(?:carbonate|bicarbonate|hydrogen carbonate|percarbonate|sesquicarbonate)', r'(?:카보네이트|바이카보네이트|퍼카보네이트|세스퀴카보네이트)', '2836', '탄산염·과탄산염'),
    (r'(?:borate|perborate|tetraborate|metaborate)', r'(?:보레이트|퍼보레이트|테트라보레이트)', '2840', '붕산염·과붕산염'),
    (r'(?:aluminate|permanganate|manganate|stannate|titanate|molybdate|tungstate|chromate|dichromate|zincate|vanadate)', r'(?:알루미네이트|퍼망가네이트|스타네이트|티타네이트|몰리브데이트)', '2841', '산화금속산염'),
    (r'(?:silicate|metasilicate|trisilicate)', r'(?:실리케이트|메타실리케이트|트라이실리케이트)', '2839', '규산염'),
]
def _inorganic(ko, en):
    # 특수한 것 먼저(호 문구에 물질명이 따로 있는 것)
    if re.fullmatch(r'sodium chloride|sea salt|salt|table salt', en) or re.fullmatch(r'소듐클로라이드|염화나트륨|소금|해염|정제염|천일염', ko): return '2501', '제2501호 소금·순염화나트륨'
    if re.fullmatch(r'(sodium|potassium) hydroxide', en) or re.fullmatch(r'(소듐|포타슘)하이드록사이드|수산화(나트륨|칼륨)', ko): return '2815', '제2815호 수산화나트륨·수산화칼륨'
    if re.fullmatch(r'(magnesium (hydroxide|peroxide)|(strontium|barium) (oxide|hydroxide|peroxide))', en) or re.fullmatch(r'마그네슘(하이드록사이드|퍼옥사이드)|수산화마그네슘|(스트론튬|바륨)(옥사이드|하이드록사이드|퍼옥사이드)', ko): return '2816', '제2816호 마그네슘의 수산화물·과산화물, 스트론튬·바륨의 산화물'
    if re.fullmatch(r'zinc (oxide|peroxide)', en) or re.fullmatch(r'징크(옥사이드|퍼옥사이드)|산화아연', ko): return '2817', '제2817호 산화아연·과산화아연'
    if re.fullmatch(r'(aluminum|aluminium) (oxide|hydroxide)|alumina', en) or re.fullmatch(r'알루미늄(옥사이드|하이드록사이드)|알루미나|산화알루미늄|수산화알루미늄', ko): return '2818', '제2818호 산화알루미늄·수산화알루미늄'
    if re.fullmatch(r'chromium (oxide|hydroxide)( greens?)?', en) or re.fullmatch(r'(크로뮴|크롬)(옥사이드|하이드록사이드)(그린)?|산화크롬', ko): return '2819', '제2819호 크로뮴의 산화물·수산화물'
    if re.fullmatch(r'manganese (di)?oxide', en) or re.fullmatch(r'(망가니즈|망간)(다이)?옥사이드', ko): return '2820', '제2820호 망가니즈의 산화물'
    if re.fullmatch(r'(iron|ferric|ferrous) (oxides?|hydroxide)', en) or re.fullmatch(r'(아이언|철)(옥사이드|하이드록사이드)|산화철|(적색|황색|흑색)산화철', ko): return '2821', '제2821호 철의 산화물·수산화물'
    if re.fullmatch(r'cobalt (oxide|hydroxide)', en): return '2822', '제2822호 코발트의 산화물'
    if re.fullmatch(r'titanium (di)?oxide', en) or re.fullmatch(r'티타늄(디|다이)?옥사이드|(이)?산화티타늄', ko): return '2823', '제2823호 티타늄의 산화물'
    if re.fullmatch(r'magnesium oxide|magnesia|magnesite', en) or re.fullmatch(r'마그네슘옥사이드|산화마그네슘|마그네시아|마그네사이트', ko): return '2519', '제2519호 마그네시아(산화마그네슘)'
    if re.fullmatch(r'hydrogen peroxide', en) or re.fullmatch(r'하이드로젠퍼옥사이드|과산화수소', ko): return '2847', '제2847호 과산화수소'
    if re.fullmatch(r'(colloidal )?(silver|gold|platinum|palladium)( nitrate| chloride| oxide| sulfate| citrate)?', en) or re.fullmatch(r'(콜로이드성?)?(은|금|백금|실버|골드|플래티넘)(콜로이드)?|(실버|은)(나이트레이트|클로라이드|옥사이드)|질산은', ko): return '2843', '제2843호 콜로이드 귀금속·귀금속 화합물'
    if re.fullmatch(r'(cerium|lanthanum|yttrium|neodymium)( \w+)?', en) or re.fullmatch(r'(세륨|란타넘|이트륨)\S*', ko): return '2846', '제2846호 희토류 금속 화합물'
    if re.fullmatch(r'ammonium nitrate', en) or re.fullmatch(r'암모늄나이트레이트|질산암모늄', ko): return '3102', '제3102호 질소비료(질산암모늄)'
    if re.fullmatch(r'potassium chloride', en) or re.fullmatch(r'포타슘클로라이드|염화칼륨', ko): return '3104', '제3104호 칼륨비료(염화칼륨)'
    if re.fullmatch(r'(silica|silicon dioxide|hydrated silica|fumed silica|silicic acid|carbon dioxide|nitrous oxide|sulfur dioxide|hydro(bromic|fluoric|iodic) acid)', en) or re.fullmatch(r'실리카|하이드레이티드실리카|이산화규소|무수규산|카본다이옥사이드|이산화탄소|나이트러스옥사이드|아산화질소|하이드로(브로믹|플루오릭)애씨드', ko): return '2811', '제2811호 그 밖의 무기산·비금속의 무기산소화합물'
    for pe, pk, h, why in [(r'hydrochloric acid', r'하이드로클로릭애씨드|염산', '2806', '염화수소(염산)'), (r'sulfuric acid', r'설퓨릭애씨드|황산', '2807', '황산'), (r'nitric acid', r'나이트릭애씨드|질산', '2808', '질산'),
                           (r'(phosphoric|polyphosphoric|metaphosphoric) acid|phosphorus pentoxide', r'(포스포릭|폴리포스포릭)애씨드|인산', '2809', '인산·폴리인산'), (r'boric acid|boron oxide', r'보릭애씨드|붕산', '2810', '붕소의 산화물·붕산'),
                           (r'ammonia|ammonium hydroxide', r'암모니아|암모늄하이드록사이드|암모니아수', '2814', '암모니아'), (r'(colloidal |precipitated |sublimed )?sulfur', r'황|승화황|침강황|설퍼|콜로이드성?황|설퍼콜로이드', '2802', '승화황·침강황·콜로이드황'),
                           (r'carbon black|charcoal powder|carbon', r'카본블랙|카본', '2803', '탄소(카본블랙)')]:
        if re.fullmatch(pe, en) or re.fullmatch(pk, ko): return h, f'제{h}호 {why}'
    if re.fullmatch(rf'{CAT_EN}( {CAT_EN})* (alumino)?silicate|{CAT_EN} {CAT_EN} (sulfate|phosphate|carbonate)', en) and len(re.findall(CAT_EN, en)) >= 2 and 'silicate' in en:
        return '2842', '제2842호 그 밖의 무기산염(복규산염·알루미노규산염)'
    for ae, ak, h, why in ANION:
        if re.fullmatch(rf'{PRE_EN}{CAT_EN}( {CAT_EN})? (?:hydrogen |di|tri|bi)?{ae}', en) or re.fullmatch(rf'{PRE_KO}{CAT_KO}{CAT_KO}?(?:하이드로젠|바이|다이|트라이)?{ak}', ko):
            return h, f'제{h}호 {why}'
    if re.fullmatch(rf'{CAT_EN} (oxide|dioxide|hydroxide|peroxide)', en) or re.fullmatch(rf'{CAT_KO}(옥사이드|다이옥사이드|하이드록사이드|퍼옥사이드)', ko): return '2825', '제2825호 그 밖의 무기염기·금속산화물·수산화물'
    return None

# ── 그 밖의 류: (영문 패턴, 국문 패턴, 호, 호 문구).  re.search 로 찾음(영문은 소문자)
TABLE = [
    # 제25류 광물
    (r'^quartz$|^quartz powder$', r'^석영$', '2506', '석영'), (r'^kaolin$|^china clay$', r'^카올린$|^고령토$', '2507', '고령토'),
    (r'^(bentonite|fuller.?s earth|attapulgite|sillimanite|kyanite|andalusite|mullite|montmorillonite|illite|.* clay)$', r'^(벤토나이트|풀러토|규선석|남정석|홍주석|멀라이트|몬모릴로나이트|일라이트|.*클레이|.*점토)$', '2508', '그 밖의 점토·홍주석·남정석·규선석'),
    (r'^chalk$', r'^백악$|^초크$', '2509', '백악'), (r'^(barite|barytes|witherite)$', r'^(중정석|바라이트)$', '2511', '천연 황산바륨'),
    (r'^(diatomaceous earth|diatomite|kieselguhr|tripolite)$', r'^(규조토|다이아토마이트)$', '2512', '규조토'), (r'^(pumice|emery|corundum|garnet)( powder)?$', r'^(부석|경석|퍼미스|금강사|강옥)(가루)?$', '2513', '부석·금강사·천연 연마재'),
    (r'^dolomite$', r'^돌로마이트$|^백운석$', '2518', '백운석'), (r'^(gypsum|anhydrite|plaster)$', r'^(석고|집섬)$', '2520', '석고'),
    (r'^(mica|sericite|muscovite|phlogopite)$', r'^(마이카|운모|세리사이트|견운모)$', '2525', '운모'), (r'^(talc|steatite|soapstone)$', r'^(탤크|탈크|활석|스테아타이트)$', '2526', '천연 스테아타이트·활석'),
    (r'^(feldspar|fluorspar|nepheline|leucite)$', r'^(장석|형석)$', '2529', '장석·형석'), (r'^(perlite|vermiculite|sepiolite|pyrophyllite|wollastonite|zeolite|tourmaline)$', r'^(펄라이트|질석|세피올라이트|파이로필라이트|규회석|제올라이트|토르말린)$', '2530', '따로 분류되지 않은 광물'),
    (r'^graphite$', r'^흑연$|^그래파이트$', '2504', '천연 흑연'),
    # 제11·17류 전분·당
    (r'^[a-z ()\-]*\b(corn|maize|wheat|rice|potato|tapioca|manihot|cassava|sweet potato|oryza sativa|zea mays|triticum vulgare|solanum tuberosum)[a-z ()\-]*\bstarch$|^starch$', r'^(옥수수|밀|쌀|감자|타피오카|고구마)?전분$', '1108', '전분'),
    (r'^(dextrin|maltodextrin|cyclodextrin|.*(hydroxypropyl|distarch|acetylated|octenylsuccinate|modified|pregelatinized).*starch.*)$', r'^(덱스트린|말토덱스트린|사이클로덱스트린|변성.*전분|.*(하이드록시프로필|다이스타치).*)$', '3505', '덱스트린·변성전분'),
    (r'^(sucrose|sugar|cane sugar|beet sugar)$', r'^(수크로오스|수크로스|설탕|자당)$', '1701', '사탕수수당·사탕무당, 화학적으로 순수한 자당'),
    (r'^(lactose|glucose|dextrose|fructose|maltose|invert sugar|glucose syrup|maple syrup|caramel|high fructose corn syrup|isomaltulose)$', r'^(락토오스|락토스|유당|글루코오스|글루코스|포도당|프럭토오스|프럭토스|과당|말토오스|말토스|맥아당|전화당|캐러멜)$', '1702', '그 밖의 당류(유당·맥아당·포도당·과당)'),
    (r'^(galactose|xylose|mannose|ribose|arabinose|rhamnose|fucose|trehalose|raffinose|sorbose|tagatose|melezitose|cellobiose|gentiobiose|[a-z]{3,}ose)$(?<!cellulose)(?<!glucose)(?<!lactose)(?<!maltose)(?<!sucrose)(?<!fructose)(?<!dextrose)(?<!amylose)', r'^(갈락토오스|갈락토스|자일로오스|자일로스|만노오스|만노스|리보오스|리보스|아라비노오스|아라비노스|람노오스|람노스|푸코오스|푸코스|트레할로오스|트레할로스|라피노오스|라피노스)$', '2940', '화학적으로 순수한 당류(자당·유당·맥아당·포도당·과당 제외)'),
    # 제13류
    (r'^(lac|shellac|gum arabic|acacia senegal gum|acacia gum|tragacanth|gum tragacanth|karaya gum|natural gum|gum benzoin|myrrh|olibanum|frankincense|dammar|mastic|copal)$', r'^(락|쉘락|셸락|아라비아검|아카시아검|트라가칸트|카라야검)$', '1301', '락·천연 검·수지'),
    (r'^(pectin|agar|carrageenan|guar gum|cyamopsis tetragonoloba \(guar\) gum|locust bean gum|ceratonia siliqua \(carob\) gum|tara gum|opium|.* (leaf|root|flower|fruit|seed|bark|stem|bud|peel|rhizome|sprout|callus|cell|herb|whole plant) extract)$', r'^(펙틴|한천|카라기난|구아검|로커스트빈검|타라검)$', '1302', '식물성 수액·추출물, 펙틴질, 한천·점질물'),
    # 제15류 유지·납
    (r'^(lanolin|lanolin wax|lanolin oil|wool fat|wool grease|adeps lanae)$', r'^(라놀린|라놀린왁스|라놀린오일|양모지)$', '1505', '울그리스·라놀린'),
    (r'^(fish oil|cod liver oil|shark liver oil|.*fish.* oil)$', r'^(어유|피쉬오일|대구간유|상어간유)$', '1504', '어류의 지방과 기름'),
    (r'^(emu oil|mink oil|horse (fat|oil)|egg oil|turtle oil|ostrich oil|snake oil)$', r'^(에뮤오일|밍크오일|마유|호스오일|난황오일|타조오일)$', '1506', '그 밖의 동물성 지방과 기름'),
    (r'^(tallow|lard|beef tallow|mutton tallow)$', r'^(우지|탤로우|라드|돈지)$', '1502', '소·양의 지방(1501 돼지)'),
    (r'^hydrogenated [a-z ()/\-]*\b(oil|oils|butter|fat)$|^(hydrogenated vegetable oil|interesterified .*|re-esterified .*)$', r'^(하이드로제네이티드|수소첨가|경화).*(오일|유|버터)$|.*경화유$', '1516', '수소첨가·에스터교환한 유지'),
    (r'\b(soybean|glycine soja|glycine max)\b.*\boil$', r'^(대두유|콩기름|돌콩오일|대두오일|소이빈오일)$', '1507', '대두유'), (r'\b(peanut|arachis hypogaea)\b.*\boil$', r'^(땅콩오일|땅콩기름|낙화생유|피넛오일)$', '1508', '땅콩기름'),
    (r'\b(olive|olea europaea)\b.*\bfruit oil$|^olive oil$', r'^(올리브오일|올리브유)$', '1509', '올리브유'), (r'^(palm oil|elaeis guineensis \(palm\) oil|elaeis guineensis oil)$', r'^(팜오일|팜유)$', '1511', '팜유'),
    (r'\b(sunflower|helianthus annuus|safflower|carthamus tinctorius|cotton|gossypium)\b.*\boil$', r'^(해바라기씨오일|해바라기유|홍화씨오일|잇꽃씨오일|면실유|목화씨오일)$', '1512', '해바라기씨유·잇꽃유·목화씨유'),
    (r'\b(coconut|cocos nucifera|palm kernel|babassu|orbignya)\b.*\boil$', r'^(코코넛오일|코코넛 오일|야자유|팜커넬오일|팜핵유|바바수오일|바바수씨오일)$', '1513', '야자유·팜핵유·바바수유'),
    (r'\b(rapeseed|canola|colza|brassica (napus|campestris)|mustard)\b.*\boil$', r'^(카놀라오일|카놀라유|유채씨오일|유채유|채종유|겨자씨오일)$', '1514', '유채유·겨자유'),
    (r'^(beeswax|bees wax|cera alba|cera flava|synthetic beeswax|.*\b(carnauba|candelilla|rice bran|japan|sumac|bayberry|ouricury|jojoba|sunflower seed|berry)\b.* wax|spermaceti|insect wax|chinese wax|shellac wax|myrica cerifera \(bayberry\) fruit wax)$', r'^(비즈왁스|밀랍|밀납|카나우바왁스|카르나우바왁스|칸데릴라왁스|쌀겨왁스|라이스왁스|목랍|옻나무열매왁스|경랍)$', '1521', '식물성 납·밀랍·그 밖의 곤충랍'),
    # 제27류
    (r'^(mineral oil|paraffinum liquidum|liquid paraffin|white mineral oil|light mineral oil)$', r'^(미네랄오일|유동파라핀|광유)$', '2710', '석유·역청유(광유)'),
    # 제29류 주: 비타민(2936)·알칼로이드(2939)·당 에터·에스터(2940)·항생물질(2941)
    (r'^(tocopherol|tocopheryl (acetate|linoleate|nicotinate|succinate|glucoside)|tocotrienols?|retinol|retinal|retinyl (acetate|palmitate|propionate|linoleate)|ascorbic acid|(sodium|calcium|magnesium|potassium) ascorbate|ascorbyl (palmitate|stearate|dipalmitate|tetraisopalmitate)|panthenol|dexpanthenol|dl-panthenol|d-panthenol|pantothenic acid|calcium pantothenate|niacinamide|nicotinamide|niacin|nicotinic acid|biotin|pyridoxine( hcl| hydrochloride)?|riboflavin|thiamine( hcl| nitrate)?|cyanocobalamin|folic acid|(ergo|chole)calciferol|menadione|phytonadione|vitamin [a-k]\d*( acetate| palmitate)?|beta-carotene)$',
     r'^(토코페롤|토코페릴(아세테이트|리놀리에이트|니코티네이트|석시네이트)|토코트라이에놀|레티놀|레틴알|레티닐(아세테이트|팔미테이트|프로피오네이트|리놀리에이트)|아스코빅애씨드|아스코르빈산|(소듐|칼슘|마그네슘|포타슘)아스코베이트|아스코빌(팔미테이트|스테아레이트|다이팔미테이트|테트라아이소팔미테이트)|판테놀|덱스판테놀|판토테닉애씨드|칼슘판토테네이트|나이아신아마이드|니코틴아마이드|나이아신|니코티닉애씨드|바이오틴|비오틴|피리독신(에이치씨엘)?|리보플라빈|티아민(에이치씨엘)?|사이아노코발아민|시아노코발라민|폴릭애씨드|엽산|(에르고|콜레)칼시페롤|메나디온|파이토나디온|비타민\S*|베타카로틴|베타-카로틴)$', '2936', '프로비타민·비타민과 그 유도체'),
    (r'^(caffeine|theobromine|theophylline|nicotine|quinine|ephedrine|berberine|piperine|capsaicin|atropine|morphine|codeine|papaverine|colchicine|emetine|glaucine|tomatidine|solanine|sanguinarine|matrine|[a-z]+ine alkaloid)$', r'^(카페인|테오브로민|테오필린|니코틴|퀴닌|에페드린|베르베린|피페린|캡사이신|글라우신|토마티딘|마트린)$', '2939', '알칼로이드'),
    (r'^(sucrose (stearate|laurate|palmitate|distearate|cocoate|acetate isobutyrate|polystearate|tristearate|myristate|oleate))$', r'^수크로오스(스테아레이트|라우레이트|팔미테이트|다이스테아레이트|코코에이트|폴리스테아레이트|트라이스테아레이트|미리스테이트|올리에이트)$', '2940', '당 에터·당 에스터'),
    # 제33·34류
    (r'^(fragrance|parfum|perfume|aroma|flavor|fragrance mix|perfume compound|mixture of odoriferous substances)$', r'^(향료|조합향료|착향제|퍼퓸|방향성 물질의 혼합물|방향성물질의혼합물)$', '3302', '방향성 물질의 혼합물'),
    (r'^sorbitan [a-z]+ate$|^sorbitan (sesqui|tri|di)[a-z]+ate$|\b(betaine|sultaine|amphoacetate|amphodiacetate|amphopropionate|sulfosuccinate|isethionate|sarcosinate|taurate|amine oxide)$(?<!^betaine)|^cocamide (mea|dea|mipa)$|^(lauramide|cocamide|oleamide|stearamide|myristamide) (mea|dea|mipa)$|^(decyl|lauryl|coco|caprylyl/capryl|cetearyl|myristyl|arachidyl) glucoside$|^(sodium|ammonium|tea|potassium|magnesium|mea|tipa|disodium)[ -](lauryl|laureth|myreth|coco|cocoyl|c\d+-\d+ [a-z]+|trideceth|pareth|cetearyl|cetyl|myristyl|oleth|deceth)[a-z\-0-9 ]* (sulfate|sulfonate|ether sulfate)$|^polysorbate[ -]?\d+$|^(peg|ppg)-\d+[a-z/ \-0-9]* (stearate|laurate|oleate|isostearate|distearate|dilaurate|dioleate|cocoate|castor oil|hydrogenated castor oil|glyceryl [a-z]+|sorbitan [a-z]+|glycerides|dimethicone)$|^[a-z]+eth-\d+( phosphate| carboxylate| sulfate)?$',
     r'^솔비탄\S*(에이트|레이트)$|(?<!^)(베타인|설테인|암포아세테이트|암포다이아세테이트|설포석시네이트|이세티오네이트|사코시네이트|타우레이트|아민옥사이드)$|^(코카|라우라|올레아|스테아라|미리스타)마이드(엠이에이|디이에이|엠아이피에이)$|^(데실|라우릴|코코|카프릴릴/카프릴|세테아릴|미리스틸|아라키딜)글루코사이드$|^(소듐|암모늄|티이에이|포타슘|마그네슘|다이소듐)(라우릴|라우레스|미레스|코코|코코일|트라이데세스|파레스)\S*(설페이트|설포네이트)$|^폴리솔베이트 ?\d+$|^(피이지|피피지)-\d+\S*(스테아레이트|라우레이트|올리에이트|아이소스테아레이트|다이스테아레이트|코코에이트|캐스터오일)$|^[가-힣]+레스-\d+$|^[가-힣]+세스-\d+$',
     '3402', '유기계면활성제(음이온·양이온·비이온·양쪽성)'),
    # 제35류
    (r'^(casein|sodium caseinate|calcium caseinate)$', r'^(카제인|카세인|소듐카제이네이트)$', '3501', '카세인·카세인산염'), (r'^(albumin|albumen|serum albumin|egg albumin|ovalbumin|lactalbumin|whey protein)$', r'^(알부민|세럼알부민|난백알부민|유청단백질)$', '3502', '알부민'),
    (r'^(gelatin|gelatine|isinglass)$', r'^(젤라틴)$', '3503', '젤라틴'),
    # 제38류
    (r'^(rosin|colophony|gum rosin|rosinates?|.* rosinates?|glyceryl rosinate|hydrogenated rosin|.*ester gum)$', r'^(로진|송진|콜로포니|\S*로지네이트|하이드로제네이티드로진)$', '3806', '로진·수지산과 그 유도체'), (r'^tall oil$', r'^톨유$|^톨오일$', '3803', '톨유'),
    (r'^(activated (carbon|charcoal)|charcoal)$', r'^(활성탄|활성탄소|숯)$', '3802', '활성탄'),
    # 호 문구에 물질명이 그대로 나오는 그 밖의 것 (기존 이름 규칙의 2712·3203·3507을 사전으로 옮김)
    (r'^(petrolatum|vaseline|petroleum jelly|paraffin|paraffin wax|microcrystalline wax|cera microcristallina|ceresin|ozokerite|montan wax|peat wax|slack wax)$', r'^(페트롤라툼|바셀린|파라핀|파라핀왁스|마이크로크리스탈린왁스|마이크로크리스탈린 왁스|세레신|오조케라이트|몬탄왁스)$', '2712', '석유젤리·파라핀왁스·마이크로크리스탈린 왁스·오조케라이트'),
    (r'^(heparin|heparin sodium|sodium heparin)$', r'^(헤파린|헤파린소듐)$', '3001', '헤파린과 그 염'),
    (r'^(cochineal|carmine|annatto|chlorophyll|chlorophyllin[a-z\- ]*|sepia|natural indigo|beet(root)? red|paprika (oleoresin|extract color)|curcumin|turmeric color|gardenia (yellow|blue)|natural colou?rant)$', r'^(코치닐|카민|카르민|안나토|클로로필|세피아|비트레드|커큐민|치자황색소|치자청색소|천연색소)$', '3203', '식물성·동물성 착색제'),
    (r'^(papain|bromelain|ficin|subtilisin|sutilains?|lipase|protease|amylase|lysozyme|pepsin|trypsin|catalase|superoxide dismutase|lactoperoxidase|glucose oxidase|rennet|[a-z]+ase)$(?<!base)(?<!phase)', r'^(파파인|브로멜라인|피신|서틸레인|리파아제|리파제|프로테아제|프로테아제|아밀라아제|아밀레이스|라이소자임|펩신|트립신|카탈라아제|효소)$', '3507', '효소'),
    (r'^(artificial wax|prepared wax|synthetic wax|polyethylene wax|peg wax)$', r'^(인조왁스|조제왁스|합성왁스)$', '3404', '인조 왁스·조제 왁스'),
    (r'^(disinfectant|fungicide|insecticide|herbicide|[a-z ]*preservative( blend| preparation)?)$', r'^(살균제|소독제|살충제|\S*\s?보존제)$', '3808', '살충제·살균제·소독제 등 조제품'),
    (r'^(glass beads?|glass microspheres?|glass powder|glass flakes?)$', r'^(글라스비드|유리구슬|글라스파우더|유리가루)$', '7018', '유리구슬·유리 소구(유리가루는 3207)'),
    # 제39류 중합체
    (r'^(polyethylene|oxidized polyethylene|ethylene/[a-z]+ copolymer|ethylene/va copolymer|ldpe|hdpe)$', r'^(폴리에틸렌|산화폴리에틸렌)$', '3901', '에틸렌의 중합체'),
    (r'^(polypropylene|polybutene|polyisobutene|polyisobutylene|polydecene|polyisoprene|hydrogenated (polyisobutene|polydecene|polybutene|poly\(c\d+-\d+ olefin\)|polyisoprene)|poly\(?alpha\)?olefin|polyolefin)$', r'^(폴리프로필렌|폴리부텐|폴리아이소부텐|폴리이소부텐|폴리데센|하이드로제네이티드(폴리아이소부텐|폴리데센|폴리부텐))$', '3902', '프로필렌·그 밖의 올레핀의 중합체'),
    (r'^(polystyrene|styrene polymer|styrene/[a-z ]+ copolymer|styrene/acrylates copolymer|abs)$', r'^(폴리스타이렌|폴리스티렌|스티렌 ?중합체|스타이렌/\S+코폴리머)$', '3903', '스티렌의 중합체'),
    (r'^(polyvinyl chloride|pvc|polyvinylidene (di)?(fluoride|chloride)|ptfe|polytetrafluoroethylene|polychlorotrifluoroethylene)$', r'^(폴리비닐클로라이드|폴리비닐리덴다이플루오라이드|폴리테트라플루오로에틸렌|피티에프이)$', '3904', '염화비닐·할로겐화 올레핀의 중합체'),
    (r'^(polyvinyl (acetate|alcohol|butyral)|pvp|polyvinylpyrrolidone|vp/[a-z ]+ copolymer|pva|pvm/ma copolymer|vinyl acetate/[a-z ]+ copolymer)$', r'^(폴리비닐(아세테이트|알코올|부티랄)|피브이피|폴리비닐피롤리돈|브이피/\S+코폴리머)$', '3905', '초산비닐·그 밖의 비닐의 중합체'),
    (r'^(polyurethane(-\d+)?|polyamide(-\d+)?|nylon(-\d+)?)$', r'^(폴리우레탄(-\d+)?|나일론(-\d+)?|폴리아마이드(-\d+)?)$', '3909', '폴리우레탄(나일론·폴리아미드는 3908)'),
    (r'^(dimethicone|dimethiconol|[a-z ]*methicone|[a-z ]*dimethicone|[a-z/ \-0-9]*dimethicone[a-z/ \-0-9]* (copolymer|crosspolymer)|silicone (oil|resin|rubber|elastomer)|methyl phenyl silicone|polysilicone-\d+|trimethylsiloxysilicate|polymethylsilsesquioxane|[a-z ]*silsesquioxane)$(?<!cyclomethicone)', r'^(다이메티콘|디메치콘|다이메티콘올|\S*메티콘|메틸페닐실리콘|실리콘(오일|레진|고무)|폴리실리콘-\d+|트라이메틸실록시실리케이트|폴리메틸실세스퀴옥세인)$(?<!사이클로메티콘)', '3910', '실리콘(일차제품)'),
    (r'^(cellulose|microcrystalline cellulose|cellulose gum|(hydroxyethyl|hydroxypropyl|methyl|ethyl|carboxymethyl|hydroxypropyl methyl)cellulose|cellulose acetate( butyrate| propionate)?|nitrocellulose|[a-z ]+ cellulose|polyquaternium-(4|10|24))$', r'^(셀룰로오스|마이크로크리스탈린셀룰로오스|셀룰로오스검|\S*셀룰로오스|나이트로셀룰로오스|폴리쿼터늄-(4|10|24))$', '3912', '셀룰로오스와 그 화학적 유도체'),
    (r'^(alginic acid|algin|[a-z ]*alginate|chitosan|chitin|hyaluronic acid|(sodium|potassium) hyaluronate|hydrolyzed hyaluronic acid|xanthan gum|pullulan|dextran|gellan gum|sclerotium gum|natural rubber derivatives?|hardened proteins?)$', r'^(알지닉애씨드|알긴산|알진|(소듐|포타슘|칼슘|암모늄|프로필렌글라이콜)?알지네이트|키토산|키틴|하이알루로닉애씨드|히알루론산|(소듐|포타슘)하이알루로네이트|잔탄검|풀루란|덱스트란|젤란검)$', '3913', '천연 중합체·변성한 천연 중합체'),
]

def lookup(ko='', en=''):
    ko = re.sub(r'\s+', '', str(ko or '')); en = re.sub(r'\s+', ' ', str(en or '')).strip().lower()
    r = _inorganic(ko, en)
    if r: return r
    for pe, pk, h, why in TABLE:
        if (en and re.search(pe, en)) or (ko and pk and re.search(pk, ko)): return h, f'제{h}호 {why}'
    return None
