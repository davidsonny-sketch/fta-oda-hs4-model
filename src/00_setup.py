# 원재료 모델 공통 준비 (Colab): 인증 → 공통 코드·데이터 내려받기 → /content/rawmodel 에 단계 폴더 만들기
# 각 노트북 첫 칸에서 이 파일을 내려받아 exec로 실행
import os, io, re, pandas as pd, gspread, google.auth
from google.colab import auth
from googleapiclient.discovery import build
auth.authenticate_user()
creds, _ = google.auth.default()
d = build('drive', 'v3', credentials=creds); gc = gspread.authorize(creds)

FOLDER = '1bNwzZnTPU0T64xoycbgGE2iJAlTpeTgo'                    # 원재료 모델 폴더
SHEET_ID = '1ulSdeUfojfHPUg2vaKogQiquMzOossbo6RVyAdwOIfk'       # 원재료 모델 테스트셋 (시트1, R열 데이터구분)
PRODUCT_SHEET_ID = '1LCQdz9yt0K72Cu51ViX7plTkb1MP2k8KTjQvTG70qoU'  # 완제품 모델 데이터_사례_전처리 후 (읽기만)
MFDS_ID, PUBCHEM_ID = '1SUliA8_fiMixwFTkbXjEWUG0XchyvScp', '119yqXTmUEJH_1MgEJWx9jpezvuAP29hC'

def child(name, parent):
    q = f"name = '{name}' and '{parent}' in parents and trashed = false"
    r = d.files().list(q=q, fields='files(id,name)', supportsAllDrives=True, includeItemsFromAllDrives=True).execute()['files']
    if not r: raise FileNotFoundError(name)
    return r[0]['id']
get = lambda fid: d.files().get_media(fileId=fid).execute()

BASE = '/content/rawmodel'
for sub in ['01_baseline', '02_name_normalization', '03_text_classification', '04_knowledge_structure', '05_reranking', '06_final_test']:
    os.makedirs(f'{BASE}/{sub}', exist_ok=True)
os.chdir(BASE)
common = child('00_공통', FOLDER)
for fn in ['common.py', 'hs4_bilingual.py', 'name_norm.py', 'bom_check.py', 'test_b_v2.csv', 'hs_lexicon.py', 'sheet1_corrections.csv']:   # 2026-11-11 v2: 호 용어 사전·시트1 정답 수정표
    open(fn, 'wb').write(get(child(fn, common)))
open('struct_rules.py', 'wb').write(get('1-ajDj_NsRNSMe8lgrSLCFwrjy55p8QGQ'))   # 원재료 모델 폴더에 이미 있는 구조 규칙
open('03_text_classification/run.py', 'wb').write(get(child('run_03.py', common)))   # 05단계가 03의 모델 함수를 재사용
open('mfds_db.csv', 'wb').write(get(MFDS_ID)); open('pubchem.csv', 'wb').write(get(PUBCHEM_ID))
try: open('real_pubchem.csv', 'wb').write(get(child('real_pubchem.csv', common)))   # 실제 성분·HK 추가분 구조(있으면)
except FileNotFoundError: pass

# 시트1 → gold_814_split.csv (R열 '데이터구분': 학습/시험A)
ws = gc.open_by_key(SHEET_ID).worksheet('시트1')
g = pd.DataFrame(ws.get_all_records(numericise_ignore=['all'])).astype(str)
assert '데이터구분' in g.columns, '시트1에 R열 데이터구분이 없어요'
g.to_csv('gold_814_split.csv', index=False)

# 실제 성분표(완제품 시트, 읽기만) → 국문·영문 성분 짝 → 시험 B 120종을 뺀 연습용
from bom_check import split_ingredients
p = pd.DataFrame(gc.open_by_key(PRODUCT_SHEET_ID).worksheet('사례_전처리 완료').get_all_records()).astype(str)
fix = lambda s: re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', s)
rows = []
for i, r in p.iterrows():
    if not r['Main Ingredients (KR)'].strip(): continue
    K, E = split_ingredients(fix(r['Main Ingredients (KR)'])), split_ingredients(fix(r['Main Ingredients (EN)']))
    for j, (k, _) in enumerate(K): rows.append({'제품': i, 'KR': k, 'EN': E[j][0] if len(K) == len(E) else ''})
R = pd.DataFrame(rows); R.to_csv('real_ingredients_v2.csv', index=False)
tb = set(pd.read_csv('test_b_v2.csv', dtype=str).fillna('')['입력 국문명'])
R[~R.KR.isin(tb)].to_csv('dev_ingredients.csv', index=False)
print('준비 완료 | 시트1', len(g), '행 (학습', (g['데이터구분'] == '학습').sum(), '/ 시험A', (g['데이터구분'] == '시험A').sum(), ') | 실제 성분', len(R), '개')
