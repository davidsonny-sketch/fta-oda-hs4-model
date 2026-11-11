"""드라이브·Colab 없이 원재료 HS4 모델을 쓰는 진입점 (서버·플랫폼 어댑터용)
저장소 구조: src/ (코드) + data/ (시트1 내보내기·식약처 DB·PubChem 구조·정답 수정표)
    import sys; sys.path.insert(0, '<repo>/src')
    from standalone import load_model
    m = load_model()            # 기본 data 폴더 = <repo>/data
    m.one('나이아신아마이드'); m.bom('정제수, 글리세린, ...')"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
if HERE not in sys.path: sys.path.insert(0, HERE)

def load_model(data_dir=DATA):
    from hs4_final import HS4Final
    need = ['gold_814_split.csv', 'mfds_db.csv', 'pubchem.csv']
    miss = [f for f in need if not os.path.exists(os.path.join(data_dir, f))]
    if miss: raise FileNotFoundError(f'원재료 모델 데이터 파일이 없습니다: {miss} (폴더: {data_dir})')
    return HS4Final(data_dir)
