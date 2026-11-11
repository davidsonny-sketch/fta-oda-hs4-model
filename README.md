# fta-oda-hs4-model

원재료 HS4 판정 모델 (FTA ODA 프로젝트)

- `원재료 HS4 판정 모델 (최종).ipynb`: 최종 모델 노트북 (설명 · 준비 · 원료명/성분표 판정 · 최종 평가)

## 작업 규칙
1. **main에 바로 올리지 않는다.** 모든 작업은 `hyunsoo/<작업내용>` 브랜치에서 한다. (`scripts/new_branch.sh <작업내용>`)
2. **브랜치마다 아래 「변경 기록」에 항목을 하나 추가한다.** 무엇을 · 왜 · 바뀐 파일 · 검증 · 영향을 적는다.
3. 작업이 끝나면 PR을 열고, 현수가 확인한 뒤 main에 합친다. 합치면 항목의 상태를 `병합됨`으로 바꾼다.
4. 다른 팀원 저장소(lin1118-code/FTA-platform, lin1118-code/HS-complete-product)는 읽기만 한다. 바꿀 일이 있으면 먼저 상의한다.

PR에 README 「변경 기록」 수정이 없으면 GitHub Actions 검사(`README 변경 기록 확인`)가 실패한다.

## 변경 기록
<!-- 새 항목은 이 줄 바로 아래에, 최신이 위로 오게 추가 -->

### 2026-11-11 · `hyunsoo/branch-readme-setup` · 상태: 작업 중
- **무엇을**: 브랜치 작업 규칙과 변경 기록 양식을 만들고, PR에서 기록 누락을 막는 검사를 추가
- **왜**: main을 직접 바꾸지 않고, 브랜치마다 무엇이 바뀌었는지 README에 남기기 위해
- **바뀐 파일**: `README.md`(작업 규칙 · 변경 기록), `CLAUDE.md` · `AGENTS.md`(AI 작업 지침), `.github/pull_request_template.md`, `.github/workflows/readme-changelog-check.yml`, `scripts/new_branch.sh`
- **검증**: `scripts/new_branch.sh` 로 브랜치 생성·기록 양식 추가 시험, 검사 스크립트를 README 수정 있음/없음 두 경우로 시험
- **영향**: 모델 코드 · 노트북 · 점수 변화 없음
