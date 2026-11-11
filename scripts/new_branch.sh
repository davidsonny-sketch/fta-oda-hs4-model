#!/usr/bin/env bash
# 사용: scripts/new_branch.sh <작업내용>   예) scripts/new_branch.sh synonym-table
# main 최신에서 hyunsoo/<작업내용> 브랜치를 만들고 README 「변경 기록」에 빈 양식을 추가
set -euo pipefail
[ $# -ge 1 ] || { echo "사용: scripts/new_branch.sh <작업내용>"; exit 1; }
B="hyunsoo/$1"; D="$(date +%Y-%m-%d)"
git fetch -q origin main 2>/dev/null || true
git checkout -q -b "$B" "$(git rev-parse --verify -q origin/main || echo main)"
python3 - "$B" "$D" << 'PY'
import sys; b, d = sys.argv[1], sys.argv[2]
mark = '<!-- 새 항목은 이 줄 바로 아래에, 최신이 위로 오게 추가 -->'
s = open('README.md', encoding='utf-8').read()
entry = (f"\n\n### {d} · `{b}` · 상태: 작업 중\n- **무엇을**: \n- **왜**: \n- **바뀐 파일**: \n- **검증**: \n- **영향**: ")
assert mark in s, 'README.md에 변경 기록 표시줄이 없습니다'
open('README.md', 'w', encoding='utf-8').write(s.replace(mark, mark + entry, 1))
PY
echo "✅ $B 브랜치를 만들고 README 「변경 기록」에 양식을 추가했습니다. 내용을 채운 뒤 커밋하세요."
