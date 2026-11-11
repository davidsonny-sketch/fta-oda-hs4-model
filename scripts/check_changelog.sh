#!/usr/bin/env bash
# 사용: scripts/check_changelog.sh <비교 기준 커밋> <브랜치명>
# PR(또는 로컬 브랜치)이 README.md 「변경 기록」에 이 브랜치 항목을 추가했는지 확인
set -euo pipefail
BASE="${1:-main}"; BRANCH="${2:-$(git rev-parse --abbrev-ref HEAD)}"
if ! git diff --name-only "$BASE"...HEAD | grep -qx 'README.md'; then
  echo "❌ README.md가 바뀌지 않았습니다. 「변경 기록」에 이 브랜치($BRANCH) 항목을 추가하세요."; exit 1
fi
if ! git diff "$BASE"...HEAD -- README.md | grep -E '^\+### ' | grep -qF "\`$BRANCH\`"; then
  echo "❌ README.md 「변경 기록」에 \`$BRANCH\` 항목(### 날짜 · \`$BRANCH\` · 상태: ...)이 없습니다."; exit 1
fi
echo "✅ README.md 「변경 기록」에 \`$BRANCH\` 항목이 있습니다."
