#!/bin/sh
# Релиз для git-зависимости anion: коммит с собранным lib/ поверх текущего
# HEAD и тег на него. Сама ветка lib/ не содержит и не меняется — так при
# установке ничего не собирается (bun не запускает prepare у зависимостей).
#
#   scripts/release.sh 1.1.0-anion.2   # version в package.json должна совпадать
#   git push origin main v1.1.0-anion.2
#
# В anion: "anime4k-webgpu": "github:HeilungZeit/anionWebGPU#v1.1.0-anion.2"
set -e
VERSION=$1
[ -n "$VERSION" ] || { echo "usage: scripts/release.sh <version>"; exit 1; }
git diff --quiet && git diff --cached --quiet || { echo "есть незакоммиченные изменения"; exit 1; }
[ "$(node -p "require('./package.json').version")" = "$VERSION" ] \
  || { echo "version в package.json не $VERSION"; exit 1; }

npm run check
npm run build

BRANCH=$(git rev-parse --abbrev-ref HEAD)
git checkout -q --detach
git add -f lib
git commit -q -m "release v$VERSION: собранный lib/"
git tag -a "v$VERSION" -m "v$VERSION"
git checkout -q "$BRANCH"
# Checkout ветки убирает отслеживаемый в релизе lib/ — собираем заново.
npm run build >/dev/null
echo "Тег v$VERSION готов: git push origin v$VERSION"
