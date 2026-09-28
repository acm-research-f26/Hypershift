#!/usr/bin/env bash
# Fetches only what we need from the RSR repo (~100 MB instead of ~340 MB). See decision node D1 on failure.
set -euo pipefail
export MSYS_NO_PATHCONV=1   # Git Bash would otherwise rewrite "/data/..." below into a Windows path
DEST=data/raw/rsr
mkdir -p data/raw
if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none --no-checkout --depth 1 \
    https://github.com/fulifeng/Temporal_Relational_Stock_Ranking.git "$DEST"
fi
cd "$DEST"
git sparse-checkout init --no-cone
git sparse-checkout set "/data/2013-01-01/*" "/data/*.csv" "/data/relation.tar.gz"
git checkout master
tar xzf data/relation.tar.gz -C data/
ls data/relation/sector_industry data/relation/wikidata
