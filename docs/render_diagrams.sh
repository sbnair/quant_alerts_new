#!/bin/bash
# Re-render article.md's mermaid blocks to docs/diagrams/*.svg.
# Requires @mermaid-js/mermaid-cli (npx) and Chrome.
set -e
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

python3 - "$TMP" <<'EOF'
import re, sys
tmp = sys.argv[1]
text = open('article.md').read()
# order must match the mermaid blocks' order of appearance in article.md
names = ['ml-pipeline', 'context-flow', 'architecture',
         'scan-sequence', 'scanrun-states', 'data-model']
blocks = re.findall(r'```mermaid\n(.*?)```', text, flags=re.S)
assert len(blocks) == len(names), f"expected {len(names)} blocks, found {len(blocks)}"
for name, block in zip(names, blocks):
    open(f'{tmp}/{name}.mmd', 'w').write(block)
print('\n'.join(names))
EOF

echo '{"executablePath": "/usr/bin/google-chrome", "args": ["--no-sandbox"]}' > "$TMP/pcfg.json"
mkdir -p docs/diagrams
for f in "$TMP"/*.mmd; do
  name=$(basename "$f" .mmd)
  npx --no-install -q @mermaid-js/mermaid-cli -p "$TMP/pcfg.json" \
      -i "$f" -o "docs/diagrams/$name.svg" -b transparent
  echo "rendered docs/diagrams/$name.svg"
done
