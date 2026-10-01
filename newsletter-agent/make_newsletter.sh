#!/bin/zsh

SCRIPT_DIR="$(cd "$(dirname "${(%):-%x}")" && pwd)"
cd "$SCRIPT_DIR" || exit

export PATH="/opt/homebrew/bin:/opt/homebrew/share/google-cloud-sdk/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export PYTHONUTF8=1

# GOOGLE_API_KEY kommt aus .env (gitignored) statt hartcodiert im Script
set -a
source "$SCRIPT_DIR/.env"
set +a

export GOOGLE_GENAI_USE_VERTEXAI=FALSE
unset GOOGLE_CLOUD_PROJECT
unset GOOGLE_APPLICATION_CREDENTIALS

DATE=$(date +"%Y-%m-%d")
# Same default as app/portfolio.py; NEWSLETTER_OUTPUT_DIR overrides it (e.g. for the demo example)
OUTPUT_DIR="${NEWSLETTER_OUTPUT_DIR:-$SCRIPT_DIR/Newsletter_output}"

mkdir -p "$OUTPUT_DIR"

uvx google-agents-cli run "Erstelle meinen heutigen Newsletter" > "$OUTPUT_DIR/newsletter_$DATE.md"

python3 - <<EOF
from pathlib import Path
import re

path = Path("$OUTPUT_DIR/newsletter_$DATE.md")
text = path.read_text(errors="ignore")

start = text.find("# Daily Personal Finance & AI Brief")
if start != -1:
    text = text[start:]

text = re.sub(r"\n?Session: .*", "", text, flags=re.DOTALL)

path.write_text(text)
EOF

pandoc "$OUTPUT_DIR/newsletter_$DATE.md" \
  -o "$OUTPUT_DIR/newsletter_$DATE.html" \
  --standalone \
  --embed-resources \
  --css="$SCRIPT_DIR/style.css"