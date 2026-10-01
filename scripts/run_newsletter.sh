#!/bin/bash

# Falls back to the in-repo newsletter-agent dir; override via env var
# to point at a different location.
NEWSLETTER_AGENT_DIR="${NEWSLETTER_AGENT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/newsletter-agent}"

unset VIRTUAL_ENV

cd "$NEWSLETTER_AGENT_DIR" || exit 1
./make_newsletter.sh
