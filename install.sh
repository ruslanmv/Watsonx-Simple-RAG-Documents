#!/usr/bin/env bash
set -euo pipefail

# colors
BLUE="\033[1;34m"; GREEN="\033[1;32m"; YELLOW="\033[1;33m"; RED="\033[1;31m"; NC="\033[0m"

echo -e "${BLUE}🔄  Updating apt cache…${NC}"
sudo apt-get update -y

echo -e "${BLUE}📥  Adding deadsnakes PPA for Python 3.12…${NC}"
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y

echo -e "${BLUE}🐍  Installing Python 3.12 and venv support…${NC}"
sudo apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-distutils \
    curl \
    poppler-utils \
    ghostscript \
    jq
echo -e "${GREEN}✅  Python check:${NC}"
printf "   • python3.12 → %s\n" "$(python3.12 --version)"
printf "   • pip3.12   → %s\n" "$(python3.12 -m pip --version || echo 'pip not yet installed')"

echo -e "${BLUE}🧹  Cleaning up…${NC}"
sudo apt-get autoremove -y

# Create & activate a Python 3.12 venv
if [[ -d ".venv" ]]; then
  echo -e "${YELLOW}⚠️   .venv exists; recreating with Python3.12...${NC}"
  rm -rf .venv
fi

echo -e "${BLUE}🐍  Creating virtual environment (.venv) with Python 3.12…${NC}"
python3.12 -m venv .venv

echo -e "${BLUE}🔐  Activating .venv…${NC}"
# shellcheck disable=SC1091
source .venv/bin/activate

echo -e "${BLUE}⬆️   Upgrading pip in the venv…${NC}"
pip install --upgrade pip

# Install project dependencies (This part is typically for requirements.txt, now we'll focus on Poetry)
if [[ -f "requirements.txt" ]]; then
  echo -e "${BLUE}📦  Installing Python dependencies from requirements.txt…${NC}"
  pip install -r requirements.txt
else
  echo -e "${YELLOW}📄  No requirements.txt found; skipping pip install from requirements.txt.${NC}"
fi