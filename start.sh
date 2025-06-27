#!/bin/bash

# ==============================================================================
#  AI Tutor Service Starter (Bash-Only Edition)
#  - Prepares data and runs the Streamlit UI for the AI Tutor project.
#  - Usage:
#      bash start.sh pipeline    (to run the data processing pipeline)
#      bash start.sh             (to start the web UI)
#      bash start.sh stop        (to stop the web UI)
# ==============================================================================

# --- Configuration ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- Script Logic ---
# Assumes the script is run from the project's root directory
PROJECT_ROOT=$(pwd)
APP_PID_FILE="$PROJECT_ROOT/app.pid"

# --- Stop Functionality ---
if [ "$1" == "stop" ]; then
    echo -e "${YELLOW}🛑 Stopping AI Tutor Web UI...${NC}"
    if [ -f "$APP_PID_FILE" ]; then
        echo "   - Stopping app (PID: $(cat $APP_PID_FILE))..."
        kill $(cat $APP_PID_FILE)
        rm $APP_PID_FILE
        echo -e "${GREEN}✅ Web UI stopped.${NC}"
    else
        echo "   - Web UI not running via this script (no PID file)."
    fi
    exit 0
fi

# --- Data Pipeline Functionality ---
if [ "$1" == "pipeline" ]; then
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║      Running Data Prep Pipeline      ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
    echo ""

    echo "▶️  Step 1: Chunking PDF document..."
    python step1_extract_chunks.py
    echo "▶️  Step 2: Generating embeddings for chunks..."
    python step2_generate_embeddings_ibm.py

    echo -e "\n${GREEN}✅ Data preparation complete.${NC}"
    echo -e "You can now ask questions using the command-line interface:"
    echo -e "${YELLOW}python step3_vector_search.py${NC}\n"
    exit 0
fi


# --- Header for Starting UI ---
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Starting AI Tutor Web UI         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# --- Pre-run Checks ---
if [ ! -f "app.py" ]; then
    echo -e "${YELLOW}Error: 'app.py' not found."
    echo "Please run this script from the project's root directory."
    exit 1
fi

echo "🚀 Launching Web UI in the background..."

# --- Service Launch ---
# Start Streamlit App as a background process
streamlit run app.py >/dev/null 2>&1 &
# '$!' gets the PID of the last background process. We save it to a file.
echo $! > "$APP_PID_FILE"


# --- Final Instructions ---
echo -e "\n${GREEN}✅ Web UI is launching in the background.${NC}"
echo -e "   - Access the interface at ${YELLOW}http://localhost:8501${NC}."
echo -e "\nTo stop the Web UI, run this command:"
echo -e "${YELLOW}bash start.sh stop${NC}\n"