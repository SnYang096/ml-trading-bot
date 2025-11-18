#!/bin/bash
#
# Install Git hooks from .githooks/ to .git/hooks/
# This script should be run after cloning the repository
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the project root directory
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
GITHOOKS_DIR="$PROJECT_ROOT/.githooks"
GIT_HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

# Check if we're in a git repository
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo -e "${RED}❌ Error: Not a git repository${NC}"
    echo -e "${YELLOW}💡 Please run this script from within a git repository${NC}"
    exit 1
fi

# Check if .githooks directory exists
if [ ! -d "$GITHOOKS_DIR" ]; then
    echo -e "${RED}❌ Error: .githooks directory not found${NC}"
    echo -e "${YELLOW}💡 Expected directory: $GITHOOKS_DIR${NC}"
    exit 1
fi

# Create .git/hooks directory if it doesn't exist
mkdir -p "$GIT_HOOKS_DIR"

# Install hooks
echo -e "${YELLOW}📦 Installing Git hooks...${NC}"

HOOKS_INSTALLED=0
for hook in "$GITHOOKS_DIR"/*; do
    if [ -f "$hook" ] && [ -x "$hook" ]; then
        hook_name=$(basename "$hook")
        target_hook="$GIT_HOOKS_DIR/$hook_name"
        
        # Copy hook and make it executable
        cp "$hook" "$target_hook"
        chmod +x "$target_hook"
        
        echo -e "${GREEN}  ✅ Installed: $hook_name${NC}"
        HOOKS_INSTALLED=$((HOOKS_INSTALLED + 1))
    fi
done

if [ $HOOKS_INSTALLED -eq 0 ]; then
    echo -e "${YELLOW}⚠️  No hooks found in .githooks/${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Successfully installed $HOOKS_INSTALLED Git hook(s)${NC}"
echo -e "${YELLOW}💡 Hooks will now run automatically on git commit${NC}"

