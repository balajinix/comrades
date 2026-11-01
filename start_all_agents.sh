#!/bin/bash

echo "Starting COMRADES Multi-Agent System with MCP Servers..."

mkdir -p logs

echo "Cleaning up previous log files..."
rm -f logs/*.log logs/*.pid

GRAPHQL_DIR="agents/graphql_query_agent"
SCHEMA_DIR="agents/schema_mapping_agent"
DAL_AGENT_DIR="agents/dal_agent"
HOST_DIR="agents/a2a_host_agent"
ORCHESTRATOR_DIR="orchestrator"

ROOT_DIR="$(pwd)"
# External mcp_dal server (data access layer backends). Override via env if relocated.
MCP_DAL_DIR="${MCP_DAL_DIR:-$HOME/src/mcp_dal}"
MCP_DAL_PORT="${MCP_DAL_PORT:-8201}"

run_with_venv() {
    local dir=$1
    local cmd=$2
    local log_file=$3
    local pid_file=$4
    
    echo "Starting in $dir..."
    
    # Store original directory
    local original_dir=$(pwd)
    
    # Check if directory exists before changing to it
    if [ ! -d "$dir" ]; then
        echo "Error: Directory $dir not found"
        echo "Current working directory: $(pwd)"
        echo "Available directories: $(ls -la | grep ^d)"
        return 1
    fi
    
    cd "$dir"
    
    # Check if venv exists
    if [ ! -d ".venv" ]; then
        echo "Error: .venv not found in $dir"
        echo "Please run 'uv venv' and 'uv sync --no-editable --link-mode=copy' in $dir first"
        # Return to original directory
        cd "$original_dir"
        return 1
    fi
    
    # Deactivate any existing venv first
    if [ ! -z "$VIRTUAL_ENV" ]; then
        deactivate 2>/dev/null || true
        unset VIRTUAL_ENV
    fi
    
    # Sync dependencies without editable install
    echo "Syncing dependencies in $dir..."
    source .venv/bin/activate
    uv sync --no-editable --link-mode=copy

    
    # Run command
    nohup $cmd > "$original_dir/$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$original_dir/$pid_file"
    echo "Started with PID: $pid"
    
    # Deactivate to clean up for next agent
    deactivate 2>/dev/null || true
    unset VIRTUAL_ENV
    
    cd "$original_dir"
    return 0
}

echo "=== Starting Data Access Layer (mcp_dal) ==="
if [ -x "$MCP_DAL_DIR/venv/bin/dal-mcp-server" ]; then
    echo "Starting mcp_dal on port $MCP_DAL_PORT..."
    ( cd "$MCP_DAL_DIR" && \
      LITELLM_API_KEY="${LITELLM_API_KEY:-dummy-key}" \
      LITELLM_BASE_URL="${LITELLM_BASE_URL:-http://localhost:9999}" \
      nohup venv/bin/dal-mcp-server --transport streamable-http --port "$MCP_DAL_PORT" \
        > "$ROOT_DIR/logs/mcp_dal.log" 2>&1 & echo $! > "$ROOT_DIR/logs/mcp_dal.pid" )
    sleep 6
    echo "Started mcp_dal with PID: $(cat logs/mcp_dal.pid 2>/dev/null || echo unknown)"
else
    echo "Warning: mcp_dal not found at $MCP_DAL_DIR/venv/bin/dal-mcp-server."
    echo "  Run its setup.sh first. The DAL Agent will report errors until mcp_dal is up."
fi

echo "=== Starting Main Agents ==="

echo "Starting GraphQL Query Agent..."
if ! run_with_venv "$GRAPHQL_DIR" "uv run --active app/__main__.py" "logs/graphql_agent.log" "logs/graphql_agent.pid"; then
    exit 1
fi
GRAPHQL_PID=$(cat logs/graphql_agent.pid)

echo "Starting Schema Mapping Agent..."
if ! run_with_venv "$SCHEMA_DIR" "uv run --active app/__main__.py" "logs/schema_agent.log" "logs/schema_agent.pid"; then
    exit 1
fi
SCHEMA_PID=$(cat logs/schema_agent.pid)

echo "Starting DAL Agent (mcp_dal via federated kg_agent)..."
if ! run_with_venv "$DAL_AGENT_DIR" "uv run --active app/__main__.py" "logs/dal_agent.log" "logs/dal_agent.pid"; then
    echo "Warning: DAL Agent failed to start (continuing)."
fi
DAL_AGENT_PID=$(cat logs/dal_agent.pid 2>/dev/null || echo "failed")

echo "Waiting for LangGraph agents to initialize..."
sleep 5

echo "Starting Host Agent Orchestrator..."
cd "$HOST_DIR"

if [ ! -d ".venv" ]; then
    echo "Error: .venv not found in $HOST_DIR"
    echo "Please run 'uv venv' and 'uv sync --no-editable --link-mode=copy' in $HOST_DIR first"
    exit 1
fi

if [ ! -z "$VIRTUAL_ENV" ]; then
    deactivate 2>/dev/null || true
    unset VIRTUAL_ENV
fi

source .venv/bin/activate
echo "Syncing dependencies in $HOST_DIR..."
uv sync --no-editable --link-mode=copy

nohup uv run --active adk web --host 0.0.0.0 --port 8080 . > ../logs/a2a_host_agent.log 2>&1 &
ORCHESTRATOR_PID=$!
echo $ORCHESTRATOR_PID > ../logs/a2a_host_agent.pid
echo "Started a2a_host_agent with PID: $ORCHESTRATOR_PID"
cd ..

echo ""
echo "=== COMRADES Multi-Agent System Started ==="
echo ""
echo "Main Agents:"
echo "  GraphQL Query Agent:      $GRAPHQL_PID (Port: 8001)"
echo "  Schema Mapping Agent:     $SCHEMA_PID (Port: 8002)"
echo "  DAL Agent:                $DAL_AGENT_PID (Port: 8005)"
echo "  Host Agent Orchestrator:  $ORCHESTRATOR_PID (Port: 8080)"
echo ""
echo "Agent URLs:"
echo "  GraphQL Query Agent:      http://localhost:8001"
echo "  Schema Mapping Agent:     http://localhost:8002"
echo "  DAL Agent:                http://localhost:8005"
echo "  mcp_dal (data access):    http://localhost:$MCP_DAL_PORT/mcp"
echo "  Web Interface:            http://internal-host:8080/dev-ui"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/graphql_agent.log"
echo "  tail -f logs/schema_agent.log"
echo "  tail -f logs/a2a_host_agent.log"
echo ""
