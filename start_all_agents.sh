#!/bin/bash

echo "Starting COMRADES Multi-Agent System with MCP Servers..."

mkdir -p logs

echo "Cleaning up previous log files..."
rm -f logs/*.log logs/*.pid

GRAPHQL_DIR="agents/graphql_query_agent"
SCHEMA_DIR="agents/schema_mapping_agent"
DASHBOARD_DIR="agents/dashboard_agent"
TRACE_PRUNING_DIR="agents/trace_pruning_agent"
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

start_mcp_server() {
    local server_name=$1
    local server_file=$2
    local port=$3
    local log_file=$4
    local pid_file=$5
    
    echo "Starting $server_name MCP Server on port $port..."
    
    # Check if the server file exists
    if [ ! -f "$TRACE_PRUNING_DIR/$server_file" ]; then
        echo "Warning: $server_file not found in $TRACE_PRUNING_DIR"
        echo "Skipping $server_name MCP Server"
        cd "$(dirname "$0")" 2>/dev/null || cd "$PWD"
        return 1
    fi
    
    cd "$TRACE_PRUNING_DIR"
    
    # Check if venv exists
    if [ ! -d ".venv" ]; then
        echo "Error: .venv not found in $TRACE_PRUNING_DIR for MCP server"
        cd ../..
        return 1
    fi

    # Activate venv and start server
    source .venv/bin/activate
    nohup python $server_file --port $port > ../../$log_file 2>&1 &
    local pid=$!
    echo $pid > ../../$pid_file
    echo "Started $server_name MCP Server with PID: $pid"

    deactivate 2>/dev/null || true
    unset VIRTUAL_ENV
    cd ../..
    return 0
}


start_k8scost_port_forward() {
    local log_file=$1
    local pid_file=$2
    local port=$3

    echo "Starting K8sCost port-forward on port $port..."

    # Store the original directory
    local original_dir=$(pwd)
    
    # Change to the data directory where the .env and kubeconfig files are located
    if [ ! -d "$TRACE_PRUNING_DIR" ]; then
        echo "Warning: $TRACE_PRUNING_DIR directory not found"
        echo "Skipping K8sCost port-forward"
        return 1
    fi
    
    cd "$TRACE_PRUNING_DIR"
    
    if [ -f ".env" ]; then
        echo "Sourcing .env file in $TRACE_PRUNING_DIR ..."
        source .env
    else
        echo "Warning: .env file not found in $TRACE_PRUNING_DIR. KUBECONFIG might not be set."
        echo "Skipping K8sCost port-forward"
        cd "$original_dir"
        return 1
    fi
    
    if [ -z "$KUBECONFIG" ]; then
        echo "Warning: KUBECONFIG environment variable not set"
        echo "Skipping K8sCost port-forward"
        cd "$original_dir"
        return 1
    fi
    
    if [ ! -f "$KUBECONFIG" ]; then
        echo "Warning: KUBECONFIG file not found at $KUBECONFIG"
        echo "Skipping K8sCost port-forward"
        cd "$original_dir"
        return 1
    fi

    echo "Using KUBECONFIG: $KUBECONFIG"

    echo "Testing kubectl connectivity..."
    if ! kubectl get nodes > /dev/null 2>&1; then
        echo "Warning: kubectl cannot connect to cluster. Check your kubeconfig."
        echo "Skipping K8sCost port-forward"
        cd "$original_dir"
        return 1
    fi

    if ! kubectl get namespace k8scost > /dev/null 2>&1; then
        echo "Warning: k8scost namespace not found in cluster"
        echo "Skipping K8sCost port-forward"
        cd "$original_dir"
        return 1
    fi

    if ! kubectl get deployment k8scost-cost-analyzer -n k8scost > /dev/null 2>&1; then
        echo "Warning: k8scost-cost-analyzer deployment not found in k8scost namespace"
        echo "Skipping K8sCost port-forward"
        cd "$original_dir"
        return 1
    fi

    echo "Starting kubectl port-forward for k8scost-cost-analyzer..."
    nohup kubectl port-forward --namespace k8scost deployment/k8scost-cost-analyzer $port:9090 --address 0.0.0.0 > "$original_dir/$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$original_dir/$pid_file"
    echo "Started K8sCost port-forward with PID: $pid"
    
    cd "$original_dir"
    return 0
}


echo "=== Starting MCP Servers ==="

start_mcp_server "K8sCost" "k8scost_mcp.py" "8101" "logs/k8scost_mcp.log" "logs/k8scost_mcp.pid"
K8SCOST_MCP_PID=$(cat logs/k8scost_mcp.pid 2>/dev/null || echo "failed")

start_mcp_server "TraceSampling" "trace_sampling_mcp.py" "8102" "logs/trace_sampling_mcp.log" "logs/trace_sampling_mcp.pid"
TRACE_SAMPLING_PID=$(cat logs/trace_sampling_mcp.pid 2>/dev/null || echo "failed")

echo "Waiting for MCP servers to initialize..."
sleep 3

start_k8scost_port_forward "logs/k8scost_port_forward.log" "logs/k8scost_port_forward.pid" "9090"
K8SCOST_PORT_FORWARD_PID=$(cat logs/k8scost_port_forward.pid 2>/dev/null || echo "failed")

echo "Waiting for K8sCost port-forward to establish..."
sleep 5

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

echo "Starting Dashboard Visualization Agent..."
if ! run_with_venv "$DASHBOARD_DIR" "uv run --active app/__main__.py" "logs/dashboard_agent.log" "logs/dashboard_agent.pid"; then
    exit 1
fi
DASHBOARD_PID=$(cat logs/dashboard_agent.pid)

echo "Starting Trace Pruning Agent..."
if ! run_with_venv "$TRACE_PRUNING_DIR" "uv run --active app/__main__.py" "logs/trace_pruning_agent.log" "logs/trace_pruning_agent.pid"; then
    exit 1
fi
TRACE_PRUNING_PID=$(cat logs/trace_pruning_agent.pid)

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
echo "MCP Servers:"
echo "  K8sCost MCP Server:      $K8SCOST_MCP_PID (Port: 8101)"
echo "  Trace Sampling MCP:       $TRACE_SAMPLING_PID (Port: 8102)"
echo ""
echo "External Servers:"
echo "  K8sCost Port-Forward:    $K8SCOST_PORT_FORWARD_PID (Port: 9090)"
echo ""
echo "Main Agents:"
echo "  GraphQL Query Agent:      $GRAPHQL_PID (Port: 8001)"
echo "  Schema Mapping Agent:     $SCHEMA_PID (Port: 8002)"
echo "  Dashboard Visualization:  $DASHBOARD_PID (Port: 8003)"
echo "  Trace Pruning Agent:      $TRACE_PRUNING_PID (Port: 8004)"
echo "  Host Agent Orchestrator:  $ORCHESTRATOR_PID (Port: 8080)"
echo ""
echo "Agent URLs:"
echo "  GraphQL Query Agent:      http://localhost:8001"
echo "  Schema Mapping Agent:     http://localhost:8002"
echo "  Dashboard Visualization:  http://localhost:8003"
echo "  Trace Pruning Agent:      http://localhost:8004"
echo "  DAL Agent:                http://localhost:8005"
echo "  mcp_dal (data access):    http://localhost:$MCP_DAL_PORT/mcp"
echo "  Web Interface:            http://internal-host:8080/dev-ui"
echo ""
echo "MCP Server URLs:"
echo "  K8sCost MCP:             http://localhost:8101/sse"
echo "  Trace Sampling MCP:       http://localhost:8102/sse"
echo ""
echo "External Server URLs:"
echo "  K8sCost UI:              http://localhost:9090"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/k8scost_mcp.log"
echo "  tail -f logs/trace_sampling_mcp.log"
echo "  tail -f logs/k8scost_port_forward.log" # New log file
echo "  tail -f logs/graphql_agent.log"
echo "  tail -f logs/schema_agent.log"
echo "  tail -f logs/dashboard_agent.log"
echo "  tail -f logs/trace_pruning_agent.log"
echo "  tail -f logs/a2a_host_agent.log"
echo ""
