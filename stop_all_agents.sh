#!/bin/bash

# stop_all_agents.sh - Stop all gateway DAL agents

echo "Stopping gateway DAL Multi-Agent System..."

# Stop processes by PID files
if [ -d "logs" ]; then
    echo "Stopping agents by PID files..."
    for pidfile in logs/*.pid; do
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            agent_name=$(basename "$pidfile" .pid)
            
            if kill -0 "$pid" 2>/dev/null; then
                echo "Stopping $agent_name (PID: $pid)"
                kill "$pid"
                
                # Wait for graceful shutdown
                sleep 2
                
                # Force kill if still running
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Force stopping $agent_name"
                    kill -9 "$pid"
                fi
                
                echo "$agent_name stopped"
            else
                echo "$agent_name was not running"
            fi
            
            rm "$pidfile"
        fi
    done
else
    echo "No logs directory found"
fi

# Also kill by port (backup method for all services)
echo "Checking for any remaining processes on known ports..."
declare -A port_names=(
    [8001]="GraphQL Agent"
    [8002]="Schema Mapping Agent"
    [8005]="DAL Agent"
    [8080]="Host Agent Orchestrator"
)

for port in 8001 8002 8005 8080; do 
    pid=$(lsof -ti:$port 2>/dev/null || true)
    if [ ! -z "$pid" ]; then
        service_name=${port_names[$port]}
        echo "Killing $service_name process $pid on port $port"
        kill -9 "$pid" 2>/dev/null || true
    fi
done

echo "All agents stopped"
