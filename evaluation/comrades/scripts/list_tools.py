import json
import urllib.request
import sys

import os; URL = os.getenv("MCP_DAL_URL", "http://localhost:8002/mcp")
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

def list_tools():
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }
    
    print("Listing tools from live MCP server...")
    req = urllib.request.Request(URL, data=json.dumps(payload).encode('utf-8'), headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data
    except Exception as e:
        print(f"Error listing tools: {e}", file=sys.stderr)
        return None

def main():
    res = list_tools()
    if not res:
        return
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
