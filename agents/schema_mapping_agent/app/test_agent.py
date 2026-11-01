# test_agent.py 
# Used to individually test the schema mapping agent that has been integrated with LangGraph and A2A protocol. 
# ----------------------------------------------------------------------------------------------------------

import requests
import json

def test_agent():
    """Basic test for the A2A GraphQL Mapping Agent"""
    base_url = "http://localhost:8002"
    
    # Test health endpoint
    print("Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Health check: {response.json()}")
    
    # Test capabilities endpoint
    print("\nTesting capabilities endpoint...")
    response = requests.get(f"{base_url}/capabilities")
    print(f"Capabilities: {json.dumps(response.json(), indent=2)}")
    
    # Test execute endpoint
    print("\nTesting execute endpoint...")
    test_request = {
        "task": "Map resilience vulnerabilities to optimizer entities",
        "context": {}
    }
    
    response = requests.post(f"{base_url}/execute", json=test_request)
    result = response.json()
    print(f"Execute result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    test_agent()
