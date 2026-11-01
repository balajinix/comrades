import requests
import json
import os

def test_agent():
    """Basic test for the A2A Data Visualization Agent"""
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test capabilities endpoint
    print("\nTesting capabilities endpoint...")
    try:
        response = requests.get(f"{base_url}/capabilities")
        print(f"Capabilities: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Capabilities test failed: {e}")
        return
    
    # Use your own CSV file
    print("\nUsing your CSV file...")
    # Replace this path with your actual CSV file path
    test_csv = "../data/apm_application_metrics.csv"
    
    # Test process endpoint
    print("\nTesting process endpoint...")
    test_request = {
        "input_csv": os.path.abspath(test_csv),
        "output_dir": "./test_output",
        "nl_query": "Give me an overview of the apm application metrics."
    }
    
    try:
        response = requests.post(f"{base_url}/process", json=test_request, timeout=60)
        result = response.json()
        print(f"Process result: {json.dumps(result, indent=2)}")
        
        if result.get("status") == "success":
            print("✅ Test passed!")
        else:
            print("❌ Test failed!")
            
    except Exception as e:
        print(f"Process test failed: {e}")
    

if __name__ == "__main__":
    test_agent()