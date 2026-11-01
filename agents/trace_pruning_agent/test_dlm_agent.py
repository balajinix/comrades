import requests
import json
import uuid
import time

DLM_AGENT_BASE_URL = "http://localhost:8004"  # DLM Agent runs on port 8004

def generate_rpc_id() -> str:
    return str(uuid.uuid4())

def test_dlm_agent_a2a():
    """Tests the DLM Agent using A2A protocol"""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # 1. Get Agent Card
    print(f"--- Testing {DLM_AGENT_BASE_URL}/.well-known/agent.json ---")
    try:
        response = requests.get(f"{DLM_AGENT_BASE_URL}/.well-known/agent.json", timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        agent_card = response.json()
        print("Agent Card retrieved successfully:")
        print(json.dumps(agent_card, indent=2))
        if not agent_card.get("name") == "Data Lifecycle Management Agent (DLMAgent)":
            print("WARNING: Agent name might not be as expected.")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not get agent card: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not parse agent card JSON: {e}")
        print(f"Response text: {response.text if response else 'No response object'}")
        return
    print("-" * 70)

    # 2. Send a message (create a task)
    print(f"\n--- Testing {DLM_AGENT_BASE_URL}/ (message/send) ---")
    user_query = "get the allocation data for last 7 days"

    client_suggested_task_id = generate_rpc_id() # ID the client suggests for the task
    message_id = generate_rpc_id()
    context_id = generate_rpc_id()

    message_send_payload = {
        "jsonrpc": "2.0",
        "id": generate_rpc_id(), # RPC call ID for this specific request
        "method": "message/send",
        "params": {
            "id": client_suggested_task_id, # Client's suggestion for the task's ID
            "message": {
                "role": "user",
                "messageId": message_id,
                "contextId": context_id,
                "parts": [{"type": "text", "text": user_query}]
            },
            "configuration": {"acceptedOutputModes": ["text"]}
        }
    }

    task_info = None
    server_assigned_task_id = None
    try:
        print(f"Sending query to DLM agent: \"{user_query}\" (Client suggested Task ID: {client_suggested_task_id})")
        response = requests.post(DLM_AGENT_BASE_URL, json=message_send_payload, headers=headers, timeout=45) # Increased timeout
        response.raise_for_status()
        rpc_response = response.json()
        print("Response from message/send:")
        print(json.dumps(rpc_response, indent=2))

        if "error" in rpc_response:
            print(f"ERROR in message/send RPC response: {rpc_response['error']}")
            return
        if "result" not in rpc_response or not isinstance(rpc_response["result"], dict):
            print("ERROR: 'result' field missing or not a dictionary in message/send RPC response.")
            return

        task_info = rpc_response["result"]
        server_assigned_task_id = task_info.get("id")

        if not server_assigned_task_id:
            print("ERROR: Server response for message/send is missing a 'result.id' for the task.")
            return

        if server_assigned_task_id != client_suggested_task_id:
            print(f"WARNING: Task ID mismatch. Client suggested '{client_suggested_task_id}', but server assigned '{server_assigned_task_id}'.")
            print(f"Proceeding with server-assigned Task ID: {server_assigned_task_id}")
        else:
            print(f"Task created successfully with ID: {server_assigned_task_id} (matches client suggestion).")

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not send message to agent: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not parse message/send response JSON: {e}")
        print(f"Response text: {response.text if response else 'No response object'}")
        return
    print("-" * 70)

    # 3. Poll for task completion (tasks/get)
    if not server_assigned_task_id: # This check is now based on the ID we extracted
        print("Skipping task polling because task creation failed or server did not return a task ID.")
        return

    actual_task_id_to_poll = server_assigned_task_id # Explicitly use the ID from server for polling
    print(f"\n--- Polling task {actual_task_id_to_poll} status using tasks/get ---")
    max_polls = 70  # Increased polling time slightly
    poll_interval = 2

    final_task_data = None
    for i in range(max_polls):
        print(f"Polling attempt {i+1}/{max_polls} for task {actual_task_id_to_poll}...")
        tasks_get_payload = {
            "jsonrpc": "2.0",
            "id": generate_rpc_id(),
            "method": "tasks/get",
            "params": {"id": actual_task_id_to_poll}
        }
        try:
            response = requests.post(DLM_AGENT_BASE_URL, json=tasks_get_payload, headers=headers, timeout=10)
            response.raise_for_status()
            rpc_response = response.json()

            if "error" in rpc_response:
                print(f"ERROR in tasks/get RPC response: {rpc_response['error']}")
                break
            if "result" not in rpc_response or not isinstance(rpc_response["result"], dict):
                print("ERROR: 'result' field missing or invalid in tasks/get RPC response.")
                break

            task_data = rpc_response["result"]
            current_task_state = task_data.get("status", {}).get("state", "UNKNOWN")
            print(f"Current task status: {current_task_state}")

            if current_task_state == "completed":
                print("Task completed!")
                final_task_data = task_data
                break
            elif current_task_state == "failed":
                print("Task failed!")
                error_message_text = "Unknown failure reason."
                # Attempt to extract error message more robustly
                status_block = task_data.get("status", {})
                if isinstance(status_block, dict):
                    message_block = status_block.get("message", {})
                    if isinstance(message_block, dict):
                        parts = message_block.get("parts", [])
                        if parts and isinstance(parts, list) and isinstance(parts[0], dict):
                             part_text = parts[0].get("text")
                             if part_text:
                                error_message_text = part_text
                print(f"Failure reason: {error_message_text}")
                final_task_data = task_data
                break
            elif current_task_state == "input-required":
                 print("Task requires input. This test script doesn't handle this.")
                 final_task_data = task_data
                 break
            elif current_task_state not in ["working", "pending", "queued"]: # Add any other transient states
                 print(f"Task is in an unexpected or unhandled state: {current_task_state}")


        except requests.exceptions.RequestException as e:
            print(f"ERROR: Could not poll task status: {e}")
            break
        except json.JSONDecodeError as e:
            print(f"ERROR: Could not parse tasks/get response JSON: {e}")
            print(f"Response text: {response.text if response else 'No response object'}")
            break

        time.sleep(poll_interval)
    else: # Loop finished without break (polling timeout)
        print(f"Polling timeout: Task {actual_task_id_to_poll} did not reach 'completed' or 'failed' state after {max_polls * poll_interval} seconds.")

    if final_task_data:
        print("\nFinal task data:")
        print(json.dumps(final_task_data, indent=2))
        if final_task_data.get("status", {}).get("state") == "completed":
            print("\nExtracted Artifacts (if any):")
            artifacts = final_task_data.get("artifacts", [])
            if not isinstance(artifacts, list): artifacts = [] 

            found_text_artifact = False
            for artifact_item in artifacts:
                if not isinstance(artifact_item, dict): continue
                artifact_parts = artifact_item.get("parts", [])
                if not isinstance(artifact_parts, list): continue

                for part in artifact_parts:
                    if isinstance(part, dict) and part.get("kind") == "text":
                        print(f"  Artifact Name: {artifact_item.get('name')}")
                        print(f"  Text Content: {part.get('text')}")
                        found_text_artifact = True
            if not found_text_artifact:
                print("  No text artifacts found in the completed task.")
    print("-" * 70)

if __name__ == "__main__":
    print("Starting DLM Agent A2A Test Script...")
    print("Ensure the DLM agent is running and its dependencies (MCPs, RITS if used) are accessible.")
    test_dlm_agent_a2a()