from typing import Callable

import httpx
from a2a.client import A2AClient
from a2a.types import (
    AgentCard,
    SendMessageRequest,
    SendMessageResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from dotenv import load_dotenv

load_dotenv()

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


class RemoteAgentConnections:
    def __init__(self, agent_card: AgentCard, agent_url: str):
        self._httpx_client = httpx.AsyncClient(timeout=120)
        self.agent_client = A2AClient(self._httpx_client, agent_card, url=agent_url)
        self.card = agent_card
        self.agent_url = agent_url
        self.conversation_name = None
        self.conversation = None
        self.pending_tasks = set()

    def get_agent(self) -> AgentCard:
        return self.card

    def get_agent_url(self) -> str:
        return self.agent_url

    async def send_message(self, message_request: SendMessageRequest) -> SendMessageResponse:
        try:
            print(f"Sending message to {self.card.name} at {self.agent_url}")
            print(f"Request ID: {message_request.id}")
            
            response = await self.agent_client.send_message(message_request)
            
            print(f"Response received from {self.card.name}")
            
            return response
            
        except httpx.TimeoutException as e:
            print(f"Timeout sending message to {self.card.name}: {e}")
            print("The agent may still be processing the request")
            raise
            
        except httpx.ConnectError as e:
            print(f"Connection error to {self.card.name} at {self.agent_url}: {e}")
            print("Check if the agent is running and the URL is correct")
            raise
            
        except Exception as e:
            print(f"Error sending message to {self.card.name}: {e}")
            print(f"Error type: {type(e).__name__}")
            raise

    async def close(self):
        if self._httpx_client:
            await self._httpx_client.aclose()
