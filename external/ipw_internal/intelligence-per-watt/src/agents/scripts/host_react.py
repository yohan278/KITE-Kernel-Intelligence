from click import command, option
from agno.models.google import Gemini

from agents.serve import AgentServer
from agents import React


@command("host-react")
@option("--model", type=str, required=True, help="The model to use.")
@option("--api-key", type=str, required=True, help="The API key to use.")
def host_react(model_name: str, api_key: str):
    """Host the React orchestrator."""
    model = Gemini(id=model_name, api_key=api_key)

    orchestrater = React(model, [])
    server = AgentServer(orchestrater)

    server.serve()


if __name__ == "__main__":
    host_react()
