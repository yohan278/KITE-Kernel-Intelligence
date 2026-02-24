"""Unit tests for MCP servers."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from agents.mcp import (
    BaseMCPServer,
    MCPToolResult,
    CalculatorServer,
    OllamaMCPServer,
    OpenAIMCPServer,
    AnthropicMCPServer,
    OpenRouterMCPServer,
)


class TestCalculatorServer:
    """Test calculator server (no external dependencies)."""

    def test_simple_addition(self):
        """Test basic addition."""
        calc = CalculatorServer()
        result = calc.execute("2 + 2")

        assert result.content == "4"
        assert result.cost_usd == 0.0
        assert result.usage["total_tokens"] == 0

    def test_complex_expression(self):
        """Test complex mathematical expression."""
        calc = CalculatorServer()
        result = calc.execute("(10 + 5) * 2 - 3")

        assert result.content == "27"

    def test_natural_language_prompt(self):
        """Test extraction from natural language."""
        calc = CalculatorServer()
        result = calc.execute("what is 100 / 4?")

        assert result.content == "25.0"

    def test_functions(self):
        """Test mathematical functions."""
        calc = CalculatorServer()
        result = calc.execute("sqrt(16) + abs(-5)")

        assert result.content == "9.0"

    def test_invalid_expression(self):
        """Test error handling for invalid expressions."""
        calc = CalculatorServer()
        result = calc.execute("invalid expression !!!")

        assert "Error:" in result.content

    def test_telemetry_captured(self):
        """Test that telemetry samples are captured."""
        calc = CalculatorServer()
        result = calc.execute("1 + 1")

        # Telemetry samples should be empty or populated (depends on energy monitor availability)
        assert isinstance(result.telemetry_samples, list)
        assert result.latency_seconds > 0.0


class TestOllamaMCPServer:
    """Test Ollama server (mocked)."""

    def test_server_initialization(self):
        """Test server can be initialized."""
        server = OllamaMCPServer(model_name="llama3.2:1b", base_url="http://localhost:11434")

        assert server.model_name == "llama3.2:1b"
        assert server.name == "ollama:llama3.2:1b"

    @patch("agents.mcp.ollama_server.Client")
    def test_execute_generates_response(self, mock_client_class):
        """Test execute generates response (mocked)."""
        # Mock Ollama client
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock streaming response
        mock_chunk_1 = Mock()
        mock_chunk_1.response = "Hello"
        mock_chunk_1.done = False

        mock_chunk_2 = Mock()
        mock_chunk_2.response = " world"
        mock_chunk_2.done = False

        mock_chunk_3 = Mock()
        mock_chunk_3.response = None
        mock_chunk_3.done = True
        mock_chunk_3.prompt_eval_count = 10
        mock_chunk_3.eval_count = 5

        mock_client.generate.return_value = [mock_chunk_1, mock_chunk_2, mock_chunk_3]

        # Create server and execute
        server = OllamaMCPServer(model_name="llama3.2:1b")
        result = server.execute("test prompt")

        assert result.content == "Hello world"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.cost_usd == 0.0  # Local model, no cost


class TestOpenAIMCPServer:
    """Test OpenAI server (mocked)."""

    def test_cost_calculation(self):
        """Test cost calculation for GPT-4o."""
        server = OpenAIMCPServer(model_name="gpt-4o", api_key="test_key")

        cost = server._calculate_cost(prompt_tokens=1000, completion_tokens=500)

        # gpt-4o: $2.50/1M input, $10.00/1M output
        expected_cost = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert abs(cost - expected_cost) < 0.0001

    def test_pricing_fallback(self):
        """Test fallback to default pricing for unknown model."""
        server = OpenAIMCPServer(model_name="unknown-model", api_key="test_key")

        # Should use gpt-4o pricing as fallback
        assert server.pricing["input"] == 2.50
        assert server.pricing["output"] == 10.00


class TestAnthropicMCPServer:
    """Test Anthropic server (mocked)."""

    def test_cost_calculation(self):
        """Test cost calculation for Claude Sonnet 4.5."""
        server = AnthropicMCPServer(
            model_name="claude-sonnet-4-5-20250929",
            api_key="test_key"
        )

        cost = server._calculate_cost(input_tokens=2000, output_tokens=1000)

        # Sonnet 4.5: $3.00/1M input, $15.00/1M output
        expected_cost = (2000 / 1_000_000) * 3.00 + (1000 / 1_000_000) * 15.00
        assert abs(cost - expected_cost) < 0.0001


class TestOpenRouterMCPServer:
    """Test OpenRouter server (mocked)."""

    # Models from trajectory_generation_moonlight_50k.yaml config
    CONFIG_MODELS = [
        "qwen/qwen-math-72b",
        "qwen/qwen-coder-32b",
        "deepseek/deepseek-coder",
    ]

    def test_server_initialization(self):
        """Test server can be initialized with test API key."""
        server = OpenRouterMCPServer(
            model_name="qwen/qwen-math-72b",
            api_key="test_key"
        )

        assert server.model_name == "qwen/qwen-math-72b"
        assert server.name == "openrouter:qwen/qwen-math-72b"

    @pytest.mark.parametrize("model", CONFIG_MODELS)
    def test_initialization_all_config_models(self, model):
        """Test initialization for each model in the config."""
        server = OpenRouterMCPServer(model_name=model, api_key="test_key")

        assert server.model_name == model
        assert server.name == f"openrouter:{model}"

    def test_cost_calculation_math_model(self):
        """Test cost calculation for qwen/qwen-math-72b."""
        server = OpenRouterMCPServer(
            model_name="qwen/qwen-math-72b",
            api_key="test_key"
        )

        cost = server._calculate_cost(prompt_tokens=1000, completion_tokens=500)

        # qwen/qwen-math-72b: $0.40/1M input, $0.40/1M output
        expected_cost = (1000 / 1_000_000) * 0.40 + (500 / 1_000_000) * 0.40
        assert abs(cost - expected_cost) < 0.0001

    def test_cost_calculation_coder_model(self):
        """Test cost calculation for qwen/qwen-coder-32b."""
        server = OpenRouterMCPServer(
            model_name="qwen/qwen-coder-32b",
            api_key="test_key"
        )

        cost = server._calculate_cost(prompt_tokens=2000, completion_tokens=1000)

        # qwen/qwen-coder-32b: $0.20/1M input, $0.20/1M output
        expected_cost = (2000 / 1_000_000) * 0.20 + (1000 / 1_000_000) * 0.20
        assert abs(cost - expected_cost) < 0.0001

    def test_cost_calculation_deepseek_coder(self):
        """Test cost calculation for deepseek/deepseek-coder."""
        server = OpenRouterMCPServer(
            model_name="deepseek/deepseek-coder",
            api_key="test_key"
        )

        cost = server._calculate_cost(prompt_tokens=5000, completion_tokens=2000)

        # deepseek/deepseek-coder: $0.14/1M input, $0.28/1M output
        expected_cost = (5000 / 1_000_000) * 0.14 + (2000 / 1_000_000) * 0.28
        assert abs(cost - expected_cost) < 0.0001

    def test_pricing_fallback_unknown_model(self):
        """Test fallback to default pricing for unknown model."""
        server = OpenRouterMCPServer(
            model_name="unknown/unknown-model",
            api_key="test_key"
        )

        # Should use default pricing
        assert server.pricing == {"input": 1.00, "output": 3.00}

    def test_api_key_required(self):
        """Test that API key is required."""
        # Ensure env var is not set for this test
        import os
        original = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key required"):
                OpenRouterMCPServer(model_name="qwen/qwen-math-72b")
        finally:
            if original:
                os.environ["OPENROUTER_API_KEY"] = original

    def test_api_key_from_env(self):
        """Test API key can be loaded from environment variable."""
        import os
        os.environ["OPENROUTER_API_KEY"] = "env_test_key"
        try:
            server = OpenRouterMCPServer(model_name="qwen/qwen-math-72b")
            assert server._client.api_key == "env_test_key"
        finally:
            os.environ.pop("OPENROUTER_API_KEY", None)

    @patch("agents.mcp.openrouter_server.OpenAI")
    def test_health_check_success(self, mock_openai_class):
        """Test health_check returns True on success."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        server = OpenRouterMCPServer(
            model_name="qwen/qwen-math-72b",
            api_key="test_key"
        )
        result = server.health_check()

        assert result is True
        mock_client.chat.completions.create.assert_called_once()

    @patch("agents.mcp.openrouter_server.OpenAI")
    def test_health_check_failure(self, mock_openai_class):
        """Test health_check returns False on failure."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_client.chat.completions.create.side_effect = Exception("API error")

        server = OpenRouterMCPServer(
            model_name="qwen/qwen-math-72b",
            api_key="test_key"
        )
        result = server.health_check()

        assert result is False

    @patch("agents.mcp.openrouter_server.OpenAI")
    def test_execute_with_streaming_response(self, mock_openai_class):
        """Test execute generates response with streaming."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock streaming response chunks
        mock_chunk_1 = Mock()
        mock_chunk_1.choices = [Mock(delta=Mock(content="Hello"))]
        mock_chunk_1.usage = None

        mock_chunk_2 = Mock()
        mock_chunk_2.choices = [Mock(delta=Mock(content=" world"))]
        mock_chunk_2.usage = None

        # Final chunk with usage info
        mock_chunk_3 = Mock()
        mock_chunk_3.choices = [Mock(delta=Mock(content="!"))]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_chunk_3.usage = mock_usage

        mock_client.chat.completions.create.return_value = iter([
            mock_chunk_1, mock_chunk_2, mock_chunk_3
        ])

        server = OpenRouterMCPServer(
            model_name="qwen/qwen-math-72b",
            api_key="test_key"
        )
        result = server.execute("test prompt")

        assert result.content == "Hello world!"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.usage["total_tokens"] == 15
        # Cost should be calculated
        expected_cost = (10 / 1_000_000) * 0.40 + (5 / 1_000_000) * 0.40
        assert abs(result.cost_usd - expected_cost) < 0.0001


@pytest.mark.integration
class TestOpenRouterIntegration:
    """Integration tests for OpenRouter (requires API key)."""

    CONFIG_MODELS = [
        "qwen/qwen-math-72b",
        "qwen/qwen-coder-32b",
        "deepseek/deepseek-coder",
    ]

    @pytest.fixture
    def api_key(self):
        """Get API key from environment."""
        import os
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            pytest.skip("OPENROUTER_API_KEY not set")
        return key

    @pytest.mark.parametrize("model", CONFIG_MODELS)
    def test_health_check_live(self, api_key, model):
        """Test health_check with live API for each config model."""
        server = OpenRouterMCPServer(model_name=model, api_key=api_key)
        result = server.health_check()

        assert result is True, f"Health check failed for {model}"

    def test_execute_live(self, api_key):
        """Test live execute with a simple prompt."""
        server = OpenRouterMCPServer(
            model_name="qwen/qwen-math-72b",
            api_key=api_key,
            max_tokens=50
        )
        result = server.execute("What is 2 + 2?")

        assert result.content is not None
        assert len(result.content) > 0
        assert result.cost_usd > 0
        assert result.usage["total_tokens"] > 0


class TestMCPToolResult:
    """Test MCPToolResult dataclass."""

    def test_result_creation(self):
        """Test creating result with minimal fields."""
        result = MCPToolResult(content="test")

        assert result.content == "test"
        assert result.usage == {}
        assert result.cost_usd is None
        assert result.telemetry_samples == []
        assert result.latency_seconds == 0.0

    def test_result_with_all_fields(self):
        """Test creating result with all fields."""
        result = MCPToolResult(
            content="response",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            cost_usd=0.005,
            telemetry_samples=[],
            latency_seconds=1.5,
            ttft_seconds=0.2,
            metadata={"model": "test"},
        )

        assert result.content == "response"
        assert result.usage["total_tokens"] == 30
        assert result.cost_usd == 0.005
        assert result.latency_seconds == 1.5
        assert result.ttft_seconds == 0.2
        assert result.metadata["model"] == "test"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
