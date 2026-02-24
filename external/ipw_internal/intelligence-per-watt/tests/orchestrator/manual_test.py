"""Manual tests for MCP servers (no pytest required)."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.mcp import CalculatorServer, MCPToolResult


def test_calculator():
    """Test calculator server."""
    print("Testing CalculatorServer...")

    calc = CalculatorServer()

    # Test 1: Simple addition
    result = calc.execute("2 + 2")
    assert result.content == "4", f"Expected '4', got '{result.content}'"
    print(f"✓ Simple addition: 2 + 2 = {result.content}")

    # Test 2: Complex expression
    result = calc.execute("(10 + 5) * 2 - 3")
    assert result.content == "27", f"Expected '27', got '{result.content}'"
    print(f"✓ Complex expression: (10 + 5) * 2 - 3 = {result.content}")

    # Test 3: Natural language
    result = calc.execute("what is 100 / 4?")
    assert result.content == "25.0", f"Expected '25.0', got '{result.content}'"
    print(f"✓ Natural language: what is 100 / 4? = {result.content}")

    # Test 4: Functions
    result = calc.execute("sqrt(16) + abs(-5)")
    assert result.content == "9.0", f"Expected '9.0', got '{result.content}'"
    print(f"✓ Functions: sqrt(16) + abs(-5) = {result.content}")

    # Test 5: Check telemetry
    assert isinstance(result.telemetry_samples, list), "telemetry_samples should be list"
    assert result.latency_seconds > 0, "latency_seconds should be positive"
    assert result.cost_usd == 0.0, "cost_usd should be 0.0 for calculator"
    print(f"✓ Telemetry captured: {len(result.telemetry_samples)} samples, {result.latency_seconds:.6f}s latency")

    print("\n✅ All calculator tests passed!\n")


def test_mcp_tool_result():
    """Test MCPToolResult dataclass."""
    print("Testing MCPToolResult...")

    # Test creation with minimal fields
    result = MCPToolResult(content="test")
    assert result.content == "test"
    assert result.usage == {}
    assert result.cost_usd is None
    print("✓ Minimal MCPToolResult created")

    # Test creation with all fields
    result = MCPToolResult(
        content="response",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        cost_usd=0.005,
        latency_seconds=1.5,
        ttft_seconds=0.2,
        metadata={"model": "test"},
    )
    assert result.content == "response"
    assert result.usage["total_tokens"] == 30
    assert result.cost_usd == 0.005
    print("✓ Full MCPToolResult created")

    print("\n✅ All MCPToolResult tests passed!\n")


def main():
    """Run all manual tests."""
    print("=" * 60)
    print("MCP Server Manual Tests")
    print("=" * 60)
    print()

    try:
        test_mcp_tool_result()
        test_calculator()

        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
