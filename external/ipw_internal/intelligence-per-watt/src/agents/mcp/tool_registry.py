"""Unified Tool Registry for ToolOrchestra + ADP tools.

Consolidates all tools from:
- ToolOrchestra: calculator, think, code_interpreter, web_search, LLM backends
- ADP domains: codeact actions, alfworld actions, mind2web actions, etc.

Provides MCP server management and tool discovery for the orchestrator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from .base import BaseMCPServer, MCPToolResult


class ToolCategory(Enum):
    """Categories of tools matching ToolOrchestra + ADP domains."""
    
    # ToolOrchestra core tools
    UTILITY = "utility"           # calculator, think
    CODE = "code"                 # code_interpreter, code execution
    SEARCH = "search"             # web_search, retrieval
    
    # LLM backends (by size/capability)
    LLM_SMALL = "llm_small"       # <3B params: fast, cheap
    LLM_MEDIUM = "llm_medium"     # 3-10B params: balanced
    LLM_LARGE = "llm_large"       # >10B params: high quality
    LLM_SPECIALIST = "llm_specialist"  # domain-specific (math, code)
    LLM_CLOUD = "llm_cloud"       # cloud APIs (OpenAI, Anthropic)
    
    # ADP domain-specific actions
    ADP_CODEACT = "adp_codeact"   # code execution actions
    ADP_ALFWORLD = "adp_alfworld" # household task actions
    ADP_MIND2WEB = "adp_mind2web" # web navigation actions
    ADP_DATABASE = "adp_database" # database query actions


@dataclass
class ToolSpec:
    """Specification for a tool in the registry."""
    
    name: str
    """Unique tool identifier (e.g., 'calculator', 'ollama:qwen2.5:1.5b')"""
    
    category: ToolCategory
    """Tool category for routing decisions"""
    
    description: str
    """Human-readable description for policy model"""
    
    server_class: Optional[Type[BaseMCPServer]] = None
    """MCP server class to instantiate"""
    
    factory: Optional[Callable[..., BaseMCPServer]] = None
    """Factory function for custom initialization"""
    
    # Cost/efficiency metadata (for routing decisions)
    estimated_latency_ms: float = 0.0
    """Estimated latency in milliseconds"""
    
    estimated_cost_usd: float = 0.0
    """Estimated cost per call in USD"""
    
    estimated_energy_joules: float = 0.0
    """Estimated energy consumption per call"""
    
    requires_api_key: Optional[str] = None
    """Environment variable name for required API key"""
    
    requires_server: Optional[str] = None
    """Required server (e.g., 'ollama', 'vllm')"""
    
    # ADP domain mapping
    adp_domains: List[str] = field(default_factory=list)
    """ADP domains this tool is relevant for"""
    
    # Capability tags for semantic matching
    capabilities: List[str] = field(default_factory=list)
    """Capability tags (e.g., 'math', 'code', 'reasoning')"""



class ToolRegistry:
    """Unified registry for all ToolOrchestra + ADP tools.
    
    Example:
        registry = ToolRegistry()
        registry.discover_tools()  # Auto-discover available tools
        
        # Get all available tools
        tools = registry.get_available_tools()
        
        # Get tools by category
        small_llms = registry.get_tools_by_category(ToolCategory.LLM_SMALL)
        
        # Get tool instance
        calc = registry.get_tool_instance("calculator")
        result = calc.execute("2 + 2")
    """
    
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        vllm_base_url: str = "http://localhost:8000",
        telemetry_collector: Optional[Any] = None,
        code_isolation: Optional[str] = "auto",
        retrieval_gpu_device: Optional[int] = None,
    ):
        """Initialize tool registry.

        Args:
            ollama_base_url: Base URL for Ollama server
            vllm_base_url: Base URL for vLLM server
            telemetry_collector: Energy monitor collector for all tools
            code_isolation: Isolation mode for code_interpreter tool.
                Options: None (no isolation), "bubblewrap", or "auto" (default).
            retrieval_gpu_device: GPU device index for neural retrieval models.
                If None, uses CPU. Set to a different GPU than the LM to avoid
                contention (e.g., if LM uses GPU 0-3, set this to 4).
        """
        self.ollama_base_url = ollama_base_url
        self.vllm_base_url = vllm_base_url
        self.telemetry_collector = telemetry_collector
        self.code_isolation = code_isolation
        self.retrieval_gpu_device = retrieval_gpu_device

        self._specs: Dict[str, ToolSpec] = {}
        self._instances: Dict[str, BaseMCPServer] = {}
        self._aliases: Dict[str, str] = {}  # Maps alias names to target tool names

        # Register all known tools
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Register all built-in tools from ToolOrchestra + ADP."""
        
        # === UTILITY TOOLS (ToolOrchestra) ===
        self.register(ToolSpec(
            name="calculator",
            category=ToolCategory.UTILITY,
            description="Evaluate mathematical expressions. Supports arithmetic, exponents, trig functions. Zero cost, instant.",
            estimated_latency_ms=1,
            estimated_cost_usd=0.0,
            capabilities=["math", "arithmetic", "computation"],
            adp_domains=["agenttuning_db", "agenttuning_kg"],
        ))
        
        self.register(ToolSpec(
            name="think",
            category=ToolCategory.UTILITY,
            description="Internal reasoning scratchpad. Break down complex problems step-by-step before delegating. Zero cost.",
            estimated_latency_ms=1,
            estimated_cost_usd=0.0,
            capabilities=["reasoning", "planning", "decomposition"],
            adp_domains=["codeact", "agenttuning_alfworld"],
        ))
        
        # === CODE TOOLS ===
        self.register(ToolSpec(
            name="code_interpreter",
            category=ToolCategory.CODE,
            description="Execute Python code in sandbox. Returns stdout/stderr. 30s timeout. Use for computation, data analysis.",
            estimated_latency_ms=1000,
            estimated_cost_usd=0.00001,
            capabilities=["code_execution", "python", "computation"],
            adp_domains=["codeact", "code_feedback", "swe-smith"],
        ))

        self.register(ToolSpec(
            name="file_read",
            category=ToolCategory.CODE,
            description="Read file contents. Supports line ranges. Zero cost, instant.",
            estimated_latency_ms=10,
            estimated_cost_usd=0.0,
            capabilities=["file_operations", "code_analysis"],
            adp_domains=["codeact", "swe-smith"],
        ))

        self.register(ToolSpec(
            name="file_write",
            category=ToolCategory.CODE,
            description="Write content to file. Supports write and append modes. Zero cost, instant.",
            estimated_latency_ms=10,
            estimated_cost_usd=0.0,
            capabilities=["file_operations", "code_generation"],
            adp_domains=["codeact", "swe-smith"],
        ))
        
        # === SEARCH TOOLS ===
        self.register(ToolSpec(
            name="web_search",
            category=ToolCategory.SEARCH,
            description="Search the web via Tavily API. Get current info, news, facts. Cost: $0.01/search.",
            estimated_latency_ms=500,
            estimated_cost_usd=0.01,
            requires_api_key="TAVILY_API_KEY",
            capabilities=["search", "retrieval", "current_info"],
            adp_domains=["agenttuning_webshop", "mind2web", "go-browse-wa"],
        ))

        # === RETRIEVAL TOOLS ===
        self.register(ToolSpec(
            name="retrieval:grep",
            category=ToolCategory.SEARCH,
            description="Fast regex/keyword search. No indexing, ~1ms latency.",
            estimated_latency_ms=1,
            estimated_cost_usd=0.0,
            capabilities=["search", "retrieval", "keyword_search", "regex"],
        ))

        self.register(ToolSpec(
            name="retrieval:bm25",
            category=ToolCategory.SEARCH,
            description="BM25 sparse retrieval. Fast, CPU-only, ~10ms latency.",
            estimated_latency_ms=10,
            estimated_cost_usd=0.0,
            capabilities=["search", "retrieval", "keyword_search"],
        ))

        self.register(ToolSpec(
            name="retrieval:dense",
            category=ToolCategory.SEARCH,
            description="Dense neural retrieval with FAISS. Semantic search, ~50ms.",
            estimated_latency_ms=50,
            estimated_cost_usd=0.0,
            capabilities=["search", "retrieval", "semantic_search"],
        ))

        self.register(ToolSpec(
            name="retrieval:hybrid",
            category=ToolCategory.SEARCH,
            description="Hybrid BM25 + Qwen3-Embedding-4B dense retrieval with RRF fusion. Best accuracy, ~100ms.",
            estimated_latency_ms=100,
            estimated_cost_usd=0.0,
            capabilities=["search", "retrieval", "semantic_search", "keyword_search"],
        ))

        # === SMALL LLMs (<3B) via Ollama ===
        small_llms = [
            ("ollama:qwen2.5:0.5b", "Qwen2.5 0.5B - Fastest, basic tasks", 300, ["basic_qa"]),
            ("ollama:qwen2.5:1.5b", "Qwen2.5 1.5B - Fast, simple reasoning", 800, ["simple_reasoning"]),
            ("ollama:qwen3:1.5b", "Qwen3 1.5B - Fast reasoning with Qwen3 architecture", 800, ["simple_reasoning"]),
            ("ollama:llama3.2:1b", "Llama3.2 1B - Fast, general tasks", 500, ["basic_qa"]),
        ]
        for name, desc, latency, caps in small_llms:
            self.register(ToolSpec(
                name=name,
                category=ToolCategory.LLM_SMALL,
                description=desc,
                estimated_latency_ms=latency,
                estimated_cost_usd=0.0,
                requires_server="ollama",
                capabilities=caps + ["text_generation"],
            ))
        
        # === MEDIUM LLMs (3-10B) via Ollama ===
        medium_llms = [
            ("ollama:qwen2.5:3b", "Qwen2.5 3B - Balanced speed/quality", 1500, ["reasoning"]),
            ("ollama:qwen2.5:7b", "Qwen2.5 7B - Good reasoning", 3000, ["reasoning", "complex_qa"]),
            ("ollama:llama3.2:3b", "Llama3.2 3B - Balanced, general", 1500, ["reasoning"]),
            ("vllm:qwen3-8b", "Qwen3 8B - High quality reasoning", 2000, ["complex_reasoning"]),
            ("vllm:llama-8b", "Llama3.1 8B - Strong general model", 2000, ["complex_reasoning"]),
        ]
        for name, desc, latency, caps in medium_llms:
            self.register(ToolSpec(
                name=name,
                category=ToolCategory.LLM_MEDIUM,
                description=desc,
                estimated_latency_ms=latency,
                estimated_cost_usd=0.0,
                requires_server="ollama",
                capabilities=caps + ["text_generation"],
            ))
        
        # === LARGE LLMs (>10B) via vLLM ===
        large_llms = [
            ("vllm:qwen3-32b", "Qwen3 32B - Best open-source quality", 5000, ["complex_reasoning", "analysis"]),
            ("vllm:llama-70b", "Llama3.1 70B - Near SOTA quality", 8000, ["complex_reasoning", "analysis"]),
        ]
        for name, desc, latency, caps in large_llms:
            self.register(ToolSpec(
                name=name,
                category=ToolCategory.LLM_LARGE,
                description=desc,
                estimated_latency_ms=latency,
                estimated_cost_usd=0.0,
                requires_server="vllm",
                capabilities=caps + ["text_generation"],
            ))
        
        # === SPECIALIST LLMs via vLLM ===
        specialist_llms = [
            ("vllm:qwen-math-7b", "Qwen Math 7B - Math specialist", 2000, ["math", "problem_solving"]),
            ("vllm:glm-4.7", "GLM-4.7 - Best math model", 8000, ["math", "complex_math"]),
            ("vllm:qwen-coder-7b", "Qwen Coder 7B - Code specialist", 2000, ["code", "programming"]),
            ("vllm:qwen3-coder-plus", "Qwen3 Coder Plus - Best code model", 5000, ["code", "complex_code"]),
        ]
        for name, desc, latency, caps in specialist_llms:
            self.register(ToolSpec(
                name=name,
                category=ToolCategory.LLM_SPECIALIST,
                description=desc,
                estimated_latency_ms=latency,
                estimated_cost_usd=0.0,
                requires_server="vllm",
                capabilities=caps + ["text_generation"],
                adp_domains=["codeact", "code_feedback"] if "code" in caps else [],
            ))

        # === LLM ALIASES (consistent naming for energy study) ===
        # These provide standardized names that map to specific backend models
        llm_aliases = [
            ("llm_small", "vllm:qwen3-1.5b", "Small LLM for fast reasoning (Qwen3 1.5B)",
             ToolCategory.LLM_SMALL, 100, ["simple_reasoning"]),
            ("llm_medium", "vllm:qwen3-8b", "Medium LLM for balanced tasks (Qwen3 8B)",
             ToolCategory.LLM_MEDIUM, 300, ["reasoning"]),
            ("llm_large", "vllm:qwen3-32b", "Large LLM for complex reasoning (Qwen3 32B)",
             ToolCategory.LLM_LARGE, 1000, ["complex_reasoning"]),
            ("llm_specialist", "vllm:qwen-coder-32b", "Specialist LLM for code (Qwen Coder 32B)",
             ToolCategory.LLM_SPECIALIST, 1000, ["code", "programming"]),
        ]
        for alias_name, target, desc, category, latency, caps in llm_aliases:
            self.register(ToolSpec(
                name=alias_name,
                category=category,
                description=desc,
                estimated_latency_ms=latency,
                estimated_cost_usd=0.0,
                requires_server="vllm",
                capabilities=caps + ["text_generation"],
            ))
            # Store alias mapping for _create_instance
            self._aliases[alias_name] = target

        # === CLOUD LLMs ===
        cloud_llms = [
            # OpenAI - Legacy models
            ("openai:gpt-5-mini-2025-08-07", "GPT-5 Mini - Fast, capable cloud", 800, 0.005, ["reasoning"], "OPENAI_API_KEY"),
            ("openai:gpt-4o", "GPT-4o - Best GPT-4 model", 1000, 0.0025, ["complex_reasoning"], "OPENAI_API_KEY"),
            ("openai:o1-mini", "o1-mini - Reasoning model", 2000, 0.003, ["deep_reasoning"], "OPENAI_API_KEY"),
            ("openai:o1", "o1 - Best reasoning model", 5000, 0.015, ["deep_reasoning"], "OPENAI_API_KEY"),
            # OpenAI - GPT-5 series
            ("openai:gpt-5.2-2025-12-11", "GPT-5.2 - Most capable OpenAI model", 2000, 0.03, ["complex_reasoning", "analysis"], "OPENAI_API_KEY"),
            ("openai:gpt-5-mini-2025-08-07", "GPT-5 Mini - Fast, capable", 800, 0.005, ["reasoning", "fast"], "OPENAI_API_KEY"),
            ("openai:gpt-5-nano-2025-08-07", "GPT-5 Nano - Fastest, cheapest", 400, 0.001, ["basic_reasoning", "fast"], "OPENAI_API_KEY"),
            # Anthropic - Legacy models
            ("anthropic:claude-3-5-haiku-20241022", "Claude 3.5 Haiku - Fast, cheap", 400, 0.0008, ["reasoning"], "ANTHROPIC_API_KEY"),
            ("anthropic:claude-sonnet-4-20250514", "Claude Sonnet 4 - Balanced", 800, 0.003, ["complex_reasoning"], "ANTHROPIC_API_KEY"),
            ("anthropic:claude-opus-4-20250514", "Claude Opus 4 - Most capable", 2000, 0.015, ["complex_reasoning", "analysis"], "ANTHROPIC_API_KEY"),
            # Anthropic - Claude 4.5 series
            ("anthropic:claude-haiku-4-5-20251001", "Claude 4.5 Haiku - Fast, cheap", 300, 0.001, ["reasoning", "fast"], "ANTHROPIC_API_KEY"),
            ("anthropic:claude-sonnet-4-5-20250929", "Claude 4.5 Sonnet - Balanced quality/speed", 600, 0.004, ["complex_reasoning"], "ANTHROPIC_API_KEY"),
            ("anthropic:claude-opus-4-5-20251101", "Claude 4.5 Opus - Most capable Anthropic model", 1500, 0.02, ["complex_reasoning", "analysis", "deep_reasoning"], "ANTHROPIC_API_KEY"),
        ]
        for name, desc, latency, cost, caps, api_key in cloud_llms:
            self.register(ToolSpec(
                name=name,
                category=ToolCategory.LLM_CLOUD,
                description=desc,
                estimated_latency_ms=latency,
                estimated_cost_usd=cost,
                requires_api_key=api_key,
                capabilities=caps + ["text_generation"],
            ))
        
        # === OPENROUTER (access to many models) ===
        openrouter_models = [
            # General models
            ("openrouter:google/gemini-2.5-flash", "Gemini 2.5 Flash via OpenRouter", 500, 0.00015, ["reasoning", "fast"]),
            ("openrouter:google/gemini-2.5-pro", "Gemini 2.5 Pro via OpenRouter", 1000, 0.00125, ["complex_reasoning"]),
            ("openrouter:anthropic/claude-sonnet-4", "Claude Sonnet 4 via OpenRouter", 800, 0.003, ["complex_reasoning"]),
            ("openrouter:openai/gpt-4o", "GPT-4o via OpenRouter", 1000, 0.0025, ["complex_reasoning"]),
            ("openrouter:openai/gpt-5-mini-2025-08-07", "GPT-5 Mini via OpenRouter", 800, 0.005, ["reasoning"]),
            ("openrouter:meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B via OpenRouter", 2000, 0.0004, ["reasoning"]),
            ("openrouter:qwen/qwen-2.5-72b-instruct", "Qwen 2.5 72B via OpenRouter", 2000, 0.00035, ["reasoning"]),
            ("openrouter:qwen/qwq-32b", "QwQ 32B reasoning model via OpenRouter", 3000, 0.00015, ["deep_reasoning"]),
            ("openrouter:deepseek/deepseek-r1", "DeepSeek R1 reasoning via OpenRouter", 3000, 0.00055, ["deep_reasoning"]),
            ("openrouter:deepseek/deepseek-chat-v3-0324", "DeepSeek Chat V3 via OpenRouter", 1000, 0.00014, ["reasoning"]),
            ("openrouter:mistralai/mistral-large-2411", "Mistral Large via OpenRouter", 1500, 0.002, ["reasoning"]),
            # Large LLMs (>10B) - replacing vLLM for cloud-only setup
            ("openrouter:qwen/qwen3-32b", "Qwen3 32B via OpenRouter - Best open-source quality", 3000, 0.0002, ["complex_reasoning", "analysis"]),
            # Specialist models - Math
            ("openrouter:z-ai/glm-4.7", "GLM-4.7 via OpenRouter - Best math model", 4000, 0.0004, ["math", "complex_math", "problem_solving"]),
            # Specialist models - Code
            ("openrouter:qwen/qwen3-coder-plus", "Qwen3 Coder Plus via OpenRouter - Best code model", 3000, 0.0002, ["code", "complex_code", "programming"]),
        ]
        for name, desc, latency, cost, caps in openrouter_models:
            self.register(ToolSpec(
                name=name,
                category=ToolCategory.LLM_CLOUD,
                description=desc,
                estimated_latency_ms=latency,
                estimated_cost_usd=cost,
                requires_api_key="OPENROUTER_API_KEY",
                capabilities=caps + ["text_generation"],
            ))
        
        # === ADP DOMAIN-SPECIFIC TOOLS ===
        # These are meta-tools that map to ADP action types
        self.register(ToolSpec(
            name="adp:codeact",
            category=ToolCategory.ADP_CODEACT,
            description="Execute code actions from ADP codeact domain. Combines reasoning + code execution.",
            capabilities=["code_execution", "reasoning"],
            adp_domains=["codeact"],
        ))
        
        self.register(ToolSpec(
            name="adp:alfworld",
            category=ToolCategory.ADP_ALFWORLD,
            description="Household task actions (go, pick, put, open, close, etc.) from ALFWorld domain.",
            capabilities=["embodied_actions", "planning"],
            adp_domains=["agenttuning_alfworld"],
        ))
        
        self.register(ToolSpec(
            name="adp:mind2web",
            category=ToolCategory.ADP_MIND2WEB,
            description="Web navigation actions (click, type, scroll, etc.) from Mind2Web domain.",
            capabilities=["web_navigation", "ui_interaction"],
            adp_domains=["mind2web", "agenttuning_mind2web"],
        ))
        
        self.register(ToolSpec(
            name="adp:database",
            category=ToolCategory.ADP_DATABASE,
            description="Database query actions from ADP database domain.",
            capabilities=["sql", "query"],
            adp_domains=["agenttuning_db"],
        ))
    
    def register(self, spec: ToolSpec):
        """Register a tool specification.
        
        Args:
            spec: Tool specification to register
        """
        self._specs[spec.name] = spec
    
    def get_spec(self, name: str) -> Optional[ToolSpec]:
        """Get tool specification by name.
        
        Args:
            name: Tool name
            
        Returns:
            ToolSpec if found, None otherwise
        """
        return self._specs.get(name)
    
    def get_all_specs(self) -> List[ToolSpec]:
        """Get all registered tool specifications."""
        return list(self._specs.values())
    
    def get_specs_by_category(self, category: ToolCategory) -> List[ToolSpec]:
        """Get tool specifications by category.
        
        Args:
            category: Tool category to filter by
            
        Returns:
            List of matching tool specifications
        """
        return [s for s in self._specs.values() if s.category == category]
    
    def get_specs_for_domain(self, domain: str) -> List[ToolSpec]:
        """Get tool specifications relevant for an ADP domain.
        
        Args:
            domain: ADP domain name (e.g., 'codeact', 'alfworld')
            
        Returns:
            List of tool specifications relevant for the domain
        """
        return [s for s in self._specs.values() if domain in s.adp_domains]
    
    def get_specs_by_capability(self, capability: str) -> List[ToolSpec]:
        """Get tool specifications by capability tag.
        
        Args:
            capability: Capability tag (e.g., 'math', 'code')
            
        Returns:
            List of matching tool specifications
        """
        return [s for s in self._specs.values() if capability in s.capabilities]
    
    def discover_available_tools(self) -> List[str]:
        """Discover which tools are actually available (servers running, API keys set).
        
        Returns:
            List of available tool names
        """
        available = []
        
        for name, spec in self._specs.items():
            # Check API key requirements
            if spec.requires_api_key:
                if not os.environ.get(spec.requires_api_key):
                    continue
            
            # Check server requirements (simplified - just mark as available for now)
            # Full implementation would ping the servers
            
            available.append(name)
        
        return available
    
    def get_tool_instance(self, name: str) -> Optional[BaseMCPServer]:
        """Get or create a tool instance.
        
        Args:
            name: Tool name
            
        Returns:
            MCP server instance if available, None otherwise
        """
        # Return cached instance if available
        if name in self._instances:
            return self._instances[name]
        
        spec = self._specs.get(name)
        if not spec:
            return None
        
        # Create instance based on tool type
        instance = self._create_instance(name, spec)
        if instance:
            self._instances[name] = instance
        
        return instance
    
    def _create_instance(self, name: str, spec: ToolSpec) -> Optional[BaseMCPServer]:
        """Create a tool instance from specification.

        Args:
            name: Tool name
            spec: Tool specification

        Returns:
            MCP server instance or None
        """
        try:
            # Check for alias first - redirect to the target tool
            if name in self._aliases:
                target_name = self._aliases[name]
                target_spec = self._specs.get(target_name)
                return self._create_instance(target_name, target_spec)

            # Utility tools
            if name == "calculator":
                from .tool_server import CalculatorServer
                return CalculatorServer(telemetry_collector=self.telemetry_collector)
            
            elif name == "think":
                from .tool_server import ThinkServer
                return ThinkServer(telemetry_collector=self.telemetry_collector)
            
            elif name == "code_interpreter":
                from .tool_server import CodeInterpreterServer
                return CodeInterpreterServer(
                    telemetry_collector=self.telemetry_collector,
                    isolation=self.code_isolation,
                )
            
            elif name == "web_search":
                from .tool_server import WebSearchServer
                return WebSearchServer(telemetry_collector=self.telemetry_collector)

            elif name == "file_read":
                from .tool_server import FileReadServer
                return FileReadServer(telemetry_collector=self.telemetry_collector)

            elif name == "file_write":
                from .tool_server import FileWriteServer
                return FileWriteServer(telemetry_collector=self.telemetry_collector)
            
            # Ollama models
            elif name.startswith("ollama:"):
                from .ollama_server import OllamaMCPServer
                model_name = name.split(":", 1)[1]
                return OllamaMCPServer(
                    model=model_name,
                    base_url=self.ollama_base_url,
                    telemetry_collector=self.telemetry_collector,
                )
            
            # vLLM models
            elif name.startswith("vllm:"):
                from .vllm_server import VLLMMCPServer
                model_name = name.split(":", 1)[1]
                return VLLMMCPServer(
                    model_name=model_name,
                    base_url=self.vllm_base_url,
                    telemetry_collector=self.telemetry_collector,
                )
            
            # OpenAI models
            elif name.startswith("openai:"):
                from .openai_server import OpenAIMCPServer
                model_name = name.split(":", 1)[1]
                return OpenAIMCPServer(
                    model_name=model_name,
                    telemetry_collector=self.telemetry_collector,
                )
            
            # Anthropic models
            elif name.startswith("anthropic:"):
                from .anthropic_server import AnthropicMCPServer
                model_name = name.split(":", 1)[1]
                return AnthropicMCPServer(
                    model_name=model_name,
                    telemetry_collector=self.telemetry_collector,
                )
            
            # OpenRouter models
            elif name.startswith("openrouter:"):
                from .openrouter_server import OpenRouterMCPServer
                model_name = name.split(":", 1)[1]
                return OpenRouterMCPServer(
                    model_name=model_name,
                    telemetry_collector=self.telemetry_collector,
                )

            # ADP domain tools - these are passthrough/simulation tools
            elif name.startswith("adp:"):
                return ADPDomainServer(
                    domain=name.split(":", 1)[1],
                    telemetry_collector=self.telemetry_collector,
                )

            # Retrieval tools
            elif name.startswith("retrieval:"):
                retrieval_type = name.split(":", 1)[1]
                if retrieval_type == "grep":
                    from .retrieval import GrepRetrievalServer
                    return GrepRetrievalServer(
                        telemetry_collector=self.telemetry_collector,
                    )
                elif retrieval_type == "bm25":
                    from .retrieval import BM25RetrievalServer
                    return BM25RetrievalServer(
                        telemetry_collector=self.telemetry_collector,
                    )
                elif retrieval_type == "dense":
                    from .retrieval import DenseRetrievalServer
                    # Use separate GPU for retrieval if configured
                    use_gpu = self.retrieval_gpu_device is not None
                    return DenseRetrievalServer(
                        telemetry_collector=self.telemetry_collector,
                        use_gpu=use_gpu,
                        gpu_device=self.retrieval_gpu_device or 0,
                    )
                elif retrieval_type == "hybrid":
                    from .retrieval import HybridRetrievalServer
                    # Use separate GPU for retrieval if configured
                    use_gpu = self.retrieval_gpu_device is not None
                    return HybridRetrievalServer(
                        model_name="Qwen/Qwen3-Embedding-4B",
                        telemetry_collector=self.telemetry_collector,
                        use_gpu=use_gpu,
                        gpu_device=self.retrieval_gpu_device or 0,
                    )

        except ImportError as e:
            print(f"Warning: Could not import server for '{name}': {e}")
        except Exception as e:
            print(f"Warning: Could not create instance for '{name}': {e}")
        
        return None
    
    def get_tool_descriptions(self, tools: Optional[List[str]] = None) -> str:
        """Get formatted tool descriptions for prompting.
        
        Args:
            tools: List of tool names to describe (None = all available)
            
        Returns:
            Formatted string with tool descriptions
        """
        if tools is None:
            tools = self.discover_available_tools()
        
        lines = ["Available tools:"]
        for name in tools:
            spec = self._specs.get(name)
            if spec:
                cost_info = f"${spec.estimated_cost_usd:.4f}" if spec.estimated_cost_usd > 0 else "free"
                lines.append(f"- {name}: {spec.description} ({cost_info}, ~{spec.estimated_latency_ms}ms)")
        
        return "\n".join(lines)


class ADPDomainServer(BaseMCPServer):
    """Passthrough server for ADP domain-specific actions.
    
    These are used for training on ADP trajectories where the actual
    execution is simulated based on the trajectory data.
    """
    
    def __init__(
        self,
        domain: str,
        telemetry_collector: Optional[Any] = None,
    ):
        super().__init__(
            name=f"adp:{domain}",
            telemetry_collector=telemetry_collector,
        )
        self.domain = domain
    
    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Execute ADP domain action (passthrough for training).
        
        For actual execution during inference, this would need to be
        connected to the appropriate environment (e.g., ALFWorld, browser).
        """
        # During training, the actual response comes from the trajectory data
        # This is a passthrough that returns the prompt as acknowledgment
        return MCPToolResult(
            content=f"[ADP:{self.domain}] Action executed: {prompt[:100]}...",
            usage={},
            cost_usd=0.0,
            metadata={
                "tool": f"adp:{self.domain}",
                "domain": self.domain,
                "action": prompt,
            },
        )


# Global registry singleton
_global_registry: Optional[ToolRegistry] = None
_global_registry_kwargs: Dict[str, Any] = {}


def get_registry(**kwargs) -> ToolRegistry:
    """Get or create the global tool registry.

    If kwargs differ from the previously used kwargs, the registry is recreated.
    This ensures the registry is updated when retrieval_gpu_device changes.

    Args:
        **kwargs: Arguments to pass to ToolRegistry constructor

    Returns:
        Global ToolRegistry instance
    """
    global _global_registry, _global_registry_kwargs

    # Check if we need to recreate (kwargs changed)
    if _global_registry is not None and kwargs != _global_registry_kwargs:
        _global_registry = None  # Force recreation

    if _global_registry is None:
        _global_registry = ToolRegistry(**kwargs)
        _global_registry_kwargs = kwargs.copy()

    return _global_registry


def reset_registry() -> None:
    """Reset the global registry singleton.

    Call this when you need to recreate the registry with different parameters.
    """
    global _global_registry, _global_registry_kwargs
    _global_registry = None
    _global_registry_kwargs = {}
