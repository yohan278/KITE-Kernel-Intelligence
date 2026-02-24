"""Prompt registry — single source of truth for system prompts and tool definitions.

All training, generation, evaluation, and inference code should import from here.

Usage:
    from prompt_registry import AVAILABLE_TOOLS, build_system_prompt
"""

from __future__ import annotations

from typing import Dict, List, Optional


PROMPT_VERSION = "2.0"

SYSTEM_PROMPT_TEMPLATE = """You are an intelligent orchestrator that solves tasks by delegating to the most appropriate tools.

Your job is to SELECT THE BEST TOOL for each task based on the tool's strengths.

=== AVAILABLE TOOLS ===
{tools_description}

=== TOOL SELECTION GUIDE ===
{tool_selection_guide}

=== RESPONSE FORMAT ===
You MUST respond in this EXACT format:

THOUGHT: <analyze the task and explain which tool is best and why>
TOOL: <exact tool name from the list>
INPUT: <input for the tool>

After getting tool results, either use another tool or give final answer:
THOUGHT: <analyze the result>
FINAL_ANSWER: <your final answer>

=== CRITICAL RULES ===
1. You MUST use at least one tool for EVERY task - never answer directly
2. Match the tool to the task type (see guide above)
3. For LLM tools, write clear prompts that will get good responses
4. Prefer specialized tools when available (math specialists for math, code specialists for code)
5. For simple factual questions, use fast/cheap models when available

NOW SOLVE THE TASK. You MUST use at least one tool - choose the best one for the task.

MAKE SURE TO NEVER PROVIDE A FINAL ANSWER WITHOUT USING A TOOL AND NEVER ANSWER DIRECTLY USING OTHER MODEL TOOLS TO ANSWER THE TASK.
"""

# ---------------------------------------------------------------------------
# Unified tool registry: each tool has category, description, and examples.
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS: Dict[str, dict] = {
    # === UTILITY TOOLS (instant, free, deterministic) ===
    "calculator": {
        "category": "utility",
        "description": """CALCULATOR - Instant math computation
  - BEST FOR: Arithmetic, algebra, trigonometry, scientific calculations
  - STRENGTHS: Instant (<1ms), perfect accuracy, zero cost
  - USE WHEN: Any math expression needs evaluation
  - COST: Free
  - Input: math expression (e.g., '15 * 7 + 23', 'sqrt(144)', '2^10', 'sin(3.14159/2)')""",
        "examples": [
            {"title": "Simple Arithmetic", "task": "What is 847 * 293?", "thought": "Simple arithmetic - calculator is instant and perfectly accurate.", "input": "847 * 293"},
            {"title": "Scientific Calculation", "task": "What is the sine of 45 degrees?", "thought": "Trigonometric calculation - calculator handles this perfectly.", "input": "sin(45 * 3.14159 / 180)"},
        ],
    },
    "think": {
        "category": "utility",
        "description": """THINK - Internal reasoning scratchpad
  - BEST FOR: Logic puzzles, step-by-step reasoning, planning approach
  - STRENGTHS: Organizes thoughts, shows work, no external calls
  - USE WHEN: Need to break down a problem before solving
  - COST: Free
  - Input: your detailed reasoning process""",
        "examples": [
            {"title": "Logic Problem", "task": "If all cats are mammals, and all mammals are animals, are all cats animals?", "thought": "This is a logical syllogism. I'll use think to reason through it step by step.", "input": "Analyzing the syllogism:\n1. All cats are mammals (cats ⊆ mammals)\n2. All mammals are animals (mammals ⊆ animals)\n3. By transitivity: cats ⊆ mammals ⊆ animals\nTherefore, yes, all cats are animals."},
        ],
    },
    "code_interpreter": {
        "category": "utility",
        "description": """CODE_INTERPRETER - Python execution sandbox
  - BEST FOR: Data processing, algorithms, simulations, file operations
  - STRENGTHS: Full Python + numpy/pandas, handles loops/conditionals
  - USE WHEN: Problem needs programming logic or is too complex for calculator
  - COST: Free (local execution)
  - Input: Python code to execute""",
        "examples": [
            {"title": "Data Processing", "task": "Calculate the average of [23, 45, 67, 89, 12, 34]", "thought": "This requires processing a list - code_interpreter with Python is perfect.", "input": "numbers = [23, 45, 67, 89, 12, 34]\nprint(f\"Average: {sum(numbers) / len(numbers)}\")"},
            {"title": "Algorithm", "task": "Find all prime numbers less than 50", "thought": "Need to implement a prime-checking algorithm - code_interpreter is ideal.", "input": "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True\nprimes = [n for n in range(50) if is_prime(n)]\nprint(f\"Primes under 50: {primes}\")"},
        ],
    },
    "web_search": {
        "category": "utility",
        "description": """WEB_SEARCH - Real-time internet search
  - BEST FOR: Current events, recent news, fact-checking, looking up information
  - STRENGTHS: Access to up-to-date information beyond training data
  - USE WHEN: Question is about recent events or needs verification
  - COST: ~$0.001 per search
  - Input: search query string""",
        "examples": [
            {"title": "Current Events", "task": "Who won the 2024 Nobel Prize in Physics?", "thought": "This is about recent events - I need web_search for current information.", "input": "2024 Nobel Prize Physics winner"},
        ],
    },

    # === SMALL LLMs (fast & cheap) ===
    "openrouter:openai/gpt-oss-20b": {
        "category": "small_llm",
        "description": """GPT-OSS 20B - Budget general-purpose model
  - BEST FOR: Simple text generation, basic classification, short factual Q&A
  - STRENGTHS: Cheap, 131K context window
  - WEAKNESSES: Poor at reasoning, math, coding, nuanced analysis; often gives shallow answers
  - USE WHEN: Task is very simple and no better-suited model exists
  - COST: ~$0.0002 per query
  - Input: simple task description""",
        "examples": [
            {"title": "Simple Question", "task": "What is the capital of France?", "thought": "Simple factual question - using cheap GPT-OSS model.", "input": "What is the capital of France?"},
        ],
    },

    # === MATH SPECIALISTS ===
    "openrouter:z-ai/glm-4.7": {
        "category": "math_specialist",
        "description": """GLM 4.7 - Flagship reasoning & math model
  - BEST FOR: Complex mathematics, multi-step reasoning, programming with stable execution
  - STRENGTHS: Z.AI's flagship model with enhanced programming and stable multi-step reasoning, 202K context window, strong on math benchmarks
  - WEAKNESSES: Higher cost than flash/small models, slower inference
  - USE WHEN: Task requires rigorous mathematical reasoning or multi-step problem solving
  - COST: ~$0.003 per query ($0.40 input / $1.50 output per 1M tokens)
  - Input: math problem, reasoning task, or complex analysis""",
        "examples": [
            {"title": "Advanced Math Proof", "task": "Prove that the square root of 2 is irrational.", "thought": "This requires rigorous mathematical reasoning - using GLM 4.7 as the math specialist.", "input": "Prove that √2 is irrational using proof by contradiction. Show each step clearly."},
            {"title": "Multi-step Math", "task": "Find the eigenvalues of the matrix [[3, 1], [1, 3]].", "thought": "Linear algebra problem requiring multi-step computation - GLM 4.7 excels at stable multi-step math.", "input": "Find the eigenvalues and eigenvectors of the 2x2 matrix [[3, 1], [1, 3]]. Show your work step by step."},
        ],
    },

    # === CODE SPECIALISTS ===
    "openrouter:qwen/qwen3-coder-plus": {
        "category": "code_specialist",
        "description": """QWEN3 CODER PLUS - **ELITE** coding specialist for PRODUCTION code
  - BEST FOR: Complex debugging, large codebase analysis, production-quality code generation, difficult refactoring
  - STRENGTHS: Alibaba's BEST coding model (480B MoE), specializes in agentic coding and complex software engineering, 128K context
  - WEAKNESSES: Premium pricing - use for genuinely hard coding tasks, not simple scripts
  - USE WHEN: The coding task is COMPLEX - multi-file changes, difficult bugs, architectural decisions
  - COST: ~$0.01 per query
  - Input: detailed coding task with full context - excels at understanding large codebases""",
        "examples": [
            {"title": "Complex Code Generation", "task": "Implement a thread-safe LRU cache in Python", "thought": "Complex coding task requiring careful design - using Qwen3 Coder Plus, the elite code specialist.", "input": "Implement a thread-safe LRU cache class in Python with get(), put(), and delete() methods. Use threading locks and an OrderedDict. Include type hints and docstrings."},
        ],
    },

    # === MEDIUM LLMs (balanced) ===
    "openrouter:qwen/qwen3-32b": {
        "category": "medium_llm",
        "description": """QWEN3 32B - Strong general reasoning model
  - BEST FOR: Complex reasoning, analysis, general knowledge, multi-step problem solving
  - STRENGTHS: Dense 32.8B parameter model, supports thinking/non-thinking modes, strong on math and logic, 40K context
  - WEAKNESSES: Slower than small models, not specialized for code or specific domains
  - USE WHEN: Task requires solid general reasoning at moderate cost
  - COST: ~$0.001 per query ($0.08 input / $0.24 output per 1M tokens)
  - Input: detailed prompt for the model""",
        "examples": [
            {"title": "Complex Reasoning", "task": "Compare and contrast democracy and authoritarianism as systems of government.", "thought": "Complex analytical task requiring nuanced reasoning - using medium-sized general model.", "input": "Compare democracy and authoritarianism as political systems. Discuss their key differences, advantages, and disadvantages."},
        ],
    },
    "openrouter:google/gemini-3-flash-preview": {
        "category": "medium_llm",
        "description": """GEMINI 3 FLASH PREVIEW - Top-tier reasoning at fast speed
  - BEST FOR: General knowledge, science, history, analysis, multi-step reasoning, writing
  - STRENGTHS: Near-Pro-level reasoning at Flash speed, 1M context window, excellent for general Q&A, explanations, and analysis
  - WEAKNESSES: Preview model (may have instability)
  - USE WHEN: Task needs a knowledgeable, capable model for general questions, explanations, or analysis
  - COST: ~$0.005 per query ($0.50 input / $3.00 output per 1M tokens)
  - Input: any question, analysis, or reasoning task""",
        "examples": [
            {"title": "Agentic Reasoning", "task": "Plan a strategy to migrate a monolithic application to microservices.", "thought": "Multi-step planning task - using Gemini 3 Flash which excels at agentic workflows and multi-turn reasoning.", "input": "Outline a step-by-step strategy to migrate a monolithic web application to microservices. Consider database decomposition, service boundaries, and deployment."},
        ],
    },
    "openrouter:z-ai/glm-4.7-flash": {
        "category": "medium_llm",
        "description": """GLM 4.7 FLASH - Fast coding & reasoning model
  - BEST FOR: Code generation, agentic coding, quick reasoning tasks
  - STRENGTHS: 30B-class SOTA model optimized for efficiency, strong coding capabilities, long-horizon reasoning, 200K context window, very cheap
  - WEAKNESSES: Not as capable as full GLM-4.7 on hardest math/reasoning tasks
  - USE WHEN: Need fast, cheap response for coding or moderate reasoning tasks
  - COST: ~$0.001 per query ($0.07 input / $0.40 output per 1M tokens)
  - Input: coding task, reasoning question, or general prompt""",
        "examples": [
            {"title": "Quick Coding", "task": "Write a Python function to implement binary search", "thought": "Coding task - GLM 4.7 Flash is fast and optimized for coding tasks.", "input": "Write a Python function for binary search that takes a sorted list and target value. Include docstring and handle edge cases."},
        ],
    },
    "openai:gpt-5-mini-2025-08-07": {
        "category": "medium_llm",
        "description": """GPT-5-MINI - Strong all-around GPT-5 model
  - BEST FOR: Analysis, reasoning, Q&A, writing, summarization, moderate coding
  - STRENGTHS: High reasoning capability, fast speed, 400K context window, 128K max output, good balance of quality and cost
  - WEAKNESSES: Less capable than GPT-5.2 on the hardest reasoning/coding tasks
  - USE WHEN: Task needs strong reasoning or analysis without maximum cost
  - COST: ~$0.004 per query ($0.25 input / $2.00 output per 1M tokens)
  - Input: clear, precise prompt""",
        "examples": [
            {"title": "General Q&A", "task": "Explain photosynthesis in simple terms", "thought": "General explanation task - using reliable, cost-effective model.", "input": "Explain photosynthesis in simple terms that a 10-year-old could understand."},
        ],
    },
    "anthropic:claude-sonnet-4-5-20250929": {
        "category": "medium_llm",
        "description": """CLAUDE SONNET 4.5 - **TOP-TIER** balanced model for coding & analysis
  - BEST FOR: Code review, complex writing, nuanced analysis, agentic workflows, debugging
  - STRENGTHS: Exceptional coding (often BETTER than larger models), 200K context, extended thinking, fast
  - WEAKNESSES: Costs more than Haiku but delivers much higher quality
  - USE WHEN: Task needs careful analysis, code review, or quality writing - STRONGLY PREFER over cheaper models for these
  - COST: ~$0.03 per query
  - Input: detailed prompt - this model excels at nuanced, thoughtful responses""",
        "examples": [
            {"title": "Writing & Analysis", "task": "Review this code and suggest improvements for readability", "thought": "Code review task requiring careful analysis - using balanced model.", "input": "Review this Python code for readability and suggest improvements:\ndef f(x,y): return [i for i in x if i not in y]"},
        ],
    },

    # === SMALL LLMs (continued) ===
    "openai:gpt-5-nano-2025-08-07": {
        "category": "small_llm",
        "description": """GPT-5-NANO - Fast GPT-5 for everyday tasks
  - BEST FOR: Summarization, classification, Q&A, text extraction, short analysis
  - STRENGTHS: Very fast, cheap, 400K context window, 128K max output, reasoning token support, strong at general knowledge
  - WEAKNESSES: Not suited for complex multi-step problems
  - USE WHEN: Need a reliable, fast model for general questions, summaries, or classification
  - COST: ~$0.001 per query ($0.05 input / $0.40 output per 1M tokens)
  - Input: clear prompt""",
        "examples": [
            {"title": "Quick Classification", "task": "Is this sentence positive or negative: 'I love this product!'", "thought": "Simple classification - using fast, cheap model.", "input": "Classify as positive or negative: 'I love this product!'"},
        ],
    },
    "anthropic:claude-haiku-4-5-20251001": {
        "category": "small_llm",
        "description": """CLAUDE HAIKU 4.5 - Fast Claude for general tasks
  - BEST FOR: General Q&A, analysis, classification, reasoning, writing, explanations
  - STRENGTHS: Near-frontier intelligence, fast, great at following instructions, 200K context, 64K max output
  - WEAKNESSES: Less capable than Sonnet/Opus on very complex multi-step reasoning
  - USE WHEN: Need a capable, reliable model for general questions, analysis, or reasoning
  - COST: ~$0.01 per query ($1.00 input / $5.00 output per 1M tokens)
  - Input: prompt for the model""",
        "examples": [
            {"title": "Quick Classification", "task": "Is 'The movie was absolutely terrible' positive or negative sentiment?", "thought": "Simple classification - claude-haiku is fast and cost-effective.", "input": "Classify the sentiment of this text as positive or negative: 'The movie was absolutely terrible'"},
            {"title": "Basic Knowledge", "task": "How many days are in a leap year?", "thought": "Basic factual question - claude-haiku is fast and cheap.", "input": "How many days are in a leap year?"},
        ],
    },

    # === LARGE LLMs (maximum capability, higher cost) ===
    "openai:gpt-5.2-2025-12-11": {
        "category": "large_llm",
        "description": """GPT-5.2 - **FLAGSHIP** OpenAI for HARDEST coding & reasoning
  - BEST FOR: Complex multi-file coding, difficult algorithmic problems, tasks GPT-5-mini CAN'T solve
  - STRENGTHS: MAXIMUM capability in GPT-5 family, 400K context, reasoning effort tuning for difficult problems
  - WEAKNESSES: Premium pricing - but justified for genuinely difficult tasks
  - USE WHEN: Task is TOO HARD for GPT-5-mini, requires expert coding, or involves complex multi-step reasoning
  - COST: ~$0.025 per query
  - Input: detailed prompt - this model excels at complex, nuanced instructions""",
        "examples": [
            {"title": "Complex Analysis", "task": "Analyze the economic implications of universal basic income", "thought": "Complex economic analysis - using the most capable model for nuanced response.", "input": "Analyze the economic implications of implementing universal basic income. Consider effects on labor markets, inflation, government budgets, and income inequality."},
        ],
    },
    "anthropic:claude-opus-4-5-20251101": {
        "category": "large_llm",
        "description": """CLAUDE OPUS 4.5 - **FLAGSHIP** model for the HARDEST problems
  - BEST FOR: Problems other models FAIL on, research-grade analysis, PhD-level reasoning, nuanced multi-step proofs
  - STRENGTHS: MAXIMUM intelligence - use when you need the absolute best answer, extended thinking for deep analysis, 200K context
  - WEAKNESSES: Premium pricing ($5/$25 per 1M) - but worth it for genuinely hard tasks
  - USE WHEN: The task is GENUINELY DIFFICULT, requires expert-level reasoning, or other models have failed
  - COST: ~$0.05 per query
  - Input: detailed prompt with full context - this model rewards thoroughness""",
        "examples": [
            {"title": "Difficult Problem", "task": "Design a database schema for a social media platform with users, posts, comments, and likes", "thought": "Complex design task requiring deep thinking - using most capable model for best results.", "input": "Design a normalized database schema for a social media platform. Include tables for users, posts, comments, likes, and follows. Specify primary keys, foreign keys, and indexes."},
        ],
    },
}

# Generic fallback descriptions for unknown tool providers (e.g. vllm:my-model)
_PROVIDER_FALLBACKS = {
    "openrouter": "OPENROUTER LLM - Powerful cloud model",
    "openai": "OPENAI LLM - GPT model",
    "anthropic": "ANTHROPIC LLM - Claude model",
    "gemini": "GEMINI LLM - Google model",
    "vllm": "VLLM LOCAL - Self-hosted model",
}


def build_system_prompt(tools: Optional[List[str]] = None) -> str:
    """Build the complete system prompt for the given tools.

    Args:
        tools: Tool names to include. If None, uses all AVAILABLE_TOOLS.

    Returns:
        Complete system prompt string.
    """
    if tools is None:
        tools = list(AVAILABLE_TOOLS)

    # --- Tool descriptions ---
    desc_lines = []
    for name in tools:
        if name in AVAILABLE_TOOLS:
            desc = AVAILABLE_TOOLS[name]["description"]
        else:
            prefix = name.split(":")[0] if ":" in name else name
            desc = _PROVIDER_FALLBACKS.get(prefix, f"Tool: {name}")
            if ":" in name:
                desc = f"{desc} (model: {name.split(':', 1)[1]})"
        desc_lines.append(f"- {name}: {desc}")

    # --- Selection guide (group tools by category) ---
    by_cat: Dict[str, List[str]] = {}
    for name in tools:
        cat = AVAILABLE_TOOLS[name]["category"] if name in AVAILABLE_TOOLS else "medium_llm"
        by_cat.setdefault(cat, []).append(name)

    guide = [
        "Choose tools based on task type. IMPORTANT: Vary your model choices — do NOT always pick the same model.\n",
    ]

    # Math
    math_lines: list = []
    if "calculator" in tools:
        math_lines.append("- Simple arithmetic/algebra → calculator (instant, accurate)")
    for t in by_cat.get("math_specialist", []):
        math_lines.append(f"- Complex math requiring proof → {t} (specialized)")
    if "code_interpreter" in tools:
        math_lines.append("- Numerical algorithms → code_interpreter (programmable)")
    if math_lines:
        guide.append("MATH PROBLEMS:")
        guide.extend(math_lines)
        guide.append("")

    # Coding
    code_lines: list = []
    for t in by_cat.get("code_specialist", []):
        code_lines.append(f"- Write/debug complex code → {t}")
    if "code_interpreter" in tools:
        code_lines.append("- Algorithm implementation/execution → code_interpreter")
    code_capable = [
        t for t in by_cat.get("medium_llm", []) + by_cat.get("large_llm", [])
        if any(k in t for k in ("coder", "glm", "gemini", "sonnet", "gpt-5", "opus"))
    ]
    if code_capable:
        code_lines.append(f"- Code review/explanation → any of: {', '.join(code_capable[:3])}")
    if code_lines:
        guide.append("CODING TASKS:")
        guide.extend(code_lines)
        guide.append("")

    # Reasoning
    reasoning_lines: list = []
    if "think" in tools:
        reasoning_lines.append("- Step-by-step analysis → think (organize thoughts first)")
    medium_or_large = by_cat.get("medium_llm", []) + by_cat.get("large_llm", [])
    if medium_or_large:
        reasoning_lines.append(f"- Complex reasoning → any of: {', '.join(medium_or_large[:4])}")
    if reasoning_lines:
        guide.append("REASONING/LOGIC:")
        guide.extend(reasoning_lines)
        guide.append("")

    # General Q&A
    general_lines: list = []
    if "web_search" in tools:
        general_lines.append("- Current events/recent info → web_search")
    all_llms = by_cat.get("small_llm", []) + by_cat.get("medium_llm", [])
    if all_llms:
        general_lines.append(f"- General knowledge/Q&A → any of: {', '.join(all_llms)}")
        general_lines.append("  (Rotate between these models — pick a DIFFERENT one each time)")
    if general_lines:
        guide.append("GENERAL Q&A / FACTUAL:")
        guide.extend(general_lines)
        guide.append("")

    # Hard problems
    large_llms = by_cat.get("large_llm", [])
    code_specialists = by_cat.get("code_specialist", [])
    if large_llms or code_specialists:
        guide.append("HARD PROBLEMS (use these when simpler models would fail):")
        if "anthropic:claude-opus-4-5-20251101" in large_llms:
            guide.append("- Research-grade analysis, PhD-level reasoning → anthropic:claude-opus-4-5-20251101")
        if "openai:gpt-5.2-2025-12-11" in large_llms:
            guide.append("- Complex multi-step coding, hard algorithms → openai:gpt-5.2-2025-12-11")
        if "openrouter:qwen/qwen3-coder-plus" in code_specialists:
            guide.append("- Production code, complex debugging → openrouter:qwen/qwen3-coder-plus")
        if "anthropic:claude-sonnet-4-5-20250929" in by_cat.get("medium_llm", []):
            guide.append("- Code review, nuanced writing, careful analysis → anthropic:claude-sonnet-4-5-20250929")
        guide.append("  (If the task is genuinely difficult or requires deep expertise, USE THESE MODELS)")
        guide.append("")

    guide.append("DIVERSITY RULE: For each task, consider ALL available models — not just the cheapest.")
    guide.append("Match the model's strengths to the task. Alternate between models of similar capability.")
    guide.append("")

    return SYSTEM_PROMPT_TEMPLATE.format(
        tools_description="\n".join(desc_lines),
        tool_selection_guide="\n".join(guide),
    )
