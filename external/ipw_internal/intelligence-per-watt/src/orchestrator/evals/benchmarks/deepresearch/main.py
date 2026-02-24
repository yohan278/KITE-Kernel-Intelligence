"""DeepResearch-Bench benchmark runner.

Runs evaluation on the DeepResearch-Bench for long-form research generation.
Uses the RACE (Reference-based Adaptive Criteria-driven Evaluation) framework
with Gemini 2.5 Pro as the judge, exactly matching the original implementation:
https://github.com/Ayanami0730/deep_research_bench

Dimensions: Comprehensiveness, Insight, Instruction Following, Readability
Scoring: Comparative (target vs reference), normalized as target/(target+reference) -> 0-1
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .dataset import DeepResearchDataset, DeepResearchSample
from ..registry import register_benchmark
from ...utils import extract_final_answer, print_result_box, print_task_header

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RACE dimensions (original benchmark)
# ---------------------------------------------------------------------------
RACE_DIMENSIONS = ["comprehensiveness", "insight", "instruction_following", "readability"]

# ---------------------------------------------------------------------------
# Scoring prompts — exact copies from the original repo:
# https://github.com/Ayanami0730/deep_research_bench/blob/main/prompt/score_prompt_en.py
# https://github.com/Ayanami0730/deep_research_bench/blob/main/prompt/score_prompt_zh.py
# ---------------------------------------------------------------------------

SCORE_PROMPT_EN = """
<system_role>You are a strict, meticulous, and objective research article evaluation expert. You excel at using specific assessment criteria to deeply compare two articles on the same task, providing precise scores and clear justifications.</system_role>

<user_prompt>
**Task Background**
There is a deep research task, and you need to evaluate two research articles written for this task. We will assess the articles across four dimensions: Comprehensiveness, Insight, Instruction Following, and Readability. The content is as follows:
<task>
"{task_prompt}"
</task>

**Articles to Evaluate**
<article_1>
"{article_1}"
</article_1>

<article_2>
"{article_2}"
</article_2>

**Evaluation Criteria**
Now, you need to evaluate and compare these two articles based on the following **evaluation criteria list**, providing comparative analysis and scoring each on a scale of 0-10. Each criterion includes an explanation, please understand carefully.

<criteria_list>
{criteria_list}
</criteria_list>

<Instruction>
**Your Task**
Please strictly evaluate and compare `<article_1>` and `<article_2>` based on **each criterion** in the `<criteria_list>`. You need to:
1.  **Analyze Each Criterion**: Consider how each article fulfills the requirements of each criterion.
2.  **Comparative Evaluation**: Analyze how the two articles perform on each criterion, referencing the content and criterion explanation.
3.  **Score Separately**: Based on your comparative analysis, score each article on each criterion (0-10 points).

**Scoring Rules**
For each criterion, score both articles on a scale of 0-10 (continuous values). The score should reflect the quality of performance on that criterion:
*   0-2 points: Very poor performance. Almost completely fails to meet the criterion requirements.
*   2-4 points: Poor performance. Minimally meets the criterion requirements with significant deficiencies.
*   4-6 points: Average performance. Basically meets the criterion requirements, neither good nor bad.
*   6-8 points: Good performance. Largely meets the criterion requirements with notable strengths.
*   8-10 points: Excellent/outstanding performance. Fully meets or exceeds the criterion requirements.

**Output Format Requirements**
Please **strictly** follow the `<output_format>` below for each criterion evaluation. **Do not include any other unrelated content, introduction, or summary**. Start with "Standard 1" and proceed sequentially through all criteria:
</Instruction>

<output_format>
{{
    "comprehensiveness": [
        {{
            "criterion": [Text content of the first comprehensiveness evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
}},
{{
            "criterion": [Text content of the second comprehensiveness evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        }},
        ...
    ],
    "insight": [
        {{
            "criterion": [Text content of the first insight evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        }},
        ...
    ],
    ...
}}
</output_format>

Now, please evaluate the two articles based on the research task and criteria, providing detailed comparative analysis and scores according to the requirements above. Ensure your output follows the specified `<output_format>` and that the JSON format is parsable, with all characters that might cause JSON parsing errors properly escaped.
</user_prompt>
""".strip()

SCORE_PROMPT_ZH = """
<system_role>\u4f60\u662f\u4e00\u540d\u4e25\u683c\u3001\u7ec6\u81f4\u3001\u5ba2\u89c2\u7684\u8c03\u7814\u6587\u7ae0\u8bc4\u4f30\u4e13\u5bb6\u3002\u4f60\u64c5\u957f\u6839\u636e\u5177\u4f53\u7684\u8bc4\u4f30\u6807\u51c6\uff0c\u6df1\u5165\u6bd4\u8f83\u4e24\u7bc7\u9488\u5bf9\u540c\u4e00\u4efb\u52a1\u7684\u6587\u7ae0\uff0c\u5e76\u7ed9\u51fa\u7cbe\u786e\u7684\u8bc4\u5206\u548c\u6e05\u6670\u7684\u7406\u7531\u3002</system_role>

<user_prompt>
**\u4efb\u52a1\u80cc\u666f**
\u6709\u4e00\u4e2a\u6df1\u5ea6\u8c03\u7814\u4efb\u52a1\uff0c\u4f60\u9700\u8981\u8bc4\u4f30\u9488\u5bf9\u8be5\u4efb\u52a1\u64b0\u5199\u7684\u4e24\u7bc7\u8c03\u7814\u6587\u7ae0\u3002\u6211\u4eec\u4f1a\u4ece\u4ee5\u4e0b\u56db\u4e2a\u7ef4\u5ea6\u8bc4\u4f30\u6587\u7ae0\uff1a\u5168\u9762\u6027\u3001\u6d1e\u5bdf\u529b\u3001\u6307\u4ee4\u9075\u5faa\u80fd\u529b\u548c\u53ef\u8bfb\u6027\u3002\u5185\u5bb9\u5982\u4e0b\uff1a
<task>
"{task_prompt}"
</task>

**\u5f85\u8bc4\u4f30\u6587\u7ae0**
<article_1>
"{article_1}"
</article_1>

<article_2>
"{article_2}"
</article_2>

**\u8bc4\u4f30\u6807\u51c6**
\u73b0\u5728\uff0c\u4f60\u9700\u8981\u6839\u636e\u4ee5\u4e0b**\u8bc4\u5224\u6807\u51c6\u5217\u8868**\uff0c\u9010\u6761\u8bc4\u4f30\u5e76\u6bd4\u8f83\u8fd9\u4e24\u7bc7\u6587\u7ae0\u7684\u8868\u73b0\uff0c\u8f93\u51fa\u5bf9\u6bd4\u5206\u6790\uff0c\u7136\u540e\u7ed9\u51fa0-10\u7684\u5206\u6570\u3002\u6bcf\u4e2a\u6807\u51c6\u90fd\u9644\u6709\u5176\u89e3\u91ca\uff0c\u8bf7\u4ed4\u7ec6\u7406\u89e3\u3002

<criteria_list>
{criteria_list}
</criteria_list>

**\u4f60\u7684\u4efb\u52a1**
\u8bf7\u4e25\u683c\u6309\u7167 `<criteria_list>` \u4e2d\u7684**\u6bcf\u4e00\u6761\u6807\u51c6**\uff0c\u5bf9\u6bd4\u8bc4\u4f30 `<article_1>` \u548c `<article_2>` \u5728\u8be5\u6807\u51c6\u4e0a\u7684\u5177\u4f53\u8868\u73b0\u3002\u4f60\u9700\u8981\uff1a
1. **\u9010\u6761\u5206\u6790**\uff1a\u9488\u5bf9\u5217\u8868\u4e2d\u7684\u6bcf\u4e00\u6761\u6807\u51c6\uff0c\u5206\u522b\u601d\u8003\u4e24\u7bc7\u6587\u7ae0\u662f\u5982\u4f55\u6ee1\u8db3\u8be5\u6807\u51c6\u8981\u6c42\u7684\u3002
2. **\u5bf9\u6bd4\u8bc4\u4f30**\uff1a\u7ed3\u5408\u6587\u7ae0\u5185\u5bb9\u4e0e\u6807\u51c6\u89e3\u91ca\uff0c\u5bf9\u6bd4\u5206\u6790\u4e24\u7bc7\u6587\u7ae0\u5728\u6bcf\u4e00\u6761\u6807\u51c6\u4e0a\u7684\u8868\u73b0\u3002
3. **\u5206\u522b\u6253\u5206**\uff1a\u57fa\u4e8e\u4f60\u7684\u5bf9\u6bd4\u5206\u6790\uff0c\u4e3a\u4e24\u7bc7\u6587\u7ae0\u5728\u8be5\u6761\u6807\u51c6\u4e0a\u7684\u8868\u73b0\u5206\u522b\u6253\u5206\uff080-10\u5206\uff09\u3002

**\u6253\u5206\u89c4\u5219**
\u5bf9\u6bcf\u4e00\u6761\u6807\u51c6\uff0c\u5206\u522b\u4e3a\u4e24\u7bc7\u6587\u7ae0\u6253\u5206\uff0c\u6253\u5206\u8303\u56f4\u4e3a 0-10 \u5206\uff08\u8fde\u7eed\u7684\u6570\u503c\uff09\u3002\u5206\u6570\u9ad8\u4f4e\u5e94\u4f53\u73b0\u6587\u7ae0\u5728\u8be5\u6807\u51c6\u4e0a\u8868\u73b0\u7684\u597d\u574f\uff1a
* 0-2\u5206\uff1a\u8868\u73b0\u5f88\u5dee\u3002\u51e0\u4e4e\u5b8c\u5168\u4e0d\u7b26\u5408\u6807\u51c6\u8981\u6c42\u3002
* 2-4\u5206\uff1a\u8868\u73b0\u8f83\u5dee\u3002\u5c11\u91cf\u7b26\u5408\u6807\u51c6\u8981\u6c42\uff0c\u4f46\u6709\u660e\u663e\u4e0d\u8db3\u3002
* 4-6\u5206\uff1a\u8868\u73b0\u4e2d\u7b49\u3002\u57fa\u672c\u7b26\u5408\u6807\u51c6\u8981\u6c42\uff0c\u4e0d\u597d\u4e0d\u574f\u3002
* 6-8\u5206\uff1a\u8868\u73b0\u8f83\u597d\u3002\u5927\u90e8\u5206\u7b26\u5408\u6807\u51c6\u8981\u6c42\uff0c\u6709\u53ef\u53d6\u4e4b\u5904\u3002
* 8-10\u5206\uff1a\u8868\u73b0\u51fa\u8272/\u6781\u597d\u3002\u5b8c\u5168\u6216\u8d85\u9884\u671f\u7b26\u5408\u6807\u51c6\u8981\u6c42\u3002

**\u8f93\u51fa\u683c\u5f0f\u8981\u6c42**
\u8bf7**\u4e25\u683c**\u6309\u7167\u4e0b\u5217`<output_format>`\u683c\u5f0f\u8f93\u51fa\u6bcf\u4e00\u6761\u6807\u51c6\u7684\u8bc4\u4f30\u7ed3\u679c\uff0c**\u4e0d\u8981\u5305\u542b\u4efb\u4f55\u5176\u4ed6\u65e0\u5173\u5185\u5bb9\u3001\u5f15\u8a00\u6216\u603b\u7ed3**\u3002\u4ece\u201c\u6807\u51c6 1\u201d\u5f00\u59cb\uff0c\u6309\u987a\u5e8f\u8f93\u51fa\u6240\u6709\u6807\u51c6\u7684\u8bc4\u4f30\uff1a

<output_format>
{{
    "comprehensiveness": [
        {{
            "criterion": [\u5168\u9762\u6027\u7ef4\u5ea6\u7684\u7b2c\u4e00\u6761\u8bc4\u5224\u6807\u51c6\u6587\u672c\u5185\u5bb9],
            "analysis": [\u5bf9\u6bd4\u5206\u6790],
            "article_1_score": [0-10\u8fde\u7eed\u5206\u6570],
            "article_2_score": [0-10\u8fde\u7eed\u5206\u6570]
        }},
        ...
    ],
    "insight": [...],
    "instruction_following": [...],
    "readability": [...]
}}
</output_format>

\u73b0\u5728\uff0c\u8bf7\u6839\u636e\u8c03\u7814\u4efb\u52a1\u548c\u6807\u51c6\uff0c\u5bf9\u4e24\u7bc7\u6587\u7ae0\u8fdb\u884c\u8bc4\u4f30\uff0c\u5e76\u6309\u7167\u4e0a\u8ff0\u8981\u6c42\u7ed9\u51fa\u8be6\u7ec6\u7684\u5bf9\u6bd4\u5206\u6790\u548c\u8bc4\u5206\uff0c\u8bf7\u786e\u4fdd\u8f93\u51fa\u683c\u5f0f\u9075\u5b88\u4e0a\u8ff0`<output_format>`\uff0c\u800c\u4e14\u4fdd\u8bc1\u5176\u4e2d\u7684json\u683c\u5f0f\u53ef\u4ee5\u89e3\u6790\uff0c\u6ce8\u610f\u6240\u6709\u53ef\u80fd\u5bfc\u81f4json\u89e3\u6790\u9519\u8bef\u7684\u8981\u8f6c\u4e49\u7684\u7b26\u53f7\u3002
</user_prompt>
""".strip()


# ---------------------------------------------------------------------------
# Article cleaning prompts — from the original repo:
# https://github.com/Ayanami0730/deep_research_bench/blob/main/prompt/clean_prompt.py
# ---------------------------------------------------------------------------

CLEAN_PROMPT_EN = """
<system_role>You are a professional article editor who is good at cleaning and refining article content.</system_role>

<user_prompt>
Please help me clean the following research article, removing all citation links, citation marks (such as [1], [2], 1, 2, etc. or other complex citation formats), reference lists, footnotes, and ensuring the content is coherent and smooth.
Keep all other original content of the article, removing only the citations. If the content of the citation mark is used as part of a sentence in the article, keep the text content and remove other marks.

Article content:
"{article}"

Please return the cleaned article in full, without adding any additional comments or explanations.
</user_prompt>
""".strip()

CLEAN_PROMPT_ZH = """
<system_role>\u4f60\u662f\u4e00\u540d\u4e13\u4e1a\u7684\u6587\u7ae0\u7f16\u8f91\uff0c\u64c5\u957f\u6574\u7406\u548c\u6e05\u6d17\u6587\u7ae0\u5185\u5bb9\u3002</system_role>

<user_prompt>
\u8bf7\u5e2e\u6211\u6e05\u6d17\u4ee5\u4e0b\u7814\u7a76\u6587\u7ae0\uff0c\u53bb\u9664\u6240\u6709\u5f15\u7528\u94fe\u63a5\u3001\u5f15\u7528\u6807\u8bb0\uff08\u5982[1]\u3001[2]\u30011\u30012 \u7b49\u6216\u5176\u4ed6\u590d\u6742\u5f15\u7528\u683c\u5f0f\uff09\u3001\u53c2\u8003\u6587\u732e\u5217\u8868\u3001\u811a\u6ce8\uff0c\u5e76\u786e\u4fdd\u6587\u7ae0\u5185\u5bb9\u8fde\u8d2f\u6d41\u7545\u3002
\u4fdd\u7559\u6587\u7ae0\u7684\u6240\u6709\u5176\u4ed6\u539f\u672c\u5185\u5bb9\u3001\u53ea\u79fb\u9664\u5f15\u7528\u3002\u5982\u679c\u6587\u7ae0\u4e2d\u4f7f\u7528\u5f15\u7528\u6807\u8bb0\u4e2d\u7684\u5185\u5bb9\u4f5c\u4e3a\u8bed\u53e5\u7684\u4e00\u90e8\u5206\uff0c\u4fdd\u7559\u8fd9\u5176\u4e2d\u7684\u6587\u5b57\u5185\u5bb9\uff0c\u79fb\u9664\u5176\u4ed6\u6807\u8bb0\u3002

\u6587\u7ae0\u5185\u5bb9\uff1a
"{article}"

\u8bf7\u8fd4\u56de\u6e05\u6d17\u540e\u7684\u6587\u7ae0\u5168\u6587\uff0c\u4e0d\u8981\u6dfb\u52a0\u4efb\u4f55\u989d\u5916\u8bf4\u660e\u6216\u8bc4\u8bba\u3002
</user_prompt>
""".strip()

MIN_VALID_CLEAN_LENGTH = 100


def _clean_text(article: str, language: str = "en", max_retries: int = 3) -> Optional[str]:
    """Try to clean article text via Gemini. Returns None on failure (e.g. token limit).

    Ported from: ArticleCleaner._clean_text()
    """
    clean_prompt_template = CLEAN_PROMPT_ZH if language == "zh" else CLEAN_PROMPT_EN
    user_prompt = clean_prompt_template.format(article=article)

    for retry in range(max_retries):
        try:
            result = _call_gemini(user_prompt)
            if result and len(result.strip()) >= MIN_VALID_CLEAN_LENGTH:
                return result.strip()
            logger.warning(f"Article cleaning returned invalid result, retry {retry + 1}/{max_retries}")
        except Exception as e:
            error_str = str(e).lower()
            if "tokens" in error_str and "less than" in error_str:
                logger.info("Article too long for single cleaning call, needs chunking")
                return None
            logger.warning(f"Article cleaning failed: {e}, retry {retry + 1}/{max_retries}")

    return None


def _chunk_clean_article(article: str, language: str = "en") -> Optional[str]:
    """Split a long article into two chunks, clean each, then merge.

    Ported from: ArticleCleaner.chunk_clean_article()
    """
    logger.info("Attempting to process article in 2 chunks")
    chunk_size = len(article) // 2

    chunks = []
    for i in range(2):
        start = i * chunk_size
        end = len(article) if i == 1 else chunk_size

        # For the first chunk, split at a sentence boundary near the midpoint
        if i == 0:
            search_start = max(0, end - 200)
            for j in range(end, search_start, -1):
                if j < len(article) and article[j] in ".?!。？！\n":
                    end = j + 1
                    break

        chunks.append(article[start:end])

    cleaned_chunks = []
    for idx, chunk in enumerate(chunks):
        result = _clean_text(chunk, language)
        if result is None:
            logger.error(f"Chunk {idx + 1}/2 cleaning failed")
            return None
        cleaned_chunks.append(result)

    return "".join(cleaned_chunks)


def _clean_article(article: str, language: str = "en") -> str:
    """Clean a research article by removing citations/references via Gemini.

    Matches the original ArticleCleaner from:
    https://github.com/Ayanami0730/deep_research_bench/blob/main/utils/clean_article.py

    Falls back to the original article if cleaning fails.
    Supports chunked cleaning for articles that exceed the token limit.
    """
    if not article or len(article.strip()) < MIN_VALID_CLEAN_LENGTH:
        return article

    # Try normal cleaning first
    result = _clean_text(article, language)

    # If cleaning returned None (e.g. token limit), try chunked cleaning
    if result is None:
        logger.info("Falling back to chunked cleaning for long article")
        result = _chunk_clean_article(article, language)

    if result and len(result.strip()) >= MIN_VALID_CLEAN_LENGTH:
        return result.strip()

    # Fallback: return original article if all cleaning attempts fail
    logger.warning("Article cleaning failed after all attempts, using original article")
    return article


# ---------------------------------------------------------------------------
# Helper functions ported from the original repo
# ---------------------------------------------------------------------------

def format_criteria_list(criteria_data: Dict[str, Any]) -> str:
    """Format evaluation criteria as JSON string, without weight information.

    Ported from: deepresearch_bench_race.py::format_criteria_list()
    """
    criteria_for_prompt = {}
    criterions_dict = criteria_data.get("criterions", {})

    for dim, criterions_list in criterions_dict.items():
        if not isinstance(criterions_list, list):
            continue
        criteria_for_prompt[dim] = []
        for crit_item in criterions_list:
            if isinstance(crit_item, dict) and "criterion" in crit_item and "explanation" in crit_item:
                criteria_for_prompt[dim].append({
                    "criterion": crit_item["criterion"],
                    "explanation": crit_item["explanation"],
                })
    return json.dumps(criteria_for_prompt, ensure_ascii=False, indent=2)


def extract_json_from_markdown(text: str) -> Optional[str]:
    """Extract JSON from a response that may be wrapped in markdown code blocks.

    Ported from: utils/json_extractor.py::extract_json_from_markdown()
    Uses all 6 extraction methods from the original implementation.
    """
    if not isinstance(text, str) or not text:
        return None

    # Method 0: Try parsing the complete text directly if it looks like JSON
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

    # Method 1: String operations to extract from ```json ... ``` blocks
    if "```json" in text and "```" in text[text.find("```json") + 7:]:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            candidate = text[start:end].strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

    # Method 2: Regex matching for ```json ... ```
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Method 3: Try entire text as JSON
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    # Method 4: Nesting-aware brace matching
    start = text.find("{")
    if start != -1:
        level = 0
        for i, char in enumerate(text[start:]):
            if char == "{":
                level += 1
            elif char == "}":
                level -= 1
                if level == 0:
                    candidate = text[start:start + i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        pass
                    break

    # Method 5: Simple first { to last } matching
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Method 6: Keyword-based fallback — reconstruct JSON from scattered scoring data
    if "comprehensiveness" in text and "article_1_score" in text and "article_2_score" in text:
        try:
            dimensions = ["comprehensiveness", "insight", "instruction_following", "readability"]
            result = {}
            for dim in dimensions:
                if dim not in text:
                    continue
                result[dim] = []
                dim_start = text.find(f'"{dim}"')
                if dim_start == -1:
                    dim_start = text.find(f"'{dim}'")
                if dim_start == -1:
                    dim_start = text.find(dim)
                if dim_start == -1:
                    continue
                # Find next dimension start or end of text
                next_dim_start = len(text)
                for next_dim in dimensions:
                    if next_dim != dim:
                        pos = text.find(f'"{next_dim}"', dim_start + len(dim))
                        if pos == -1:
                            pos = text.find(next_dim, dim_start + len(dim))
                        if pos != -1 and pos < next_dim_start:
                            next_dim_start = pos
                dim_content = text[dim_start:next_dim_start]
                criteria = [m.group(1) for m in re.finditer(r'"criterion"\s*:\s*"([^"]+)"', dim_content)]
                scores1 = [float(m.group(1)) for m in re.finditer(r'"article_1_score"\s*:\s*(\d+\.?\d*)', dim_content)]
                scores2 = [float(m.group(1)) for m in re.finditer(r'"article_2_score"\s*:\s*(\d+\.?\d*)', dim_content)]
                for idx in range(min(len(criteria), len(scores1), len(scores2))):
                    result[dim].append({
                        "criterion": criteria[idx],
                        "article_1_score": scores1[idx],
                        "article_2_score": scores2[idx],
                    })
            if any(len(scores) > 0 for scores in result.values()):
                return json.dumps(result)
        except Exception:
            pass

    return None


def calculate_weighted_scores(
    llm_output_json: Dict[str, Any],
    criteria_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculate weighted scores based on LLM output and criteria weights.

    Ported from: utils/score_calculator.py::calculate_weighted_scores()

    Returns:
        {"target": {"dims": {...}, "total": float},
         "reference": {"dims": {...}, "total": float}}
    """
    results = {
        "target": {"dims": {}, "total": 0.0},
        "reference": {"dims": {}, "total": 0.0},
    }
    total_target_score = 0.0
    total_reference_score = 0.0

    dimension_weights = criteria_data.get("dimension_weight", {})

    # Build criterion -> weight mapping per dimension
    criterion_weights: Dict[str, Dict[str, float]] = {}
    for dim, criterions in criteria_data.get("criterions", {}).items():
        criterion_weights[dim] = {
            crit["criterion"]: crit["weight"]
            for crit in criterions
            if isinstance(crit, dict) and "criterion" in crit and "weight" in crit
        }

    for dim, scores_list in llm_output_json.items():
        if not isinstance(scores_list, list):
            continue
        if dim not in dimension_weights or dim not in criterion_weights:
            continue

        dim_criteria_map = criterion_weights.get(dim, {})
        if not dim_criteria_map:
            continue

        dim_target_weighted_sum = 0.0
        dim_reference_weighted_sum = 0.0
        dim_total_weight = 0.0

        for score_item in scores_list:
            if not isinstance(score_item, dict):
                continue

            criterion_text_raw = score_item.get("criterion")
            criterion_text = criterion_text_raw.strip() if isinstance(criterion_text_raw, str) else None

            article_1_score_raw = score_item.get("article_1_score")
            article_2_score_raw = score_item.get("article_2_score")

            try:
                article_1_score = float(article_1_score_raw) if article_1_score_raw is not None else None
                article_2_score = float(article_2_score_raw) if article_2_score_raw is not None else None
            except (ValueError, TypeError):
                continue

            if criterion_text and article_1_score is not None:
                # Look up weight: exact match, then case-insensitive, then substring
                weight = dim_criteria_map.get(criterion_text)
                if weight is None:
                    criterion_lower = criterion_text.lower()
                    for key, val in dim_criteria_map.items():
                        if key.lower() == criterion_lower:
                            weight = val
                            break
                if weight is None:
                    criterion_lower = criterion_text.lower()
                    for key, val in dim_criteria_map.items():
                        if criterion_lower in key.lower() or key.lower() in criterion_lower:
                            weight = val
                            break
                if weight is None:
                    # Fallback: average weight for dimension
                    weight = sum(dim_criteria_map.values()) / len(dim_criteria_map)

                dim_target_weighted_sum += article_1_score * weight
                dim_total_weight += weight

                if article_2_score is not None:
                    dim_reference_weighted_sum += article_2_score * weight

        if dim_total_weight > 0:
            dim_target_avg = dim_target_weighted_sum / dim_total_weight
            dim_reference_avg = dim_reference_weighted_sum / dim_total_weight
        else:
            dim_target_avg = 0.0
            dim_reference_avg = 0.0

        results["target"]["dims"][f"{dim}_weighted_avg"] = dim_target_avg
        results["reference"]["dims"][f"{dim}_weighted_avg"] = dim_reference_avg

        dim_weight = dimension_weights.get(dim, 0)
        total_target_score += dim_target_avg * dim_weight
        total_reference_score += dim_reference_avg * dim_weight

    results["target"]["total"] = total_target_score
    results["reference"]["total"] = total_reference_score

    return results


# ---------------------------------------------------------------------------
# Gemini judge
# ---------------------------------------------------------------------------

MAX_RETRIES = 10


def _call_gemini(prompt: str) -> str:
    """Call Gemini 2.5 Pro via the google-genai SDK.

    Matches the original AIClient configuration:
    - Model: gemini-2.5-pro
    - Thinking budget: 16000
    - Timeout: 600s
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required")

    client = genai.Client(api_key=api_key, http_options={"timeout": 600000})
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=16000),
        ),
    )
    return response.text


def judge_research_report(
    target_report: str,
    prompt_text: str,
    reference_article: str,
    criteria_data: Dict[str, Any],
    language: str = "en",
) -> Tuple[Dict[str, float], float]:
    """Score a research report using the RACE framework with Gemini 2.5 Pro.

    Follows the exact scoring pipeline from the original DeepResearch-Bench:
    1. Format criteria (without weights) for the judge prompt
    2. Call Gemini with the comparative scoring prompt (target=article_1, reference=article_2)
    3. Parse JSON output with per-criterion scores for both articles
    4. Calculate weighted scores and normalize: target / (target + reference)

    Args:
        target_report: The model's generated research report
        prompt_text: The original research prompt
        reference_article: Reference article (OpenAI Deep Research output)
        criteria_data: Per-task criteria dict with dimension_weight and criterions
        language: Language of the task ('en' or 'zh')

    Returns:
        (dimension_scores, overall_score) where scores are 0-1 ratios
    """
    empty_dims = {dim: 0.0 for dim in RACE_DIMENSIONS}

    # Format criteria list for the prompt (without weights)
    criteria_list_str = format_criteria_list(criteria_data)

    # Select language-appropriate prompt
    score_prompt_template = SCORE_PROMPT_ZH if language == "zh" else SCORE_PROMPT_EN

    user_prompt = score_prompt_template.format(
        task_prompt=prompt_text,
        article_1=target_report,
        article_2=reference_article,
        criteria_list=criteria_list_str,
    )

    # Call Gemini with retries
    llm_response_str = None
    llm_output_json = None
    success = False

    for retry in range(MAX_RETRIES):
        try:
            llm_response_str = _call_gemini(user_prompt)

            json_str = extract_json_from_markdown(llm_response_str)
            if not json_str:
                raise ValueError("Failed to extract JSON from Gemini response")

            llm_output_json = json.loads(json_str)

            # Validate that ALL expected dimensions exist (matching original)
            missing_dims = [dim for dim in RACE_DIMENSIONS if dim not in llm_output_json]
            if missing_dims:
                raise ValueError(f"Missing expected dimensions: {missing_dims}. Keys: {list(llm_output_json.keys())}")

            success = True
            break

        except Exception as e:
            logger.warning(f"Judge retry {retry + 1}/{MAX_RETRIES}: {e}")
            if retry < MAX_RETRIES - 1:
                time.sleep(1.5 ** (retry + 1))

    if not success or llm_output_json is None:
        return empty_dims, 0.0

    # Calculate weighted scores
    try:
        scores = calculate_weighted_scores(llm_output_json, criteria_data)

        target_total = scores["target"]["total"]
        reference_total = scores["reference"]["total"]

        # Overall score = target / (target + reference)
        overall_score = (
            target_total / (target_total + reference_total)
            if (target_total + reference_total) > 0
            else 0.0
        )

        # Per-dimension normalized scores
        dimension_scores = {}
        for dim in RACE_DIMENSIONS:
            dim_key = f"{dim}_weighted_avg"
            target_score = scores["target"]["dims"].get(dim_key, 0.0)
            reference_score = scores["reference"]["dims"].get(dim_key, 0.0)
            if target_score + reference_score > 0:
                dimension_scores[dim] = target_score / (target_score + reference_score)
            else:
                dimension_scores[dim] = 0.0

        return dimension_scores, overall_score

    except Exception as e:
        logger.error(f"Error calculating scores: {e}")
        return empty_dims, 0.0


# ---------------------------------------------------------------------------
# Result / Metrics dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DeepResearchResult:
    """Result for a single DeepResearch evaluation."""

    sample_id: str
    """Sample identifier"""

    category: str
    """Task category"""

    overall_score: float
    """Overall quality score: target/(target+reference), range 0-1"""

    dimension_scores: Dict[str, float]
    """Per-dimension normalized scores (0-1): comprehensiveness, insight, instruction_following, readability"""

    predicted_report: str
    """Model's generated research report"""

    latency_seconds: float
    """Time taken for this task"""

    error: Optional[str] = None
    """Error message if evaluation failed"""

    model_response: Optional[str] = None
    """Full model response for debugging"""

    # Metadata from orchestrator
    num_turns: int = 0
    """Number of orchestrator turns"""

    tools_used: List[str] = field(default_factory=list)
    """Tools that were called"""

    tools_successful: int = 0
    """Number of successful tool calls"""

    tools_failed: int = 0
    """Number of failed tool calls"""

    conversation: List[Dict[str, str]] = field(default_factory=list)
    """Full conversation history"""

    raw_responses: List[str] = field(default_factory=list)
    """Raw model responses per turn"""


@dataclass
class DeepResearchMetrics:
    """Aggregate metrics for DeepResearch-Bench evaluation."""

    overall_score: float
    """Average overall score (0-1): target/(target+reference)"""

    comprehensiveness_score: float
    """Average comprehensiveness score (0-1)"""

    insight_score: float
    """Average insight score (0-1)"""

    instruction_following_score: float
    """Average instruction following score (0-1)"""

    readability_score: float
    """Average readability score (0-1)"""

    avg_latency: float
    """Average latency per task"""

    total_tasks: int
    """Total number of tasks evaluated"""

    category_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Per-category metrics"""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@register_benchmark(
    name="deepresearch",
    description="DeepResearch-Bench - long-form research generation benchmark (RACE evaluation)",
    domains=["all", "en", "zh"],
    default_domain="all",
    metrics=["overall_score", "comprehensiveness_score", "insight_score",
             "instruction_following_score", "readability_score"],
)
class DeepResearchRunner:
    """Runner for DeepResearch-Bench evaluation using the RACE framework.

    Follows the original implementation exactly:
    https://github.com/Ayanami0730/deep_research_bench

    Example:
        runner = DeepResearchRunner(limit=50, seed=42)
        results = runner.run(model_fn=my_model_inference)
        print(f"Overall Score: {results.overall_score:.4f}")
    """

    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        seed: Optional[int] = None,
        domain: str = "all",
        verbose: bool = False,
        output_dir: Optional[str] = None,
        save_interval_minutes: int = 10,
    ):
        self.split = split
        self.limit = limit
        self.seed = seed
        self.verbose = verbose
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_interval_minutes = save_interval_minutes

        # Convert domain to language filter
        language = None
        if domain in ("en", "zh"):
            language = domain

        self.dataset = DeepResearchDataset(
            split=split,
            limit=limit,
            seed=seed,
            language=language,
        )

    def _load_existing_results(self) -> tuple[list[DeepResearchResult], set[str]]:
        """Load previously saved results for auto-resume."""
        if not self.output_dir:
            return [], set()
        results_file = self.output_dir / "deepresearch_results.jsonl"
        if not results_file.exists():
            return [], set()

        results = []
        completed_ids = set()
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                completed_ids.add(data["sample_id"])
                results.append(DeepResearchResult(
                    sample_id=data["sample_id"],
                    category=data.get("category", ""),
                    overall_score=data.get("overall_score", 0.0),
                    dimension_scores=data.get("dimension_scores", {}),
                    predicted_report=data.get("predicted_report", ""),
                    latency_seconds=data.get("latency_seconds", 0.0),
                    error=data.get("error"),
                    model_response=data.get("model_response"),
                    num_turns=data.get("num_turns", 0),
                    tools_used=data.get("tools_used", []),
                    tools_successful=data.get("tools_successful", 0),
                    tools_failed=data.get("tools_failed", 0),
                    conversation=data.get("conversation", []),
                    raw_responses=data.get("raw_responses", []),
                ))
        return results, completed_ids

    def run(
        self,
        model_fn: Callable[[str, List[Dict]], str],
        orchestrator: bool = True,
    ) -> DeepResearchMetrics:
        """Run evaluation on the benchmark.

        Args:
            model_fn: Model inference function (system_prompt, messages) -> response
            orchestrator: Whether orchestrator mode is enabled. Selects the appropriate system prompt.
        """
        # Auto-resume: load any previously saved results
        existing_results, completed_ids = self._load_existing_results()
        results: List[DeepResearchResult] = list(existing_results)
        if completed_ids:
            print(f"  [Resume] Loaded {len(completed_ids)} existing results from {self.output_dir}")

        last_save_time = time.time()

        # Orchestrator mode builds its own system prompt internally;
        # non-orchestrator uses the model's default.
        system_prompt = ""

        total = len(self.dataset)
        for i, sample in enumerate(self.dataset):
            # Skip already-completed samples (auto-resume)
            if sample.sample_id in completed_ids:
                continue

            print_task_header(
                index=i, total=total,
                task_id=sample.sample_id,
                question=sample.get_prompt(),
                metadata=sample.category,
                verbose=self.verbose,
            )

            result = self._evaluate_sample(sample, model_fn, system_prompt)
            results.append(result)

            self._print_result(result, sample)

            # Periodic intermediate save
            if self.output_dir and (time.time() - last_save_time) >= self.save_interval_minutes * 60:
                self._save_results(results, self._compute_metrics(results))
                print(f"  [Auto-saved {len(results)} results to {self.output_dir}]")
                last_save_time = time.time()

        metrics = self._compute_metrics(results)

        if self.output_dir:
            self._save_results(results, metrics)

        return metrics

    def _evaluate_sample(
        self,
        sample: DeepResearchSample,
        model_fn: Callable,
        system_prompt: str,
    ) -> DeepResearchResult:
        """Evaluate a single sample."""
        start_time = time.time()
        empty_dims = {dim: 0.0 for dim in RACE_DIMENSIONS}

        try:
            prompt = sample.get_prompt()
            messages = [{"role": "user", "content": prompt}]

            response = model_fn(system_prompt, messages)
            latency = time.time() - start_time

            predicted = extract_final_answer(response)

            # Clean both articles before judging (matches original pipeline
            # where both target and reference go through ArticleCleaner)
            if predicted:
                predicted = _clean_article(predicted, language=sample.language)

            # Judge only if we have both criteria and reference
            if sample.criteria and sample.reference_article:
                # Skip reference cleaning if already pre-cleaned
                if self.dataset.references_precleaned:
                    cleaned_reference = sample.reference_article
                else:
                    cleaned_reference = _clean_article(
                        sample.reference_article, language=sample.language
                    )
                dimension_scores, overall_score = judge_research_report(
                    target_report=predicted,
                    prompt_text=sample.prompt,
                    reference_article=cleaned_reference,
                    criteria_data=sample.criteria,
                    language=sample.language,
                )
            else:
                logger.warning(f"Sample {sample.sample_id}: missing criteria or reference, scoring as 0")
                dimension_scores = empty_dims
                overall_score = 0.0

            # Extract orchestrator metadata if available
            num_turns = 0
            tools_used = []
            tools_successful = 0
            tools_failed = 0
            conversation = []
            raw_responses = []

            if hasattr(model_fn, "last_result") and model_fn.last_result is not None:
                orch_result = model_fn.last_result
                num_turns = orch_result.num_turns
                tools_used = orch_result.tools_used
                conversation = orch_result.conversation
                raw_responses = orch_result.raw_responses
                for entry in orch_result.conversation:
                    if entry.get("role") == "tool":
                        content = entry.get("content", "")
                        if content.startswith("Error"):
                            tools_failed += 1
                        else:
                            tools_successful += 1

            return DeepResearchResult(
                sample_id=sample.sample_id,
                category=sample.category,
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                predicted_report=predicted,
                latency_seconds=latency,
                model_response=response,
                num_turns=num_turns,
                tools_used=tools_used,
                tools_successful=tools_successful,
                tools_failed=tools_failed,
                conversation=conversation,
                raw_responses=raw_responses,
            )

        except Exception as e:
            return DeepResearchResult(
                sample_id=sample.sample_id,
                category=sample.category,
                overall_score=0.0,
                dimension_scores=empty_dims,
                predicted_report="",
                latency_seconds=time.time() - start_time,
                error=str(e),
            )

    def _print_result(self, result: DeepResearchResult, sample: DeepResearchSample):
        """Print result using shared formatting."""
        dims = result.dimension_scores
        extra_lines = [
            f"Dimensions: C={dims.get('comprehensiveness', 0):.4f} "
            f"I={dims.get('insight', 0):.4f} "
            f"IF={dims.get('instruction_following', 0):.4f} "
            f"R={dims.get('readability', 0):.4f}",
            f"Report preview: {result.predicted_report[:500]}...",
        ]
        print_result_box(
            status=f"Score: {result.overall_score:.4f}",
            latency_seconds=result.latency_seconds,
            num_turns=result.num_turns,
            tools_used=result.tools_used,
            tools_successful=result.tools_successful,
            tools_failed=result.tools_failed,
            model_response=result.model_response,
            raw_responses=result.raw_responses,
            extra_lines=extra_lines,
            verbose=self.verbose,
        )

    def _compute_metrics(self, results: List[DeepResearchResult]) -> DeepResearchMetrics:
        """Compute aggregate metrics from results."""
        total = len(results)
        if total == 0:
            return DeepResearchMetrics(
                overall_score=0.0, comprehensiveness_score=0.0, insight_score=0.0,
                instruction_following_score=0.0, readability_score=0.0,
                avg_latency=0.0, total_tasks=0,
            )

        overall_score = sum(r.overall_score for r in results) / total

        def avg_dim(dim: str) -> float:
            scores = [r.dimension_scores.get(dim, 0.0) for r in results]
            return sum(scores) / len(scores) if scores else 0.0

        avg_latency = sum(r.latency_seconds for r in results) / total

        # Per-category metrics
        category_metrics = {}
        categories = set(r.category for r in results if r.category)
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            c_total = len(cat_results)
            category_metrics[cat] = {
                "overall_score": sum(r.overall_score for r in cat_results) / c_total,
                "comprehensiveness": sum(r.dimension_scores.get("comprehensiveness", 0) for r in cat_results) / c_total,
                "insight": sum(r.dimension_scores.get("insight", 0) for r in cat_results) / c_total,
                "instruction_following": sum(r.dimension_scores.get("instruction_following", 0) for r in cat_results) / c_total,
                "readability": sum(r.dimension_scores.get("readability", 0) for r in cat_results) / c_total,
                "total": c_total,
            }

        return DeepResearchMetrics(
            overall_score=overall_score,
            comprehensiveness_score=avg_dim("comprehensiveness"),
            insight_score=avg_dim("insight"),
            instruction_following_score=avg_dim("instruction_following"),
            readability_score=avg_dim("readability"),
            avg_latency=avg_latency,
            total_tasks=total,
            category_metrics=category_metrics,
        )

    def _save_results(self, results: List[DeepResearchResult], metrics: DeepResearchMetrics):
        """Save results with full metadata to output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results_file = self.output_dir / "deepresearch_results.jsonl"
        with open(results_file, "w") as f:
            for result in results:
                f.write(json.dumps({
                    "sample_id": result.sample_id,
                    "category": result.category,
                    "overall_score": result.overall_score,
                    "dimension_scores": result.dimension_scores,
                    "predicted_report": result.predicted_report[:5000],
                    "model_response": result.model_response,
                    "latency_seconds": result.latency_seconds,
                    "error": result.error,
                    "num_turns": result.num_turns,
                    "tools_used": result.tools_used,
                    "tools_successful": result.tools_successful,
                    "tools_failed": result.tools_failed,
                    "conversation": result.conversation,
                    "raw_responses": result.raw_responses,
                }) + "\n")

        metrics_file = self.output_dir / "deepresearch_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({
                "overall_score": metrics.overall_score,
                "comprehensiveness_score": metrics.comprehensiveness_score,
                "insight_score": metrics.insight_score,
                "instruction_following_score": metrics.instruction_following_score,
                "readability_score": metrics.readability_score,
                "avg_latency": metrics.avg_latency,
                "total_tasks": metrics.total_tasks,
                "category_metrics": metrics.category_metrics,
            }, f, indent=2)

        print(f"Results saved to {self.output_dir}")
