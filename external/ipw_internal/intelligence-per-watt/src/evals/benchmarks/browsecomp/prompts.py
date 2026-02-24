# Prompts for BrowseComp benchmark.
# From openai/simple-evals (https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py)

import re

from .types import ParsedResponse

# Query template from OpenAI simple-evals
_QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

# Judge prompt from OpenAI simple-evals (called GRADER_TEMPLATE in their code)
_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect."""

# Custom instruction for browsing-enabled mode
BROWSING_INSTRUCTION = (
    "You have access to web browsing tools to help find the answer. "
    "Use web_search to search for information, web_open to read web pages, "
    "web_find to search within page content, and web_click to follow links."
)


def format_query(question: str) -> str:
    """Format a question into the query template."""
    return _QUERY_TEMPLATE.format(Question=question)


def format_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    """Format the judge prompt for grading."""
    return _JUDGE_PROMPT.format(question=question, response=response, correct_answer=correct_answer)


def parse_model_response(response: str) -> ParsedResponse:
    """Parse model response to extract Explanation, Exact Answer, and Confidence."""
    errors = []
    
    explanation = re.search(r"Explanation:\s*(.+?)(?=Exact Answer:|Confidence:|$)", response, re.IGNORECASE | re.DOTALL)
    explanation = explanation.group(1).strip() if explanation else ""
    
    exact_answer = re.search(r"Exact Answer:\s*(.+?)(?=Confidence:|$)", response, re.IGNORECASE | re.DOTALL)
    if exact_answer:
        exact_answer = re.sub(r"\s+", " ", exact_answer.group(1).strip())
    else:
        exact_answer = ""
        errors.append("Could not extract 'Exact Answer' field")
    
    confidence = 100.0
    conf_match = re.search(r"Confidence:\s*([0-9.]+)\s*%?", response, re.IGNORECASE)
    if conf_match:
        try:
            conf = float(conf_match.group(1))
            confidence = max(0.0, min(100.0, conf * 100 if conf <= 1.0 and "." in conf_match.group(1) else conf))
        except ValueError:
            errors.append(f"Could not parse confidence: {conf_match.group(1)}")
    else:
        errors.append("Could not extract 'Confidence' field, defaulting to 100")
    
    return ParsedResponse(explanation, exact_answer, confidence, response, errors)
