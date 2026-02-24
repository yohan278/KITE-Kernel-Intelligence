from __future__ import annotations

# System instruction for financial QA
SYSTEM_INSTRUCTION = (
    "You are a financial analyst assistant. Answer questions about financial documents "
    "accurately and concisely based on the provided context or your knowledge."
)

# System instruction requiring Explanation, Exact Answer, Confidence format
SYSTEM_EXACT_ANSWER = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your final answer}\n"
    "Exact Answer: {your succinct, final answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)

# Query template - {Question} is replaced with the question, {Context} with optional evidence
QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

# Query template with context/evidence
QUERY_TEMPLATE_WITH_CONTEXT = """
Context from financial document:
{Context}

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

# Judge prompt for LLM-based grading
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer]. For numerical answers, allow for minor rounding differences. For financial metrics, ensure the values and units match. Focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""

JUDGE_SYSTEM_INSTRUCTION = (
    "You are a strict evaluator for financial question answering. Your task is to determine "
    "whether a model's answer matches the correct reference answer. Focus on semantic equivalence "
    "and correctness of financial values, not on formatting or verbosity."
)


def format_query(question: str, context: str = "") -> str:
    """Format a question into the query template."""
    if context:
        return QUERY_TEMPLATE_WITH_CONTEXT.format(Question=question, Context=context)
    return QUERY_TEMPLATE.format(Question=question)


def format_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    """Format the judge prompt for grading."""
    return JUDGE_PROMPT.format(question=question, response=response, correct_answer=correct_answer)
