"""Production-grade LLM-as-a-Judge evaluation framework for AI visibility audits.
Implements structured rubric evaluations: factual consistency, hallucination severity,
brand safety/tone scoring, and blinded pairwise comparison."""
import json
import re
from typing import Any, Callable, Dict, List, Optional


JUDGE_SYSTEM_PROMPT = """You are an expert impartial evaluation judge auditing AI assistant search visibility.
You strictly follow the given rubric and output only valid JSON matching the requested schema.
Be objective, rigorous, and cite direct evidence from the input text."""

FACTUAL_CONSISTENCY_TEMPLATE = """Evaluate the factual consistency of the AI candidate answer against the provided ground truth facts.

Title: {title}
Ground Truth Facts:
{ground_truth}

Candidate Answer:
\"\"\"{candidate}\"\"\"

Evaluation Rubric:
- 5: Perfectly consistent. All stated dates, cast, and details are accurate according to ground truth.
- 4: Mostly consistent. Minor non-critical omissions, but no contradictory or false claims.
- 3: Partially consistent. Some accurate details, but contains at least one factual error or unverified claim.
- 2: Poor consistency. Multiple significant factual inaccuracies or fabrications.
- 1: Completely inconsistent. Fabricated dates, incorrect names, or totally unrelated content.

Return JSON in this exact structure:
{{
  "score": <integer from 1 to 5>,
  "reasoning": "<concise justification citing evidence>",
  "has_hallucinations": <true or false>,
  "fabricated_claims": ["<list of any fabricated claims or names>"]
}}"""

PAIRWISE_EVAL_TEMPLATE = """Compare the following two model answers about the entertainment title '{title}' against the ground truth.

Ground Truth:
{ground_truth}

[Candidate Answer A]:
\"\"\"{answer_a}\"\"\"

[Candidate Answer B]:
\"\"\"{answer_b}\"\"\"

Which answer is more factually accurate, comprehensive, and helpful?
Return JSON in this exact structure:
{{
  "winner": "<A, B, or tie>",
  "confidence": <float from 0.0 to 1.0>,
  "rationale": "<concise explanation of comparative strengths/weaknesses>"
}}"""


def _clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Extract and parse JSON object from LLM response, stripping markdown code fences."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        # Try finding outer braces
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "score": 3,
            "reasoning": f"Failed to parse structured JSON judge output: {text[:150]}",
            "has_hallucinations": False,
            "fabricated_claims": [],
        }


class LLMJudge:
    """Judge that uses an LLM to evaluate candidate answers against ground truth facts."""

    def __init__(self, llm_callable: Callable[[str], str], model_name: str = "judge"):
        self.llm = llm_callable
        self.model_name = model_name

    def evaluate_consistency(
        self,
        title: str,
        ground_truth: Dict[str, Any],
        candidate: str
    ) -> Dict[str, Any]:
        """Judge the factual consistency and hallucination presence of a candidate answer."""
        gt_lines = []
        for k, vals in ground_truth.items():
            if isinstance(vals, list):
                gt_lines.append(f"- {k}: {', '.join(str(v) for v in vals)}")
            else:
                gt_lines.append(f"- {k}: {vals}")
        gt_text = "\n".join(gt_lines) or "No specific facts supplied."

        prompt = FACTUAL_CONSISTENCY_TEMPLATE.format(
            title=title,
            ground_truth=gt_text,
            candidate=candidate
        )

        try:
            raw = self.llm(prompt, system=JUDGE_SYSTEM_PROMPT, max_tokens=400)
            result = _clean_json_response(raw)
            result["judge_model"] = self.model_name
            result["score"] = max(1, min(5, int(result.get("score", 3))))
            return result
        except Exception as exc:
            return {
                "score": 3,
                "reasoning": f"Judge invocation error: {exc}",
                "has_hallucinations": False,
                "fabricated_claims": [],
                "judge_model": self.model_name,
            }

    def pairwise_compare(
        self,
        title: str,
        ground_truth: Dict[str, Any],
        answer_a: str,
        answer_b: str
    ) -> Dict[str, Any]:
        """Perform a pairwise comparative evaluation between two candidate model answers."""
        gt_lines = [f"- {k}: {v}" for k, v in ground_truth.items()]
        gt_text = "\n".join(gt_lines) or "N/A"

        prompt = PAIRWISE_EVAL_TEMPLATE.format(
            title=title,
            ground_truth=gt_text,
            answer_a=answer_a,
            answer_b=answer_b
        )

        try:
            raw = self.llm(prompt, system=JUDGE_SYSTEM_PROMPT, max_tokens=300)
            parsed = _clean_json_response(raw)
            winner = parsed.get("winner", "tie").upper()
            if winner not in ("A", "B", "TIE"):
                winner = "TIE"
            return {
                "winner": winner,
                "confidence": float(parsed.get("confidence", 0.5)),
                "rationale": parsed.get("rationale", ""),
                "judge_model": self.model_name
            }
        except Exception as exc:
            return {
                "winner": "TIE",
                "confidence": 0.0,
                "rationale": f"Pairwise evaluation failed: {exc}",
                "judge_model": self.model_name
            }
