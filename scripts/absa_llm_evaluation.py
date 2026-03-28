"""
Automated ABSA pipeline evaluation using an external LLM via API.

Validates Aspect-Based Sentiment Analysis outputs from the aspect-based-sentiment
pipeline by sending structured prompts to an LLM and computing metrics from
the parsed JSON responses. Uses the runtime prompt and JSON schema defined in
docs/methodology/RUNTIME EVALUATION PROMPT — ABSA QUALI.md.

Cost estimate (gpt-4o-mini; based on aspect_sentiment_results.parquet):
  - Mean review length: ~151 tokens (chars/4); range 50–472.
  - Per call: ~1,050 input tokens (900 fixed prompt + taxonomy + review), ~400 output.
  - 100 reviews: ~105k input + 40k output → ~$0.04 (input $0.016 + output $0.024).
  - Pricing: input $0.15/1M, output $0.60/1M (check platform.openai.com/docs/pricing).
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

# -----------------------------------------------------------------------------
# Runtime prompt loading
# -----------------------------------------------------------------------------

DEFAULT_RUNTIME_PROMPT_PATH = "docs/methodology/RUNTIME EVALUATION PROMPT — ABSA QUALI.md"


def load_runtime_evaluation_prompt(prompt_path: Optional[Union[str, Path]] = None) -> str:
    """
    Load the runtime evaluation prompt from the methodology docs.

    Args:
        prompt_path: Path to the markdown file. If None, resolves relative to
            project root (when available) or uses embedded fallback.

    Returns:
        Full text of the runtime evaluation prompt.
    """
    if prompt_path is None:
        prompt_path = _resolve_default_prompt_path()
    path = Path(prompt_path)
    if path.exists():
        return path.read_text(encoding="utf-8")
    # Fallback: minimal prompt if file not found (e.g. when run from different cwd)
    return _get_embedded_runtime_prompt()


def _resolve_default_prompt_path() -> Path:
    """Resolve default path to runtime prompt (project root / docs/methodology/...)."""
    try:
        from scripts.project_utils import find_project_root
        root = find_project_root(confirm=False)
        return root / DEFAULT_RUNTIME_PROMPT_PATH
    except Exception:
        pass
    return Path(DEFAULT_RUNTIME_PROMPT_PATH)


def _get_embedded_runtime_prompt() -> str:
    """Embedded copy of the runtime prompt for when the file is not found."""
    return """# RUNTIME EVALUATION PROMPT — ABSA QUALITY ASSESSMENT

You are evaluating an Aspect-Based Sentiment Analysis (ABSA) pipeline applied to tourism-related reviews.

--------------------------------------------------
INPUTS
--------------------------------------------------

REVIEW:
"{review_text}"

PREDICTED ASPECTS & SENTIMENTS:
{formatted_predictions}

ASPECT TAXONOMY:
{aspect_definitions}

--------------------------------------------------
EVALUATION TASK
--------------------------------------------------

For EACH predicted aspect:

1. Determine whether the aspect is ACTUALLY discussed in the review.
   - Answer: true / false

2. If the aspect is present:
   - Determine whether the predicted sentiment is correct.
   - Answer: true / false
   - Provide the correct (gold) sentiment.

Additionally:

3. Identify any MISSING aspects that SHOULD have been detected.
4. Identify any FALSE POSITIVE aspects that should NOT have been detected.

--------------------------------------------------
EVALUATION GUIDELINES
--------------------------------------------------

- Be strict but fair
- Only evaluate aspects in the provided taxonomy
- Ignore unrelated topics
- Handle negation carefully (e.g., "not bad" = mildly positive)
- For mixed sentiment, select the dominant sentiment

--------------------------------------------------
RESPONSE FORMAT (STRICT JSON)
--------------------------------------------------

Return ONLY valid JSON in this exact structure:

{
  "predicted_aspects_evaluation": {
    "aspect_name": {
      "is_present": true | false,
      "sentiment_correct": true | false | null,
      "gold_sentiment": "positive" | "negative" | "neutral" | null,
      "reasoning": "brief explanation"
    }
  },
  "missed_aspects": [
    {
      "aspect": "aspect_name",
      "sentiment": "positive | negative | neutral",
      "evidence": "short quote from the review"
    }
  ],
  "false_positives": ["aspect_name1", "aspect_name2"],
  "overall_quality": "excellent | good | fair | poor"
}

--------------------------------------------------
IMPORTANT
--------------------------------------------------

- Do NOT include explanations outside JSON
- Do NOT include markdown
- Do NOT include additional keys
- Output must be machine-parseable
"""


def format_predictions_for_prompt(predicted_aspects: Dict[str, Any]) -> str:
    """
    Format pipeline aspect output for inclusion in the runtime prompt.

    Args:
        predicted_aspects: Dict mapping aspect name to at least {"sentiment": str}
            (e.g. from pipeline result "aspects" field).

    Returns:
        Human-readable string listing each aspect and its predicted sentiment.
    """
    if not predicted_aspects:
        return "(none)"
    lines = []
    for aspect, info in predicted_aspects.items():
        sentiment = info.get("sentiment", "unknown")
        confidence = info.get("confidence")
        if confidence is not None:
            lines.append(f"  - {aspect}: {sentiment} (confidence: {confidence:.2f})")
        else:
            lines.append(f"  - {aspect}: {sentiment}")
    return "\n".join(lines)


def format_aspect_taxonomy_for_prompt(aspect_definitions: Dict[str, str]) -> str:
    """Format aspect taxonomy (name -> definition) for the prompt."""
    if not aspect_definitions:
        return "(none)"
    return "\n".join(f"  - {k}: {v}" for k, v in aspect_definitions.items())


def build_evaluation_prompt(
    review_text: str,
    predicted_aspects: Dict[str, Any],
    aspect_definitions: Dict[str, str],
    prompt_template: Optional[str] = None,
) -> str:
    """
    Build the full evaluation prompt by filling the runtime template.

    Args:
        review_text: Raw review text.
        predicted_aspects: Pipeline output for this review (aspect name -> sentiment/confidence).
        aspect_definitions: Aspect taxonomy (name -> definition).
        prompt_template: Full prompt text with placeholders. If None, loaded via load_runtime_evaluation_prompt().

    Returns:
        Filled prompt string to send to the LLM.
    """
    if prompt_template is None:
        prompt_template = load_runtime_evaluation_prompt()
    formatted_predictions = format_predictions_for_prompt(predicted_aspects)
    aspect_defs = format_aspect_taxonomy_for_prompt(aspect_definitions)
    return prompt_template.format(
        review_text=review_text,
        formatted_predictions=formatted_predictions,
        aspect_definitions=aspect_defs,
    )


# -----------------------------------------------------------------------------
# LLM response parsing and validation
# -----------------------------------------------------------------------------

EXPECTED_QUALITY_VALUES = frozenset({"excellent", "good", "fair", "poor"})


def _strip_markdown_json(raw: str) -> str:
    """Remove markdown code fences and leading/trailing whitespace."""
    text = raw.strip()
    # Remove ```json ... ``` or ``` ... ```
    match = re.search(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return text


def parse_and_validate_llm_response(raw_response: str) -> Dict[str, Any]:
    """
    Parse LLM response string into JSON and validate expected schema.

    Args:
        raw_response: Raw string returned by the LLM.

    Returns:
        Parsed dict with keys: predicted_aspects_evaluation, missed_aspects,
        false_positives, overall_quality. Missing or invalid fields are normalized.

    Raises:
        ValueError: If JSON is invalid or structure is unusable.
    """
    text = _strip_markdown_json(raw_response)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("LLM response root must be a JSON object")

    out: Dict[str, Any] = {}
    out["predicted_aspects_evaluation"] = data.get("predicted_aspects_evaluation")
    if out["predicted_aspects_evaluation"] is None:
        out["predicted_aspects_evaluation"] = {}
    elif not isinstance(out["predicted_aspects_evaluation"], dict):
        out["predicted_aspects_evaluation"] = {}

    out["missed_aspects"] = data.get("missed_aspects")
    if out["missed_aspects"] is None:
        out["missed_aspects"] = []
    elif not isinstance(out["missed_aspects"], list):
        out["missed_aspects"] = []

    out["false_positives"] = data.get("false_positives")
    if out["false_positives"] is None:
        out["false_positives"] = []
    elif not isinstance(out["false_positives"], list):
        out["false_positives"] = []

    quality = data.get("overall_quality")
    if quality is None or not isinstance(quality, str):
        out["overall_quality"] = "fair"
    else:
        q = quality.strip().lower()
        out["overall_quality"] = q if q in EXPECTED_QUALITY_VALUES else "fair"

    return out


# -----------------------------------------------------------------------------
# LLM client protocol and adapters
# -----------------------------------------------------------------------------


class LLMEvalClient(Protocol):
    """Protocol for an LLM client used by the evaluation pipeline."""

    def complete(self, prompt: str, *, max_tokens: int = 2048, timeout: float = 60.0) -> str:
        """
        Send the prompt to the LLM and return the raw response text.

        Args:
            prompt: Full evaluation prompt.
            max_tokens: Maximum tokens in the response.
            timeout: Request timeout in seconds.

        Returns:
            Raw response body (e.g. content of the first message).
        """
        ...


def create_openai_client(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> LLMEvalClient:
    """
    Create an LLM client that uses the OpenAI API.

    Requires: pip install openai
    Set OPENAI_API_KEY in the environment or pass api_key.

    Args:
        model: Model name (e.g. gpt-4o-mini for cost efficiency).
        api_key: API key; defaults to os.environ["OPENAI_API_KEY"].

    Returns:
        An object implementing complete(prompt, max_tokens=..., timeout=...) -> str.
    """
    try:
        import openai
    except ImportError as e:
        raise ImportError("OpenAI client requires: pip install openai") from e
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI API key not set. Set OPENAI_API_KEY or pass api_key.")
    client = openai.OpenAI(api_key=key)

    class _OpenAIClient:
        def complete(self, prompt: str, *, max_tokens: int = 2048, timeout: float = 60.0) -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                timeout=timeout,
            )
            if not resp.choices:
                raise RuntimeError("OpenAI returned no choices")
            return (resp.choices[0].message.content or "").strip()

    return _OpenAIClient()


def create_anthropic_client(
    model: str = "claude-3-5-haiku-20241022",
    api_key: Optional[str] = None,
) -> LLMEvalClient:
    """
    Create an LLM client that uses the Anthropic API.

    Requires: pip install anthropic
    Set ANTHROPIC_API_KEY in the environment or pass api_key.

    Args:
        model: Model name (e.g. claude-3-5-haiku for cost efficiency).
        api_key: API key; defaults to os.environ["ANTHROPIC_API_KEY"].

    Returns:
        An object implementing complete(prompt, max_tokens=..., timeout=...) -> str.
    """
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("Anthropic client requires: pip install anthropic") from e
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Anthropic API key not set. Set ANTHROPIC_API_KEY or pass api_key.")
    client = anthropic.Anthropic(api_key=key)

    class _AnthropicClient:
        def complete(self, prompt: str, *, max_tokens: int = 2048, timeout: float = 60.0) -> str:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if not resp.content:
                return ""
            return (resp.content[0].text if hasattr(resp.content[0], "text") else str(resp.content[0])).strip()

    return _AnthropicClient()


# -----------------------------------------------------------------------------
# Single-sample evaluation
# -----------------------------------------------------------------------------


def llm_evaluate_sample(
    review: str,
    predicted_aspects: Dict[str, Any],
    client: LLMEvalClient,
    aspect_definitions: Dict[str, str],
    *,
    prompt_template: Optional[str] = None,
    max_tokens: int = 2048,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    Evaluate one review's ABSA predictions using the LLM.

    Formats the runtime evaluation prompt, calls the LLM API, parses and
    validates the JSON response. Handles malformed responses by returning
    a structured error payload instead of raising.

    Args:
        review: Full review text.
        predicted_aspects: Pipeline output for this review (aspect name -> {sentiment, confidence}).
        client: LLM client with complete(prompt, max_tokens=..., timeout=...) -> str.
        aspect_definitions: Aspect taxonomy (name -> definition).
        prompt_template: Optional full prompt text; if None, loaded from file.
        max_tokens: Max response tokens.
        timeout: Request timeout in seconds.

    Returns:
        Dict with at least:
          - "review_preview": first 100 chars of review (for logging).
          - "predicted_aspects_evaluation": per-aspect evaluation from LLM.
          - "missed_aspects": list of missed aspect dicts.
          - "false_positives": list of aspect names.
          - "overall_quality": excellent|good|fair|poor.
          - "parse_error": present and truthy if JSON parsing failed.
          - "api_error": present and truthy if API call failed.
    """
    result: Dict[str, Any] = {
        "review_preview": (review[:100] + "..." if len(review) > 100 else review),
        "predicted_aspects_evaluation": {},
        "missed_aspects": [],
        "false_positives": [],
        "overall_quality": "fair",
    }
    prompt = build_evaluation_prompt(
        review_text=review,
        predicted_aspects=predicted_aspects,
        aspect_definitions=aspect_definitions,
        prompt_template=prompt_template,
    )
    try:
        raw = client.complete(prompt, max_tokens=max_tokens, timeout=timeout)
    except Exception as e:
        result["api_error"] = str(e)
        return result
    try:
        parsed = parse_and_validate_llm_response(raw)
        result.update(parsed)
    except ValueError as e:
        result["parse_error"] = str(e)
    return result


# -----------------------------------------------------------------------------
# Stratified sampling
# -----------------------------------------------------------------------------

RANDOM_SEED = 42


def _get_result_aspects(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract 'aspects' dict from a pipeline result; empty if missing."""
    aspects = result.get("aspects")
    if aspects is None or not isinstance(aspects, dict):
        return {}
    return aspects


def _mean_confidence(aspects: Dict[str, Any]) -> float:
    """Mean confidence of predicted aspects; 0.0 if no aspects."""
    if not aspects:
        return 0.0
    confs = [a.get("confidence") for a in aspects.values() if isinstance(a, dict) and a.get("confidence") is not None]
    if not confs:
        return 0.0
    return sum(confs) / len(confs)


def stratified_sample_indices(
    results: List[Dict[str, Any]],
    sample_size: int,
    *,
    high_confidence_threshold: float = 0.8,
    low_confidence_max: float = 0.7,
    many_aspects_min: int = 4,
    random_seed: int = RANDOM_SEED,
) -> List[int]:
    """
    Select indices for stratified evaluation sampling.

    Strata: random, high-confidence, low-confidence, no-aspect, many-aspect.
    Each stratum is filled up to a fair share; remainder is filled from random.

    Args:
        results: List of pipeline results (each has "text" and "aspects").
        sample_size: Total number of samples to select.
        high_confidence_threshold: Mean confidence >= this => high-confidence.
        low_confidence_max: Mean confidence <= this (and has aspects) => low-confidence.
        many_aspects_min: Number of aspects >= this => many-aspect.
        random_seed: For reproducibility.

    Returns:
        List of indices into results, length <= sample_size.
    """
    rng = random.Random(random_seed)
    n = len(results)
    if n == 0 or sample_size <= 0:
        return []

    no_aspect: List[int] = []
    many_aspect: List[int] = []
    high_conf: List[int] = []
    low_conf: List[int] = []

    for i, res in enumerate(results):
        aspects = _get_result_aspects(res)
        num_aspects = len(aspects)
        mean_conf = _mean_confidence(aspects)

        if num_aspects == 0:
            no_aspect.append(i)
        elif num_aspects >= many_aspects_min:
            many_aspect.append(i)
        elif mean_conf >= high_confidence_threshold:
            high_conf.append(i)
        elif mean_conf <= low_confidence_max and num_aspects > 0:
            low_conf.append(i)

    all_indices = list(range(n))
    rng.shuffle(no_aspect)
    rng.shuffle(many_aspect)
    rng.shuffle(high_conf)
    rng.shuffle(low_conf)
    rng.shuffle(all_indices)

    per_stratum = max(1, sample_size // 5)
    chosen: List[int] = []
    seen: set = set()

    def take_from(pool: List[int], k: int) -> None:
        for idx in pool:
            if len(chosen) >= sample_size:
                return
            if idx not in seen:
                seen.add(idx)
                chosen.append(idx)
                k -= 1
                if k <= 0:
                    return

    take_from(no_aspect, per_stratum)
    take_from(many_aspect, per_stratum)
    take_from(high_conf, per_stratum)
    take_from(low_conf, per_stratum)
    take_from(all_indices, sample_size - len(chosen))

    return chosen[:sample_size]


# -----------------------------------------------------------------------------
# Automated evaluation pipeline
# -----------------------------------------------------------------------------


def automated_evaluation_pipeline(
    results: List[Dict[str, Any]],
    reviews: Optional[List[str]] = None,
    dataset_name: str = "default",
    sample_size: int = 50,
    client: Optional[LLMEvalClient] = None,
    aspect_definitions: Optional[Dict[str, str]] = None,
    *,
    prompt_template: Optional[str] = None,
    rate_limit_delay_seconds: float = 1.0,
    max_tokens: int = 2048,
    timeout: float = 60.0,
    random_seed: int = RANDOM_SEED,
    early_terminate_after_errors: Optional[int] = 5,
) -> List[Dict[str, Any]]:
    """
    Run stratified sampling and evaluate sampled results via the LLM.

    Performs stratified sampling (random, high/low confidence, no-aspect,
    many-aspect), then for each sample calls the LLM evaluator. Tracks
    progress and API usage; optionally respects rate limits and early
    termination after consecutive errors.

    Args:
        results: List of pipeline results (each "text" and "aspects").
        reviews: Optional list of review texts; if None, derived from results (result["text"]).
        dataset_name: Name for logging and output files.
        sample_size: Number of samples to evaluate.
        client: LLM client. If None, attempts to create one from OPENAI_API_KEY or ANTHROPIC_API_KEY.
        aspect_definitions: Aspect taxonomy. If None, caller must ensure prompt has definitions.
        prompt_template: Optional full runtime prompt text.
        rate_limit_delay_seconds: Sleep between API calls.
        max_tokens: Max tokens per response.
        timeout: Request timeout per call.
        random_seed: For stratified sampling.
        early_terminate_after_errors: Stop after this many consecutive API/parse errors; None = do not stop early.

    Returns:
        List of evaluation dicts (one per sampled item), each including
        review_preview, predicted_aspects_evaluation, missed_aspects,
        false_positives, overall_quality, and optional parse_error/api_error.
    """
    if not results:
        return []

    if reviews is None:
        reviews = [r.get("text", "") for r in results]
    if len(reviews) != len(results):
        raise ValueError("reviews length must match results length")

    if aspect_definitions is None:
        aspect_definitions = {}

    if client is None:
        if os.environ.get("OPENAI_API_KEY"):
            client = create_openai_client()
        elif os.environ.get("ANTHROPIC_API_KEY"):
            client = create_anthropic_client()
        else:
            raise ValueError("No LLM client provided and no OPENAI_API_KEY or ANTHROPIC_API_KEY set.")

    indices = stratified_sample_indices(results, sample_size, random_seed=random_seed)
    evaluations: List[Dict[str, Any]] = []
    consecutive_errors = 0
    api_calls = 0
    free_tier_warn_threshold = 50  # Warn when approaching common free-tier limits

    try:
        from tqdm import tqdm
        iterator = tqdm(indices, desc=f"Evaluating {dataset_name}", unit="sample")
    except ImportError:
        iterator = indices

    for idx in iterator:
        if early_terminate_after_errors is not None and consecutive_errors >= early_terminate_after_errors:
            break
        review = reviews[idx]
        result = results[idx]
        predicted_aspects = _get_result_aspects(result)
        eval_out = llm_evaluate_sample(
            review=review,
            predicted_aspects=predicted_aspects,
            client=client,
            aspect_definitions=aspect_definitions,
            prompt_template=prompt_template,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        eval_out["_sample_index"] = idx
        eval_out["_num_predicted_aspects"] = len(predicted_aspects)
        evaluations.append(eval_out)
        api_calls += 1
        if api_calls == free_tier_warn_threshold:
            print(f"\n⚠️  Approaching free-tier limit ({free_tier_warn_threshold} API calls). Consider early termination or reducing sample_size.\n")

        if eval_out.get("parse_error") or eval_out.get("api_error"):
            consecutive_errors += 1
        else:
            consecutive_errors = 0

        if rate_limit_delay_seconds > 0:
            time.sleep(rate_limit_delay_seconds)

    return evaluations


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def calculate_metrics(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aspect detection and sentiment metrics from evaluation results.

    - Aspect detection: Precision, Recall, F1 (based on is_present, missed_aspects, false_positives).
    - Sentiment accuracy: Among predicted aspects that are present (true positives), fraction with correct sentiment.
    - Coverage: Fraction of evaluated reviews with at least one predicted aspect.

    Args:
        evaluations: List of evaluation dicts from automated_evaluation_pipeline.

    Returns:
        Dict with precision, recall, f1, sentiment_accuracy, coverage, and
        supporting counts (tp, fp, fn, total_reviews, etc.).
    """
    metrics: Dict[str, Any] = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "sentiment_accuracy": 0.0,
        "coverage": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "total_evaluated": len(evaluations),
        "reviews_with_at_least_one_aspect": 0,
        "sentiment_correct_total": 0,
        "sentiment_denominator": 0,
    }

    if not evaluations:
        return metrics

    tp = fp = fn = 0
    reviews_with_aspect = 0
    sentiment_correct = 0
    sentiment_denom = 0

    for ev in evaluations:
        if ev.get("parse_error") or ev.get("api_error"):
            continue
        pred_eval = ev.get("predicted_aspects_evaluation") or {}
        missed = ev.get("missed_aspects") or []
        false_pos = ev.get("false_positives") or []

        for aspect, info in pred_eval.items():
            if not isinstance(info, dict):
                continue
            present = info.get("is_present", False)
            if present:
                tp += 1
                sent_ok = info.get("sentiment_correct")
                if sent_ok is True:
                    sentiment_correct += 1
                if sent_ok is not None:
                    sentiment_denom += 1
            else:
                fp += 1

        fn += len(missed)
        # Note: false_positives list is a subset of aspects with is_present=false; do not double-count fp

        num_pred = ev.get("_num_predicted_aspects")
        if num_pred is None:
            num_pred = len(pred_eval) + len(false_pos)
        if num_pred > 0:
            reviews_with_aspect += 1

    total_eval = len([e for e in evaluations if not e.get("parse_error") and not e.get("api_error")])
    if total_eval == 0:
        total_eval = len(evaluations)

    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["total_evaluated"] = len(evaluations)
    metrics["reviews_with_at_least_one_aspect"] = reviews_with_aspect
    metrics["coverage"] = reviews_with_aspect / len(evaluations) if evaluations else 0.0

    if tp + fp > 0:
        metrics["precision"] = tp / (tp + fp)
    if tp + fn > 0:
        metrics["recall"] = tp / (tp + fn)
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])

    metrics["sentiment_correct_total"] = sentiment_correct
    metrics["sentiment_denominator"] = sentiment_denom
    if sentiment_denom > 0:
        metrics["sentiment_accuracy"] = sentiment_correct / sentiment_denom

    return metrics


# -----------------------------------------------------------------------------
# Report and error analysis
# -----------------------------------------------------------------------------


def print_evaluation_report(metrics: Dict[str, Any], dataset_name: str) -> None:
    """
    Print a clean console report with metrics, quality assessment, and recommendation.

    Computes a weighted overall score and outputs a clear recommendation.
    """
    p = metrics.get("precision", 0)
    r = metrics.get("recall", 0)
    f1 = metrics.get("f1", 0)
    sent_acc = metrics.get("sentiment_accuracy", 0)
    cov = metrics.get("coverage", 0)

    # Weighted overall: F1 and sentiment matter most
    overall = 0.4 * f1 + 0.35 * sent_acc + 0.15 * (p + r) / 2 if (p + r) > 0 else 0.0
    overall += 0.1 * cov

    print("=" * 60)
    print(f"ABSA Evaluation Report — {dataset_name}")
    print("=" * 60)
    print()
    print("Aspect detection")
    print(f"  Precision:  {p:.3f}")
    print(f"  Recall:     {r:.3f}")
    print(f"  F1:         {f1:.3f}")
    print()
    print("Sentiment (on true positives)")
    print(f"  Accuracy:   {sent_acc:.3f}")
    print()
    print("Coverage")
    print(f"  Reviews with ≥1 aspect: {cov:.1%}")
    print()
    print("Weighted overall score (0–1):", f"{overall:.3f}")
    if overall >= 0.75:
        quality = "excellent"
    elif overall >= 0.55:
        quality = "good"
    elif overall >= 0.35:
        quality = "fair"
    else:
        quality = "poor"
    print("Quality assessment:", quality)
    print()
    if quality == "excellent":
        print("Recommendation: Pipeline is ready for production use.")
    elif quality == "good":
        print("Recommendation: Pipeline is usable; consider tuning thresholds or expanding taxonomy.")
    elif quality == "fair":
        print("Recommendation: Review error analysis and improve aspect detection or sentiment model.")
    else:
        print("Recommendation: Significant improvements needed; check data and model configuration.")
    print("=" * 60)


def create_error_analysis(
    evaluations: List[Dict[str, Any]],
    dataset_name: str,
    *,
    max_examples_per_aspect: int = 3,
) -> Dict[str, Any]:
    """
    Aggregate sentiment errors, false positives, and missed aspects by aspect;
    include representative examples.

    Args:
        evaluations: List of evaluation dicts from the pipeline.
        dataset_name: Label for the report.
        max_examples_per_aspect: Max number of example reviews to keep per aspect per category.

    Returns:
        Dict with sentiment_errors_by_aspect, false_positives_by_aspect,
        missed_aspects_by_aspect, and representative_examples (per category).
    """
    sentiment_errors: Dict[str, List[Dict[str, Any]]] = {}
    false_positives_by_aspect: Dict[str, List[str]] = {}
    missed_by_aspect: Dict[str, List[Dict[str, Any]]] = {}

    for ev in evaluations:
        if ev.get("parse_error") or ev.get("api_error"):
            continue
        pred_eval = ev.get("predicted_aspects_evaluation") or {}
        missed = ev.get("missed_aspects") or []
        false_pos = ev.get("false_positives") or []
        preview = ev.get("review_preview", "")

        for aspect, info in pred_eval.items():
            if not isinstance(info, dict):
                continue
            if info.get("is_present") and info.get("sentiment_correct") is False:
                sentiment_errors.setdefault(aspect, [])
                if len(sentiment_errors[aspect]) < max_examples_per_aspect:
                    sentiment_errors[aspect].append({
                        "review_preview": preview,
                        "predicted_sentiment": info.get("gold_sentiment"),
                        "reasoning": info.get("reasoning", ""),
                    })

        for asp in false_pos:
            false_positives_by_aspect.setdefault(asp, [])
            if len(false_positives_by_aspect[asp]) < max_examples_per_aspect:
                false_positives_by_aspect[asp].append(preview)

        for item in missed:
            if not isinstance(item, dict):
                continue
            asp = item.get("aspect")
            if not asp:
                continue
            missed_by_aspect.setdefault(asp, [])
            if len(missed_by_aspect[asp]) < max_examples_per_aspect:
                missed_by_aspect[asp].append({
                    "review_preview": preview,
                    "sentiment": item.get("sentiment"),
                    "evidence": item.get("evidence", ""),
                })

    return {
        "dataset_name": dataset_name,
        "sentiment_errors_by_aspect": sentiment_errors,
        "false_positives_by_aspect": false_positives_by_aspect,
        "missed_aspects_by_aspect": missed_by_aspect,
        "representative_examples": {
            "sentiment_errors": sentiment_errors,
            "false_positives": false_positives_by_aspect,
            "missed_aspects": missed_by_aspect,
        },
    }


# -----------------------------------------------------------------------------
# Output and main entry
# -----------------------------------------------------------------------------


def save_evaluation_results(
    evaluations: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    error_analysis: Dict[str, Any],
    dataset_name: str,
    output_dir: Union[str, Path, None] = None,
) -> Path:
    """
    Write evaluation_{dataset_name}.json with full results, metrics, and error analysis.

    Args:
        evaluations: Raw evaluation list.
        metrics: From calculate_metrics.
        error_analysis: From create_error_analysis.
        dataset_name: Used in filename and inside the payload.
        output_dir: Directory for the file; default current directory.

    Returns:
        Path to the written file.
    """
    output_dir = Path(output_dir or ".")
    safe_name = re.sub(r"[^\w\-]", "_", dataset_name).strip("_") or "default"
    path = output_dir / f"evaluation_{safe_name}.json"
    payload = {
        "dataset_name": dataset_name,
        "evaluations": evaluations,
        "metrics": metrics,
        "error_analysis": error_analysis,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_results_from_parquet(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load pipeline results from a parquet file produced by the aspect-based-sentiment notebook.

    Expects columns: "text", "aspects" (dict of aspect name -> {sentiment, confidence}).

    Args:
        path: Path to the parquet file (e.g. aspect_sentiment_results.parquet).

    Returns:
        List of dicts with "text" and "aspects", suitable for automated_evaluation_pipeline.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("Loading parquet requires pandas: pip install pandas") from e
    df = pd.read_parquet(path)
    if "text" not in df.columns or "aspects" not in df.columns:
        raise ValueError("Parquet must contain 'text' and 'aspects' columns")
    return df[["text", "aspects"]].to_dict("records")


def run_evaluation(
    results: List[Dict[str, Any]],
    reviews: Optional[List[str]] = None,
    dataset_name: str = "default",
    aspect_definitions: Optional[Dict[str, str]] = None,
    sample_size: int = 50,
    client: Optional[LLMEvalClient] = None,
    output_dir: Union[str, Path, None] = None,
    **pipeline_kw: Any,
) -> Dict[str, Any]:
    """
    Full evaluation run: sample, evaluate, compute metrics, print report, save JSON.

    Args:
        results: Pipeline results (list of {text, aspects}).
        reviews: Optional; if None, taken from result["text"].
        dataset_name: Name for report and output file.
        aspect_definitions: Aspect taxonomy; required for meaningful evaluation.
        sample_size: Number of samples to evaluate.
        client: LLM client; optional if OPENAI_API_KEY or ANTHROPIC_API_KEY set.
        output_dir: Where to write evaluation_{dataset}.json.
        **pipeline_kw: Passed to automated_evaluation_pipeline.

    Returns:
        Dict with keys: evaluations, metrics, error_analysis, report_path.
    """
    evaluations = automated_evaluation_pipeline(
        results=results,
        reviews=reviews,
        dataset_name=dataset_name,
        sample_size=sample_size,
        client=client,
        aspect_definitions=aspect_definitions or {},
        **pipeline_kw,
    )
    metrics = calculate_metrics(evaluations)
    error_analysis = create_error_analysis(evaluations, dataset_name)
    print_evaluation_report(metrics, dataset_name)
    report_path = save_evaluation_results(
        evaluations, metrics, error_analysis, dataset_name, output_dir=output_dir
    )
    return {
        "evaluations": evaluations,
        "metrics": metrics,
        "error_analysis": error_analysis,
        "report_path": report_path,
    }


# -----------------------------------------------------------------------------
# Runnable example (use from notebook or: python -m scripts.absa_llm_evaluation)
# -----------------------------------------------------------------------------

def _example_aspect_taxonomy() -> Dict[str, str]:
    """Minimal aspect taxonomy for demo; use EVALUATIVE_ASPECTS from the notebook in practice."""
    return {
        "food quality": "quality and taste of food and meals",
        "service quality": "speed, efficiency, and attentiveness of service",
        "cleanliness": "hygiene and tidiness standards",
        "location": "convenience and accessibility of location",
        "value for money": "price fairness relative to quality received",
        "crowding": "crowding, crowded spaces, overcrowding",
    }


if __name__ == "__main__":
    import sys
    # Allow running from project root or from notebooks/users/kristina/shared
    _script_dir = Path(__file__).resolve().parent
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir.parent))
    parquet_path = _script_dir.parent / "aspect_sentiment_results.parquet"
    if not parquet_path.exists() and len(sys.argv) > 1:
        parquet_path = Path(sys.argv[1])
    if not parquet_path.exists():
        print("Usage: python -m scripts.absa_llm_evaluation [path/to/aspect_sentiment_results.parquet]")
        print("Or run from notebook: from scripts.absa_llm_evaluation import run_evaluation, ...")
        sys.exit(0)
    results = load_results_from_parquet(parquet_path)
    aspect_definitions = _example_aspect_taxonomy()
    print(f"Loaded {len(results)} results from {parquet_path}")
    run_evaluation(
        results=results,
        dataset_name="tripadvisor_nyc",
        aspect_definitions=aspect_definitions,
        sample_size=min(20, len(results)),
        output_dir=_script_dir.parent,
    )
