import re
import json
from typing import Optional, Union
from qwen_agent.tools.base import BaseTool, register_tool
from rouge_score import rouge_scorer as _rouge_scorer

_scorer = _rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)


def _get_content(message: dict, key: str = "content") -> str:
    """Extract text content from a message dict (handles string or list formats)."""
    value = message.get(key, "")
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        if isinstance(value[0], dict):
            text = value[0].get("text") or ""
            if key == "content":
                recipient = message.get("recipient")
                name = message.get("name")
                if name:
                    text = f"[Tool Response: {name}]\n{text}"
                elif recipient:
                    text = f"[Tool Call: {recipient}]\n{text}"
            return text
    return ""


def _rouge_l_recall(query: str, text: str) -> float:
    """ROUGE-L recall of query against text."""
    if not query or not text:
        return 0.0
    return _scorer.score(query, text)['rougeL'].recall


def truncate_text(text: str, max_words: int = 150) -> str:
    """Truncate text to first n words."""
    if not text:
        return ""
    count = 0
    for m in re.finditer(r'\S+', text):
        count += 1
        if count == max_words:
            return text[:m.end()] + '\n[... truncated]'
    return text


def _count_tokens_approx(messages: list, chars_per_token: float = 4.0) -> int:
    """Approximate token count for a list of messages."""
    total_chars = 0
    for msg in messages:
        for key in ["role", "reasoning_content", "reasoning", "content"]:
            value = msg.get(key)
            if isinstance(value, str):
                total_chars += len(value)
        tool_calls = msg.get("tool_calls")
        if tool_calls is not None:
            total_chars += len(json.dumps(tool_calls, ensure_ascii=False))
    return int(total_chars / chars_per_token)


def format_metadata(trajectories: list) -> str:
    """Format trajectory metadata for inclusion in user prompt."""
    blocks = []
    for i, traj in enumerate(trajectories):
        num_steps = len(traj)
        approx_tokens = _count_tokens_approx(traj)

        tool_counts: dict[str, int] = {}
        for msg in traj:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tc = msg["tool_calls"][0]
                func = tc.get("function", {})
                if not isinstance(func, dict):
                    try:
                        func = func.to_dict()
                    except Exception:
                        func = {}
                name = func.get("name")
                if name:
                    tool_counts[name] = tool_counts.get(name, 0) + 1

        tool_str = ", ".join(f"{n}×{c}" for n, c in sorted(tool_counts.items())) if tool_counts else "none"
        blocks.append(f"Trajectory {i + 1}: {num_steps} steps, ~{approx_tokens:,} tokens | tools: {tool_str}")

    return "\n\n".join(blocks)


@register_tool("get_solution", allow_overwrite=True)
class GetSolutionTool(BaseTool):
    name = "get_solution"
    description = "Retrieves the final content from trajectories' last step. Returns a list of {trajectory_id, content} entries."
    parameters = {
        "type": "object",
        "properties": {
            "trajectory_id": {"type": "integer", "description": "Trajectory index. Omit to retrieve all trajectories."}
        },
        "required": []
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs) -> list:
        trajectories = kwargs.get("trajectories", [])
        trajectory_id = params.get("trajectory_id") if isinstance(params, dict) else None

        if trajectory_id is not None:
            n = len(trajectories)
            if trajectory_id < 1 or trajectory_id > n:
                return f"[get_solution] 'trajectory_id' must be 1-{n}"
            trajs = [(trajectory_id - 1, trajectories[trajectory_id - 1])]
        else:
            trajs = list(enumerate(trajectories))

        results = []
        for i, traj in trajs:
            content = (_get_content(traj[-1]) if traj else None) or ""
            results.append({"trajectory_id": i + 1, "content": content})
        return results

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }


@register_tool("get_segment", allow_overwrite=True)
class GetSegmentTool(BaseTool):
    name = "get_segment"
    description = "Shows a contiguous range of steps (max 5) to reveal the flow around a step -- what the agent did before and after it. Each step is shortened to 600 words, which is below the typical observation length, so this is for context only: never conclude a value is absent from what this tool shows. Use read_tool_output to read any single step's evidence in full."
    parameters = {
        "type": "object",
        "properties": {
            "trajectory_id": {"type": "integer", "description": "Trajectory index."},
            "start_step": {"type": "integer", "description": "Start step (inclusive)."},
            "end_step": {"type": "integer", "description": "End step (inclusive); end_step - start_step ≤ 4."}
        },
        "required": ["trajectory_id", "start_step", "end_step"]
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            trajectory_id = params["trajectory_id"]
            start_step = params["start_step"]
            end_step = params["end_step"]
        except Exception:
            return "[get_segment] Invalid request format: Input must be a JSON object containing 'trajectory_id', 'start_step', and 'end_step' field"

        trajectories = kwargs.get("trajectories", [])
        n_traj = len(trajectories)
        if trajectory_id < 1 or trajectory_id > n_traj:
            return f"[get_segment] 'trajectory_id' must be 1-{n_traj}"
        traj = trajectories[trajectory_id - 1]
        n = len(traj)
        start_step = max(1, min(start_step, n))
        end_step = max(1, min(end_step, n))
        if start_step > end_step:
            start_step = end_step
        if end_step - start_step > 4:
            end_step = start_step + 4
        start_0 = start_step - 1
        end_0 = end_step - 1

        result = []
        for step_idx in range(start_0, end_0 + 1):
            step = traj[step_idx]
            entry = {"step": step_idx + 1, "role": step.get("role", "")}
            content = _get_content(step)
            reasoning = _get_content(step, "reasoning_content") or _get_content(step, "reasoning")
            tool_calls = step.get("tool_calls")
            if content:
                entry["content"] = truncate_text(content, 600)
            if reasoning:
                entry["reasoning"] = truncate_text(reasoning, 600)
            if tool_calls:
                entry["tool_calls"] = tool_calls
            result.append(entry)
        return result

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }


@register_tool("search_tool_outputs", allow_overwrite=True)
class SearchToolOutputsTool(BaseTool):
    name = "search_tool_outputs"
    description = (
        "Searches ONLY the content of tool steps -- the raw observations the environment "
        "returned. The agent's own reasoning is excluded, so a match here means a tool "
        "really returned the term, not that the agent wrote it. This is the tool to use "
        "when tracing a claim to evidence. Each result reports 'is_complete': when false, "
        "the snippet is a window into a longer observation and you must call "
        "read_tool_output(step_id) before concluding anything about what the step does or "
        "does not contain."
    )
    SNIPPET_WORDS = 60
    MAX_K = 10

    parameters = {
        "type": "object",
        "properties": {
            "trajectory_id": {"type": "integer", "description": "Trajectory index."},
            "query": {"type": "string", "description": "The concrete value to look for -- a number, name, or date as it would appear in the source, not a paraphrase."},
            "k": {"type": "integer", "description": "Max matches to return (default 5, max 10)."},
        },
        "required": ["trajectory_id", "query", "k"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    @classmethod
    def _snippet(cls, text: str, query: str) -> tuple[str, bool]:
        """Window of text around the best literal hit for query. Returns (snippet, is_complete)."""
        words = text.split()
        if len(words) <= cls.SNIPPET_WORDS:
            return text.strip(), True

        # centre the window on the longest query token that appears literally
        pos = -1
        for tok in sorted(set(re.findall(r"\S+", query)), key=len, reverse=True):
            if len(tok) < 2:
                continue
            pos = text.lower().find(tok.lower())
            if pos != -1:
                break

        if pos == -1:
            chunk = " ".join(words[: cls.SNIPPET_WORDS])
            return chunk.strip() + " ...", False

        word_idx = len(text[:pos].split())
        start = max(0, word_idx - cls.SNIPPET_WORDS // 2)
        end = min(len(words), start + cls.SNIPPET_WORDS)
        chunk = " ".join(words[start:end]).strip()
        prefix = "... " if start > 0 else ""
        suffix = " ..." if end < len(words) else ""
        return f"{prefix}{chunk}{suffix}", False

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            trajectory_id = params["trajectory_id"]
            query = params["query"]
        except Exception:
            return ("[search_tool_outputs] Invalid request format: Input must be a JSON object "
                    "containing 'trajectory_id' and 'query' field")
        try:
            k = int(params.get("k") or 5)
        except (TypeError, ValueError):
            k = 5
        k = max(1, min(k, self.MAX_K))

        trajectories = kwargs.get("trajectories", [])
        n_traj = len(trajectories)
        if not isinstance(trajectory_id, int) or trajectory_id < 1 or trajectory_id > n_traj:
            return f"[search_tool_outputs] 'trajectory_id' must be 1-{n_traj}"
        if not query or not str(query).strip():
            return "[search_tool_outputs] 'query' must not be empty"
        traj = trajectories[trajectory_id - 1]

        scored = []
        for step_idx, step in enumerate(traj):
            if step.get("role") != "tool":
                continue
            content = _get_content(step)
            if not content:
                continue
            # literal containment is the strongest signal for verification; fall back to
            # ROUGE-L recall so near-misses (formatting, thousands separators) still rank
            literal = str(query).lower() in content.lower()
            score = 1.0 if literal else _rouge_l_recall(str(query), content)
            if score > 0:
                scored.append((score, step_idx, step, content))

        if not scored:
            return (f"[search_tool_outputs] No tool output matches {query!r}. "
                    f"Before treating this as absent, read the most likely step verbatim "
                    f"with read_tool_output -- this search only sees whole-content matches.")

        scored.sort(key=lambda x: (-x[0], x[1]))
        results = []
        for score, step_idx, step, content in scored[:k]:
            snippet, complete = self._snippet(content, str(query))
            results.append({
                "step_id": step_idx + 1,
                "snippet": snippet,
                "is_complete": complete,
            })
        return results

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }


@register_tool("read_tool_output", allow_overwrite=True)
class ReadToolOutputTool(BaseTool):
    name = "read_tool_output"
    description = (
        "Reads one step's content verbatim. This is the only tool that shows a whole "
        "observation: search_tool_outputs returns a 60-word window, get_segment caps each "
        "step at 600 words, and search_trajectory at 150 -- so a value can be absent from "
        "all of their output while still being present in the step. Use this to settle "
        "whether a claim really appears in a long observation such as a scraped page or "
        "PDF. One page holds 8000 words, which covers essentially every step; if a step "
        "is longer still, the result reports a 'next_offset' to pass back."
    )
    # 8000 words covers the longest tool step observed in the shipped rollouts
    # (browsecomp-plus max 4101, deepsearchqa max 5273), so pagination is a safety net
    # rather than something the model has to be relied on to follow up on.
    WORDS_PER_PAGE = 8000

    parameters = {
        "type": "object",
        "properties": {
            "trajectory_id": {"type": "integer", "description": "Trajectory index."},
            "step_id": {"type": "integer", "description": "Step number to read, as reported by search_trajectory or get_segment."},
            "offset": {"type": "integer", "description": "Word offset to start from. Pass 0 for the beginning, then the 'next_offset' from the previous call to continue."},
        },
        "required": ["trajectory_id", "step_id", "offset"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    @staticmethod
    def _slice_words(text: str, offset: int, limit: int) -> tuple[str, int, int]:
        """Return (chunk, total_words, next_offset); next_offset is -1 when exhausted."""
        bounds = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        total = len(bounds)
        if offset >= total:
            return "", total, -1
        start_char = bounds[offset][0]
        end_idx = min(offset + limit, total)
        end_char = bounds[end_idx - 1][1]
        return text[start_char:end_char], total, (end_idx if end_idx < total else -1)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            trajectory_id = params["trajectory_id"]
            step_id = params["step_id"]
            offset = int(params.get("offset") or 0)
        except Exception:
            return ("[read_tool_output] Invalid request format: Input must be a JSON object "
                    "containing 'trajectory_id', 'step_id', and 'offset' field")

        trajectories = kwargs.get("trajectories", [])
        n_traj = len(trajectories)
        if not isinstance(trajectory_id, int) or trajectory_id < 1 or trajectory_id > n_traj:
            return f"[read_tool_output] 'trajectory_id' must be 1-{n_traj}"
        traj = trajectories[trajectory_id - 1]
        n = len(traj)
        if not isinstance(step_id, int) or step_id < 1 or step_id > n:
            return f"[read_tool_output] 'step_id' must be 1-{n}"
        if offset < 0:
            offset = 0

        step = traj[step_id - 1]
        content = _get_content(step)
        reasoning = _get_content(step, "reasoning_content") or _get_content(step, "reasoning")
        entry: dict = {"step": step_id, "role": step.get("role", "")}

        if content:
            chunk, total, nxt = self._slice_words(content, offset, self.WORDS_PER_PAGE)
            entry["content"] = chunk
            entry["content_words_total"] = total
            entry["words_shown"] = f"{offset}-{offset + len(chunk.split())} of {total}"
            if nxt >= 0:
                entry["next_offset"] = nxt
                entry["note"] = (f"Truncated at the page limit, not the end of the step. "
                                 f"Call read_tool_output again with offset={nxt} for the rest.")
        elif offset:
            entry["content"] = ""
            entry["note"] = "This step has no text content; 'offset' does not apply."

        # reasoning is only shown on the first page, and never for tool steps
        if reasoning and not offset and step.get("role") != "tool":
            entry["reasoning"] = reasoning
        if step.get("tool_calls"):
            entry["tool_calls"] = step["tool_calls"]
        return [entry]

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }


@register_tool("search_trajectory", allow_overwrite=True)
class SearchTrajectoriesTool(BaseTool):
    name = "search_trajectory"
    description = "Searches for keywords or phrases within a single trajectory. Returns top matching steps ranked by relevance score."
    parameters = {
        "type": "object",
        "properties": {
            "trajectory_id": {"type": "integer", "description": "Trajectory index to search within."},
            "query": {"type": "string", "description": "Search term or phrase."},
            "role": {"type": "string", "enum": ["tool", "assistant"], "description": "Optional. Filter to 'tool' steps (actual environment observations) or 'assistant' steps only. Omit to search all steps."},
            "k": {"type": "integer", "description": "Max matches to return (default 5, max 10).", "default": 5}
        },
        "required": ["trajectory_id", "query"]
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def _score_traj(self, traj_idx, traj, query, role_filter=None):
        scored = []
        for step_idx, step in enumerate(traj):
            if role_filter is not None and step.get("role", "") != role_filter:
                continue
            content = _get_content(step) or ""
            reasoning_content = _get_content(step, "reasoning_content") or _get_content(step, "reasoning") or ""
            tool_calls_str = json.dumps(step.get("tool_calls"), ensure_ascii=False) if step.get("tool_calls") else ""

            score = max(
                _rouge_l_recall(query, content),
                _rouge_l_recall(query, reasoning_content),
                _rouge_l_recall(query, tool_calls_str),
            )
            if score > 0:
                scored.append((score, traj_idx, step_idx, step))
        return scored

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except Exception:
            return "[search_trajectory] Invalid request format: Input must be a JSON object containing 'query' field"

        trajectory_id = params.get("trajectory_id")
        if trajectory_id is None:
            return "[search_trajectory] 'trajectory_id' is required"
        max_results = min(params.get("k", 5), 10)
        role_filter = params.get("role", None)
        trajectories = kwargs.get("trajectories", [])
        n = len(trajectories)

        if trajectory_id < 1 or trajectory_id > n:
            return f"[search_trajectory] 'trajectory_id' must be 1-{n}"

        scored = self._score_traj(trajectory_id - 1, trajectories[trajectory_id - 1], query, role_filter=role_filter)
        scored.sort(key=lambda x: -x[0])

        matches = []
        for score, traj_idx, step_idx, step in scored[:max_results]:
            content = _get_content(step)
            reasoning_content = _get_content(step, "reasoning_content") or _get_content(step, "reasoning")
            tool_calls = step.get("tool_calls")
            match_entry = {
                "trajectory_id": traj_idx + 1,
                "step": step_idx + 1,
                "role": step.get("role", ""),
                "score": round(score, 3),
            }
            if content:
                match_entry["content"] = truncate_text(content)
            if reasoning_content:
                match_entry["reasoning"] = truncate_text(reasoning_content)
            if tool_calls:
                match_entry["tool_calls"] = tool_calls
            matches.append(match_entry)

        if not matches:
            role_msg = f" (role={role_filter})" if role_filter else ""
            return f"[search_trajectory] No matches found for '{query}'{role_msg} in trajectory {trajectory_id}"
        return matches

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }


@register_tool("finish", allow_overwrite=True)
class FinishTool(BaseTool):
    name = "finish"

    _PROPERTIES = {
        "solution": {"type": "string", "description": "A comprehensive, standalone solution as a single string with two XML sections: <explanation>detailed reasoning leading to the answer</explanation><answer>the exact answer</answer>. The explanation must be self-contained and make sense without any reference to trajectories or aggregation."},
        "solution_report": {"type": "string", "description": "The synthesized long-form response. Write it as a complete, standalone report — do not reference trajectories, agents, or aggregation. Cite using <cite url=\"...\">...</cite> if necessary."},
        "reason": {"type": "string", "description": "Meta-reasoning explaining your decision: how you evaluated the trajectories, what evidence you relied on, and how you resolved any conflicts or inconsistencies."},
        "support_score": {
            "type": "number",
            "description": "How strongly the recorded tool observations establish the final answer, from 0 to 1. 1.0 = fully established; 0.7-0.9 = strongly supported, only non-critical or redundant evidence missing; 0.4-0.7 = partially supported, an important identification or reasoning link unverified; 0.1-0.4 = weak, only isolated parts supported; 0.0 = the observations establish a different answer or contradict this one (including a refuted intermediate claim the answer depends on). Weight missing evidence by how necessary it is, rather than counting clues. At most 0.1 if the agent gave no answer; at most 0.2 if the retrieved material merely echoes the answer (benchmark item, answer key).",
        },
        "confidence": {
            "type": "integer",
            "description": "Confidence in the score you assigned, 0-100. A separate axis from 'score': that rates the answer's grounding, this rates your certainty about your own rating. Only exceed 90 if you read the decisive raw tool output yourself.",
        },
        "verified_answer": {
            "type": "string",
            "description": "The answer as you re-derived it from the tool observations. At score 0 this is the corrected answer the evidence actually supports. If the observations do not let you derive any answer (typical at score 1), use the empty string rather than repeating the agent's claim. Never put a non-answer here (e.g. 'unable to determine', 'not found') -- that is what the empty string is for.",
        },
        "evidence": {
            "type": "string",
            "description": "The specific observations you relied on, each tied to where it was found, e.g. 'trajectory 3 step 60: PDF shows Total imports $948,399,094'. Cite raw tool output, not the agent's reasoning about it. Use the empty string only if you located no relevant observation at all.",
        },
    }

    _QWEN_SOLUTION_DESCRIPTION = (
        "A comprehensive, standalone solution as a single string in the following format:\n"
        "Explanation: {detailed reasoning leading to the answer}\n"
        "Exact Answer: {the exact answer}\n"
        "The explanation must be self-contained and make sense without any reference to trajectories or aggregation."
    )

    # support_score is continuous in [0, 1]; these bands are only for human-readable
    # labelling. Downstream thresholds are a consumer's choice, not fixed here -- the
    # point of a continuous score is that the cut-off can be tuned after the fact.
    SUPPORT_BANDS = (
        (1.0, "fully_established"),
        (0.7, "strongly_supported"),
        (0.4, "partially_supported"),
        (0.1, "weak_support"),
        (0.0, "contradicted_or_none"),
    )

    @classmethod
    def support_label(cls, score: float) -> str:
        for lo, name in cls.SUPPORT_BANDS:
            if score >= lo:
                return name
        return "contradicted_or_none"

    def __init__(self, cfg: Optional[dict] = None, variant: str = "", model: str = ""):
        # variant: "" (default), "long_form", or "verify"
        super().__init__(cfg)
        self.variant = variant
        self.use_qwen_solution = "qwen" in model.lower()
        if variant == "long_form":
            required = ["solution_report", "reason"]
            self.description = "Submits the final synthesized long-form response."
        elif variant == "verify":
            required = ["support_score", "verified_answer", "evidence", "reason"]
            self.description = "Submits the support score (0-1) for the candidate answer."
        else:
            required = ["solution", "reason"]
            self.description = "Submits the final synthesized solution."
        properties = {k: v for k, v in self._PROPERTIES.items() if k in required}
        if self.use_qwen_solution and "solution" in required:
            properties = dict(properties)
            properties["solution"] = {"type": "string", "description": self._QWEN_SOLUTION_DESCRIPTION}
        self.parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        required = self.parameters["required"]
        missing = [f for f in required if f not in params]
        if missing:
            return f"[finish] Invalid request format: missing field(s): {', '.join(missing)}"
        if self.variant == "long_form":
            if not params.get("solution_report", "").strip():
                return "[finish] Invalid format: 'solution_report' must not be empty."
            return {"solution": params["solution_report"], "reason": params.get("reason", "")}
        if self.variant == "verify":
            raw = params["support_score"]
            try:
                score = float(raw)
            except (TypeError, ValueError):
                return (f"[finish] Invalid 'support_score': {raw!r}. "
                        f"Must be a number between 0 and 1.")
            if score != score or not 0.0 <= score <= 1.0:   # NaN or out of range
                return (f"[finish] Invalid 'support_score': {raw!r}. "
                        f"Must be between 0 and 1 inclusive (not a 0-100 percentage).")
            if not str(params.get("reason", "")).strip():
                return "[finish] Invalid format: 'reason' must not be empty."
            return {
                "support_score": score,
                "label": self.support_label(score),
                "verified_answer": params.get("verified_answer", ""),
                "evidence": params.get("evidence", ""),
                "reason": params.get("reason", ""),
            }
        # default: solution + reason
        solution = params.get("solution", "")
        if self.use_qwen_solution:
            explanation_match = re.search(r'Explanation:\s*(.*?)(?=\nExact Answer:|\Z)', solution, re.DOTALL)
            answer_match = re.search(r'Exact Answer:\s*(\S.*)', solution, re.DOTALL)
            if not explanation_match or not explanation_match.group(1).strip():
                return "[finish] Invalid solution format: missing or empty 'Explanation:' section."
            if not answer_match or not answer_match.group(1).strip():
                return "[finish] Invalid solution format: missing or empty 'Exact Answer:' section."
        else:
            explanation_match = re.search(r'<explanation>(.*?)</explanation>', solution, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', solution, re.DOTALL)
            if not explanation_match or not explanation_match.group(1).strip():
                return "[finish] Invalid solution format: missing or empty <explanation>...</explanation> section."
            if not answer_match or not answer_match.group(1).strip():
                return "[finish] Invalid solution format: missing or empty <answer>...</answer> section."
        return {f: params[f] for f in required}

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }
