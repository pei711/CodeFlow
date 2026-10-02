"""多步 agent 运行时使用的轻量工作记忆。

session history 负责保存完整事件流；这个模块只保存更小的一层工作集：
当前任务摘要、最近接触的文件、文件短摘要，以及少量跨轮笔记。
这样下一轮 prompt 还能接上上一轮，但不会被整段历史塞满。
"""

import hashlib
import math
import os
import re
import threading
from datetime import datetime
from pathlib import Path

from ..workspace import clip, now, workspace_state_path

WORKING_FILE_LIMIT = 8
EPISODIC_NOTE_LIMIT = 12
FILE_SUMMARY_LIMIT = 6
TURN_SUMMARY_LIMIT = 200
TURN_SUMMARY_TEXT_LIMIT = 800
HYBRID_RETRIEVAL_CANDIDATES = 10
RRF_RANK_CONSTANT = 60
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
_EMBEDDING_MODEL = None
_EMBEDDING_MODEL_NAME = None
_EMBEDDING_LOCK = threading.Lock()
_EMBEDDING_CACHE = {}
_EMBEDDING_CACHE_LIMIT = 512

DURABLE_TOPIC_DEFAULTS = {
    "project-conventions": {
        "title": "Project Conventions",
        "summary": "Stable repository conventions.",
        "tags": ["convention"],
    },
    "key-decisions": {
        "title": "Key Decisions",
        "summary": "Long-lived decisions and rationale anchors.",
        "tags": ["decision"],
    },
    "dependency-facts": {
        "title": "Dependency Facts",
        "summary": "Stable dependency and environment facts.",
        "tags": ["dependency"],
    },
    "user-preferences": {
        "title": "User Preferences",
        "summary": "Stable user preferences.",
        "tags": ["preference"],
    },
}


def default_memory_state():
    # 用一个小而结构化的状态，而不是一大段自由文本摘要。
    return {
        "working": {
            "task_summary": "",
            "recent_files": [],
        },
        "episodic_notes": [],
        "turn_summaries": [],
        "file_summaries": {},
        "task": "",
        "files": [],
        "notes": [],
        "next_note_index": 0,
    }


class DurableMemoryStore:
    def __init__(self, root):
        self.root = Path(root)
        self.index_path = self.root / "MEMORY.md"
        self.topics_dir = self.root / "topics"

    def topic_slugs(self):
        return [topic["topic"] for topic in self.load_index()]

    def load_index(self):
        if not self.index_path.exists():
            return []
        lines = self.index_path.read_text(encoding="utf-8").splitlines()
        topics = []
        current = None
        for raw in lines:
            line = raw.strip()
            match = re.match(r"- \[([^\]]+)\]\([^)]+\):\s*(.+)", line)
            if match:
                current = {
                    "topic": match.group(1).strip(),
                    "title": match.group(2).strip(),
                    "summary": "",
                    "tags": [],
                }
                topics.append(current)
                continue
            if current is None:
                continue
            summary_match = re.match(r"- summary:\s*(.+)", line)
            if summary_match:
                current["summary"] = summary_match.group(1).strip()
                continue
            tags_match = re.match(r"- tags:\s*(.+)", line)
            if tags_match:
                current["tags"] = [tag.strip() for tag in tags_match.group(1).split(",") if tag.strip()]
        return topics

    def load_topic_notes(self, topic):
        path = self.topics_dir / f"{topic}.md"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        notes = []
        capture = False
        updated_at = ""
        tags = []
        for raw in lines:
            line = raw.strip()
            if line.startswith("- tags:"):
                tags = [tag.strip() for tag in line.split(":", 1)[1].split(",") if tag.strip()]
            elif line.startswith("- updated_at:"):
                updated_at = line.split(":", 1)[1].strip()
            elif line == "## Notes":
                capture = True
            elif capture and line.startswith("- "):
                notes.append(
                    {
                        "text": line[2:].strip(),
                        "tags": tags,
                        "source": topic,
                        "created_at": updated_at or now(),
                        "kind": "durable",
                    }
                )
        return notes

    @staticmethod
    def _subject_key(text):
        text = str(text).strip()
        patterns = (
            r"^(.+?)\s+is\s+.+$",
            r"^(.+?)\s+are\s+.+$",
            r"^(.+?)\s+uses?\s+.+$",
            r"^(.+?)\s+should\s+.+$",
            r"^(.+?)是.+$",
            r"^(.+?)使用.+$",
        )
        for pattern in patterns:
            match = re.match(pattern, text, re.I)
            if match:
                subject = " ".join(_tokenize(match.group(1)))
                return subject or None
        return None

    def retrieval_candidates(self, query, limit=3):
        query_tokens = _tokenize(query)
        ranked = []
        for topic in self.load_index():
            notes = self.load_topic_notes(topic["topic"])
            for note in notes:
                note_tags = {tag.lower() for tag in note.get("tags", [])}
                note_tokens = _tokenize(note.get("text", "")) | _tokenize(topic.get("title", "")) | note_tags
                exact_tag_match = int(bool(query_tokens & note_tags))
                keyword_overlap = len(query_tokens & note_tokens)
                if exact_tag_match == 0 and keyword_overlap == 0:
                    continue
                recency = _parse_timestamp(note.get("created_at"))
                ranked.append(((exact_tag_match, keyword_overlap, recency), note))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [note for _, note in ranked[:limit]]

    def _write_index(self, topics):
        self.root.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        lines = ["# Durable Memory Index", ""]
        for topic in topics:
            lines.append(f"- [{topic['topic']}](topics/{topic['topic']}.md): {topic['title']}")
            lines.append(f"  - summary: {topic['summary']}")
            lines.append(f"  - tags: {', '.join(topic['tags'])}")
        self.index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _write_topic(self, topic, notes):
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        meta = DURABLE_TOPIC_DEFAULTS[topic]
        lines = [
            f"# {meta['title']}",
            "",
            f"- topic: {topic}",
            f"- summary: {meta['summary']}",
            f"- tags: {', '.join(meta['tags'])}",
            f"- updated_at: {now()}",
            "",
            "## Notes",
        ]
        for note in notes:
            lines.append(f"- {note}")
        (self.topics_dir / f"{topic}.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def promote(self, promotions):
        if not promotions:
            return [], []
        topics = {topic["topic"]: topic for topic in self.load_index()}
        topic_notes = {slug: [note["text"] for note in self.load_topic_notes(slug)] for slug in topics}
        results = []
        superseded = []
        for topic, note_text in promotions:
            meta = DURABLE_TOPIC_DEFAULTS[topic]
            topics.setdefault(
                topic,
                {
                    "topic": topic,
                    "title": meta["title"],
                    "summary": meta["summary"],
                    "tags": list(meta["tags"]),
                },
            )
            existing = topic_notes.setdefault(topic, [])
            if note_text in existing:
                continue
            new_subject = self._subject_key(note_text)
            replaced = False
            if new_subject:
                for index, old_text in enumerate(list(existing)):
                    if self._subject_key(old_text) == new_subject:
                        superseded.append(f"{topic}: {old_text} -> {note_text}")
                        existing[index] = note_text
                        replaced = True
                        break
            if not replaced:
                existing.append(note_text)
            results.append(f"{topic}: {note_text}")
        self._write_index([topics[slug] for slug in sorted(topics)])
        for topic, notes in topic_notes.items():
            self._write_topic(topic, notes)
        return results, superseded


def _ensure_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def resolve_workspace_path(raw_path, workspace_root=None):
    path = Path(str(raw_path))
    if workspace_root is None:
        return path

    root = Path(workspace_root).resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def canonicalize_path(raw_path, workspace_root=None):
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None:
        return Path(str(raw_path)).as_posix()
    if workspace_root is None:
        return Path(str(raw_path)).as_posix()
    root = Path(workspace_root).resolve()
    return resolved.relative_to(root).as_posix()


def file_freshness(raw_path, workspace_root=None):
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _tokenize(text):
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", str(text))}


def _vector_terms(text):
    """Build sparse text-vector features, including Chinese character n-grams."""
    terms = []
    for token in re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]+", str(text).lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            terms.extend(f"c1:{char}" for char in token)
            terms.extend(f"c2:{token[index:index + 2]}" for index in range(len(token) - 1))
        else:
            terms.append(f"w:{token}")
    return terms


def _tfidf_vector(terms, document_frequency, document_count):
    counts = {}
    for term in terms:
        counts[term] = counts.get(term, 0) + 1
    vector = {}
    for term, count in counts.items():
        inverse_frequency = 1 + math.log((document_count + 1) / (document_frequency.get(term, 0) + 1))
        vector[term] = (1 + math.log(count)) * inverse_frequency
    magnitude = math.sqrt(sum(weight * weight for weight in vector.values()))
    if magnitude:
        return {term: weight / magnitude for term, weight in vector.items()}
    return {}


def _cosine_similarity(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(term, 0.0) for term, weight in left.items())


def _retrieval_terms(text):
    """Tokenize English identifiers and overlapping Chinese character bigrams."""
    terms = []
    for token in re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]+", str(text).lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            if len(token) == 1:
                terms.append(f"c1:{token}")
            else:
                terms.extend(f"c2:{token[index:index + 2]}" for index in range(len(token) - 1))
        else:
            terms.append(f"w:{token}")
    return terms


def _bm25_rank(documents, query, top_k=HYBRID_RETRIEVAL_CANDIDATES, k1=1.2, b=0.75):
    """Return (document index, BM25 score) pairs in descending relevance order."""
    query_terms = _retrieval_terms(query)
    tokenized_documents = [_retrieval_terms(text) for text in documents]
    if not query_terms or not tokenized_documents:
        return []

    document_count = len(tokenized_documents)
    document_frequency = {}
    for terms in tokenized_documents:
        for term in set(terms):
            document_frequency[term] = document_frequency.get(term, 0) + 1
    average_length = sum(map(len, tokenized_documents)) / document_count or 1.0
    scores = []
    for index, terms in enumerate(tokenized_documents):
        term_frequency = {}
        for term in terms:
            term_frequency[term] = term_frequency.get(term, 0) + 1
        score = 0.0
        document_length = len(terms)
        for term in set(query_terms):
            frequency = term_frequency.get(term, 0)
            if not frequency:
                continue
            doc_frequency = document_frequency[term]
            inverse_frequency = math.log(1 + (document_count - doc_frequency + 0.5) / (doc_frequency + 0.5))
            denominator = frequency + k1 * (1 - b + b * document_length / average_length)
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        if score > 0:
            scores.append((index, score))
    scores.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return scores[:max(0, int(top_k))]


def _get_embedding_model(model_name):
    global _EMBEDDING_MODEL, _EMBEDDING_MODEL_NAME
    if _EMBEDDING_MODEL is not None and _EMBEDDING_MODEL_NAME == model_name:
        return _EMBEDDING_MODEL
    with _EMBEDDING_LOCK:
        if _EMBEDDING_MODEL is not None and _EMBEDDING_MODEL_NAME == model_name:
            return _EMBEDDING_MODEL
        from fastembed import TextEmbedding

        _EMBEDDING_MODEL = TextEmbedding(model_name=model_name)
        _EMBEDDING_MODEL_NAME = model_name
        return _EMBEDDING_MODEL


def _cached_embedding(model, model_name, text):
    cache_key = (model_name, hashlib.sha256(text.encode("utf-8")).hexdigest())
    cached = _EMBEDDING_CACHE.get(cache_key)
    if cached is not None:
        return cached
    vector = next(iter(model.embed([text])), None)
    if vector is None:
        raise RuntimeError("embedding model returned no vector")
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    vector = tuple(float(value) for value in vector)
    _EMBEDDING_CACHE[cache_key] = vector
    while len(_EMBEDDING_CACHE) > _EMBEDDING_CACHE_LIMIT:
        _EMBEDDING_CACHE.pop(next(iter(_EMBEDDING_CACHE)))
    return vector


def _dense_cosine_similarity(left, right):
    if not left or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _semantic_rank(documents, query, top_k=HYBRID_RETRIEVAL_CANDIDATES):
    model_name = os.environ.get("CODEFLOW_MEMORY_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    model = _get_embedding_model(model_name)
    query_vector = _cached_embedding(model, model_name, str(query))
    scores = []
    for index, document in enumerate(documents):
        document_vector = _cached_embedding(model, model_name, str(document))
        similarity = _dense_cosine_similarity(query_vector, document_vector)
        if similarity > 0:
            scores.append((index, similarity))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:max(0, int(top_k))]


def _rrf_fuse(vector_hits, keyword_hits, summaries, limit):
    fused_scores = {}
    source_ranks = {}
    representatives = {}
    for source, hits in (("vector", vector_hits), ("bm25", keyword_hits)):
        seen_in_source = set()
        for rank, (index, score) in enumerate(hits, start=1):
            identity = " ".join(str(summaries[index].get("text", "")).split()).casefold()
            if not identity or identity in seen_in_source:
                continue
            seen_in_source.add(identity)
            representatives.setdefault(identity, index)
            fused_scores[identity] = fused_scores.get(identity, 0.0) + 1.0 / (RRF_RANK_CONSTANT + rank)
            source_ranks.setdefault(identity, {})[source] = {"rank": rank, "score": round(float(score), 6)}
    ranked = sorted(
        fused_scores,
        key=lambda identity: (
            fused_scores[identity],
            _parse_timestamp(summaries[representatives[identity]].get("created_at")),
        ),
        reverse=True,
    )
    results = []
    for identity in ranked[:max(0, int(limit))]:
        index = representatives[identity]
        item = summaries[index]
        summary_id = str(item.get("summary_id", "")).strip() or hashlib.sha256(
            item["text"].encode("utf-8")
        ).hexdigest()[:16]
        results.append({
            **item,
            "kind": "conversation_summary",
            "source": item.get("turn_id", "conversation_turn"),
            "retrieval": {
                "mode": "hybrid_rrf",
                "summary_id": summary_id,
                "rrf_score": round(fused_scores[identity], 6),
                "vector": source_ranks[identity].get("vector"),
                "bm25": source_ranks[identity].get("bm25"),
            },
        })
    return results


def summarize_conversation_turn(user_message, final_answer):
    """Create a bounded extractive summary for one completed user/assistant turn."""
    user_text = clip(str(user_message).strip(), 260)
    answer_text = clip(str(final_answer).strip(), 520)
    parts = []
    if user_text:
        parts.append(f"用户目标：{user_text}")
    if answer_text:
        parts.append(f"处理结果：{answer_text}")
    return clip("\n".join(parts), TURN_SUMMARY_TEXT_LIMIT)


def _parse_timestamp(value):
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except Exception:
        return 0.0


def _normalize_note(note, index):
    if isinstance(note, str):
        text = clip(note.strip(), 500)
        return {
            "text": text,
            "tags": [],
            "source": "",
            "created_at": now(),
            "note_index": index,
            "kind": "episodic",
        }

    if not isinstance(note, dict):
        text = clip(str(note).strip(), 500)
        return {
            "text": text,
            "tags": [],
            "source": "",
            "created_at": now(),
            "note_index": index,
            "kind": "episodic",
        }

    text = clip(str(note.get("text", "")).strip(), 500)
    tags = [str(tag).strip() for tag in _ensure_list(note.get("tags", [])) if str(tag).strip()]
    source = str(note.get("source", "")).strip()
    created_at = str(note.get("created_at", "")).strip() or now()
    note_index = int(note.get("note_index", index))
    kind = str(note.get("kind", "episodic")).strip() or "episodic"
    return {
        "text": text,
        "tags": _dedupe_preserve_order(tags),
        "source": source,
        "created_at": created_at,
        "note_index": note_index,
        "kind": kind,
    }


def normalize_memory_state(state, workspace_root=None):
    if state is None:
        state = default_memory_state()
    elif not isinstance(state, dict):
        raise TypeError("memory state must be a mapping")

    # 规范化层的作用，是把“磁盘里可能长得不太一样的旧状态”
    # 统一整理成当前 runtime 可直接使用的紧凑结构。
    working = state.get("working")
    if not isinstance(working, dict):
        working = {}
    working.setdefault("task_summary", "")
    working.setdefault("recent_files", [])
    working["task_summary"] = clip(str(working.get("task_summary", "")).strip(), 300)
    working["recent_files"] = _dedupe_preserve_order(
        [
            canonicalize_path(path, workspace_root)
            for path in _ensure_list(working.get("recent_files", []))
            if str(path).strip()
        ]
    )[-WORKING_FILE_LIMIT:]
    state["working"] = working

    if not str(working["task_summary"]).strip() and state.get("task"):
        working["task_summary"] = clip(str(state.get("task", "")).strip(), 300)
    if not working["recent_files"] and state.get("files"):
        working["recent_files"] = _dedupe_preserve_order(
            [
                canonicalize_path(path, workspace_root)
                for path in _ensure_list(state.get("files", []))
                if str(path).strip()
            ]
        )[-WORKING_FILE_LIMIT:]

    episodic_notes = state.get("episodic_notes")
    if not isinstance(episodic_notes, list):
        episodic_notes = []

    if not episodic_notes and state.get("notes"):
        episodic_notes = [
            _normalize_note(note, index)
            for index, note in enumerate(_ensure_list(state.get("notes", [])))
            if str(note).strip()
        ]
    else:
        normalized_notes = []
        for index, note in enumerate(episodic_notes):
            if isinstance(note, str) and not str(note).strip():
                continue
            normalized_notes.append(_normalize_note(note, index))
        episodic_notes = normalized_notes
    episodic_notes = episodic_notes[-EPISODIC_NOTE_LIMIT:]
    state["episodic_notes"] = episodic_notes

    turn_summaries = state.get("turn_summaries")
    if not isinstance(turn_summaries, list):
        turn_summaries = []
    normalized_turn_summaries = []
    for item in turn_summaries:
        summary_id = ""
        if isinstance(item, str):
            text = clip(item.strip(), TURN_SUMMARY_TEXT_LIMIT)
            created_at = now()
            turn_id = ""
        elif isinstance(item, dict):
            text = clip(str(item.get("text", item.get("summary", ""))).strip(), TURN_SUMMARY_TEXT_LIMIT)
            created_at = str(item.get("created_at", "")).strip() or now()
            turn_id = str(item.get("turn_id", "")).strip()
            summary_id = str(item.get("summary_id", "")).strip()
        else:
            continue
        if text:
            summary_id = summary_id or hashlib.sha256(
                f"{turn_id}\0{created_at}\0{text}".encode()
            ).hexdigest()[:16]
            normalized_turn_summaries.append({
                "text": text,
                "created_at": created_at,
                "turn_id": turn_id,
                "summary_id": summary_id,
            })
    state["turn_summaries"] = normalized_turn_summaries[-TURN_SUMMARY_LIMIT:]

    file_summaries = state.get("file_summaries")
    if not isinstance(file_summaries, dict):
        file_summaries = {}
    normalized_file_summaries = {}
    for path, summary in file_summaries.items():
        path = canonicalize_path(path, workspace_root)
        if isinstance(summary, dict):
            text = clip(str(summary.get("summary", "")).strip(), 500)
            created_at = str(summary.get("created_at", "")).strip() or now()
            freshness = summary.get("freshness")
            freshness = None if freshness in (None, "") else str(freshness).strip() or None
        else:
            text = clip(str(summary).strip(), 500)
            created_at = now()
            freshness = None
        if not path or not text:
            continue
        normalized_file_summaries[path] = {
            "summary": text,
            "created_at": created_at,
            "freshness": freshness,
        }
    state["file_summaries"] = normalized_file_summaries

    next_note_index = state.get("next_note_index")
    if not isinstance(next_note_index, int) or next_note_index < 0:
        next_note_index = 0
    max_index = max([note["note_index"] for note in episodic_notes], default=-1)
    state["next_note_index"] = max(next_note_index, max_index + 1)

    state["task"] = working["task_summary"]
    state["files"] = list(working["recent_files"])
    state["notes"] = [note["text"] for note in episodic_notes]
    durable_root = workspace_state_path(workspace_root, "memory") if workspace_root is not None else None
    durable_store = DurableMemoryStore(durable_root) if durable_root is not None else None
    state["durable_topics"] = durable_store.topic_slugs() if durable_store is not None else []
    return state


def set_task_summary(state, summary, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    state["working"]["task_summary"] = clip(str(summary).strip(), 300)
    state["task"] = state["working"]["task_summary"]
    return state


def remember_file(state, path, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    if not path:
        return state
    files = [item for item in state["working"]["recent_files"] if item != path]
    files.append(path)
    state["working"]["recent_files"] = files[-WORKING_FILE_LIMIT:]
    state["files"] = list(state["working"]["recent_files"])
    return state


def append_note(state, text, tags=(), source="", created_at=None, workspace_root=None, kind="episodic"):
    state = normalize_memory_state(state, workspace_root)
    text = clip(str(text).strip(), 500)
    if not text:
        return state

    normalized_tags = _dedupe_preserve_order(
        [str(tag).strip() for tag in _ensure_list(tags) if str(tag).strip()]
    )
    note = {
        "text": text,
        "tags": normalized_tags,
        "source": str(source).strip(),
        "created_at": str(created_at).strip() if created_at else now(),
        "note_index": int(state.get("next_note_index", 0)),
        "kind": str(kind).strip() or "episodic",
    }
    state["next_note_index"] = note["note_index"] + 1

    notes = [item for item in state["episodic_notes"] if item["text"] != note["text"]]
    notes.append(note)
    state["episodic_notes"] = notes[-EPISODIC_NOTE_LIMIT:]
    state["notes"] = [item["text"] for item in state["episodic_notes"]]
    return state


def append_turn_summary(state, summary, turn_id="", created_at=None, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    text = clip(str(summary).strip(), TURN_SUMMARY_TEXT_LIMIT)
    if not text:
        return state
    timestamp = str(created_at).strip() if created_at else now()
    summary_id = hashlib.sha256(
        f"{turn_id}\0{timestamp}\0{text}".encode()
    ).hexdigest()[:16]
    state["turn_summaries"].append(
        {"text": text, "created_at": timestamp, "turn_id": str(turn_id).strip(), "summary_id": summary_id}
    )
    state["turn_summaries"] = state["turn_summaries"][-TURN_SUMMARY_LIMIT:]
    return state
def set_file_summary(state, path, summary, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    summary = clip(str(summary).strip(), 500)
    if not path or not summary:
        return state
    state["file_summaries"][path] = {
        "summary": summary,
        "created_at": now(),
        "freshness": file_freshness(path, workspace_root),
    }
    return state


def invalidate_file_summary(state, path, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    if not path:
        return state
    state["file_summaries"].pop(path, None)
    return state


def invalidate_stale_file_summaries(state, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    invalidated = []
    for path, summary in list(state["file_summaries"].items()):
        current_freshness = file_freshness(path, workspace_root)
        if summary.get("freshness") == current_freshness:
            continue
        invalidated.append(path)
        state["file_summaries"].pop(path, None)
    return state, invalidated


def summarize_read_result(result, limit=180):
    # 我们不会把完整文件内容塞进记忆层，
    # 这里只保留足够提醒下一轮“刚刚读到了什么”的短摘要。
    lines = [line.strip() for line in str(result).splitlines() if line.strip()]
    if not lines:
        return "(empty)"
    if lines[0].startswith("# "):
        lines = lines[1:]
    if not lines:
        return "(empty)"
    summary = " | ".join(lines[:3])
    return clip(summary, limit)


def retrieval_candidates(state, query, limit=3, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    query_tokens = _tokenize(query)
    ranked = []
    for note in state["episodic_notes"]:
        # 召回逻辑故意保持简单透明：先看 tag 精确命中，
        # 再看关键词重叠，最后看新旧程度。这里不引入 embedding。
        note_tags = {tag.lower() for tag in note.get("tags", [])}
        note_tokens = _tokenize(note.get("text", "")) | _tokenize(note.get("source", "")) | note_tags
        exact_tag_match = int(bool(query_tokens & note_tags))
        keyword_overlap = len(query_tokens & note_tokens)
        if exact_tag_match == 0 and keyword_overlap == 0:
            continue
        recency = _parse_timestamp(note.get("created_at"))
        note_index = int(note.get("note_index", 0))
        ranked.append(((exact_tag_match, keyword_overlap, recency, note_index), note))

    if workspace_root is not None:
        durable_store = DurableMemoryStore(workspace_state_path(workspace_root, "memory"))
        for note in durable_store.retrieval_candidates(query, limit=limit):
            note_tags = {tag.lower() for tag in note.get("tags", [])}
            note_tokens = _tokenize(note.get("text", "")) | _tokenize(note.get("source", "")) | note_tags
            exact_tag_match = int(bool(query_tokens & note_tags))
            keyword_overlap = len(query_tokens & note_tokens)
            recency = _parse_timestamp(note.get("created_at"))
            ranked.append(((exact_tag_match, keyword_overlap, recency, -1), note))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [note for _, note in ranked[:limit]]


def retrieval_view(state, query, limit=3, workspace_root=None):
    candidates = retrieval_candidates(state, query, limit=limit, workspace_root=workspace_root)
    lines = ["Relevant memory:"]
    if not candidates:
        lines.append("- none")
        return "\n".join(lines)
    for note in candidates:
        lines.append(f"- {note['text']}")
    return "\n".join(lines)


def retrieve_turn_summaries(state, query, limit=3, workspace_root=None):
    """Retrieve summaries with semantic vectors and BM25, fused by RRF."""
    state = normalize_memory_state(state, workspace_root)
    summaries = state["turn_summaries"]
    if not summaries or not str(query).strip():
        return []

    documents = [item["text"] for item in summaries]
    if not _retrieval_terms(query):
        return []
    keyword_hits = _bm25_rank(documents, query)
    try:
        vector_hits = _semantic_rank(documents, query)
        vector_mode = "semantic_embedding"
    except Exception:  # noqa: BLE001 - embedding backends expose different runtime errors.
        # Keep retrieval available if the optional model runtime is missing or
        # its model cannot be downloaded on the first run.
        terms_by_document = [_vector_terms(text) for text in documents]
        document_frequency = {}
        for terms in terms_by_document:
            for term in set(terms):
                document_frequency[term] = document_frequency.get(term, 0) + 1
        query_vector = _tfidf_vector(_vector_terms(query), document_frequency, len(documents))
        vector_hits = []
        for index, terms in enumerate(terms_by_document):
            score = _cosine_similarity(
                query_vector,
                _tfidf_vector(terms, document_frequency, len(documents)),
            )
            if score > 0:
                vector_hits.append((index, score))
        vector_hits.sort(
            key=lambda item: (item[1], _parse_timestamp(summaries[item[0]].get("created_at"))),
            reverse=True,
        )
        vector_hits = vector_hits[:HYBRID_RETRIEVAL_CANDIDATES]
        vector_mode = "tfidf_fallback"

    fused = _rrf_fuse(vector_hits, keyword_hits, summaries, limit)
    for item in fused:
        item["retrieval"]["vector_mode"] = vector_mode
    return fused


def render_memory_text(state, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    # 这里渲染的是给模型看的紧凑“仪表盘”，不是完整回放。
    # 笔记正文默认不展开，只有在相关召回时才按需拿出来。
    lines = [
        "Memory:",
        f"- task: {state['working']['task_summary'] or '-'}",
        f"- recent_files: {', '.join(state['working']['recent_files']) or '-'}",
    ]

    summaries = []
    for path in state["working"]["recent_files"][:FILE_SUMMARY_LIMIT]:
        summary = state["file_summaries"].get(path, {})
        current_freshness = file_freshness(path, workspace_root)
        if summary.get("summary", "") and summary.get("freshness") == current_freshness:
            summaries.append(f"- {path}: {summary['summary']}")
    if summaries:
        lines.append("- file_summaries:")
        lines.extend(f"  {line}" for line in summaries)
    else:
        lines.append("- file_summaries: -")

    lines.append(f"- episodic_notes: {len(state['episodic_notes'])}")
    lines.append(f"- conversation_turn_summaries: {len(state['turn_summaries'])}")
    durable_topics = state.get("durable_topics", [])
    lines.append(f"- durable_topics: {', '.join(durable_topics) or '-'}")
    return "\n".join(lines)


def is_effectively_empty(state, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    return (
        not str(state["working"]["task_summary"]).strip()
        and not state["working"]["recent_files"]
        and not state["episodic_notes"]
        and not state["turn_summaries"]
        and not state["file_summaries"]
    )


class LayeredMemory:
    def __init__(self, state=None, workspace_root=None):
        self.workspace_root = workspace_root
        self.state = normalize_memory_state(state, workspace_root)
        self.durable_store = DurableMemoryStore(workspace_state_path(workspace_root, "memory")) if workspace_root is not None else None

    def to_dict(self):
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return self.state

    def canonical_path(self, path):
        return canonicalize_path(path, self.workspace_root)

    def set_task_summary(self, summary):
        self.state = set_task_summary(self.state, summary, self.workspace_root)
        return self

    def remember_file(self, path):
        self.state = remember_file(self.state, path, self.workspace_root)
        return self

    def append_note(self, text, tags=(), source="", created_at=None, kind="episodic"):
        self.state = append_note(
            self.state,
            text,
            tags=tags,
            source=source,
            created_at=created_at,
            workspace_root=self.workspace_root,
            kind=kind,
        )
        return self

    def append_turn_summary(self, summary, turn_id="", created_at=None):
        self.state = append_turn_summary(
            self.state,
            summary,
            turn_id=turn_id,
            created_at=created_at,
            workspace_root=self.workspace_root,
        )
        return self

    def set_file_summary(self, path, summary):
        self.state = set_file_summary(self.state, path, summary, self.workspace_root)
        return self

    def invalidate_file_summary(self, path):
        self.state = invalidate_file_summary(self.state, path, self.workspace_root)
        return self

    def invalidate_stale_file_summaries(self):
        self.state, invalidated = invalidate_stale_file_summaries(self.state, self.workspace_root)
        return invalidated

    def retrieval_candidates(self, query, limit=3):
        return retrieval_candidates(self.state, query, limit=limit, workspace_root=self.workspace_root)

    def relevant_turn_summaries(self, query, limit=3):
        return retrieve_turn_summaries(self.state, query, limit=limit, workspace_root=self.workspace_root)

    def retrieval_view(self, query, limit=3):
        return retrieval_view(self.state, query, limit=limit, workspace_root=self.workspace_root)

    def render_memory_text(self):
        return render_memory_text(self.state, self.workspace_root)

    def promote_durable(self, promotions):
        if self.durable_store is None:
            return [], []
        self.state = normalize_memory_state(self.state, self.workspace_root)
        promoted, superseded = self.durable_store.promote(promotions)
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return promoted, superseded
