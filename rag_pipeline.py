

from collections import Counter, defaultdict
import math
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import os
from openai import OpenAI
import json
import re

BASE_DIR = Path(__file__).resolve().parent
QDRANT_PATH = BASE_DIR / "qdrant_data"
FINES_CSV_PATH = BASE_DIR / "gdpr_csv" / "updated_gdpr_fines.csv"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
collection_name = "gdpr_data"

PRIMARY_GUIDANCE_TOP_N = 3
PRIMARY_TOP_CHUNKS_FOR_SCORE = 3
PRIMARY_CONTEXT_LIMITS_BY_RANK = {
    1: 20,
    2: 3,
    3: 2,
}
PRIMARY_FIRST_DOCUMENT_HARD_LIMIT = 24
PRIMARY_CHUNK_MAX_GAP_FROM_BEST = 0.08
PRIMARY_DIVERSITY_ENABLED = True
PRIMARY_DIVERSITY_RELEVANCE_WEIGHT = 0.75
PRIMARY_DIVERSITY_SIMILARITY_WEIGHT = 0.25
PRIMARY_DUPLICATE_SIMILARITY_THRESHOLD = 0.97
PRIMARY_NEIGHBOUR_EXPANSION_ENABLED = True
PRIMARY_NEIGHBOUR_SCORE_TOLERANCE = 0.05
PRIMARY_MAX_NEIGHBOUR_SHARE = 0.30
SUPPLEMENTARY_MAX_CHUNKS_PER_DOCUMENT = 2
SUPPLEMENTARY_CONTEXT_LIMIT = 5
SUPPLEMENTARY_MIN_CHUNK_SCORE = 0.38
PRIMARY_MAX_SCORE_GAP = 0.12
PRIMARY_MAX_SCORE_WEIGHT = 0.50
PRIMARY_AVERAGE_TOP_CHUNKS_WEIGHT = 0.30
PRIMARY_METADATA_SCORE_WEIGHT = 0.20
PRIMARY_STRONG_TITLE_THRESHOLD = 0.58
PRIMARY_MODERATE_TITLE_THRESHOLD = 0.42
PRIMARY_MODERATE_CHUNK_THRESHOLD = 0.38
PRIMARY_STRONG_CHUNK_THRESHOLD = 0.48
PRIMARY_DOCUMENT_FILTER_FIELDS = ("document_id", "filename", "source", "title")

LEGAL_CSV_RETRIEVAL_LIMIT = 8
LEGAL_CSV_CONTEXT_LIMIT = 5
LEGAL_CASES_IN_ANSWER_LIMIT = 2
PDF_CONTEXT_LIMIT = 35
SCORE_COMPARISON_EPSILON = 1e-12

load_dotenv(BASE_DIR / ".env")

# These objects are intentionally initialized once when the module is loaded.
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
qdrant_client = QdrantClient(path=str(QDRANT_PATH))


def _load_fines_from_qdrant():
    records = []
    offset = None
    csv_filter = Filter(
        must=[
            FieldCondition(
                key="document_type",
                match=MatchValue(value="csv"),
            )
        ]
    )

    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=csv_filter,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            records.append(
                {
                    "country": payload.get("country"),
                    "sector": payload.get("sector"),
                    "fine_eur": payload.get("fine_amount"),
                    "controller_processor": payload.get(
                        "controller_processor", "N/A"
                    ),
                    "date": payload.get("date", "N/A"),
                    "summary": payload.get(
                        "summary",
                        payload.get("violation description", "N/A"),
                    ),
                }
            )

        if offset is None:
            break

    if not records:
        raise RuntimeError(
            f"No enforcement records were found in Qdrant collection "
            f"{collection_name!r} at {QDRANT_PATH}."
        )

    return pd.DataFrame.from_records(records)


def _load_fines_dataframe():
    if FINES_CSV_PATH.is_file():
        return pd.read_csv(FINES_CSV_PATH)

    return _load_fines_from_qdrant()


fines_df = _load_fines_dataframe()


DOCUMENT_URL_FIELDS = ("source_url", "url", "link")


def get_document_url(payload):
    for field_name in DOCUMENT_URL_FIELDS:
        value = payload.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def is_safe_document_url(url):
    return (
        isinstance(url, str)
        and url.strip().lower().startswith(("https://", "http://"))
    )


def escape_markdown_link_text(text):
    return str(text).replace("[", "\\[").replace("]", "\\]")


def format_document_link(title: str, url: str | None) -> str:
    title = str(title or "").strip() or "Untitled document"

    if not is_safe_document_url(url):
        return title

    return f"[{escape_markdown_link_text(title)}]({url.strip()})"


def cosine_similarity(vector_a, vector_b):
    values_a = vector_a.tolist() if hasattr(vector_a, "tolist") else vector_a
    values_b = vector_b.tolist() if hasattr(vector_b, "tolist") else vector_b
    numerator = sum(a * b for a, b in zip(values_a, values_b))
    norm_a = math.sqrt(sum(a * a for a in values_a))
    norm_b = math.sqrt(sum(b * b for b in values_b))

    if not norm_a or not norm_b:
        return 0

    return numerator / (norm_a * norm_b)


def get_primary_guidance(query, pdf_results, top_n=PRIMARY_GUIDANCE_TOP_N):

    grouped = defaultdict(list)
    for result in pdf_results:
        payload = result.payload or {}
        title = payload.get("title")

        if not title:
            continue

        if "REGULATION (EU) 2016/679" in title:
            continue

        grouped[title].append(result)

    if not grouped:
        return [], []

    query_vector = model.encode(query)
    candidates = []

    for title, chunks in grouped.items():
        ranked_chunks = sorted(
            chunks,
            key=lambda chunk: chunk.score or 0,
            reverse=True,
        )
        top_chunks = ranked_chunks[:PRIMARY_TOP_CHUNKS_FOR_SCORE]
        top_scores = [chunk.score or 0 for chunk in top_chunks]
        max_chunk_score = max(top_scores) if top_scores else 0
        average_top_3_score = (
            sum(top_scores) / len(top_scores)
            if top_scores
            else 0
        )
        payload = ranked_chunks[0].payload or {}
        topic = str(payload.get("topic") or "").strip()
        title_score = cosine_similarity(query_vector, model.encode(title))
        topic_score = (
            cosine_similarity(query_vector, model.encode(topic))
            if topic
            else 0
        )
        metadata_score = max(title_score, topic_score)
        document_score = (
            PRIMARY_MAX_SCORE_WEIGHT * max_chunk_score
            + PRIMARY_AVERAGE_TOP_CHUNKS_WEIGHT * average_top_3_score
            + PRIMARY_METADATA_SCORE_WEIGHT * metadata_score
        )

        strong_metadata_and_moderate_chunks = (
            metadata_score >= PRIMARY_STRONG_TITLE_THRESHOLD
            and average_top_3_score >= PRIMARY_MODERATE_CHUNK_THRESHOLD
        )
        moderate_metadata_and_strong_chunk = (
            metadata_score >= PRIMARY_MODERATE_TITLE_THRESHOLD
            and max_chunk_score >= PRIMARY_STRONG_CHUNK_THRESHOLD
        )
        accepted = (
            strong_metadata_and_moderate_chunks
            or moderate_metadata_and_strong_chunk
        )
        rejection_reason = ""
        if not accepted:
            rejection_reason = "below metadata/chunk relevance thresholds"

        candidates.append(
            {
                "title": title,
                "url": get_document_url(payload),
                "score": document_score,
                "document_score": document_score,
                "title_score": title_score,
                "topic_score": topic_score,
                "metadata_score": metadata_score,
                "max_chunk_score": max_chunk_score,
                "average_top_3_score": average_top_3_score,
                "retrieved_chunk_count": len(chunks),
                "chunks": ranked_chunks,
                "accepted": accepted,
                "rejection_reason": rejection_reason,
            }
        )

    candidates = sorted(
        candidates,
        key=lambda candidate: candidate["document_score"],
        reverse=True,
    )
    accepted_candidates = [
        candidate for candidate in candidates
        if candidate["accepted"]
    ]

    if accepted_candidates:
        best_score = accepted_candidates[0]["document_score"]
        for candidate in accepted_candidates:
            if best_score - candidate["document_score"] > PRIMARY_MAX_SCORE_GAP:
                candidate["accepted"] = False
                candidate["rejection_reason"] = "score gap from best accepted document"

    primary_guidance = [
        candidate for candidate in accepted_candidates
        if candidate["accepted"]
    ][:top_n]

    return primary_guidance, candidates
# Weighted pdf context


def get_primary_context_limit_for_rank(rank):
    return PRIMARY_CONTEXT_LIMITS_BY_RANK.get(
        rank,
        PRIMARY_CONTEXT_LIMITS_BY_RANK[3],
    )


def get_primary_hard_limit_for_rank(rank):
    return PRIMARY_FIRST_DOCUMENT_HARD_LIMIT if rank == 1 else (
        get_primary_context_limit_for_rank(rank)
    )


def get_chunk_id(chunk):
    return getattr(chunk, "id", id(chunk))


def get_chunk_text(chunk):
    return str((chunk.payload or {}).get("text", ""))


def get_chunk_vector(chunk):
    vector = getattr(chunk, "vector", None)
    if isinstance(vector, dict):
        vector = next(iter(vector.values()), None)

    if vector is None:
        return None

    return vector.tolist() if hasattr(vector, "tolist") else vector


def tokenize_for_similarity(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def cosine_similarity_from_counters(counter_a, counter_b):
    if not counter_a or not counter_b:
        return 0

    numerator = sum(
        counter_a[token] * counter_b.get(token, 0)
        for token in counter_a
    )
    norm_a = math.sqrt(sum(value * value for value in counter_a.values()))
    norm_b = math.sqrt(sum(value * value for value in counter_b.values()))

    if not norm_a or not norm_b:
        return 0

    return numerator / (norm_a * norm_b)


def text_overlap_similarity(text_a, text_b):
    tokens_a = set(tokenize_for_similarity(text_a))
    tokens_b = set(tokenize_for_similarity(text_b))

    if not tokens_a or not tokens_b:
        return 0

    return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))


def calculate_chunk_similarity(chunk_a, chunk_b):
    vector_a = get_chunk_vector(chunk_a)
    vector_b = get_chunk_vector(chunk_b)

    if vector_a is not None and vector_b is not None:
        return cosine_similarity(vector_a, vector_b)

    text_a = get_chunk_text(chunk_a)
    text_b = get_chunk_text(chunk_b)
    counter_similarity = cosine_similarity_from_counters(
        Counter(tokenize_for_similarity(text_a)),
        Counter(tokenize_for_similarity(text_b)),
    )
    overlap_similarity = text_overlap_similarity(text_a, text_b)

    return max(counter_similarity, overlap_similarity)


def get_chunk_order_value(chunk):
    payload = chunk.payload or {}
    for field_name in ("chunk_index", "chunk_id", "position", "sequence_number"):
        value = payload.get(field_name)
        if isinstance(value, int):
            return value

        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())

    return None


def get_adjacent_chunks(chunk, all_document_chunks):
    order_value = get_chunk_order_value(chunk)
    if order_value is None:
        return []

    title = (chunk.payload or {}).get("title")
    adjacent_chunks = []

    for candidate in all_document_chunks:
        if candidate is chunk:
            continue

        payload = candidate.payload or {}
        if payload.get("title") != title:
            continue

        candidate_order = get_chunk_order_value(candidate)
        if candidate_order is None:
            continue

        if abs(candidate_order - order_value) == 1:
            adjacent_chunks.append(candidate)

    return sorted(
        adjacent_chunks,
        key=lambda candidate: get_chunk_order_value(candidate) or 0,
    )


def select_diverse_primary_chunks(
    chunks,
    limit,
    minimum_eligible_score,
    all_document_chunks=None,
    relevance_weight=PRIMARY_DIVERSITY_RELEVANCE_WEIGHT,
    similarity_weight=PRIMARY_DIVERSITY_SIMILARITY_WEIGHT,
    duplicate_threshold=PRIMARY_DUPLICATE_SIMILARITY_THRESHOLD,
    enable_neighbours=PRIMARY_NEIGHBOUR_EXPANSION_ENABLED,
):
    ranked_chunks = sorted(
        chunks,
        key=lambda chunk: chunk.score or 0,
        reverse=True,
    )
    selection_debug = {
        "diversity_selection_enabled": PRIMARY_DIVERSITY_ENABLED,
        "candidate_chunks_before_deduplication": len(ranked_chunks),
        "near_duplicate_chunks_skipped": 0,
        "diverse_chunks_selected": 0,
        "independent_chunks_added": 0,
        "max_neighbour_chunks": math.floor(
            limit * PRIMARY_MAX_NEIGHBOUR_SHARE
        ),
        "neighbour_chunks_added": 0,
        "neighbour_cap_reached": False,
        "final_chunks_added": 0,
        "selected_chunks": [],
    }

    if not ranked_chunks or limit <= 0:
        return [], selection_debug

    if not PRIMARY_DIVERSITY_ENABLED:
        selected = ranked_chunks[:limit]
        selection_debug["diverse_chunks_selected"] = len(selected)
        selection_debug["independent_chunks_added"] = len(selected)
        selection_debug["final_chunks_added"] = len(selected)
        selection_debug["selected_chunks"] = [
            {
                "chunk_id": get_chunk_id(chunk),
                "chunk_score": chunk.score or 0,
                "selection_method": "top_score",
                "maximum_similarity_to_previous_selected": 0,
            }
            for chunk in selected
        ]
        return selected, selection_debug

    independent_target = limit - selection_debug["max_neighbour_chunks"]
    selected = [ranked_chunks[0]]
    selected_ids = {get_chunk_id(ranked_chunks[0])}
    remaining = ranked_chunks[1:]
    best_score = ranked_chunks[0].score or 0
    score_range = max(best_score - minimum_eligible_score,
                      SCORE_COMPARISON_EPSILON)
    selection_debug["diverse_chunks_selected"] = 1
    selection_debug["independent_chunks_added"] = 1
    selection_debug["selected_chunks"].append(
        {
            "chunk_id": get_chunk_id(ranked_chunks[0]),
            "chunk_score": ranked_chunks[0].score or 0,
            "selection_method": "top_score",
            "maximum_similarity_to_previous_selected": 0,
        }
    )

    if (
        enable_neighbours
        and len(selected) < limit
        and selection_debug["independent_chunks_added"] >= independent_target
    ):
        neighbour_minimum_score = (
            minimum_eligible_score - PRIMARY_NEIGHBOUR_SCORE_TOLERANCE
        )
        for neighbour in get_adjacent_chunks(
            ranked_chunks[0],
            all_document_chunks or chunks,
        ):
            if len(selected) >= limit:
                break

            if (
                selection_debug["neighbour_chunks_added"]
                >= selection_debug["max_neighbour_chunks"]
            ):
                selection_debug["neighbour_cap_reached"] = True
                break

            neighbour_id = get_chunk_id(neighbour)
            if neighbour_id in selected_ids:
                continue

            if (neighbour.score or 0) < neighbour_minimum_score:
                continue

            maximum_similarity = max(
                calculate_chunk_similarity(neighbour, selected_chunk)
                for selected_chunk in selected
            )
            if maximum_similarity >= duplicate_threshold:
                selection_debug["near_duplicate_chunks_skipped"] += 1
                continue

            selected.append(neighbour)
            selected_ids.add(neighbour_id)
            remaining = [
                candidate for candidate in remaining
                if get_chunk_id(candidate) != neighbour_id
            ]
            selection_debug["neighbour_chunks_added"] += 1
            selection_debug["selected_chunks"].append(
                {
                    "chunk_id": neighbour_id,
                    "chunk_score": neighbour.score or 0,
                    "selection_method": "neighbour",
                    "maximum_similarity_to_previous_selected": maximum_similarity,
                }
            )

    while remaining and len(selected) < limit:
        best_candidate = None
        best_candidate_score = None
        best_candidate_similarity = 0
        near_duplicates = []

        for candidate in remaining:
            maximum_similarity = max(
                calculate_chunk_similarity(candidate, selected_chunk)
                for selected_chunk in selected
            )

            if maximum_similarity >= duplicate_threshold:
                near_duplicates.append(candidate)
                continue

            normalized_relevance_score = (
                (candidate.score or 0) - minimum_eligible_score
            ) / score_range
            normalized_relevance_score = max(
                0,
                min(normalized_relevance_score, 1),
            )
            mmr_score = (
                relevance_weight * normalized_relevance_score
                - similarity_weight * maximum_similarity
            )

            if best_candidate is None or mmr_score > best_candidate_score:
                best_candidate = candidate
                best_candidate_score = mmr_score
                best_candidate_similarity = maximum_similarity

        if best_candidate is None:
            selection_debug["near_duplicate_chunks_skipped"] += len(
                near_duplicates)
            break

        selected.append(best_candidate)
        selected_ids.add(get_chunk_id(best_candidate))
        remaining = [
            candidate for candidate in remaining
            if get_chunk_id(candidate) != get_chunk_id(best_candidate)
        ]
        selection_debug["near_duplicate_chunks_skipped"] += len(
            near_duplicates)
        selection_debug["diverse_chunks_selected"] += 1
        selection_debug["independent_chunks_added"] += 1
        selection_debug["selected_chunks"].append(
            {
                "chunk_id": get_chunk_id(best_candidate),
                "chunk_score": best_candidate.score or 0,
                "selection_method": "diversity",
                "maximum_similarity_to_previous_selected": best_candidate_similarity,
            }
        )

        if (
            not enable_neighbours
            or len(selected) >= limit
            or selection_debug["independent_chunks_added"] < independent_target
        ):
            continue

        neighbour_minimum_score = (
            minimum_eligible_score - PRIMARY_NEIGHBOUR_SCORE_TOLERANCE
        )
        for neighbour in get_adjacent_chunks(
            best_candidate,
            all_document_chunks or chunks,
        ):
            if len(selected) >= limit:
                break

            if (
                selection_debug["neighbour_chunks_added"]
                >= selection_debug["max_neighbour_chunks"]
            ):
                selection_debug["neighbour_cap_reached"] = True
                break

            neighbour_id = get_chunk_id(neighbour)
            if neighbour_id in selected_ids:
                continue

            if (neighbour.score or 0) < neighbour_minimum_score:
                continue

            maximum_similarity = max(
                calculate_chunk_similarity(neighbour, selected_chunk)
                for selected_chunk in selected
            )
            if maximum_similarity >= duplicate_threshold:
                selection_debug["near_duplicate_chunks_skipped"] += 1
                continue

            selected.append(neighbour)
            selected_ids.add(neighbour_id)
            remaining = [
                candidate for candidate in remaining
                if get_chunk_id(candidate) != neighbour_id
            ]
            selection_debug["neighbour_chunks_added"] += 1
            selection_debug["selected_chunks"].append(
                {
                    "chunk_id": neighbour_id,
                    "chunk_score": neighbour.score or 0,
                    "selection_method": "neighbour",
                    "maximum_similarity_to_previous_selected": maximum_similarity,
                }
            )

    selection_debug["final_chunks_added"] = len(selected)
    return selected, selection_debug


def build_weighted_pdf_context(
    pdf_results,
    primary_guidance=None,
    max_chunks=PDF_CONTEXT_LIMIT,
):

    grouped = defaultdict(list)
    for p in pdf_results:
        title = p.payload.get("title", "Unknown")
        if not title:
            continue

        if "REGULATION (EU) 2016/679" in title:
            continue

        if title:
            grouped[title].append(p)

    document_scores = {}

    for title, chunks in grouped.items():
        top_scores = sorted(
            (p.score or 0 for p in chunks),
            reverse=True,
        )[:3]

        document_scores[title] = (
            sum(top_scores) / len(top_scores)
            if top_scores
            else 0
        )

    ranked_titles = sorted(
        document_scores,
        key=document_scores.get,
        reverse=True
    )

    selected_chunks = []
    selected_ids = set()
    primary_guidance = primary_guidance or []
    primary_titles = {guidance["title"] for guidance in primary_guidance}
    primary_allocation = []
    supplementary_chunk_count = 0

    for rank, guidance in enumerate(primary_guidance, start=1):
        ranked_chunks = sorted(
            guidance.get("chunks", []),
            key=lambda chunk: chunk.score or 0,
            reverse=True,
        )
        assigned_limit = get_primary_context_limit_for_rank(rank)
        best_score = (ranked_chunks[0].score or 0) if ranked_chunks else 0
        minimum_eligible_score = best_score - PRIMARY_CHUNK_MAX_GAP_FROM_BEST
        eligible_chunks = [
            chunk for chunk in ranked_chunks
            if (chunk.score or 0) >= minimum_eligible_score - SCORE_COMPARISON_EPSILON
        ]
        hard_limit = get_primary_hard_limit_for_rank(rank)
        selected_initial_chunks, selection_debug = select_diverse_primary_chunks(
            eligible_chunks,
            assigned_limit,
            minimum_eligible_score,
            all_document_chunks=ranked_chunks,
        )

        primary_allocation.append(
            {
                "rank": rank,
                "title": guidance["title"],
                "assigned_chunk_limit": assigned_limit,
                "hard_limit": hard_limit,
                "best_chunk_score": best_score,
                "minimum_eligible_score": minimum_eligible_score,
                "retrieved_chunk_count": len(ranked_chunks),
                "eligible_chunk_count": len(eligible_chunks),
                "global_chunks_for_document": guidance.get(
                    "global_chunks_for_document",
                    len(ranked_chunks),
                ),
                "document_filtered_chunks_retrieved": guidance.get(
                    "document_filtered_chunks_retrieved",
                    0,
                ),
                "merged_unique_chunks": guidance.get(
                    "merged_unique_chunks",
                    len(ranked_chunks),
                ),
                "eligible_chunks_after_merge": len(eligible_chunks),
                "document_filter_field": guidance.get("document_filter_field"),
                "document_search_limit": guidance.get("document_search_limit"),
                "initially_added_chunk_count": 0,
                "redistributed_added_chunk_count": 0,
                "added_chunk_count": 0,
                "eligible_chunks": eligible_chunks,
                "ranked_chunks": ranked_chunks,
                "selected_initial_chunks": selected_initial_chunks,
                "selection_debug": selection_debug,
            }
        )

    for allocation in primary_allocation:
        if len(selected_chunks) >= max_chunks:
            break

        for chunk in allocation["selected_initial_chunks"]:
            if len(selected_chunks) >= max_chunks:
                break

            chunk_id = get_chunk_id(chunk)
            if chunk_id in selected_ids:
                continue

            selected_chunks.append(chunk)
            selected_ids.add(chunk_id)
            allocation["initially_added_chunk_count"] += 1
            allocation["added_chunk_count"] += 1

    unused_primary_capacity = sum(
        max(
            allocation["assigned_chunk_limit"]
            - allocation["initially_added_chunk_count"],
            0,
        )
        for allocation in primary_allocation[1:]
    )
    redistribution_occurred = False

    if primary_allocation and unused_primary_capacity > 0:
        first_allocation = primary_allocation[0]
        selected_with_redistribution, redistribution_selection_debug = (
            select_diverse_primary_chunks(
                first_allocation["eligible_chunks"],
                PRIMARY_FIRST_DOCUMENT_HARD_LIMIT,
                first_allocation["minimum_eligible_score"],
                all_document_chunks=first_allocation["ranked_chunks"],
            )
        )
        redistribution_debug_by_id = {
            selected_chunk["chunk_id"]: selected_chunk
            for selected_chunk in redistribution_selection_debug["selected_chunks"]
        }
        first_extra_capacity = min(
            unused_primary_capacity,
            PRIMARY_FIRST_DOCUMENT_HARD_LIMIT
            - first_allocation["added_chunk_count"],
            max_chunks - len(selected_chunks),
        )

        for chunk in selected_with_redistribution:
            if first_extra_capacity <= 0:
                break

            chunk_id = get_chunk_id(chunk)
            if chunk_id in selected_ids:
                continue

            selected_chunks.append(chunk)
            selected_ids.add(chunk_id)
            first_allocation["redistributed_added_chunk_count"] += 1
            first_allocation["added_chunk_count"] += 1
            first_extra_capacity -= 1
            redistribution_occurred = True
            redistributed_chunk_debug = redistribution_debug_by_id.get(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "chunk_score": chunk.score or 0,
                    "selection_method": "diversity",
                    "maximum_similarity_to_previous_selected": 0,
                },
            )
            first_allocation["selection_debug"]["selected_chunks"].append(
                redistributed_chunk_debug
            )
            if redistributed_chunk_debug["selection_method"] == "neighbour":
                first_allocation["selection_debug"]["neighbour_chunks_added"] += 1
            else:
                first_allocation["selection_debug"]["independent_chunks_added"] += 1

        first_allocation["selection_debug"]["near_duplicate_chunks_skipped"] += (
            redistribution_selection_debug["near_duplicate_chunks_skipped"]
        )
        first_allocation["selection_debug"]["max_neighbour_chunks"] = (
            redistribution_selection_debug["max_neighbour_chunks"]
        )
        first_allocation["selection_debug"]["neighbour_cap_reached"] = (
            first_allocation["selection_debug"]["neighbour_cap_reached"]
            or redistribution_selection_debug["neighbour_cap_reached"]
        )
        first_allocation["selection_debug"]["final_chunks_added"] = (
            first_allocation["added_chunk_count"]
        )

    for allocation in primary_allocation:
        allocation["selection_debug"]["final_chunks_added"] = (
            allocation["added_chunk_count"]
        )
        allocation.pop("ranked_chunks", None)

    for title in ranked_titles:
        if title in primary_titles:
            continue

        if supplementary_chunk_count >= SUPPLEMENTARY_CONTEXT_LIMIT:
            break

        chunks = sorted(
            grouped[title],
            key=lambda p: p.score or 0,
            reverse=True
        )
        chunks_to_add = [
            chunk for chunk in chunks
            if (chunk.score or 0) >= SUPPLEMENTARY_MIN_CHUNK_SCORE
        ][:SUPPLEMENTARY_MAX_CHUNKS_PER_DOCUMENT]

        for chunk in chunks_to_add:
            if supplementary_chunk_count >= SUPPLEMENTARY_CONTEXT_LIMIT:
                break

            chunk_id = get_chunk_id(chunk)
            if chunk_id in selected_ids:
                continue

            selected_chunks.append(chunk)
            selected_ids.add(chunk_id)
            supplementary_chunk_count += 1

            if len(selected_chunks) >= max_chunks:
                break

        if len(selected_chunks) >= max_chunks:

            break

    selected_chunks = selected_chunks[:max_chunks]

    pdf_context = "\n\n".join([

        f"[PDF score: {p.score:.4f} | title: {p.payload.get('title', 'Unknown')}]\n"

        f"{p.payload.get('text', '')}"

        for p in selected_chunks

    ])

    allocation_metadata = {
        "primary_allocations": primary_allocation,
        "primary_chunk_count": sum(
            allocation["added_chunk_count"]
            for allocation in primary_allocation
        ),
        "supplementary_chunk_count": supplementary_chunk_count,
        "pdf_context_limit": max_chunks,
        "primary_chunk_max_gap_from_best": PRIMARY_CHUNK_MAX_GAP_FROM_BEST,
        "primary_first_document_hard_limit": PRIMARY_FIRST_DOCUMENT_HARD_LIMIT,
        "unused_primary_capacity": unused_primary_capacity,
        "redistribution_occurred": redistribution_occurred,
        "supplementary_context_limit": SUPPLEMENTARY_CONTEXT_LIMIT,
        "supplementary_min_chunk_score": SUPPLEMENTARY_MIN_CHUNK_SCORE,
    }

    return pdf_context, selected_chunks, allocation_metadata


# Fine statistics from the full dataset

def calculate_fine_statistics_from_dataset(query):
    df = fines_df.copy()
    query_lower = query.lower()
    typo_replacements = {
        "austra": "austria",
        "belgum": "belgium",
        "bulgery": "bulgaria",
        "croacia": "croatia",
        "czeck republic": "czech republic",
        "czec republic": "czech republic",
        "chech republic": "czech republic",
        "denmak": "denmark",
        "estiona": "estonia",
        "finalnd": "finland",
        "finlad": "finland",
        "franse": "france",
        "geramny": "germany",
        "gremany": "germany",
        "germnay": "germany",
        "greecee": "greece",
        "hungry": "hungary",
        "icland": "iceland",
        "irealnd": "ireland",
        "itlay": "italy",
        "lativa": "latvia",
        "lithunia": "lithuania",
        "luxemberg": "luxembourg",
        "netherland": "netherlands",
        "nederlands": "netherlands",
        "norwey": "norway",
        "polan": "poland",
        "portugual": "portugal",
        "romainia": "romania",
        "slovkia": "slovakia",
        "slovania": "slovenia",
        "spaine": "spain",
        "swedan": "sweden",
        "unighted kingdom": "united kingdom",
        "untied kingdom": "united kingdom",
        "united kindgom": "united kingdom",
        "endland": "england",
        "emploemnt": "employment",
        "employement": "employment",
        "emplyment": "employment",
    }
    for typo, replacement in typo_replacements.items():
        query_lower = query_lower.replace(typo, replacement)

    def query_contains(term):
        escaped = re.escape(term.lower())
        return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", query_lower) is not None

    def format_eur(value):
        if pd.isna(value):
            return "N/A"
        return f"EUR {value:,.0f}"

    def format_case(row):
        return (
            f"{format_eur(row.get('fine_eur'))} | "
            f"{row.get('country', 'N/A')} | "
            f"{row.get('sector', 'N/A')} | "
            f"{row.get('controller_processor', 'N/A')} | "
            f"{row.get('date', 'N/A')} | "
            f"{row.get('summary', 'N/A')}"
        )

    if "fine_eur" not in df.columns:
        return "Fine statistics are unavailable because the dataset has no fine_eur column."

    df = df.copy()
    df["fine_eur"] = pd.to_numeric(df["fine_eur"], errors="coerce")
    df = df.dropna(subset=["fine_eur"])

    if df.empty:
        return "No valid fine data found in the dataset."

    country_aliases = {
        "austria": "Austria",
        "austrian": "Austria",
        "osterreich": "Austria",
        "oesterreich": "Austria",
        "belgium": "Belgium",
        "belgian": "Belgium",
        "belgie": "Belgium",
        "belgique": "Belgium",
        "bulgaria": "Bulgaria",
        "bulgarian": "Bulgaria",
        "croatia": "Croatia",
        "croatian": "Croatia",
        "hrvatska": "Croatia",
        "cyprus": "Cyprus",
        "cypriot": "Cyprus",
        "czech republic": "Czech Republic",
        "czechia": "Czech Republic",
        "czech": "Czech Republic",
        "cesko": "Czech Republic",
        "denmark": "Denmark",
        "danish": "Denmark",
        "danmark": "Denmark",
        "estonia": "Estonia",
        "estonian": "Estonia",
        "eesti": "Estonia",
        "finland": "Finland",
        "finnish": "Finland",
        "suomi": "Finland",
        "france": "France",
        "french": "France",
        "germany": "Germany",
        "german": "Germany",
        "deutschland": "Germany",
        "alemania": "Germany",
        "allemagne": "Germany",
        "германия": "Germany",
        "германии": "Germany",
        "германский": "Germany",
        "германская": "Germany",
        "greece": "Greece",
        "greek": "Greece",
        "hellas": "Greece",
        "ellada": "Greece",
        "hungary": "Hungary",
        "hungarian": "Hungary",
        "magyarorszag": "Hungary",
        "iceland": "Iceland",
        "icelandic": "Iceland",
        "island": "Iceland",
        "ireland": "Ireland",
        "irish": "Ireland",
        "eire": "Ireland",
        "italy": "Italy",
        "italian": "Italy",
        "italia": "Italy",
        "latvia": "Latvia",
        "latvian": "Latvia",
        "latvija": "Latvia",
        "liechtenstein": "Liechtenstein",
        "lithuania": "Lithuania",
        "lithuanian": "Lithuania",
        "lietuva": "Lithuania",
        "luxembourg": "Luxembourg",
        "luxemburg": "Luxembourg",
        "malta": "Malta",
        "maltese": "Malta",
        "netherlands": "Netherlands",
        "the netherlands": "Netherlands",
        "holland": "Netherlands",
        "dutch": "Netherlands",
        "norway": "Norway",
        "norwegian": "Norway",
        "norge": "Norway",
        "poland": "Poland",
        "polish": "Poland",
        "polska": "Poland",
        "portugal": "Portugal",
        "portuguese": "Portugal",
        "romania": "Romania",
        "romanian": "Romania",
        "slovakia": "Slovakia",
        "slovak": "Slovakia",
        "slovensko": "Slovakia",
        "slovenia": "Slovenia",
        "slovenian": "Slovenia",
        "slovenija": "Slovenia",
        "spain": "Spain",
        "spanish": "Spain",
        "espana": "Spain",
        "españa": "Spain",
        "sweden": "Sweden",
        "swedish": "Sweden",
        "sverige": "Sweden",
        "united kingdom": "United Kingdom",
        "uk": "United Kingdom",
        "u k": "United Kingdom",
        "u.k.": "United Kingdom",
        "britain": "United Kingdom",
        "great britain": "United Kingdom",
        "england": "United Kingdom",
        "english": "United Kingdom",
        "british": "United Kingdom",
    }

    for country in df["country"].dropna().astype(str).unique():
        country_aliases[country.lower()] = country

    country_filter = None
    for alias, country in sorted(country_aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if country in set(df["country"].dropna().astype(str).unique()) and query_contains(alias):
            country_filter = country
            break

    if country_filter:
        df = df[df["country"].astype(
            str).str.lower() == country_filter.lower()]

    if df.empty:
        return f"No matching fine data found for country: {country_filter}."

    sector_keywords = {
        "industry": "Industry and Commerce",
        "commerce": "Industry and Commerce",
        "commercial": "Industry and Commerce",
        "business": "Industry and Commerce",
        "ecommerce": "Industry and Commerce",
        "e-commerce": "Industry and Commerce",
        "media": "Media, Telecoms and Broadcasting",
        "telecom": "Media, Telecoms and Broadcasting",
        "telecoms": "Media, Telecoms and Broadcasting",
        "telecommunications": "Media, Telecoms and Broadcasting",
        "broadcasting": "Media, Telecoms and Broadcasting",
        "finance": "Finance, Insurance and Consulting",
        "financial": "Finance, Insurance and Consulting",
        "financial sector": "Finance, Insurance and Consulting",
        "финансы": "Finance, Insurance and Consulting",
        "финансов": "Finance, Insurance and Consulting",
        "финансовый": "Finance, Insurance and Consulting",
        "финансовая": "Finance, Insurance and Consulting",
        "финансовом секторе": "Finance, Insurance and Consulting",
        "финансовый сектор": "Finance, Insurance and Consulting",
        "банковский": "Finance, Insurance and Consulting",
        "банк": "Finance, Insurance and Consulting",
        "банки": "Finance, Insurance and Consulting",
        "страхование": "Finance, Insurance and Consulting",
        "консалтинг": "Finance, Insurance and Consulting",
        "bank": "Finance, Insurance and Consulting",
        "banking": "Finance, Insurance and Consulting",
        "insurance": "Finance, Insurance and Consulting",
        "consulting": "Finance, Insurance and Consulting",
        "public sector": "Public Sector and Education",
        "government": "Public Sector and Education",
        "public": "Public Sector and Education",
        "education": "Public Sector and Education",
        "school": "Public Sector and Education",
        "university": "Public Sector and Education",
        "health": "Health Care",
        "health care": "Health Care",
        "healthcare": "Health Care",
        "medical": "Health Care",
        "hospital": "Health Care",
        "clinic": "Health Care",
        "employment": "Employment",
        "employee": "Employment",
        "employees": "Employment",
        "worker": "Employment",
        "workers": "Employment",
        "hr": "Employment",
        "human resources": "Employment",
        "transport": "Transportation and Energy",
        "transportation": "Transportation and Energy",
        "energy": "Transportation and Energy",
        "utility": "Transportation and Energy",
        "utilities": "Transportation and Energy",
        "accomodation": "Accomodation and Hospitality",
        "accommodation": "Accomodation and Hospitality",
        "hotel": "Accomodation and Hospitality",
        "hospitality": "Accomodation and Hospitality",
        "travel": "Accomodation and Hospitality",
        "tourism": "Accomodation and Hospitality",
        "restaurant": "Accomodation and Hospitality",
        "real estate": "Real Estate",
        "property": "Real Estate",
        "housing": "Real Estate",
        "individual": "Individuals and Private Associations",
        "individuals": "Individuals and Private Associations",
        "association": "Individuals and Private Associations",
        "private association": "Individuals and Private Associations",
        "non-profit": "Individuals and Private Associations",
        "ngo": "Individuals and Private Associations",
        "person": "Individuals and Private Associations",
        "private person": "Individuals and Private Associations",
    }

    for sector in df["sector"].dropna().astype(str).unique():
        sector_keywords[sector.lower()] = sector

    sector_filter = None
    for keyword, sector in sorted(sector_keywords.items(), key=lambda item: len(item[0]), reverse=True):
        if query_contains(keyword):
            sector_filter = sector
            break

    if sector_filter:
        df = df[df["sector"].astype(str).str.lower() == sector_filter.lower()]

    if df.empty:
        filters = []
        if country_filter:
            filters.append(f"country: {country_filter}")
        if sector_filter:
            filters.append(f"sector: {sector_filter}")
        return f"No valid fine data found for {'; '.join(filters)}."

    max_row = df.loc[df["fine_eur"].idxmax()]
    min_row = df.loc[df["fine_eur"].idxmin()]
    mean_fine = df["fine_eur"].mean()
    median_fine = df["fine_eur"].median()
    total_fines = df["fine_eur"].sum()
    top_fines = df.sort_values("fine_eur", ascending=False).head(10)
    bottom_fines = df.sort_values("fine_eur", ascending=True).head(5)

    top_fines_text = "\n".join(
        [f"- {format_case(row)}" for _, row in top_fines.iterrows()])
    bottom_fines_text = "\n".join(
        [f"- {format_case(row)}" for _, row in bottom_fines.iterrows()])

    group_by_country_requested = any(
        phrase in query_lower
        for phrase in [
            "by country",
            "per country",
            "group by country",
            "grouped by country",
            "across countries",
            "countries",
            "по странам",
            "по стране",
            "по государствам",
        ]
    )
    group_by_sector_requested = any(
        phrase in query_lower
        for phrase in [
            "by sector",
            "per sector",
            "group by sector",
            "grouped by sector",
            "sectors",
            "industries",
            "по секторам",
            "по сектору",
            "по отраслям",
            "по отрасли",
        ]
    )

    requested_sort = "max"
    if any(word in query_lower for word in ["median", "middle", "медиан"]):
        requested_sort = "median"
    elif any(word in query_lower for word in ["average", "mean", "avg", "средн"]):
        requested_sort = "mean"
    elif any(word in query_lower for word in ["minimum", "lowest", "smallest", "min", "миним"]):
        requested_sort = "min"

    def grouped_stats(group_column, label):
        grouped = (
            df.groupby(group_column, dropna=False)["fine_eur"]
            .agg(count="count", min="min", max="max", mean="mean", median="median", total="sum")
            .reset_index()
            .sort_values(requested_sort, ascending=False)
            .head(15)
        )
        lines = []
        for _, row in grouped.iterrows():
            lines.append(
                f"- {row[group_column]} | cases: {int(row['count'])} | "
                f"min: {format_eur(row['min'])} | "
                f"max: {format_eur(row['max'])} | "
                f"mean: {format_eur(row['mean'])} | "
                f"median: {format_eur(row['median'])} | "
                f"total: {format_eur(row['total'])}"
            )
        return f"\n{label} grouped statistics, sorted by {requested_sort}:\n" + "\n".join(lines)

    grouped_sections = []
    if group_by_country_requested or not country_filter:
        grouped_sections.append(grouped_stats("country", "Country"))
    if group_by_sector_requested or not sector_filter:
        grouped_sections.append(grouped_stats("sector", "Sector"))
    grouped_text = "\n".join(grouped_sections)

    return f"""
FINE STATISTICS FROM FULL DATASET:

- Country filter: {country_filter if country_filter else "All countries"}
- Sector filter: {sector_filter if sector_filter else "All sectors"}

- Number of matching cases: {len(df)}
- Total fines in matching cases: {format_eur(total_fines)}

- Highest fine: {format_eur(max_row['fine_eur'])}
- Highest fine country: {max_row.get('country', 'N/A')}
- Highest fine sector: {max_row.get('sector', 'N/A')}
- Highest fine controller/processor: {max_row.get('controller_processor', 'N/A')}
- Highest fine date: {max_row.get('date', 'N/A')}
- Highest fine description: {max_row.get('summary', 'N/A')}

- Lowest fine: {format_eur(min_row['fine_eur'])}
- Lowest fine country: {min_row.get('country', 'N/A')}
- Lowest fine sector: {min_row.get('sector', 'N/A')}
- Lowest fine controller/processor: {min_row.get('controller_processor', 'N/A')}

- Mean fine: {format_eur(mean_fine)}
- Median fine: {format_eur(median_fine)}

Top 10 highest matching fines:

{top_fines_text}

Top 5 lowest matching fines:

{bottom_fines_text}

{grouped_text}
"""


def get_csv_case_key(point):
    payload = point.payload or {}
    for field_name in ("case_id", "id", "row_id"):
        value = payload.get(field_name)
        if value is not None:
            return f"{field_name}:{value}"

    text = " ".join(str(payload.get("text", "")).lower().split())
    if text:
        return f"text:{text}"

    fields = (
        payload.get("country"),
        payload.get("sector"),
        payload.get("fine_amount"),
        payload.get("violated_articles"),
        payload.get("violation description"),
    )
    return "fields:" + "|".join(str(value or "").lower() for value in fields)


def deduplicate_csv_results(csv_results, limit):
    selected = []
    seen = set()

    for result in csv_results:
        key = get_csv_case_key(result)
        if key in seen:
            continue

        seen.add(key)
        selected.append(result)

        if len(selected) >= limit:
            break

    return selected


def query_qdrant_by_document_type(
    query_vector,
    document_type,
    limit,
    with_vectors=False,
):
    return qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="document_type",
                    match=MatchValue(value=document_type)
                )
            ]
        ),
        with_payload=True,
        with_vectors=with_vectors,
        limit=limit,
    ).points


def get_document_filter_identity(chunks, title):
    for chunk in chunks:
        payload = chunk.payload or {}
        for field_name in PRIMARY_DOCUMENT_FILTER_FIELDS:
            value = payload.get(field_name)
            if isinstance(value, str):
                value = value.strip()
            if value:
                return field_name, value

    return "title", title


def query_qdrant_for_primary_document(
    query_vector,
    field_name,
    field_value,
    limit,
):
    return qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="document_type",
                    match=MatchValue(value="pdf")
                ),
                FieldCondition(
                    key=field_name,
                    match=MatchValue(value=field_value)
                ),
            ]
        ),
        with_payload=True,
        with_vectors=True,
        limit=limit,
    ).points


def merge_unique_chunks(*chunk_groups):
    merged_by_id = {}

    for chunks in chunk_groups:
        for chunk in chunks:
            chunk_id = get_chunk_id(chunk)
            existing = merged_by_id.get(chunk_id)
            if existing is None or (chunk.score or 0) > (existing.score or 0):
                merged_by_id[chunk_id] = chunk

    return sorted(
        merged_by_id.values(),
        key=lambda chunk: chunk.score or 0,
        reverse=True,
    )


def add_document_filtered_primary_chunks(query_vector, primary_guidance):
    for rank, guidance in enumerate(primary_guidance, start=1):
        global_chunks = guidance.get("chunks", [])
        assigned_limit = get_primary_context_limit_for_rank(rank)
        hard_limit = get_primary_hard_limit_for_rank(rank)
        document_search_limit = max(hard_limit * 3, hard_limit + 5)
        field_name, field_value = get_document_filter_identity(
            global_chunks,
            guidance["title"],
        )
        document_filtered_chunks = query_qdrant_for_primary_document(
            query_vector,
            field_name,
            field_value,
            document_search_limit,
        )
        merged_chunks = merge_unique_chunks(
            global_chunks,
            document_filtered_chunks,
        )

        guidance["chunks"] = merged_chunks
        guidance["document_filter_field"] = field_name
        guidance["document_filter_value"] = field_value
        guidance["global_chunks_for_document"] = len(global_chunks)
        guidance["document_filtered_chunks_retrieved"] = len(
            document_filtered_chunks
        )
        guidance["merged_unique_chunks"] = len(merged_chunks)
        guidance["document_search_limit"] = document_search_limit
        guidance["assigned_chunk_limit"] = assigned_limit
        guidance["hard_limit"] = hard_limit

    return primary_guidance


def build_csv_context(csv_results):
    return "\n\n".join([
        f"[CSV score: {p.score:.4f}]\n"
        f"{p.payload.get('text', '')}"
        for p in csv_results
    ])


# automatical retrive of context from the query


def retrieve_context(
    query,
    pdf_limit=50,
    csv_limit=25,
    debug=False,
    return_metadata=False,
    route="legal",
):
    query_vector = model.encode(query).tolist()
    fine_statistics = None
    fine_statistics_calculated = False
    pdf_results = []
    csv_results = []
    csv_context_results = []
    primary_guidance = []
    primary_guidance_candidates = []
    pdf_context = ""
    selected_pdf_chunks = []
    pdf_context_allocation = {
        "primary_allocations": [],
        "primary_chunk_count": 0,
        "supplementary_chunk_count": 0,
        "pdf_context_limit": PDF_CONTEXT_LIMIT,
        "primary_chunk_max_gap_from_best": PRIMARY_CHUNK_MAX_GAP_FROM_BEST,
        "primary_first_document_hard_limit": PRIMARY_FIRST_DOCUMENT_HARD_LIMIT,
        "unused_primary_capacity": 0,
        "redistribution_occurred": False,
        "supplementary_context_limit": SUPPLEMENTARY_CONTEXT_LIMIT,
        "supplementary_min_chunk_score": SUPPLEMENTARY_MIN_CHUNK_SCORE,
    }

    if route == "legal":
        csv_limit = LEGAL_CSV_RETRIEVAL_LIMIT

    if route == "legal":
        pdf_results = query_qdrant_by_document_type(
            query_vector,
            "pdf",
            pdf_limit,
            with_vectors=True,
        )
        primary_guidance, primary_guidance_candidates = get_primary_guidance(
            query,
            pdf_results,
            top_n=PRIMARY_GUIDANCE_TOP_N,
        )
        primary_guidance = add_document_filtered_primary_chunks(
            query_vector,
            primary_guidance,
        )
        pdf_context, selected_pdf_chunks, pdf_context_allocation = build_weighted_pdf_context(
            pdf_results,
            primary_guidance=primary_guidance,
            max_chunks=PDF_CONTEXT_LIMIT,
        )

        csv_results = query_qdrant_by_document_type(
            query_vector,
            "csv",
            csv_limit,
        )
        csv_context_results = deduplicate_csv_results(
            csv_results,
            LEGAL_CSV_CONTEXT_LIMIT,
        )

    elif route == "hybrid":
        csv_results = query_qdrant_by_document_type(
            query_vector,
            "csv",
            csv_limit,
        )
        csv_context_results = csv_results[:10]
        fine_statistics = calculate_fine_statistics_from_dataset(query)
        fine_statistics_calculated = True

    elif route == "analytics":
        fine_statistics = calculate_fine_statistics_from_dataset(query)
        fine_statistics_calculated = True

    metadata = {
        "primary_guidance": primary_guidance,
        "primary_guidance_titles": [
            guidance["title"] for guidance in primary_guidance
        ],
        "primary_guidance_candidates": primary_guidance_candidates,
        "route": route,
        "pdf_results_count": len(pdf_results),
        "pdf_context_chunk_count": len(selected_pdf_chunks),
        "csv_candidates_count": len(csv_results),
        "csv_context_case_count": len(csv_context_results),
        "fine_statistics_calculated": fine_statistics_calculated,
        "pdf_context_allocation": pdf_context_allocation,
    }

    if debug:
        print("\n\n================ RAG DEBUG START ================")
        print(f"QUERY: {query}")
        print(f"COLLECTION: {collection_name}")
        print(f"ROUTE: {route}")
        print(f"PDF results retrieved: {len(pdf_results)}")
        print(f"PDF chunks placed in context: {len(selected_pdf_chunks)}")
        print(
            f"Primary Guidance chunks placed in context: {pdf_context_allocation['primary_chunk_count']}")
        print(
            f"Supplementary PDF chunks placed in context: {pdf_context_allocation['supplementary_chunk_count']}")
        print(
            f"PRIMARY_CHUNK_MAX_GAP_FROM_BEST: {pdf_context_allocation['primary_chunk_max_gap_from_best']}")
        print(
            f"PRIMARY_FIRST_DOCUMENT_HARD_LIMIT: {pdf_context_allocation['primary_first_document_hard_limit']}")
        print(
            f"Unused Primary Guidance capacity: {pdf_context_allocation['unused_primary_capacity']}")
        print(
            f"Redistribution occurred: {pdf_context_allocation['redistribution_occurred']}")
        print(
            f"SUPPLEMENTARY_CONTEXT_LIMIT: {pdf_context_allocation['supplementary_context_limit']}")
        print(
            f"SUPPLEMENTARY_MIN_CHUNK_SCORE: {pdf_context_allocation['supplementary_min_chunk_score']}")
        print(
            f"PDF_CONTEXT_LIMIT: {pdf_context_allocation['pdf_context_limit']}")
        print(f"CSV candidates retrieved: {len(csv_results)}")
        print(f"CSV cases placed in context: {len(csv_context_results)}")
        print(f"Fine statistics calculated: {fine_statistics_calculated}")
        print("PRIMARY GUIDANCE CONTEXT ALLOCATION:")

        for allocation in pdf_context_allocation["primary_allocations"]:
            selection_debug = allocation.get("selection_debug", {})
            print(
                f"- rank={allocation['rank']} | "
                f"title={allocation['title']} | "
                f"assigned_limit={allocation['assigned_chunk_limit']} | "
                f"hard_limit={allocation['hard_limit']} | "
                f"best_chunk_score={allocation['best_chunk_score']:.4f} | "
                f"minimum_eligible_score={allocation['minimum_eligible_score']:.4f} | "
                f"retrieved_chunks={allocation['retrieved_chunk_count']} | "
                f"eligible_chunks={allocation['eligible_chunk_count']} | "
                f"global_chunks_for_document={allocation['global_chunks_for_document']} | "
                f"document_filtered_chunks_retrieved={allocation['document_filtered_chunks_retrieved']} | "
                f"merged_unique_chunks={allocation['merged_unique_chunks']} | "
                f"eligible_chunks_after_merge={allocation['eligible_chunks_after_merge']} | "
                f"initially_added={allocation['initially_added_chunk_count']} | "
                f"redistributed_added={allocation['redistributed_added_chunk_count']} | "
                f"final_chunks_added={allocation['added_chunk_count']} | "
                f"diversity_selection_enabled={selection_debug.get('diversity_selection_enabled')} | "
                f"candidate_chunks_before_deduplication={selection_debug.get('candidate_chunks_before_deduplication')} | "
                f"near_duplicate_chunks_skipped={selection_debug.get('near_duplicate_chunks_skipped')} | "
                f"diverse_chunks_selected={selection_debug.get('diverse_chunks_selected')} | "
                f"max_neighbour_chunks={selection_debug.get('max_neighbour_chunks')} | "
                f"neighbour_chunks_added={selection_debug.get('neighbour_chunks_added')} | "
                f"independent_chunks_added={selection_debug.get('independent_chunks_added')} | "
                f"neighbour_cap_reached={selection_debug.get('neighbour_cap_reached')} | "
                f"final_chunks_added={selection_debug.get('final_chunks_added')}"
            )
            for selected_chunk in selection_debug.get("selected_chunks", []):
                print(
                    f"  chunk_id={selected_chunk['chunk_id']} | "
                    f"chunk_score={selected_chunk['chunk_score']:.4f} | "
                    f"selection_method={selected_chunk['selection_method']} | "
                    f"maximum_similarity_to_previous_selected="
                    f"{selected_chunk['maximum_similarity_to_previous_selected']:.4f}"
                )

        print("PRIMARY GUIDANCE CANDIDATES:")

        for guidance in primary_guidance_candidates:
            print(
                f"- {guidance['title']} | "
                f"url={guidance.get('url') or 'N/A'} | "
                f"accepted={guidance['accepted']} | "
                f"document_score={guidance['document_score']:.4f} | "
                f"title_score={guidance['title_score']:.4f} | "
                f"topic_score={guidance['topic_score']:.4f} | "
                f"metadata_score={guidance['metadata_score']:.4f} | "
                f"max_chunk_score={guidance['max_chunk_score']:.4f} | "
                f"average_top_3_score={guidance['average_top_3_score']:.4f} | "
                f"retrieved_chunks={guidance['retrieved_chunk_count']} | "
                f"rejection={guidance.get('rejection_reason') or 'N/A'}"
            )

        print("\nTop PDF results:")
        for p in pdf_results[:10]:
            print(
                f"score={p.score:.4f} | "
                f"title={p.payload.get('title', 'Unknown')} | "
                f"page={p.payload.get('page', 'N/A')}"
            )

        print("\nTop CSV results:")
        for p in csv_results[:10]:
            print(
                f"score={p.score:.4f} | "
                f"text={p.payload.get('text', '')[:120]}"
            )
        print("================ RAG DEBUG END ================\n\n")

    csv_context = build_csv_context(csv_context_results)

    if route == "legal":
        context = f"""
LEGAL GUIDANCE FROM PDF DOCUMENTS: {pdf_context}

ENFORCEMENT EXAMPLES FROM CSV DATASET: {csv_context}
"""
    elif route == "hybrid":
        context = f"""
ENFORCEMENT EXAMPLES FROM CSV DATASET: {csv_context}

FINE STATISTICS: {fine_statistics}
"""
    else:
        context = f"""
FINE STATISTICS: {fine_statistics}
"""

    if return_metadata:
        return context, metadata

    return context


# connect OPEN AI API
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_recent_conversation_turns(chat_history, max_turns=3):
    """
    Return the last complete user-assistant turns in chronological order.
    A turn is one user message followed by one assistant response.
    """
    if not chat_history:
        return []

    turns = []
    pending_assistant = None

    for message in reversed(chat_history):
        role = message.get("role")
        content = str(message.get("content", ""))

        if role == "assistant" and pending_assistant is None:
            pending_assistant = content
            continue

        if role == "user" and pending_assistant is not None:
            turns.append({
                "user": content,
                "assistant": pending_assistant,
            })
            pending_assistant = None

        if len(turns) >= max_turns:
            break

    return list(reversed(turns))


def format_history_for_rewriting(chat_history, max_turns=3, max_chars_per_message=2000):
    turns = get_recent_conversation_turns(chat_history, max_turns=max_turns)

    formatted_turns = []
    for index, turn in enumerate(turns, start=1):
        user_text = turn["user"][:max_chars_per_message]
        assistant_text = turn["assistant"][:max_chars_per_message]
        formatted_turns.append(
            f"Turn {index}\n"
            f"User: {user_text}\n"
            f"Assistant: {assistant_text}"
        )

    return "\n\n".join(formatted_turns)


def rewrite_followup_question(user_question: str, chat_history: list) -> str:
    """
    Rewrite the latest user question into a standalone question using only
    the last three complete conversation turns. Fail safely to the original.
    """
    recent_history = format_history_for_rewriting(chat_history, max_turns=3)

    if not recent_history:
        return user_question

    rewriting_prompt = """
Rewrite the user's latest question into a standalone question.

Use the recent conversation only to resolve references such as:
- this
- that
- it
- the same
- this country
- this sector
- this fine
- this case
- this requirement
- what about Spain / Germany / France / etc.
- what about withdrawal / consent / legitimate interest / etc.

Do not answer the question.
Do not add new legal assumptions.
Do not add facts that are not supported by the recent conversation.
Do not change the user's intent.
If the latest question is already standalone, return it unchanged.

Return only the rewritten standalone question.
"""

    user_prompt = f"""
Recent conversation:
{recent_history}

Latest user question:
{user_question}
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": rewriting_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        rewritten_question = response.choices[0].message.content.strip()
        rewritten_question = rewritten_question.strip("\"'")
        return rewritten_question or user_question
    except Exception:
        return user_question


def should_rewrite_question(user_question: str) -> bool:
    """
    Conservative local follow-up detector.
    Prefer False when uncertain to avoid contaminating standalone questions.
    """
    question = user_question.strip()
    query_lower = question.lower()
    word_count = len(re.findall(r"\w+", query_lower))

    if not question:
        return False

    standalone_subject_terms = [
        "gdpr",
        "edpb",
        "consent",
        "legitimate interest",
        "lawful basis",
        "controller",
        "processor",
        "camera",
        "cctv",
        "video surveillance",
        "employee monitoring",
        "data transfer",
        "fine",
        "fines",
        "penalty",
        "penalties",
        "requirements",
        "obligations",
        "risk",
        "risks",
        "штраф",
        "штрафы",
        "соглас",
        "видеонаблюд",
        "требован",
        "обязан",
    ]
    standalone_question_starts = [
        "what does",
        "what are",
        "what is",
        "which country",
        "which sector",
        "can i",
        "i want",
        "i would like",
        "how should",
        "how can",
        "do i",
        "does gdpr",
        "is it",
    ]

    starts_like_standalone = any(
        query_lower.startswith(prefix)
        for prefix in standalone_question_starts
    )
    has_standalone_subject = any(
        term in query_lower
        for term in standalone_subject_terms
    )

    if starts_like_standalone and has_standalone_subject:
        return False

    if query_lower.startswith("what about") and has_standalone_subject and word_count > 4:
        return False

    followup_prefixes = [
        "and what about",
        "what about",
        "and for",
        "and the",
        "and in",
        "and with",
        "compare it",
        "compare this",
        "compare that",
        "why was it",
        "why were they",
        "why was this",
        "why was that",
        "what about this",
        "what about that",
        "а что насчет",
        "а как насчет",
        "а по",
        "а для",
        "и что насчет",
        "почему это",
        "почему он",
        "почему она",
    ]
    followup_reference_terms = [
        " it ",
        " this ",
        " that ",
        " the same ",
        " this country",
        " that country",
        " this sector",
        " that sector",
        " this fine",
        " that fine",
        " this case",
        " that case",
        " this requirement",
        " that requirement",
        " same sector",
        " same country",
    ]
    short_followup_targets = [
        "spain",
        "germany",
        "france",
        "italy",
        "consent",
        "withdrawal",
        "legitimate interest",
        "median",
        "maximum",
        "minimum",
        "average",
        "mean",
        "total",
        "sector",
        "country",
    ]

    if any(query_lower.startswith(prefix) for prefix in followup_prefixes):
        return True

    padded_query = f" {query_lower} "
    if any(term in padded_query for term in followup_reference_terms):
        return True

    if word_count <= 5 and any(target in query_lower for target in short_followup_targets):
        return True

    return False


def detect_question_type(query):
    """
    Route questions to the prompt that matches the user's intent.
    The calculation itself remains deterministic in calculate_fine_statistics_from_dataset().
    """
    query_lower = query.lower()

    analytics_keywords = [
        "maximum fine",
        "highest fine",
        "largest fine",
        "biggest fine",
        "max fine",
        "minimum fine",
        "lowest fine",
        "smallest fine",
        "min fine",
        "median fine",
        "average fine",
        "mean fine",
        "avg fine",
        "total fines",
        "sum of fines",
        "number of cases",
        "how many cases",
        "case count",
        "grouping by country",
        "group by country",
        "grouped by country",
        "by country",
        "per country",
        "top countries",
        "grouping by sector",
        "group by sector",
        "grouped by sector",
        "by sector",
        "per sector",
        "top sectors",
        "enforcement statistics",
        "fine statistics",
        "медиан",
        "максималь",
        "минималь",
        "средн",
        "сумм",
        "сколько кейсов",
        "количество кейсов",
        "по странам",
        "по стране",
        "по секторам",
        "по сектору",
        "статистик",
    ]

    hybrid_keywords = [
        "why",
        "reason",
        "imposed",
        "what happened",
        "violation",
        "violations",
        "most common",
        "types of violations",
        "enforcement pattern",
        "pattern",
        "case descriptions",
        "relevant cases",
        "examples",
        "почему",
        "за что",
        "нарушени",
        "тип",
        "паттерн",
        "тенденц",
        "пример",
        "кейсы",
    ]

    legal_keywords = [
        "edpb",
        "guidance",
        "guideline",
        "what does gdpr say",
        "what does the gdpr say",
        "requirement",
        "requirements",
        "obligation",
        "obligations",
        "controller",
        "processor",
        "consent",
        "legitimate interest",
        "lawful basis",
        "data transfer",
        "transfers",
        "cctv",
        "risk",
        "risks",
        "compliance",
        "article",
        "cookie",
        "cookies",
        "cookie banner",
        "tracking",
        "trackers",
        "tracker",
        "prior consent",
        # "google analytics",
        # "pixel",
        "strictly necessary",
        "eprivacy",
        "правов",
        "требован",
        "обязан",
        "соглас",
        "легитим",
        "контрол",
        "процессор",
        "передач",
        "руководств",
    ]

    has_analytics_intent = any(
        keyword in query_lower for keyword in analytics_keywords)
    has_hybrid_intent = any(
        keyword in query_lower for keyword in hybrid_keywords)
    has_legal_intent = any(
        keyword in query_lower for keyword in legal_keywords)

    fine_terms = ["fine", "fines", "penalty",
                  "penalties", "штраф", "штрафы", "штрафов"]
    metric_terms = [
        "maximum",
        "highest",
        "largest",
        "biggest",
        "max",
        "minimum",
        "lowest",
        "smallest",
        "min",
        "median",
        "average",
        "mean",
        "avg",
        "total",
        "sum",
        "number",
        "count",
        "top",
        "statistics",
        "максималь",
        "минималь",
        "медиан",
        "средн",
        "сумм",
        "количество",
        "сколько",
        "топ",
        "статистик",
    ]
    grouping_terms = [
        "by country",
        "per country",
        "country has",
        "countries have",
        "by sector",
        "per sector",
        "sector has",
        "sectors have",
        "по странам",
        "по стране",
        "по секторам",
        "по сектору",
    ]

    has_fine_term = any(term in query_lower for term in fine_terms)
    has_metric_term = any(term in query_lower for term in metric_terms)
    has_grouping_term = any(term in query_lower for term in grouping_terms)
    has_analytics_intent = has_analytics_intent or (
        has_fine_term and (has_metric_term or has_grouping_term)
    )

    if has_analytics_intent and has_hybrid_intent:
        return "hybrid"

    if has_analytics_intent:
        return "analytics"

    if has_legal_intent:
        return "legal"

    return "legal"


def build_legal_system_prompt():
    return f"""

You are a GDPR legal information assistant specialising in GDPR guidance,

EDPB and WP29 documents, and European enforcement practice.

SOURCE LIMITS

Answer only from the provided context.

Do not use external knowledge.

Do not speculate or fill gaps from memory.

Do not mention a legal rule, GDPR article, guidance document, authority,

country, enforcement case, or fine unless it appears in the retrieved context.

If the retrieved context does not contain enough information to answer a

material part of the question, say so clearly and briefly.

SOURCE PRIORITY

Use retrieved legal guidance from PDF documents as the primary source for

legal interpretation.

Give the greatest weight to the most relevant Primary Guidance document.

Use related guidance only where it adds information directly relevant to

the user's question.

Use enforcement cases only as practical illustrations.

Do not derive a general legal rule solely from an enforcement case.

ANSWERING STYLE

Answer the user's actual question directly.

Use clear, practical language that a non-lawyer can understand.

Explain necessary legal terminology briefly.

Prefer concrete decision rules, conditions, examples, and consequences over

abstract descriptions.

Avoid repetition.

Do not add generic compliance statements merely to make the answer longer.

Do not present information that is only loosely related to the question.

QUESTIONS ABOUT WHETHER SOMETHING IS REQUIRED

When the user asks when an assessment, notification, consultation,

appointment, record, safeguard, or other compliance step is required,

organise the legal reasoning around the following elements where they are

supported by the retrieved context:

1. Trigger:

   State the condition that activates the requirement.

2. Practical test:

   Explain how the controller determines whether that condition is met.

   Include relevant criteria, factors, indicators, lists, thresholds, or

   decision steps contained in the Primary Guidance.

3. Exceptions:

   Explain when the requirement does not apply or may not apply.

4. Required action:

   Explain what the controller must do when the trigger is met, including

   timing and any required next step.

Do not merely repeat a phrase such as "likely to result in a high risk"

without explaining the available practical criteria for applying it.

Where the Primary Guidance contains a relevant list of criteria, mandatory

cases, indicators, examples, exceptions, or decision steps, include the

most important items from that list.

Do not invent missing criteria or complete a list from external knowledge.

If the context contains only part of a relevant list, present only the

supported items and state that the retrieved context does not contain the

complete list.

CHECKLIST QUESTIONS

When the user asks which topics, items, elements, information, fields, or
sections should or must be included in a document, notice, record,
assessment, agreement, notification, or submission, treat the question as
a request for a practical checklist.

Include all materially relevant mandatory and conditional items supported
by the retrieved Primary Guidance, unless the user explicitly asks for a
brief overview.

Distinguish clearly between:
- information that must always be included;
- information that is required only where applicable;
- recommended presentation or good-practice measures.

Do not replace the requested content checklist with general advice about
clarity, accessibility, timing, or presentation. Include such advice only
after identifying the substantive items that the document should contain.

If the retrieved Primary Guidance contains only part of the relevant
checklist, provide the supported items and state briefly that the retrieved
context may not contain the complete list.

DISTINGUISH RELATED LEGAL TESTS

Keep different legal stages and thresholds separate.

For example, where the context distinguishes between:

- an initial risk and a residual risk;

- a requirement to perform an assessment and a requirement to consult;

- a general rule and an exception;

- a controller's obligation and a recommended good practice;

explain that distinction clearly.

Do not present recommended good practice as a mandatory legal obligation.

OUTPUT STRUCTURE

You MUST use exactly these markdown section headings in every answer:

## Summary

## Key GDPR Requirements

## Real GDPR Cases

## Common Mistakes

Do not skip, rename, or merge these sections.

## Summary

Answer the user's question directly in a short paragraph.

State the core legal trigger and, where relevant, the main practical

distinction needed to apply it.

Do not include background information that does not help answer the question.

## Key GDPR Requirements

Use concise bullet points.

Start with the core trigger.

Then include, where supported and relevant:

- the practical test or decision rule;

- principal mandatory situations;

- important criteria or indicators;

- exceptions;

- timing;

- the required next action.

Prioritise specific information from the most relevant Primary Guidance

document.

Do not repeat the same rule in several differently worded bullets.

Do not include every retrieved fact merely because it is available.

## Real GDPR Cases

Include no more than {LEGAL_CASES_IN_ANSWER_LIMIT} materially relevant

enforcement cases from the retrieved CSV context.

Do not add a case merely to fill the section.

For each included case, state briefly:

- what happened;

- the relevant GDPR issue;

- the country or supervisory authority, if available;

- the fine amount, if available;

- why the case is relevant to the user's question.

If the user asks about a specific country, prioritise cases from that country.

If no directly relevant case was retrieved, say so in one sentence.

Do not claim a general enforcement trend based on one case or on weakly

related cases.

Do not add vague statements such as "authorities generally take compliance

seriously" unless a broader pattern is genuinely supported by multiple

retrieved cases.

## Common Mistakes

Include only distinct practical mistakes or compliance risks that are:

- relevant to the user's question;

- supported by the retrieved context; and

- not already fully stated in the Key GDPR Requirements section.

Do not simply rewrite each legal requirement in negative form.

Use no more than four concise bullet points.

If the retrieved context does not support distinct additional mistakes,

say so briefly.

PRIMARY LEGAL GUIDANCE

Do not create a separate Primary Legal Guidance section in the answer body.

The application may append retrieved guidance titles separately.

ACCURACY RULES

Do not invent:

- legal criteria;

- GDPR article numbers;

- mandatory steps;

- exceptions;

- deadlines;

- authority powers;

- case details;

- fine amounts;

- country information.

Do not overstate the context.

Use:

- "must" only for mandatory requirements supported by the context;

- "should" for recommendations or qualified duties;

- "may" where the context indicates discretion, possibility, or uncertainty.

If the retrieved materials conflict, describe the conflict briefly instead

of silently choosing one statement.

LENGTH AND RELEVANCE

Give enough detail to explain the practical legal test, but remain focused.

A short question does not always require a short answer if the Primary

Guidance contains a necessary multi-factor test.

However:

- omit peripheral information;

- omit repeated background;

- omit weakly relevant cases;

- avoid generic concluding paragraphs;

- do not repeat the Summary at the end.

FORMATTING

Use markdown within the answer string.

Return only a valid JSON object with exactly this structure:

{{"answer": "..."}}

Do not return explanatory text outside the JSON object.

"""


def build_hybrid_system_prompt():
    return """

You are a GDPR enforcement analytics assistant.

The calculation has already been performed by the application using the structured enforcement dataset.

Use deterministic calculation results under "FINE STATISTICS" for all numbers.
Use retrieved CSV/enforcement context only for case descriptions.
Do not perform calculations yourself.
Do not invent numbers.
Do not invent cases.
Do not add EDPB legal guidance unless the user explicitly asks for legal interpretation.
Do not speculate.

Use exactly these markdown section headings:

## Result

## Data Coverage

## Relevant Enforcement Cases

## Enforcement Pattern

Section instructions:

## Result
Directly answer the calculation part of the user's question using "FINE STATISTICS".

## Data Coverage
Explain how many cases matched the filters and how many cases had disclosed fine amounts, if this information is available.
Mention country, sector, time period, or other filters used for the calculation.

## Relevant Enforcement Cases
Describe only cases that appear in the retrieved CSV/enforcement context or in "FINE STATISTICS".
If the relevant case is already named in "FINE STATISTICS", use that deterministic result.

## Enforcement Pattern
Briefly explain the enforcement pattern shown by the retrieved cases and calculated result.
Mention that the result is based on the available structured enforcement dataset and may not be exhaustive.

Return a valid JSON object with this structure:
{"answer": "..."}

"""


def build_analytics_system_prompt():
    return """

You are a GDPR enforcement analytics assistant.

The calculation has already been performed by the application using the structured enforcement dataset.

Your task is only to explain the calculated result clearly and accurately.

Use only the numbers and statistics provided under "FINE STATISTICS".
Do not perform calculations yourself.
Do not invent numbers.
Do not add GDPR legal guidance unless the user explicitly asks for legal analysis.
Do not mention EDPB guidance documents unless they were specifically retrieved for a legal question.
Do not speculate.

Use exactly these markdown section headings:

## Result

## Data Coverage

## Interpretation

Section instructions:

## Result
Directly answer the user's question. State the calculated value clearly.

## Data Coverage
Explain how many cases matched the filters and how many cases had disclosed fine amounts, if this information is available.
Mention country, sector, time period, or other filters used for the calculation.

## Interpretation
Briefly explain what the result means.
Mention that the result is based on the available structured enforcement dataset and may not be exhaustive.
If the result is based on a small number of cases, say that it should be interpreted cautiously.

You MUST use exactly these markdown section headings in every answer.
Do not skip these headings.
Do not merge them into one paragraph.

Return a valid JSON object with this structure:
{"answer": "..."}

"""


def get_system_prompt(query):
    question_type = detect_question_type(query)

    if question_type == "analytics":
        return build_analytics_system_prompt()

    if question_type == "hybrid":
        return build_hybrid_system_prompt()

    return build_legal_system_prompt()


def build_analytics_context(query):
    fine_statistics = calculate_fine_statistics_from_dataset(query)

    return f"""
FINE STATISTICS: {fine_statistics}
"""


def ask_openai(query, context, model_name="gpt-4.1-mini"):
    """
    Sends retrieved context + user query to OpenAI and returns the answer.
    """

    system_prompt = get_system_prompt(query)
    user_prompt = f"""

Context from knowledge base:
{context}

Current question:
{query}
"""

    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    response_content = response.choices[0].message.content
    parsed_response = json.loads(response_content)
    return parsed_response["answer"]


# Adding Disclaimer and aswer format:

DISCLAIMER = (
    "This response is generated by an LLM model based on retrieved "
    "GDPR-related materials and does not constitute legal advice."
)


def format_primary_guidance(metadata):
    primary_guidance = (metadata or {}).get("primary_guidance", [])
    if not primary_guidance:
        return ""

    guidance_lines = [
        f"- {format_document_link(guidance['title'], guidance.get('url'))}"
        for guidance in primary_guidance
        if guidance.get("title")
    ]

    if not guidance_lines:
        return ""

    return "\n\n---\n\n## Primary Legal Guidance\n\n" + "\n".join(guidance_lines)


def format_final_answer(answer, metadata=None, include_primary_guidance=False):
    primary_guidance_section = (
        format_primary_guidance(metadata)
        if include_primary_guidance
        else ""
    )

    return f"""

{answer}
{primary_guidance_section}

---


## Disclaimer

{DISCLAIMER}
"""
# Answering question:


def answer_question(
    query,
    pdf_limit=50,
    csv_limit=20,
    debug=False,
    chat_history=None
):
    """
    Full RAG pipeline:
    1. Rewrite follow-up question into a standalone question
    2. Route the rewritten question
    3. Retrieve fresh context from Qdrant using the rewritten question
    4. Send context + rewritten question to OpenAI
    5. Return answer and disclaimer
    """
    rewrite_triggered = should_rewrite_question(query)
    rewritten_query = (
        rewrite_followup_question(
            user_question=query,
            chat_history=chat_history or []
        )
        if rewrite_triggered
        else query
    )
    question_type = detect_question_type(rewritten_query)

    if question_type == "analytics":
        context = build_analytics_context(rewritten_query)
        metadata = {
            "primary_guidance": [],
            "primary_guidance_titles": [],
            "primary_guidance_candidates": [],
            "route": question_type,
            "pdf_results_count": 0,
            "pdf_context_chunk_count": 0,
            "csv_candidates_count": 0,
            "csv_context_case_count": 0,
            "fine_statistics_calculated": True,
        }
    else:
        context, metadata = retrieve_context(
            query=rewritten_query,
            pdf_limit=pdf_limit,
            csv_limit=csv_limit,
            debug=debug,
            return_metadata=True,
            route=question_type,
        )

    answer = ask_openai(query=rewritten_query,
                        context=context,
                        model_name="gpt-4.1-mini")

    final_answer = format_final_answer(
        answer=answer,
        metadata=metadata,
        include_primary_guidance=question_type == "legal",
    )

    if debug:
        pdf_context_allocation = metadata.get("pdf_context_allocation", {})
        primary_allocation_debug = "\n".join(
            [
                (
                    f"- rank={allocation['rank']} | "
                    f"title={allocation['title']} | "
                    f"assigned_limit={allocation['assigned_chunk_limit']} | "
                    f"hard_limit={allocation['hard_limit']} | "
                    f"best_chunk_score={allocation['best_chunk_score']:.4f} | "
                    f"minimum_eligible_score={allocation['minimum_eligible_score']:.4f} | "
                    f"retrieved_chunks={allocation['retrieved_chunk_count']} | "
                    f"eligible_chunks={allocation['eligible_chunk_count']} | "
                    f"global_chunks_for_document={allocation['global_chunks_for_document']} | "
                    f"document_filtered_chunks_retrieved={allocation['document_filtered_chunks_retrieved']} | "
                    f"merged_unique_chunks={allocation['merged_unique_chunks']} | "
                    f"eligible_chunks_after_merge={allocation['eligible_chunks_after_merge']} | "
                    f"initially_added={allocation['initially_added_chunk_count']} | "
                    f"redistributed_added={allocation['redistributed_added_chunk_count']} | "
                    f"final_chunks_added={allocation['added_chunk_count']} | "
                    f"diversity_selection_enabled="
                    f"{allocation.get('selection_debug', {}).get('diversity_selection_enabled')} | "
                    f"candidate_chunks_before_deduplication="
                    f"{allocation.get('selection_debug', {}).get('candidate_chunks_before_deduplication')} | "
                    f"near_duplicate_chunks_skipped="
                    f"{allocation.get('selection_debug', {}).get('near_duplicate_chunks_skipped')} | "
                    f"diverse_chunks_selected="
                    f"{allocation.get('selection_debug', {}).get('diverse_chunks_selected')} | "
                    f"max_neighbour_chunks="
                    f"{allocation.get('selection_debug', {}).get('max_neighbour_chunks')} | "
                    f"neighbour_chunks_added="
                    f"{allocation.get('selection_debug', {}).get('neighbour_chunks_added')} | "
                    f"independent_chunks_added="
                    f"{allocation.get('selection_debug', {}).get('independent_chunks_added')} | "
                    f"neighbour_cap_reached="
                    f"{allocation.get('selection_debug', {}).get('neighbour_cap_reached')} | "
                    f"final_chunks_added="
                    f"{allocation.get('selection_debug', {}).get('final_chunks_added')}"
                )
                for allocation in pdf_context_allocation.get("primary_allocations", [])
            ]
        ) or "No Primary Guidance chunks were allocated."
        primary_guidance_debug = "\n".join(
            [
                (
                    f"- {guidance['title']} | "
                    f"url={guidance.get('url') or 'N/A'} | "
                    f"accepted={guidance['accepted']} | "
                    f"document_score={guidance['document_score']:.4f} | "
                    f"title_score={guidance['title_score']:.4f} | "
                    f"topic_score={guidance['topic_score']:.4f} | "
                    f"metadata_score={guidance['metadata_score']:.4f} | "
                    f"max_chunk_score={guidance['max_chunk_score']:.4f} | "
                    f"average_top_3_score={guidance['average_top_3_score']:.4f} | "
                    f"retrieved_chunks={guidance['retrieved_chunk_count']} | "
                    f"rejection={guidance.get('rejection_reason') or 'N/A'}"
                )
                for guidance in metadata.get("primary_guidance_candidates", [])
            ]
        ) or "No clearly relevant guidance document was retrieved."
        debug_text = f"""

---

## Debug

- Original question: {query}
- Rewrite triggered: {rewrite_triggered}
- Rewritten standalone question: {rewritten_query if rewrite_triggered else "Not rewritten"}
- Selected route: {metadata.get("route", question_type)}
- PDF results retrieved: {metadata.get("pdf_results_count", 0)}
- PDF chunks placed in context: {metadata.get("pdf_context_chunk_count", 0)}
- Primary Guidance chunks placed in context: {pdf_context_allocation.get("primary_chunk_count", 0)}
- Supplementary PDF chunks placed in context: {pdf_context_allocation.get("supplementary_chunk_count", 0)}
- PRIMARY_CHUNK_MAX_GAP_FROM_BEST: {pdf_context_allocation.get("primary_chunk_max_gap_from_best", PRIMARY_CHUNK_MAX_GAP_FROM_BEST)}
- PRIMARY_FIRST_DOCUMENT_HARD_LIMIT: {pdf_context_allocation.get("primary_first_document_hard_limit", PRIMARY_FIRST_DOCUMENT_HARD_LIMIT)}
- Unused Primary Guidance capacity: {pdf_context_allocation.get("unused_primary_capacity", 0)}
- Redistribution occurred: {pdf_context_allocation.get("redistribution_occurred", False)}
- SUPPLEMENTARY_CONTEXT_LIMIT: {pdf_context_allocation.get("supplementary_context_limit", SUPPLEMENTARY_CONTEXT_LIMIT)}
- SUPPLEMENTARY_MIN_CHUNK_SCORE: {pdf_context_allocation.get("supplementary_min_chunk_score", SUPPLEMENTARY_MIN_CHUNK_SCORE)}
- PDF_CONTEXT_LIMIT: {pdf_context_allocation.get("pdf_context_limit", PDF_CONTEXT_LIMIT)}
- CSV candidates retrieved: {metadata.get("csv_candidates_count", 0)}
- CSV cases placed in context: {metadata.get("csv_context_case_count", 0)}
- Fine statistics calculated: {metadata.get("fine_statistics_calculated", False)}

Primary Guidance context allocation:
{primary_allocation_debug}

Primary Guidance candidates:
{primary_guidance_debug}
"""
        final_answer = f"{final_answer}{debug_text}"

    return final_answer
