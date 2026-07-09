

from collections import defaultdict
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


def get_primary_guidance(pdf_results, top_n=5):

    guidance_scores = defaultdict(float)

    for result in pdf_results:
        title = result.payload.get("title")
        score = result.score or 0

        if not title:
            continue

        if "REGULATION (EU) 2016/679" in title:
            continue

        if title:
            guidance_scores[title] += score

    if not guidance_scores:
        return []

    sorted_guidance = sorted(
        guidance_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_guidance[:top_n]
# [title for title, score in sorted_guidance[:5]]

# Weighted pdf context


def build_weighted_pdf_context(pdf_results, max_chunks=25):

    grouped = defaultdict(list)
    for p in pdf_results:
        title = p.payload.get("title", "Unknown")
        if not title:
            continue

        if "REGULATION (EU) 2016/679" in title:
            continue

        if title:
            grouped[title].append(p)

    document_scores = {
        title: sum((p.score or 0) for p in chunks)
        for title, chunks in grouped.items()
    }

    ranked_titles = sorted(
        document_scores,
        key=document_scores.get,
        reverse=True
    )

    selected_chunks = []

    for title in ranked_titles:

        chunks = sorted(
            grouped[title],
            key=lambda p: p.score or 0,
            reverse=True
        )

        # Take chunks from 'stronger' document

        if len(selected_chunks) < 8:

            selected_chunks.extend(chunks[:6])

        else:

            selected_chunks.extend(chunks[:2])

        if len(selected_chunks) >= max_chunks:

            break

    selected_chunks = selected_chunks[:max_chunks]

    pdf_context = "\n\n".join([

        f"[PDF score: {p.score:.4f} | title: {p.payload.get('title', 'Unknown')}]\n"

        f"{p.payload.get('text', '')}"

        for p in selected_chunks

    ])

    return pdf_context


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


# automatical retrive of context from the query


def retrieve_context(query, pdf_limit=50, csv_limit=20, debug=False, return_metadata=False):
    query_vector = model.encode(query).tolist()

    pdf_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="document_type",
                    match=MatchValue(value="pdf")
                )
            ]
        ),
        with_payload=True,
        limit=pdf_limit,
    ).points

    csv_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="document_type",
                    match=MatchValue(value="csv")
                )
            ]
        ),
        with_payload=True,
        limit=csv_limit,
    ).points
    fine_statistics = calculate_fine_statistics_from_dataset(query)

    primary_guidance = get_primary_guidance(
        pdf_results,
        top_n=3
    )

    metadata = {
        "primary_guidance": primary_guidance,
        "primary_guidance_titles": [title for title, _ in primary_guidance],
    }

    if debug:
        print("\n\n================ RAG DEBUG START ================")
        print(f"QUERY: {query}")
        print(f"COLLECTION: {collection_name}")
        print("PRIMARY GUIDANCE:")

        for title, score in primary_guidance:
            print(f"- {title} | total score: {score:.4f}")

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

    primary_guidance_text = (
        "\n".join([
            f"- {title} | total score: {score:.4f}"
            for title, score in primary_guidance
        ])
        if primary_guidance
        else "No clearly relevant guidance document was retrieved."
    )
    # fine_statistics = calculate_fine_statistics_from_dataset(query)

    pdf_context = build_weighted_pdf_context(
        pdf_results,
        max_chunks=25
    )

    csv_context = "\n\n".join([
        f"[CSV score: {p.score:.4f}]\n"
        f"{p.payload.get('text', '')}"
        for p in csv_results[:10]
    ])

    context = f"""
PRIMARY GUIDANCE IDENTIFIED FROM RETRIEVAL: {primary_guidance_text}

LEGAL GUIDANCE FROM PDF DOCUMENTS: {pdf_context}

ENFORCEMENT EXAMPLES FROM CSV DATASET: {csv_context}

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
        "statistics",
        "analytics",
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
        "analytics",
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
    return """

You are a GDPR legal assistant specialized in GDPR guidance, EDPB guidelines, and enforcement practice across Europe.
Answer ONLY based on the provided context.
Do NOT use external knowledge.
Do not speculate.
Do not mention GDPR articles, guidance documents, countries, fines, or enforcement actions unless they appear in the retrieved context.
If the context is insufficient, clearly state this.

Use clear and plain language. Explain GDPR concepts in a way that can be understood by a non-lawyer.
Avoid unnecessary legal jargon. When legal terminology is necessary, briefly explain it in simple words.
Prefer practical explanations over abstract legal language.

USE LEGAL GUIDANCE FROM PDF DOCUMENTS as the primary source for GDPR interpretation.
If relevant GDPR guidance documents are available, base the legal analysis primarily on those documents.
USE enforcement cases only to illustrate how the guidance has been applied in practice. Do not derive legal conclusions solely from enforcement cases.

The documents listed under "PRIMARY GUIDANCE IDENTIFIED FROM RETRIEVAL" represent the most relevant guidance documents identified by the retrieval system.
Prioritize these documents when preparing the legal analysis, summary, key GDPR requirements, and practical recommendations.

You may use other retrieved documents when relevant, but do not ignore the guidance documents identified in "PRIMARY GUIDANCE IDENTIFIED FROM RETRIEVAL".
You MUST use exactly these markdown section headings in every answer:

## Summary

## Primary Legal Guidance

## Key GDPR Requirements

## Real GDPR Cases

## Common Mistakes

Do not skip these headings.
Do not merge them into one paragraph.

Section instructions:

## Summary

Provide a short answer, directly answering the user's question.
Base the summary primarily on the guidance documents identified under "PRIMARY GUIDANCE IDENTIFIED FROM RETRIEVAL".

## Primary Legal Guidance

Provide a bullet list of the guidance documents listed under:
PRIMARY GUIDANCE IDENTIFIED FROM RETRIEVAL

Use only the exact document titles provided there.
Do not add other guidance documents from the context.
Do not provide explanations in this section.

## Key GDPR Requirements

Provide bullet points.

## Real GDPR Cases

Provide 2-3 concrete enforcement examples from the retrieved CSV context, if available.
If the user's question mentions a specific country, prioritize enforcement examples from that country.
If no relevant examples from that country are available in the retrieved context,
explicitly state this and use the most relevant examples available.

For each example, briefly explain:
- what happened;
- what GDPR issue or violation was identified;
- the fine amount, if available;
- the country or supervisory authority, if available.

After the examples, add 2-3 short sentences summarizing the general enforcement pattern.

Do not invent examples. Use only cases from the retrieved context.

## Common Mistakes

Describe the most common GDPR compliance risks or violations related to the scenario.

Use markdown formatting.

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

You MUST use exactly these markdown section headings in every answer
Do not skip these headings
Do not merge them into one paragraph



Return a valid JSON object with this structure:
{"answer": "..."}

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


def format_final_answer(answer):

    return f"""

{answer}

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
        }
    else:
        context, metadata = retrieve_context(
            query=rewritten_query,
            pdf_limit=pdf_limit,
            csv_limit=csv_limit,
            debug=debug,
            return_metadata=True
        )

    answer = ask_openai(query=rewritten_query,
                        context=context,
                        model_name="gpt-4.1-mini")

    final_answer = format_final_answer(
        answer=answer

    )

    if debug:
        primary_guidance_debug = "\n".join([
            f"- {title} | total score: {score:.4f}"
            for title, score in metadata.get("primary_guidance", [])
        ]) or "No clearly relevant guidance document was retrieved."
        debug_text = f"""

---

## Debug

- Original question: {query}
- Rewrite triggered: {rewrite_triggered}
- Rewritten standalone question: {rewritten_query if rewrite_triggered else "Not rewritten"}
- Selected route: {question_type}

Retrieved primary guidance titles:
{primary_guidance_debug}
"""
        final_answer = f"{final_answer}{debug_text}"

    return final_answer
