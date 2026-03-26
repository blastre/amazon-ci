import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv


PROCESSED_DIR = "data/processed"
ASPECTS = ["wheels", "handle", "material", "zipper", "size", "durability"]
ASPECT_SCORE_COLUMNS = [f"{aspect}_score" for aspect in ASPECTS]
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05
GREETING_WORDS = ["hello", "hi", "hey", "help"]


def apply_ui_theme() -> None:
    px.defaults.template = "plotly"
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.1rem;
            padding-bottom: 1.4rem;
        }
        div[data-testid="stMetric"] {
            border-radius: 12px;
            padding: 0.85rem 1rem;
        }
        .soft-card {
            background: #7c3aed;
            border: 1px solid #6d28d9;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.6rem;
            color: #ffffff !important;
        }
        .insight-card {
            background: #7c3aed !important;
            border: 1px solid #6d28d9;
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 10px;
            color: #ffffff !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
        }
        .soft-card,
        .soft-card *,
        .insight-card,
        .insight-card * {
            color: #ffffff !important;
        }
        div[data-testid="stChatMessage"] {
            background: #7c3aed;
            border: 1px solid #6d28d9;
            border-radius: 12px;
            padding: 0.35rem 0.65rem;
            margin-bottom: 0.5rem;
        }
        div[data-testid="stChatMessage"] * {
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_products(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_reviews(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_brand_summary(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    products_path = os.path.join(PROCESSED_DIR, "products.csv")
    reviews_path = os.path.join(PROCESSED_DIR, "reviews.csv")
    summary_path = os.path.join(PROCESSED_DIR, "brand_summary.csv")
    return load_products(products_path), load_reviews(reviews_path), load_brand_summary(summary_path)


def safe_float(value: float, default: float) -> float:
    if pd.isna(value):
        return default
    return float(value)


def parse_brand_tier(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    if df.empty or "avg_price" not in df.columns:
        df["price_tier"] = "Unknown"
        return df

    q1 = df["avg_price"].quantile(0.25)
    q3 = df["avg_price"].quantile(0.75)

    def label(price: float) -> str:
        if pd.isna(price):
            return "Unknown"
        if price >= q3:
            return "Premium"
        if price <= q1:
            return "Value"
        return "Mid-range"

    df["price_tier"] = df["avg_price"].apply(label)
    return df


def product_level_sentiment(reviews_df: pd.DataFrame) -> pd.DataFrame:
    if reviews_df.empty:
        return pd.DataFrame(columns=["brand", "title", "product_sentiment_score"])
    grouped = (
        reviews_df.groupby(["brand", "product_title"], as_index=False)["sentiment_score"]
        .mean()
        .rename(columns={"product_title": "title", "sentiment_score": "product_sentiment_score"})
    )
    return grouped


def apply_sidebar_filters(
    products: pd.DataFrame, reviews: pd.DataFrame, summary: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    st.sidebar.header("Filters")

    all_brands = sorted(products["brand"].dropna().unique().tolist())
    selected_brands = st.sidebar.multiselect("Brand", options=all_brands, default=all_brands)

    price_min = safe_float(products["price"].min(), 0.0)
    price_max = safe_float(products["price"].max(), 1000.0)
    if price_min == price_max:
        price_max = price_min + 1.0

    price_range = st.sidebar.slider(
        "Price range",
        min_value=float(price_min),
        max_value=float(price_max),
        value=(float(price_min), float(price_max)),
    )
    min_rating = st.sidebar.slider("Minimum rating", min_value=1.0, max_value=5.0, value=1.0, step=0.1)
    sentiment_range = st.sidebar.slider(
        "Sentiment score", min_value=-1.0, max_value=1.0, value=(-1.0, 1.0), step=0.01
    )

    product_sent = product_level_sentiment(reviews)
    products_f = products.merge(product_sent, on=["brand", "title"], how="left")
    products_f["product_sentiment_score"] = products_f["product_sentiment_score"].fillna(0.0)

    products_f = products_f[
        products_f["brand"].isin(selected_brands)
        & products_f["price"].between(price_range[0], price_range[1], inclusive="both")
        & (products_f["rating"] >= min_rating)
        & products_f["product_sentiment_score"].between(sentiment_range[0], sentiment_range[1], inclusive="both")
    ]

    reviews_f = reviews[
        reviews["brand"].isin(selected_brands)
        & reviews["sentiment_score"].between(sentiment_range[0], sentiment_range[1], inclusive="both")
    ]
    if "star_rating" in reviews_f.columns:
        reviews_f = reviews_f[reviews_f["star_rating"] >= min_rating]
    reviews_f = reviews_f[
        reviews_f[["brand", "product_title"]].apply(tuple, axis=1).isin(
            products_f[["brand", "title"]].rename(columns={"title": "product_title"}).apply(tuple, axis=1)
        )
    ]

    summary_f = summary[
        summary["brand"].isin(selected_brands)
        & summary["avg_price"].between(price_range[0], price_range[1], inclusive="both")
        & (summary["avg_rating"] >= min_rating)
        & summary["sentiment_score"].between(sentiment_range[0], sentiment_range[1], inclusive="both")
    ]

    filter_state = {
        "selected_brands": selected_brands,
        "price_range": price_range,
        "min_rating": min_rating,
        "sentiment_range": sentiment_range,
    }
    return products_f, reviews_f, summary_f, filter_state


def show_empty_message(name: str) -> None:
    st.info(f"No data available for {name} with current filters. Try broadening filters.")


def page_overview(products_f: pd.DataFrame, reviews_f: pd.DataFrame, summary_f: pd.DataFrame) -> None:
    st.subheader("Overview")
    if products_f.empty or summary_f.empty:
        show_empty_message("Overview")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Brands", int(summary_f["brand"].nunique()))
    m2.metric("Total Products", int(len(products_f)))
    m3.metric("Total Reviews", int(len(reviews_f)))
    avg_sent = reviews_f["sentiment_score"].mean() if not reviews_f.empty else summary_f["sentiment_score"].mean()
    m4.metric("Avg Sentiment", f"{avg_sent:.3f}")

    avg_price = summary_f[["brand", "avg_price"]].sort_values("avg_price", ascending=False)
    fig_price = px.bar(
        avg_price,
        x="brand",
        y="avg_price",
        title="Average Price by Brand",
        labels={"brand": "Brand", "avg_price": "Average Price"},
    )
    st.plotly_chart(fig_price, use_container_width=True)

    avg_discount = summary_f[["brand", "avg_discount_pct"]].sort_values("avg_discount_pct", ascending=False)
    fig_discount = px.bar(
        avg_discount,
        x="brand",
        y="avg_discount_pct",
        title="Average Discount Percentage by Brand",
        labels={"brand": "Brand", "avg_discount_pct": "Average Discount %"},
    )
    st.plotly_chart(fig_discount, use_container_width=True)

    scatter_df = parse_brand_tier(summary_f)
    fig_scatter = px.scatter(
        scatter_df,
        x="avg_price",
        y="sentiment_score",
        size="total_reviews",
        color="price_tier",
        text="brand",
        title="Sentiment Score vs Average Price",
        labels={"avg_price": "Average Price", "sentiment_score": "Sentiment Score", "total_reviews": "Reviews"},
        hover_data=["brand", "avg_rating", "avg_discount_pct"],
    )
    fig_scatter.update_traces(textposition="top center")
    st.plotly_chart(fig_scatter, use_container_width=True)

    tier_df = scatter_df[["brand", "price_tier"]].sort_values("price_tier")
    st.dataframe(tier_df, use_container_width=True)


def render_badges(summary_f: pd.DataFrame) -> None:
    if summary_f.empty:
        return

    best = {
        "Highest Rating": summary_f.loc[summary_f["avg_rating"].idxmax(), "brand"],
        "Best Sentiment": summary_f.loc[summary_f["sentiment_score"].idxmax(), "brand"],
        "Most Reviews": summary_f.loc[summary_f["total_reviews"].idxmax(), "brand"],
        "Best Discount": summary_f.loc[summary_f["avg_discount_pct"].idxmax(), "brand"],
        "Most Affordable": summary_f.loc[summary_f["avg_price"].idxmin(), "brand"],
    }
    st.markdown("**Best Performers**")
    cols = st.columns(len(best))
    for i, (metric, brand) in enumerate(best.items()):
        cols[i].markdown(
            f"<div class='soft-card'>"
            f"<b>{metric}</b><br>{brand}</div>",
            unsafe_allow_html=True,
        )


def page_brand_comparison(summary_f: pd.DataFrame) -> None:
    st.subheader("Brand Comparison")
    if summary_f.empty:
        show_empty_message("Brand Comparison")
        return

    table_cols = [
        "brand",
        "avg_price",
        "avg_discount_pct",
        "avg_rating",
        "total_reviews",
        "sentiment_score",
        "top_5_positive_themes",
        "top_5_negative_themes",
    ]
    for col in ["recurring_praise", "recurring_complaints"]:
        if col in summary_f.columns:
            table_cols.append(col)
    table_df = summary_f[table_cols].sort_values("sentiment_score", ascending=False)
    st.dataframe(table_df, use_container_width=True)

    long_df = summary_f[["brand", "avg_price", "avg_discount_pct", "avg_rating", "sentiment_score"]].melt(
        id_vars="brand", var_name="metric", value_name="value"
    )
    fig_grouped = px.bar(
        long_df,
        x="brand",
        y="value",
        color="metric",
        barmode="group",
        title="Brand Comparison Across Price, Discount, Rating, and Sentiment",
        labels={"brand": "Brand", "value": "Metric Value", "metric": "Metric"},
    )
    st.plotly_chart(fig_grouped, use_container_width=True)

    radar_df = summary_f[["brand", *ASPECT_SCORE_COLUMNS]].melt(
        id_vars="brand", var_name="aspect", value_name="score"
    )
    radar_df["aspect"] = radar_df["aspect"].str.replace("_score", "", regex=False)
    fig_radar = px.line_polar(
        radar_df,
        r="score",
        theta="aspect",
        color="brand",
        line_close=True,
        title="Aspect Sentiment Radar by Brand",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    render_badges(summary_f)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", str(text).lower())


def top_words_by_sentiment(product_reviews: pd.DataFrame, sentiment: str, limit: int = 10) -> pd.DataFrame:
    if product_reviews.empty:
        return pd.DataFrame(columns=["word", "count", "sentiment"])

    if sentiment == "positive":
        subset = product_reviews[product_reviews["sentiment_score"] > POS_THRESHOLD]
    else:
        subset = product_reviews[product_reviews["sentiment_score"] < NEG_THRESHOLD]

    stopwords = {
        "the",
        "and",
        "for",
        "this",
        "that",
        "with",
        "very",
        "but",
        "was",
        "are",
        "you",
        "all",
        "bag",
        "luggage",
        "product",
    }

    counter: Counter = Counter()
    for text in subset["review_text"].fillna(""):
        words = [w for w in tokenize(text) if len(w) > 2 and w not in stopwords]
        counter.update(words)

    rows = [{"word": w, "count": c, "sentiment": sentiment.title()} for w, c in counter.most_common(limit)]
    return pd.DataFrame(rows)


@st.cache_data
def build_rag_documents(products_df: pd.DataFrame, reviews_df: pd.DataFrame, summary_df: pd.DataFrame) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []

    for _, row in summary_df.iterrows():
        text = (
            f"Brand {row.get('brand')}: avg_price {row.get('avg_price')}, avg_discount_pct {row.get('avg_discount_pct')}, "
            f"avg_rating {row.get('avg_rating')}, total_reviews {row.get('total_reviews')}, sentiment_score {row.get('sentiment_score')}, "
            f"positive_themes {row.get('top_5_positive_themes')}, negative_themes {row.get('top_5_negative_themes')}, "
            f"wheels_score {row.get('wheels_score')}, handle_score {row.get('handle_score')}, material_score {row.get('material_score')}, "
            f"zipper_score {row.get('zipper_score')}, size_score {row.get('size_score')}, durability_score {row.get('durability_score')}, "
            f"anomaly_flag {row.get('anomaly_flag')}"
        )
        docs.append({"source": "brand_summary", "text": text})

    for _, row in products_df.iterrows():
        text = (
            f"Product {row.get('title')} by {row.get('brand')}: price {row.get('price')}, mrp {row.get('mrp')}, "
            f"discount_pct {row.get('discount_pct')}, rating {row.get('rating')}, review_count {row.get('review_count')}, asin {row.get('asin')}"
        )
        docs.append({"source": "products", "text": text})

    sampled_reviews = reviews_df.head(200)
    for _, row in sampled_reviews.iterrows():
        text = (
            f"Review for {row.get('brand')} / {row.get('product_title')}: star_rating {row.get('star_rating')}, "
            f"sentiment_score {row.get('sentiment_score')}, verified_purchase {row.get('verified_purchase')}, "
            f"wheels {row.get('wheels')}, handle {row.get('handle')}, material {row.get('material')}, zipper {row.get('zipper')}, "
            f"size {row.get('size')}, durability {row.get('durability')}. Text: {row.get('review_text')}"
        )
        docs.append({"source": "reviews", "text": text})

    return docs


def _tokenize_for_rag(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", str(text).lower())


def extract_brand_mentions(question: str, brands: List[str]) -> List[str]:
    q = question.lower()
    found = [b for b in brands if b.lower() in q]
    return sorted(set(found))


def parse_budget(question: str) -> Optional[float]:
    m = re.search(r"(?:under|below|less than|budget)\s*₹?\s*(\d+)", question.lower())
    if m:
        return float(m.group(1))
    m2 = re.search(r"₹\s*(\d+)", question.lower())
    if m2:
        return float(m2.group(1))
    return None


def retrieve_context(question: str, docs: List[Dict[str, str]], top_k: int = 5) -> List[Dict[str, str]]:
    q_tokens = set(_tokenize_for_rag(question))
    scored = []
    for doc in docs:
        d_tokens = set(_tokenize_for_rag(doc["text"]))
        overlap = len(q_tokens.intersection(d_tokens))
        if overlap > 0:
            scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def call_groq_chat(messages: List[Dict[str, str]], api_key: str, model: str = GROQ_MODEL) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def build_llm_prompt_context(contexts: List[Dict[str, str]]) -> str:
    if not contexts:
        return "No retrieved context."
    lines = []
    for i, ctx in enumerate(contexts, start=1):
        lines.append(f"{i}. ({ctx['source']}) {ctx['text'][:600]}")
    return "\n".join(lines)


def generate_llm_answer(
    user_prompt: str,
    contexts: List[Dict[str, str]],
    chat_history: List[Dict[str, str]],
    api_key: str,
) -> str:
    system = (
        "You are a data analyst assistant for an Amazon luggage dashboard. "
        "Answer only using provided context and recent chat history. "
        "Be concise, specific, and actionable. "
        "If context is insufficient, say what is missing and suggest a precise follow-up question."
    )
    history = chat_history[-6:]
    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": (
                f"User question: {user_prompt}\n\n"
                f"Retrieved context:\n{build_llm_prompt_context(contexts)}\n\n"
                "Please provide:\n"
                "1) direct answer\n"
                "2) supporting numbers/brands\n"
                "3) one next best follow-up question"
            ),
        }
    )
    return call_groq_chat(messages, api_key=api_key)


def generate_rag_answer(
    question: str,
    contexts: List[Dict[str, str]],
    reviews_f: pd.DataFrame,
    summary_f: pd.DataFrame,
    memory: Dict[str, str],
) -> Tuple[str, Dict[str, str]]:
    q = question.lower()
    new_memory = memory.copy()
    if summary_f.empty:
        return "I could not find summary data for the current filters.", new_memory

    brands = sorted(summary_f["brand"].dropna().unique().tolist())
    mentioned = extract_brand_mentions(question, brands)

    if ("that brand" in q or "it" in q) and not mentioned and memory.get("last_brand") in brands:
        mentioned = [memory["last_brand"]]

    if any(word in q for word in GREETING_WORDS) and len(q.split()) <= 5:
        return (
            "I can help with: best/worst brand by rating, sentiment, price, discount, durability, value-for-money, verified purchase ratio, and brand comparisons.\n"
            "Try: `Which brand is best value under 4000?`",
            new_memory,
        )

    if any(
        phrase in q
        for phrase in [
            "all brands",
            "everything",
            "good/bad",
            "good and bad",
            "about all",
            "tell me all you know",
            "one line summary",
            "summary of each brand",
            "each brand",
        ]
    ):
        rows = []
        ranked = summary_f.sort_values(["avg_rating", "sentiment_score"], ascending=[False, False])
        for _, r in ranked.iterrows():
            good_aspect = sorted(ASPECTS, key=lambda a: r[f"{a}_score"], reverse=True)[0]
            bad_aspect = sorted(ASPECTS, key=lambda a: r[f"{a}_score"])[0]
            rows.append(
                f"- {r['brand']}: good at {good_aspect} ({r[f'{good_aspect}_score']:.2f}), "
                f"weak at {bad_aspect} ({r[f'{bad_aspect}_score']:.2f}), "
                f"rating {r['avg_rating']:.2f}, sentiment {r['sentiment_score']:.3f}, "
                f"avg price {r['avg_price']:.0f}, discount {r['avg_discount_pct']:.1f}%"
            )

        best_rating = ranked.iloc[0]["brand"]
        best_value_df = ranked.copy()
        best_value_df["value_ratio"] = best_value_df["sentiment_score"] / best_value_df["avg_price"].replace(0, pd.NA)
        best_value_df = best_value_df.dropna(subset=["value_ratio"])
        best_value = best_value_df.iloc[0]["brand"] if not best_value_df.empty else "N/A"

        answer = (
            "Here is a full cross-brand snapshot from the current filters:\n"
            + "\n".join(rows)
            + f"\n\nOverall leaders: best rating = {best_rating}, best value-for-money = {best_value}."
        )
        return answer, new_memory

    if "best brand" in q or ("best" in q and "brand" in q):
        rank_df = summary_f.copy()
        rank_df["overall_score"] = (
            0.45 * rank_df["avg_rating"]
            + 0.35 * rank_df["sentiment_score"]
            + 0.20 * (rank_df["avg_discount_pct"] / 100.0)
        )
        best = rank_df.sort_values("overall_score", ascending=False).iloc[0]
        new_memory["last_brand"] = str(best["brand"])
        return (
            f"Best overall brand right now is {best['brand']} "
            f"(rating {best['avg_rating']:.2f}, sentiment {best['sentiment_score']:.3f}, "
            f"discount {best['avg_discount_pct']:.1f}%).",
            new_memory,
        )

    if "compare" in q or "comparison" in q:
        cmp_df = summary_f.copy()
        if mentioned:
            cmp_df = cmp_df[cmp_df["brand"].isin(mentioned)]
        cmp_df = cmp_df.sort_values(["avg_rating", "sentiment_score"], ascending=[False, False])
        if cmp_df.empty:
            return "No comparable brands in current filter scope.", new_memory
        rows = []
        for _, r in cmp_df.head(5).iterrows():
            rows.append(
                f"- {r['brand']}: rating {r['avg_rating']:.2f}, sentiment {r['sentiment_score']:.3f}, avg price {r['avg_price']:.0f}, discount {r['avg_discount_pct']:.1f}%"
            )
        best = cmp_df.iloc[0]["brand"]
        new_memory["last_brand"] = str(best)
        return "Quick comparison:\n" + "\n".join(rows), new_memory

    budget = parse_budget(question)
    if "value" in q or ("best" in q and "price" in q) or budget is not None:
        value_df = summary_f.copy()
        if budget is not None:
            value_df = value_df[value_df["avg_price"] <= budget]
        if mentioned:
            value_df = value_df[value_df["brand"].isin(mentioned)]
        value_df["value_ratio"] = value_df["sentiment_score"] / value_df["avg_price"].replace(0, pd.NA)
        value_df = value_df.dropna(subset=["value_ratio"]).sort_values("value_ratio", ascending=False)
        if value_df.empty:
            return "No brand matches that value query in current filters.", new_memory
        best = value_df.iloc[0]
        new_memory["last_brand"] = str(best["brand"])
        return (
            f"Best value-for-money: {best['brand']} (sentiment/price ratio {best['value_ratio']:.6f}). "
            f"Avg price {best['avg_price']:.0f}, sentiment {best['sentiment_score']:.3f}, rating {best['avg_rating']:.2f}.",
            new_memory,
        )

    if "highest rating" in q or "best rating" in q:
        pool = summary_f[summary_f["brand"].isin(mentioned)] if mentioned else summary_f
        best = pool.loc[pool["avg_rating"].idxmax()]
        new_memory["last_brand"] = str(best["brand"])
        return f"Highest average rating is {best['brand']} at {best['avg_rating']:.2f}.", new_memory

    if "verified" in q:
        v = reviews_f.groupby("brand", as_index=False)["verified_purchase"].mean().sort_values(
            "verified_purchase", ascending=False
        )
        if mentioned:
            v = v[v["brand"].isin(mentioned)]
        if v.empty:
            return "No verified purchase data for that query.", new_memory
        top = v.iloc[0]
        new_memory["last_brand"] = str(top["brand"])
        return f"Highest verified purchase ratio is {top['brand']} at {top['verified_purchase']:.2%}.", new_memory

    if any(k in q for k in ["durability", "wheels", "handle", "material", "zipper", "size"]):
        aspect = None
        for a in ASPECTS:
            if a in q:
                aspect = a
                break
        if aspect is None:
            aspect = "durability"
        col = f"{aspect}_score"
        pool = summary_f[summary_f["brand"].isin(mentioned)] if mentioned else summary_f
        if any(k in q for k in ["worst", "negative", "bad"]):
            target = pool.loc[pool[col].idxmin()]
            direction = "worst"
        else:
            target = pool.loc[pool[col].idxmax()]
            direction = "best"
        new_memory["last_brand"] = str(target["brand"])
        return f"{direction.title()} {aspect} sentiment is {target['brand']} with {col}={target[col]:.2f}.", new_memory

    if mentioned:
        brand = mentioned[0]
        row = summary_f[summary_f["brand"] == brand].iloc[0]
        new_memory["last_brand"] = brand
        return (
            f"{brand} snapshot: avg price {row['avg_price']:.0f}, rating {row['avg_rating']:.2f}, sentiment {row['sentiment_score']:.3f}, "
            f"reviews {int(row['total_reviews'])}, discount {row['avg_discount_pct']:.1f}%. "
            f"Top positive themes: {row['top_5_positive_themes'] or 'N/A'}.",
            new_memory,
        )

    if not contexts:
        return (
            "I can answer better if you ask a direct question like: `best brand by rating`, `worst durability`, "
            "`compare Safari vs VIP`, or `best value under 4000`."
        ), new_memory

    ranked = summary_f.sort_values(["avg_rating", "sentiment_score"], ascending=[False, False]).head(3)
    lines = ["Quick snapshot from current filters:"]
    for _, r in ranked.iterrows():
        lines.append(
            f"- {r['brand']}: rating {r['avg_rating']:.2f}, sentiment {r['sentiment_score']:.3f}, "
            f"price {r['avg_price']:.0f}, discount {r['avg_discount_pct']:.1f}%"
        )
    lines.append("Ask: 'compare all brands' or 'best value under 4000' for deeper analysis.")
    return "\n".join(lines), new_memory


def render_agent_chatbot(products_f: pd.DataFrame, reviews_f: pd.DataFrame, summary_f: pd.DataFrame) -> None:
    st.markdown("### Agent Chatbot (RAG)")
    st.caption("Ask questions about filtered brands/products/reviews. The bot retrieves relevant rows and responds with context.")

    docs = build_rag_documents(products_f, reviews_f, summary_f)
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    use_llm = bool(groq_api_key)

    st.caption(
        "Mode: Groq LLM + RAG" if use_llm else "Mode: Rule-based RAG fallback (set GROQ_API_KEY for LLM)"
    )

    if "agent_chat_messages" not in st.session_state:
        st.session_state.agent_chat_messages = [
            {
                "role": "assistant",
                "content": "Hi! I can answer questions about brand performance, sentiment, pricing, reviews, and aspects using the filtered data.",
            }
        ]
    if "agent_chat_memory" not in st.session_state:
        st.session_state.agent_chat_memory = {}

    col1, _ = st.columns([1, 6])
    with col1:
        if st.button("Clear chat"):
            st.session_state.agent_chat_messages = [
                {
                    "role": "assistant",
                    "content": "Chat memory cleared. Ask a new question.",
                }
            ]
            st.session_state.agent_chat_memory = {}
            st.rerun()

    for msg in st.session_state.agent_chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask something like: Which brand is best value for money?")
    if user_prompt:
        st.session_state.agent_chat_messages.append({"role": "user", "content": user_prompt})

        recent_user_msgs = [
            m["content"] for m in st.session_state.agent_chat_messages if m["role"] == "user"
        ][-3:]
        memory_query = " ".join(recent_user_msgs)

        contexts = retrieve_context(memory_query, docs, top_k=5)
        if use_llm:
            try:
                llm_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.agent_chat_messages
                    if m["role"] in {"user", "assistant"}
                ]
                answer = generate_llm_answer(user_prompt, contexts, llm_history, groq_api_key)
            except Exception as exc:
                answer = f"LLM call failed ({exc}). Falling back to rule-based mode."
                fallback, new_memory = generate_rag_answer(
                    user_prompt,
                    contexts,
                    reviews_f,
                    summary_f,
                    st.session_state.agent_chat_memory,
                )
                st.session_state.agent_chat_memory = new_memory
                answer = f"{answer}\n\n{fallback}"
        else:
            answer, new_memory = generate_rag_answer(
                user_prompt,
                contexts,
                reviews_f,
                summary_f,
                st.session_state.agent_chat_memory,
            )
            st.session_state.agent_chat_memory = new_memory

        st.session_state.agent_chat_messages.append({"role": "assistant", "content": answer})
        st.rerun()


def aspect_score_for_product(product_reviews: pd.DataFrame) -> pd.DataFrame:
    mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    rows = []
    for aspect in ASPECTS:
        if aspect not in product_reviews.columns:
            rows.append({"aspect": aspect, "score": 0.0})
            continue
        vals = product_reviews[aspect].map(mapping).dropna()
        score = float(vals.mean()) if not vals.empty else 0.0
        rows.append({"aspect": aspect, "score": score})
    return pd.DataFrame(rows)


def page_product_drilldown(products_f: pd.DataFrame, reviews_f: pd.DataFrame) -> None:
    st.subheader("Product Drilldown")
    if products_f.empty:
        show_empty_message("Product Drilldown")
        return

    brands = sorted(products_f["brand"].dropna().unique().tolist())
    selected_brand = st.selectbox("Select brand", brands)

    brand_products = products_f[products_f["brand"] == selected_brand].copy()
    st.dataframe(
        brand_products[["title", "price", "mrp", "discount_pct", "rating", "review_count", "asin"]],
        use_container_width=True,
    )

    if brand_products.empty:
        show_empty_message("selected brand")
        return

    selected_product = st.selectbox("Select product", brand_products["title"].tolist())
    product_row = brand_products[brand_products["title"] == selected_product].iloc[0]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Price", f"{product_row['price']:.0f}")
    m2.metric("MRP", f"{product_row['mrp']:.0f}" if pd.notna(product_row["mrp"]) else "N/A")
    m3.metric("Discount %", f"{product_row['discount_pct']:.1f}" if pd.notna(product_row["discount_pct"]) else "N/A")
    m4.metric("Rating", f"{product_row['rating']:.2f}" if pd.notna(product_row["rating"]) else "N/A")
    m5.metric("Review Count", f"{int(product_row['review_count'])}" if pd.notna(product_row["review_count"]) else "N/A")

    product_reviews = reviews_f[
        (reviews_f["brand"] == selected_brand) & (reviews_f["product_title"] == selected_product)
    ].copy()
    if product_reviews.empty:
        show_empty_message("selected product reviews")
        return

    pos_words = top_words_by_sentiment(product_reviews, "positive", limit=10)
    neg_words = top_words_by_sentiment(product_reviews, "negative", limit=10)
    words_df = pd.concat([pos_words, neg_words], ignore_index=True)

    if words_df.empty:
        st.info("No strong positive/negative words found for this product.")
    else:
        fig_words = px.bar(
            words_df,
            x="word",
            y="count",
            color="sentiment",
            barmode="group",
            title="Top 10 Positive and Negative Words",
            labels={"word": "Word", "count": "Frequency", "sentiment": "Sentiment"},
        )
        st.plotly_chart(fig_words, use_container_width=True)

    aspect_df = aspect_score_for_product(product_reviews)
    fig_aspect = px.bar(
        aspect_df,
        x="score",
        y="aspect",
        orientation="h",
        title="Aspect Sentiment Breakdown for Selected Product",
        labels={"score": "Aspect Sentiment Score", "aspect": "Aspect"},
    )
    st.plotly_chart(fig_aspect, use_container_width=True)


def insight_card(icon: str, title: str, detail: str, data_point: str, color: str) -> None:
    st.markdown(
        f"<div class='insight-card' style='background:{color};'>"
        f"<div style='font-size:20px;'>{icon} <b>{title}</b></div>"
        f"<div style='margin-top:6px;'>{detail}</div>"
        f"<div style='margin-top:6px;'><b>Data:</b> {data_point}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def page_agent_insights(products_f: pd.DataFrame, reviews_f: pd.DataFrame, summary_f: pd.DataFrame) -> None:
    st.subheader("Agent Insights")
    if summary_f.empty or reviews_f.empty:
        show_empty_message("Agent Insights")
        return

    high_rating = summary_f[summary_f["avg_rating"] > 4.0]
    if high_rating.empty:
        worst_case = summary_f.loc[summary_f["durability_score"].idxmin()]
    else:
        worst_case = high_rating.loc[high_rating["durability_score"].idxmin()]
    insight_card(
        "1",
        "Highest Rating, Weakest Durability",
        f"{worst_case['brand']} combines high rating with weaker durability sentiment.",
        f"avg_rating={worst_case['avg_rating']:.2f}, durability_score={worst_case['durability_score']:.2f}",
        "#fdecef",
    )

    value_df = summary_f.copy()
    value_df["value_ratio"] = value_df["sentiment_score"] / value_df["avg_price"].replace(0, pd.NA)
    value_df = value_df.dropna(subset=["value_ratio"])
    best_value = value_df.loc[value_df["value_ratio"].idxmax()]
    insight_card(
        "2",
        "Best Sentiment-to-Price Value",
        f"{best_value['brand']} gives the strongest sentiment per rupee.",
        f"sentiment_score={best_value['sentiment_score']:.3f}, avg_price={best_value['avg_price']:.2f}, ratio={best_value['value_ratio']:.6f}",
        "#ecfdf3",
    )

    discount_brand = summary_f.loc[summary_f["avg_discount_pct"].idxmax()]
    insight_card(
        "3",
        "Most Discount-Driven Brand",
        f"{discount_brand['brand']} appears most dependent on discounts.",
        f"avg_discount_pct={discount_brand['avg_discount_pct']:.2f}",
        "#fff7e8",
    )

    aspect_means = {aspect: float(summary_f[f"{aspect}_score"].mean()) for aspect in ASPECTS}
    worst_aspect = sorted(aspect_means.items(), key=lambda item: item[1])[0][0]
    insight_card(
        "4",
        "Worst Aspect Across Brands",
        f"{worst_aspect.title()} is the weakest aspect overall.",
        f"{worst_aspect}_score_mean={aspect_means[worst_aspect]:.3f}",
        "#eef4ff",
    )

    verified_df = reviews_f.groupby("brand", as_index=False)["verified_purchase"].mean()
    best_verified = verified_df.loc[verified_df["verified_purchase"].idxmax()]
    insight_card(
        "5",
        "Highest Verified Purchase Ratio",
        f"{best_verified['brand']} has the strongest verified purchase signal.",
        f"verified_purchase_ratio={best_verified['verified_purchase']:.2%}",
        "#f1f5f9",
    )

    render_agent_chatbot(products_f, reviews_f, summary_f)


def main() -> None:
    st.set_page_config(page_title="Amazon India Luggage Dashboard", layout="wide")
    load_dotenv()
    apply_ui_theme()
    st.title("Amazon India Luggage Competitive Intelligence Dashboard")

    try:
        products, reviews, summary = load_all_data()
    except FileNotFoundError:
        st.error("Processed CSV files not found in data/processed. Run scraper/analyzer first.")
        return

    print(f"products.csv columns: {products.columns.tolist()}")
    print(f"reviews.csv columns: {reviews.columns.tolist()}")
    print(f"brand_summary.csv columns: {summary.columns.tolist()}")

    with st.expander("Loaded Columns (Verification)", expanded=True):
        st.write("products.csv columns:", products.columns.tolist())
        st.write("reviews.csv columns:", reviews.columns.tolist())
        st.write("brand_summary.csv columns:", summary.columns.tolist())

    if products.empty or reviews.empty or summary.empty:
        st.warning("One or more CSVs are empty. Please re-run scraping and analysis.")
        return

    products_f, reviews_f, summary_f, _ = apply_sidebar_filters(products, reviews, summary)

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Brand Comparison", "Product Drilldown", "Agent Insights"],
    )

    if page == "Overview":
        page_overview(products_f, reviews_f, summary_f)
    elif page == "Brand Comparison":
        page_brand_comparison(summary_f)
    elif page == "Product Drilldown":
        page_product_drilldown(products_f, reviews_f)
    else:
        page_agent_insights(products_f, reviews_f, summary_f)


if __name__ == "__main__":
    main()
