import glob
import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

PRODUCT_COLUMNS = [
    "brand",
    "title",
    "price",
    "mrp",
    "discount_pct",
    "rating",
    "review_count",
    "product_url",
    "asin",
]

REVIEW_BASE_COLUMNS = [
    "brand",
    "product_title",
    "review_text",
    "star_rating",
    "date",
    "verified_purchase",
]

ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "wheels": ["wheel", "wheels", "roller", "rolling", "glide"],
    "handle": ["handle", "handles", "grip", "telescopic"],
    "material": ["material", "fabric", "hard", "soft", "shell", "quality"],
    "zipper": ["zipper", "zip", "zips", "lock"],
    "size": ["size", "spacious", "capacity", "fits", "compartment"],
    "durability": ["durable", "durability", "broke", "broken", "cracked", "lasted", "flimsy"],
}

POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05


def load_json_file(path: str) -> List[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Skipping {path}: {exc}")
        return []

    if not isinstance(payload, list):
        print(f"[WARN] Skipping {path}: expected list JSON")
        return []
    return payload


def filename_to_brand(path: str) -> str:
    name = os.path.basename(path)
    stem = name.replace("_products.json", "").replace("_reviews.json", "")
    return stem.replace("_", " ").title()


def to_float(value) -> float:
    if value is None:
        return float("nan")
    text = str(value)
    match = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    return float(match.group(1)) if match else float("nan")


def iter_raw_items(suffix: str):
    pattern = os.path.join(RAW_DIR, f"*_{suffix}.json")
    for path in sorted(glob.glob(pattern)):
        file_brand = filename_to_brand(path)
        for item in load_json_file(path):
            yield file_brand, item


def normalize_products() -> pd.DataFrame:
    rows: List[Dict] = []
    for file_brand, item in iter_raw_items("products"):
        rows.append(
            {
                "brand": item.get("brand") or file_brand,
                "title": item.get("title"),
                "price": to_float(item.get("price")),
                "mrp": to_float(item.get("mrp")),
                "discount_pct": to_float(item.get("discount_pct")),
                "rating": to_float(item.get("rating")),
                "review_count": to_float(item.get("review_count")),
                "product_url": item.get("product_url"),
                "asin": item.get("asin"),
            }
        )

    df = pd.DataFrame(rows, columns=PRODUCT_COLUMNS)
    if not df.empty:
        df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").astype("Int64")
    return df


def normalize_reviews() -> pd.DataFrame:
    rows: List[Dict] = []
    for file_brand, item in iter_raw_items("reviews"):
        rows.append(
            {
                "brand": item.get("brand") or file_brand,
                "product_title": item.get("product_title"),
                "review_text": item.get("review_text"),
                "star_rating": to_float(item.get("star_rating")),
                "date": item.get("date"),
                "verified_purchase": bool(item.get("verified_purchase", False)),
            }
        )

    return pd.DataFrame(rows, columns=REVIEW_BASE_COLUMNS)


def contains_keyword(text_lower: str, keyword: str) -> bool:
    return re.search(rf"\b{re.escape(keyword)}\b", text_lower) is not None


def classify_sentiment(score: float) -> str:
    if score > POS_THRESHOLD:
        return "positive"
    if score < NEG_THRESHOLD:
        return "negative"
    return "neutral"


def add_sentiment_and_aspects(df_reviews: pd.DataFrame) -> pd.DataFrame:
    if df_reviews.empty:
        return df_reviews

    analyzer = SentimentIntensityAnalyzer()
    df_reviews = df_reviews.copy()
    df_reviews["review_text"] = df_reviews["review_text"].fillna("").astype(str)
    df_reviews["sentiment_score"] = df_reviews["review_text"].apply(
        lambda text: analyzer.polarity_scores(text)["compound"]
    )

    for aspect, keywords in ASPECT_KEYWORDS.items():
        labels: List[str] = []
        for _, row in df_reviews.iterrows():
            text_lower = row["review_text"].lower()
            has_aspect = any(contains_keyword(text_lower, kw) for kw in keywords)
            if not has_aspect:
                labels.append("not_mentioned")
                continue
            labels.append(classify_sentiment(float(row["sentiment_score"])))
        df_reviews[aspect] = labels

    return df_reviews


def top_themes(brand_reviews: pd.DataFrame, polarity: str) -> str:
    if brand_reviews.empty:
        return ""

    if polarity == "positive":
        selected = brand_reviews[brand_reviews["sentiment_score"] > POS_THRESHOLD]
    else:
        selected = brand_reviews[brand_reviews["sentiment_score"] < NEG_THRESHOLD]

    if selected.empty:
        return ""

    keyword_counter: Counter = Counter()
    for text in selected["review_text"].fillna("").astype(str).str.lower():
        for keywords in ASPECT_KEYWORDS.values():
            for kw in keywords:
                if contains_keyword(text, kw):
                    keyword_counter[kw] += 1

    top5 = [kw for kw, _ in keyword_counter.most_common(5)]
    return ", ".join(top5)


def recurring_aspects(brand_reviews: pd.DataFrame, polarity: str) -> str:
    if brand_reviews.empty:
        return ""

    target = "positive" if polarity == "positive" else "negative"
    counter: Counter = Counter()
    for aspect in ASPECT_KEYWORDS:
        count = int((brand_reviews[aspect] == target).sum()) if aspect in brand_reviews else 0
        if count > 0:
            counter[aspect] = count

    if not counter:
        return ""
    return ", ".join([f"{aspect} ({count})" for aspect, count in counter.most_common(5)])


def aspect_score(series: pd.Series) -> float:
    mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    values = series.map(mapping)
    values = values.dropna()
    if values.empty:
        return 0.0
    return float(values.mean())


def build_brand_summary(df_products: pd.DataFrame, df_reviews: pd.DataFrame) -> pd.DataFrame:
    brands = sorted(set(df_products["brand"].dropna().unique()) | set(df_reviews["brand"].dropna().unique()))

    rows: List[Dict] = []
    for brand in brands:
        p = df_products[df_products["brand"] == brand] if not df_products.empty else pd.DataFrame()
        r = df_reviews[df_reviews["brand"] == brand] if not df_reviews.empty else pd.DataFrame()

        avg_price = float(p["price"].dropna().mean()) if not p.empty else 0.0
        avg_discount_pct = float(p["discount_pct"].dropna().mean()) if not p.empty else 0.0
        avg_rating = float(p["rating"].dropna().mean()) if not p.empty else 0.0
        total_reviews = int(len(r))
        sentiment_score = float(r["sentiment_score"].dropna().mean()) if not r.empty else 0.0

        row: Dict = {
            "brand": brand,
            "avg_price": avg_price,
            "avg_discount_pct": avg_discount_pct,
            "avg_rating": avg_rating,
            "total_reviews": total_reviews,
            "sentiment_score": sentiment_score,
            "top_5_positive_themes": top_themes(r, "positive"),
            "top_5_negative_themes": top_themes(r, "negative"),
            "recurring_praise": recurring_aspects(r, "positive"),
            "recurring_complaints": recurring_aspects(r, "negative"),
        }

        for aspect in ASPECT_KEYWORDS:
            row[f"{aspect}_score"] = aspect_score(r[aspect]) if not r.empty else 0.0

        row["anomaly_flag"] = bool(avg_rating > 4.0 and row["durability_score"] < 0)
        rows.append(row)

    return pd.DataFrame(rows)


def save_csvs(df_products: pd.DataFrame, df_reviews: pd.DataFrame, df_summary: pd.DataFrame) -> Tuple[int, int, int]:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    products_path = os.path.join(PROCESSED_DIR, "products.csv")
    reviews_path = os.path.join(PROCESSED_DIR, "reviews.csv")
    summary_path = os.path.join(PROCESSED_DIR, "brand_summary.csv")

    df_products.to_csv(products_path, index=False)
    df_reviews.to_csv(reviews_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    return len(df_products), len(df_reviews), len(df_summary)


def main() -> None:
    df_products = normalize_products()
    df_reviews = normalize_reviews()
    df_reviews = add_sentiment_and_aspects(df_reviews)
    df_summary = build_brand_summary(df_products, df_reviews)

    products_rows, reviews_rows, summary_rows = save_csvs(df_products, df_reviews, df_summary)

    print("[DONE] Analyzer complete")
    print(f"products.csv rows: {products_rows}")
    print(f"reviews.csv rows: {reviews_rows}")
    print(f"brand_summary.csv rows: {summary_rows}")


if __name__ == "__main__":
    main()
