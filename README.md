# Amazon India Luggage Competitive Intelligence (End-to-End)

This repository contains a full pipeline to:

1. scrape Amazon India luggage listings and reviews for multiple brands,
2. normalize and enrich data with sentiment + aspect analysis,
3. visualize insights in a multi-page Streamlit dashboard,
4. support conversational analytics with a RAG chatbot (Groq LLM + fallback mode).

## Quick Setup

```bash
conda create -n amazon-ci python=3.11 -y
conda activate amazon-ci
pip install -r requirements.txt
```

Create `.env`:

```env
SCRAPERAPI_KEY=your_scraperapi_key
GROQ_API_KEY=your_groq_api_key_optional
```

Run:

```bash
python scraper.py
python analyzer.py
streamlit run app.py
```

---

## What This Project Does

### 1) Data Collection (`scraper.py`)
- Scrapes **5 brands**:
  - Safari
  - Skybags
  - American Tourister
  - VIP
  - Nasher Miles
- Pulls product data from search URL pattern:
  - `https://www.amazon.in/s?k={brand}+luggage`
- Product extraction strategy:
  - selector: `soup.find_all('div', {'data-asin': True})`
  - filters empty `ASIN`
  - title from `h2 span`
  - price from `.a-price .a-offscreen`
- For each product, pulls reviews from:
  - `https://www.amazon.in/dp/{asin}#customerReviews`
  - requests sent with ScraperAPI **`render=true`** for review pages
- Review extraction strategy:
  - selector: `soup.select('[data-hook="review"]')`
  - fields: `review_text`, `star_rating`, `date`, `verified_purchase`
- Saves raw JSON outputs:
  - `data/raw/{brand}_products.json`
  - `data/raw/{brand}_reviews.json`
- Reliability features:
  - 2-second delay between every request
  - progress logging
  - failure skip (continues pipeline even if some requests fail)

### 2) Analysis Layer (`analyzer.py`)
- Loads all raw JSON files and normalizes into:
  - `data/processed/products.csv`
  - `data/processed/reviews.csv`
  - `data/processed/brand_summary.csv`
- Runs sentiment analysis on every review using **VADER** (`compound` score).
- Aspect tagging per review using keyword dictionaries for:
  - wheels, handle, material, zipper, size, durability
  - labels each aspect as `positive`, `negative`, `neutral`, `not_mentioned`
- Computes per-brand summary metrics:
  - `avg_price`, `avg_discount_pct`, `avg_rating`, `total_reviews`, `sentiment_score`
  - top themes (`top_5_positive_themes`, `top_5_negative_themes`)
  - recurring signals (`recurring_praise`, `recurring_complaints`)
  - aspect score columns (`*_score`)
  - anomaly flag (`True` when rating is high but durability sentiment is negative)

### 3) Dashboard (`app.py`)
- Streamlit dashboard with 4 pages:
  1. **Overview**
  2. **Brand Comparison**
  3. **Product Drilldown**
  4. **Agent Insights**
- Global sidebar filters applied across pages:
  - brand multiselect
  - price range
  - minimum rating
  - sentiment range
- Uses Plotly for all visualizations.
- Includes metric cards, bars, scatter, grouped bars, radar, and drilldown charts.

### 4) Chatbot in Agent Insights
- Retrieval layer over filtered products/reviews/summary data.
- Two modes:
  - **Groq LLM + RAG** (if `GROQ_API_KEY` is present)
  - **Rule-based RAG fallback** (if no key / API failure)
- Supports multi-turn chat with session memory.

---

## Why These Approaches Were Used

## Approach Summary

- **Scraping approach:** ScraperAPI + BeautifulSoup, with JS rendering only on review pages to balance reliability and API credits.
- **NLP approach:** VADER for sentiment and keyword-driven aspect tagging for transparent, deterministic analytics.
- **Analytics approach:** brand-level aggregation + anomaly detection + recurring praise/complaint surfacing for business readability.
- **App approach:** Streamlit + Plotly for fast interactive BI dashboarding with cross-page filters.
- **Chat approach:** RAG retrieval over filtered dataset with Groq LLM generation and a local fallback when key/API is unavailable.

### ScraperAPI + BeautifulSoup
- ScraperAPI handles anti-bot/network complexity better than plain requests.
- BeautifulSoup keeps extraction transparent and debuggable.
- `render=true` only on review pages reduces API cost while still handling JS-rendered review blocks.

### VADER for Sentiment
- Fast, deterministic, no model hosting needed.
- Works well for short user-review style text.
- Easy to reproduce and explain.

### Keyword-Based Aspect Tagging
- Simple and auditable logic.
- Easy to customize by business users.
- Produces reliable structured fields for dashboarding.

### Streamlit + Plotly
- Rapid iteration for data app UI.
- Interactive charts and filters with minimal boilerplate.
- Good fit for exploratory business intelligence workflows.

### RAG + Optional LLM (Groq)
- Retrieval ensures answers are grounded in your scraped dataset.
- LLM improves natural language interaction and follow-ups.
- Fallback mode keeps app functional even without external LLM credentials.

---

## Project Structure

```text
.
├── scraper.py
├── analyzer.py
├── app.py
├── requirements.txt
├── .env
└── data/
    ├── raw/
    │   ├── *_products.json
    │   └── *_reviews.json
    └── processed/
        ├── products.csv
        ├── reviews.csv
        └── brand_summary.csv
```

---

## Data Schemas

### `products.csv`
- `brand`
- `title`
- `price`
- `mrp`
- `discount_pct`
- `rating`
- `review_count`
- `product_url`
- `asin`

### `reviews.csv`
- `brand`
- `product_title`
- `review_text`
- `star_rating`
- `date`
- `verified_purchase`
- `sentiment_score`
- `wheels`
- `handle`
- `material`
- `zipper`
- `size`
- `durability`

### `brand_summary.csv`
- `brand`
- `avg_price`
- `avg_discount_pct`
- `avg_rating`
- `total_reviews`
- `sentiment_score`
- `top_5_positive_themes`
- `top_5_negative_themes`
- `recurring_praise`
- `recurring_complaints`
- `wheels_score`
- `handle_score`
- `material_score`
- `zipper_score`
- `size_score`
- `durability_score`
- `anomaly_flag`

---

## Setup

## 1) Create / activate environment

Using conda (recommended):

```bash
conda create -n amazon-ci python=3.11 -y
conda activate amazon-ci
```

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

## 3) Configure `.env`

```env
SCRAPERAPI_KEY=your_scraperapi_key
GROQ_API_KEY=your_groq_api_key_optional
```

`GROQ_API_KEY` is optional; without it the chatbot uses fallback mode.

---

## Run Pipeline

## Step A: Scrape raw data

```bash
python scraper.py
```

## Step B: Build processed datasets

```bash
python analyzer.py
```

## Step C: Launch dashboard

```bash
streamlit run app.py
```

---

## Dashboard Pages

### Overview
- total brands/products/reviews/avg sentiment cards
- avg price by brand
- avg discount by brand
- sentiment vs price bubble chart
- brand tier labels (Premium / Mid-range / Value)

### Brand Comparison
- sortable brand summary table
- grouped metric comparison chart
- aspect radar chart
- best performer badges

### Product Drilldown
- brand and product selectors
- product-level KPI cards
- top positive/negative word chart for selected product
- aspect sentiment breakdown for selected product

### Agent Insights
- 5 auto-generated data insight cards
- conversational chatbot for ad-hoc questions on filtered data

---

## Reliability / Operational Notes

- Scraper enforces a 2-second inter-request delay.
- Request failures are logged and skipped (pipeline continues).
- Review scraping uses JS rendering where needed.
- Analyzer and app are designed to run with partial data as long as required CSVs exist.

---

## Current Typical Output Sizes

Based on recent run in this workspace:
- `products.csv`: 50 rows
- `reviews.csv`: 248 rows
- `brand_summary.csv`: 5 rows

---

## Troubleshooting

- `No processed CSV files found`: run `python analyzer.py` after scraping.
- Empty charts after filtering: reset sidebar filters (brand/price/rating/sentiment).
- Chatbot not using LLM: ensure `GROQ_API_KEY` is set and valid.
- Amazon selector breakage: save debug HTML and re-check review/product selectors.

---

## Future Improvements (Optional)

- Better retrieval scoring (TF-IDF / BM25) for chatbot context.
- Time-series tracking across repeated scrapes.
- Model-based topic extraction (beyond keyword themes).
- Exportable report generation (PDF/CSV snapshots for stakeholders).
