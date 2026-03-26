import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


BRANDS = ["Safari", "Skybags", "American Tourister", "VIP", "Nasher Miles"]
RAW_DIR = "data/raw"
REQUEST_DELAY_SECONDS = 2
SCRAPERAPI_ENDPOINT = "http://api.scraperapi.com"
SEARCH_SUFFIX = " luggage"
_LAST_REQUEST_TS = 0.0


def safe_filename(text: str) -> str:
    return text.lower().replace(" ", "_")


def parse_float_from_text(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    cleaned = text.replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    return float(match.group(1)) if match else None


def parse_int_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    cleaned = text.replace(",", "")
    match = re.search(r"(\d+)", cleaned)
    return int(match.group(1)) if match else None


def build_search_url(brand: str) -> str:
    return f"https://www.amazon.in/s?k={quote_plus(brand + SEARCH_SUFFIX)}"


def normalize_product_url(href: str) -> str:
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return f"https://www.amazon.in{href}"
    return f"https://www.amazon.in/{href}"


def scraperapi_get_with_status(url: str, api_key: str, render_js: bool = False) -> Tuple[int, str]:
    global _LAST_REQUEST_TS

    elapsed = time.time() - _LAST_REQUEST_TS
    if elapsed < REQUEST_DELAY_SECONDS:
        time.sleep(REQUEST_DELAY_SECONDS - elapsed)

    try:
        response = requests.get(
            SCRAPERAPI_ENDPOINT,
            params={
                "api_key": api_key,
                "url": url,
                "country_code": "in",
                "render": "true" if render_js else "false",
            },
            timeout=90,
        )
        _LAST_REQUEST_TS = time.time()
        return response.status_code, response.text or ""
    except requests.RequestException as exc:
        _LAST_REQUEST_TS = time.time()
        print(f"[WARN] Failed request for URL: {url}\n       {exc}")
        return 0, ""


def scraperapi_get(url: str, api_key: str) -> Optional[str]:
    status_code, html = scraperapi_get_with_status(url, api_key)
    if status_code >= 400 or status_code == 0:
        print(f"[WARN] Failed request for URL: {url}\n       HTTP {status_code}")
        return None

    html_l = html.lower()
    blocked_signals = [
        "captcha",
        "enter the characters you see below",
        "type the characters you see in this image",
        "sorry, we just need to make sure you're not a robot",
        "robot check",
    ]
    if not html.strip() or any(signal in html_l for signal in blocked_signals):
        print("BLOCKED")
    return html


def extract_products(html: str, brand: str, limit: int = 10) -> List[Dict]:
    soup = BeautifulSoup(html, "lxml")
    cards = soup.find_all("div", {"data-asin": True})
    cards = [card for card in cards if card.get("data-asin")]

    products: List[Dict] = []
    for card in cards:
        if len(products) >= limit:
            break

        asin = card.get("data-asin", "").strip()
        if not asin:
            continue

        title_tag = card.select_one("h2 span")
        if not title_tag:
            title_tag = card.select_one(".a-size-medium")
        if not title_tag:
            title_tag = card.select_one(".a-size-base-plus")

        link_tag = card.select_one("h2 a")
        if not link_tag:
            link_tag = card.select_one("a[href]")
        if not title_tag or not link_tag or not link_tag.get("href"):
            continue

        title = title_tag.get_text(strip=True)
        href = link_tag.get("href", "")
        product_url = normalize_product_url(href)

        price_tag = card.select_one(".a-price .a-offscreen")
        mrp_tag = card.select_one(".a-price.a-text-price .a-offscreen")
        rating_tag = card.select_one(".a-icon-alt")

        review_count_tag = card.select_one(".a-size-small .a-link-normal")

        full_text = card.get_text(" ", strip=True)
        discount_match = re.search(r"(\d+)%\s*off", full_text, flags=re.IGNORECASE)

        products.append(
            {
                "title": title,
                "price": parse_float_from_text(price_tag.get_text(strip=True) if price_tag else None),
                "mrp": parse_float_from_text(mrp_tag.get_text(strip=True) if mrp_tag else None),
                "discount_pct": float(discount_match.group(1)) if discount_match else None,
                "rating": parse_float_from_text(rating_tag.get_text(strip=True) if rating_tag else None),
                "review_count": parse_int_from_text(
                    review_count_tag.get_text(strip=True) if review_count_tag else None
                ),
                "product_url": product_url,
                "brand": brand,
                "asin": asin,
            }
        )

    return products


def extract_reviews(
    html: str,
    brand: str,
    product_title: str,
    limit: int = 10,
    debug: bool = False,
) -> List[Dict]:
    soup = BeautifulSoup(html, "lxml")
    if debug:
        os.makedirs(RAW_DIR, exist_ok=True)
        debug_path = os.path.join(RAW_DIR, "debug_reviews.html")
        with open(debug_path, "w", encoding="utf-8") as debug_file:
            debug_file.write(html)

        count_data_hook_any = len(soup.find_all("div", {"data-hook": True}))
        count_review_list = len(soup.find_all("div", {"id": "cm_cr-review_list"}))
        count_review_divs = len(soup.find_all("div", {"data-hook": "review"}))
        count_data_hook_review = len(soup.select('[data-hook="review"]'))
        count_review_class = len(soup.find_all("div", class_="review"))

        print(f"[DEBUG] Saved reviews HTML -> {debug_path}")
        print(f"[DEBUG] soup.find_all('div', {{'data-hook': True}}): {count_data_hook_any}")
        print(f"[DEBUG] soup.find_all('div', {{'data-hook': 'review'}}): {count_review_divs}")
        print(
            "[DEBUG] soup.find_all('div', {'id': 'cm_cr-review_list'}): "
            f"{count_review_list}"
        )
        print(f"[DEBUG] soup.select('[data-hook=\"review\"]'): {count_data_hook_review}")
        print(f"[DEBUG] soup.find_all('div', class_='review'): {count_review_class}")
        print("[DEBUG] First 1000 characters of page HTML:")
        print(html[:1000])

    review_blocks = soup.select('[data-hook="review"]')
    reviews: List[Dict] = []

    for block in review_blocks:
        if len(reviews) >= limit:
            break

        body_tag = block.select_one('[data-hook="review-body"] span')
        if body_tag is None:
            continue

        rating_tag = block.select_one('[data-hook="review-star-rating"] span')
        if not rating_tag:
            rating_tag = block.select_one('[data-hook="cmps-review-star-rating"] span')
        date_tag = block.select_one('[data-hook="review-date"]')
        verified_tag = block.select_one('[data-hook="avp-badge"]')

        review_text = body_tag.get_text(strip=True)
        star_rating = rating_tag.get_text(strip=True) if rating_tag else None
        date_text = date_tag.get_text(strip=True) if date_tag else None

        reviews.append(
            {
                "review_text": review_text,
                "star_rating": star_rating,
                "date": date_text,
                "verified_purchase": bool(verified_tag),
                "brand": brand,
                "product_title": product_title,
            }
        )

    return reviews


def build_reviews_urls(asin: str) -> List[str]:
    return [f"https://www.amazon.in/dp/{asin}#customerReviews"]


def save_json(path: str, payload: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    load_dotenv()
    api_key = os.getenv("SCRAPERAPI_KEY", "").strip()
    if not api_key or api_key == "placeholder":
        print("[WARN] SCRAPERAPI_KEY is missing or placeholder. Scraping may fail.")

    os.makedirs(RAW_DIR, exist_ok=True)

    for brand in BRANDS:
        print(f"Scraping {brand} products...", end=" ")
        search_url = build_search_url(brand)

        search_html = scraperapi_get(search_url, api_key)
        if not search_html:
            print("done (0 found)")
            print(f"[WARN] Skipping {brand}: search request failed.")
            continue

        products = extract_products(search_html, brand=brand, limit=10)
        print(f"done ({len(products)} found)")

        all_reviews: List[Dict] = []
        for product in products:
            product_title = product.get("title", "Unknown Product")
            print(f"Scraping {product_title} reviews...", end=" ")

            reviews: List[Dict] = []
            asin = product.get("asin", "").strip()
            if not asin:
                print("done (0 found)")
                print("[WARN] Missing ASIN; skipping review fetch.")
                continue

            for url_index, reviews_url in enumerate(build_reviews_urls(asin), start=1):
                status_code, reviews_html = scraperapi_get_with_status(
                    reviews_url,
                    api_key,
                    render_js=True,
                )
                if status_code != 200:
                    print(
                        f"[WARN] {brand} {asin}: reviews URL {url_index} returned HTTP {status_code}"
                    )
                    continue

                reviews = extract_reviews(
                    reviews_html,
                    brand=brand,
                    product_title=product["title"],
                    limit=10,
                )
                break

            print(f"done ({len(reviews)} found)")
            all_reviews.extend(reviews)

        product_file = os.path.join(RAW_DIR, f"{safe_filename(brand)}_products.json")
        review_file = os.path.join(RAW_DIR, f"{safe_filename(brand)}_reviews.json")

        save_json(product_file, products)
        save_json(review_file, all_reviews)
        print(f"[INFO] Saved {product_file} and {review_file}")

    print("\n[DONE] Scraping finished.")


if __name__ == "__main__":
    main()
