"""
DATA SCIENCE PGC CAPSTONE
Project: Job Market Analytics

PHASE 1 - WEB DATA ACQUISITION

Source:
    Jobicy Public Remote Jobs API
    https://jobicy.com/api/v2/remote-jobs

Purpose:
    Collect 1,000+ unique public job listings by querying multiple
    categories/keywords, then save the raw dataset as CSV.

Important:
    - No API key is required according to Jobicy's public API documentation.
    - The API returns up to 200 jobs per request.
    - We keep the original Jobicy listing URL and source name.
    - The script deduplicates jobs using the Jobicy job ID.
"""

import time
import re
from html import unescape

import requests
import pandas as pd
from bs4 import BeautifulSoup


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

BASE_URL = "https://jobicy.com/api/v2/remote-jobs"

TARGET_RECORDS = 1000
REQUEST_COUNT = 200
REQUEST_DELAY = 2          # seconds between API calls
OUTPUT_FILE = "raw_job_market_data.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0 Safari/537.36"
    )
}


# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def clean_html(html_text):
    """Convert HTML job descriptions into readable plain text."""
    if not html_text:
        return ""

    soup = BeautifulSoup(html_text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def request_jobs(params):
    """Request jobs from Jobicy API with basic error handling."""
    response = requests.get(
        BASE_URL,
        params=params,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    return data.get("jobs", [])


def normalize_job(job, source_filter):
    """Convert one API job object into our project schema."""

    industries = job.get("jobIndustry", [])
    job_types = job.get("jobType", [])

    if isinstance(industries, list):
        industries = ", ".join(map(str, industries))

    if isinstance(job_types, list):
        job_types = ", ".join(map(str, job_types))

    return {
        "Job_ID": job.get("id"),
        "Job_Title": job.get("jobTitle"),
        "Company": job.get("companyName"),
        "Location": job.get("jobGeo"),
        "Job_Level": job.get("jobLevel"),
        "Job_Industry": industries,
        "Employment_Type": job_types,
        "Job_Description": clean_html(job.get("jobDescription")),
        "Job_Excerpt": clean_html(job.get("jobExcerpt")),
        "Publication_Date": job.get("pubDate"),
        "Salary_Min": job.get("salaryMin"),
        "Salary_Max": job.get("salaryMax"),
        "Salary_Currency": job.get("salaryCurrency"),
        "Salary_Period": job.get("salaryPeriod"),
        "Job_URL": job.get("url"),
        "Source": "Jobicy",
        "Collected_Using": source_filter
    }


# ---------------------------------------------------------
# 3. DISCOVER CURRENT TAXONOMIES
# ---------------------------------------------------------

def get_taxonomies():
    """
    Fetch current Jobicy industry and location taxonomies.
    This avoids hard-coding potentially outdated slugs.
    """

    industries_response = requests.get(
        BASE_URL,
        params={"get": "industries"},
        headers=HEADERS,
        timeout=30
    )
    industries_response.raise_for_status()

    locations_response = requests.get(
        BASE_URL,
        params={"get": "locations"},
        headers=HEADERS,
        timeout=30
    )
    locations_response.raise_for_status()

    industries_data = industries_response.json()
    locations_data = locations_response.json()

    return industries_data, locations_data


# ---------------------------------------------------------
# 4. SCRAPING PLAN
# ---------------------------------------------------------

def collect_jobs():

    # Broad keywords chosen for a general Job Market Analytics project.
    # They intentionally cover technology, analytics, business and other
    # job families instead of collecting only one role.
    keyword_queries = [
        "data",
        "analytics",
        "data science",
        "software",
        "python",
        "sql",
        "machine learning",
        "business analyst",
        "engineering",
        "marketing",
        "finance",
        "sales",
        "design",
        "customer",
        "project management",
        "product",
        "human resources",
        "operations",
        "devops",
        "cloud"
    ]

    all_jobs = {}

    print("=" * 70)
    print("JOB MARKET ANALYTICS - WEB DATA ACQUISITION")
    print("=" * 70)
    print(f"Target unique records : {TARGET_RECORDS}")
    print(f"Records per request   : {REQUEST_COUNT}")
    print()

    # -----------------------------------------------------
    # A. First request: broad dataset
    # -----------------------------------------------------

    print("Requesting broad Jobicy dataset...")

    try:
        jobs = request_jobs({
            "count": REQUEST_COUNT
        })

        for job in jobs:
            job_id = job.get("id")

            if job_id is not None:
                all_jobs[job_id] = normalize_job(job, "all_jobs")

        print(f"Unique records collected: {len(all_jobs)}")

    except requests.RequestException as error:
        print(f"Broad request failed: {error}")

    time.sleep(REQUEST_DELAY)

    # -----------------------------------------------------
    # B. Query multiple keywords
    # -----------------------------------------------------

    for keyword in keyword_queries:

        if len(all_jobs) >= TARGET_RECORDS:
            break

        print(
            f"Searching keyword: {keyword:<22} | "
            f"Current unique records: {len(all_jobs)}"
        )

        try:
            jobs = request_jobs({
                "count": REQUEST_COUNT,
                "tag": keyword
            })

            new_records = 0

            for job in jobs:
                job_id = job.get("id")

                if job_id is None:
                    continue

                if job_id not in all_jobs:
                    all_jobs[job_id] = normalize_job(
                        job,
                        f"keyword:{keyword}"
                    )
                    new_records += 1

            print(
                f"   API returned: {len(jobs)} | "
                f"New unique: {new_records} | "
                f"Total unique: {len(all_jobs)}"
            )

        except requests.RequestException as error:
            print(f"   Request failed: {error}")

        time.sleep(REQUEST_DELAY)

    # -----------------------------------------------------
    # C. Query major industries if still below target
    # -----------------------------------------------------

    industry_queries = [
        "engineering",
        "data-science",
        "information-technology",
        "marketing",
        "sales",
        "finance",
        "design-multimedia",
        "supporting",
        "management",
        "project-management",
        "product",
        "human-resources",
        "operations",
        "writing",
        "devops",
        "legal",
        "healthcare",
        "education",
        "customer-service"
    ]

    for industry in industry_queries:

        if len(all_jobs) >= TARGET_RECORDS:
            break

        print(
            f"Searching industry: {industry:<22} | "
            f"Current unique records: {len(all_jobs)}"
        )

        try:
            jobs = request_jobs({
                "count": REQUEST_COUNT,
                "industry": industry
            })

            new_records = 0

            for job in jobs:
                job_id = job.get("id")

                if job_id is None:
                    continue

                if job_id not in all_jobs:
                    all_jobs[job_id] = normalize_job(
                        job,
                        f"industry:{industry}"
                    )
                    new_records += 1

            print(
                f"   API returned: {len(jobs)} | "
                f"New unique: {new_records} | "
                f"Total unique: {len(all_jobs)}"
            )

        except requests.RequestException as error:
            print(f"   Request failed: {error}")

        time.sleep(REQUEST_DELAY)

    return list(all_jobs.values())


# ---------------------------------------------------------
# 5. SAVE RAW DATA
# ---------------------------------------------------------

def main():

    records = collect_jobs()

    df = pd.DataFrame(records)

    # Remove duplicate Job_ID values once more as a safety check.
    if "Job_ID" in df.columns:
        df = df.drop_duplicates(subset=["Job_ID"])

    # Sort newest listings first.
    if "Publication_Date" in df.columns:
        df["Publication_Date"] = pd.to_datetime(
            df["Publication_Date"],
            errors="coerce",
            utc=True
        )

        df = df.sort_values(
            by="Publication_Date",
            ascending=False,
            na_position="last"
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("SCRAPING COMPLETED")
    print("=" * 70)
    print(f"Final unique records : {len(df)}")
    print(f"Columns              : {len(df.columns)}")
    print(f"Output file          : {OUTPUT_FILE}")

    print("\nColumns collected:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nMissing values:")
    print(df.isna().sum())

    if len(df) >= TARGET_RECORDS:
        print(
            f"\nSUCCESS: The dataset contains {len(df)} unique records, "
            f"meeting the 1,000+ record requirement."
        )
    else:
        print(
            f"\nWARNING: Only {len(df)} unique records were available "
            f"from the current API snapshot. Do NOT fabricate records. "
            f"Run the scraper again later or broaden the approved "
            f"source strategy before submission."
        )


if __name__ == "__main__":
    main()
