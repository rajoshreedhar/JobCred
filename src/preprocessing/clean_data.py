"""
Data Cleaning & Transformation Pipeline for JobCred (fake_job_postings.csv).

Executes systematic cleaning, standardization, deduplication, structured parsing,
missing-value handling, and exports the cleaned dataset to data/processed/cleaned_data.csv.
"""

from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_data.csv"

# Common uppercase acronyms to preserve when standardizing department casing
ACRONYMS = {
    "IT", "HR", "QA", "QC", "UI", "UX", "BI", "PR", "PM", "SEO", "SEM", "CRM",
    "ERP", "API", "SQL", "AWS", "GCP", "R&D", "B2B", "B2C", "VP", "CEO", "CFO",
    "CTO", "COO", "CIO", "USA", "US", "UK", "EU", "EMEA", "APAC", "LATAM",
    "IOS", "AI", "ML", "NLP", "NOC", "SOC", "DEVOPS", "SAAS", "PAAS", "IAAS",
}

SENTINEL_STRINGS = {"", "na", "n/a", "none", "null", "nan", "-", "--"}
DEPARTMENT_SENTINELS = SENTINEL_STRINGS | {"unknown", "unspecified"}


def to_snake_case(name: str) -> str:
    """Convert any column header to standardized snake_case."""
    s = re.sub(r"[\s\-/]+", "_", str(name).strip())
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[^a-zA-Z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    return s.lower().strip("_")


def clean_text_value(val: object) -> object:
    """Unescape HTML entities, remove non-breaking spaces, and normalize whitespace."""
    if pd.isna(val):
        return np.nan
    text = html.unescape(str(val))
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else np.nan


def standardize_department(val: object) -> object:
    """Standardize department casing while preserving recognized acronyms."""
    if pd.isna(val):
        return np.nan
    text = str(val).strip()
    if text.lower() in DEPARTMENT_SENTINELS:
        return np.nan

    tokens = text.split()
    out_tokens = []
    for tok in tokens:
        upper = tok.upper()
        # Preserve acronyms (including slash-joined acronyms like UI/UX)
        if upper in ACRONYMS or all(part in ACRONYMS for part in upper.split("/")):
            out_tokens.append(upper)
        elif tok.isupper() or tok.islower():
            out_tokens.append(tok.capitalize())
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens)


def parse_location(series: pd.Series) -> pd.DataFrame:
    """Parse 'CC, STATE, CITY' location strings into normalized components."""
    countries, states, cities, formatted = [], [], [], []
    for val in series:
        if pd.isna(val):
            countries.append("Unknown")
            states.append("Unknown")
            cities.append("Unknown")
            formatted.append("Unknown")
            continue

        parts = [p.strip() for p in str(val).split(",")]
        country = parts[0].upper() if len(parts) >= 1 and parts[0] else "Unknown"
        raw_state = parts[1] if len(parts) >= 2 and parts[1] else ""
        # Join any trailing comma segments into city (e.g., 'Tel Aviv, Israel')
        raw_city = ", ".join(p for p in parts[2:] if p) if len(parts) >= 3 else ""

        if raw_state:
            state = raw_state.upper() if len(raw_state) <= 3 and raw_state.isalnum() else raw_state.title()
        else:
            state = "Unknown"

        if raw_city:
            city = raw_city.title() if (raw_city.islower() or raw_city.isupper()) else raw_city
        else:
            city = "Unknown"

        countries.append(country)
        states.append(state)
        cities.append(city)
        clean_parts = [p for p in (country, state, city) if p != "Unknown"]
        formatted.append(", ".join(clean_parts) if clean_parts else "Unknown")

    return pd.DataFrame(
        {
            "location": formatted,
            "country_code": countries,
            "state_region": states,
            "city": cities,
        },
        index=series.index,
    )


def parse_salary_range(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Parse salary_range strings into numeric (salary_min, salary_max), cleaning currency
    symbols/commas and rejecting Excel date-mangled strings (e.g., '10-Oct') and '0-0' placeholders.
    """
    range_pat = re.compile(r"^\s*\$?\s*(\d+(?:\.\d+)?)\s*-\s*\$?\s*(\d+(?:\.\d+)?)\s*$")
    single_pat = re.compile(r"^\s*\$?\s*(\d+(?:\.\d+)?)\s*$")

    s_min, s_max, s_clean, s_valid = [], [], [], []
    for val in series:
        if pd.isna(val):
            s_min.append(np.nan)
            s_max.append(np.nan)
            s_clean.append("Not Specified")
            s_valid.append(0)
            continue

        raw = str(val).replace(",", "").strip()
        m_range = range_pat.match(raw)
        m_single = single_pat.match(raw) if not m_range else None

        if m_range:
            lo, hi = float(m_range.group(1)), float(m_range.group(2))
            if lo > hi:
                lo, hi = hi, lo
            if lo == 0.0 and hi == 0.0:
                s_min.append(np.nan)
                s_max.append(np.nan)
                s_clean.append("Not Specified")
                s_valid.append(0)
            else:
                s_min.append(lo)
                s_max.append(hi)
                s_clean.append(f"{int(lo) if lo.is_integer() else lo}-{int(hi) if hi.is_integer() else hi}")
                s_valid.append(1)
        elif m_single:
            v = float(m_single.group(1))
            if v == 0.0:
                s_min.append(np.nan)
                s_max.append(np.nan)
                s_clean.append("Not Specified")
                s_valid.append(0)
            else:
                s_min.append(v)
                s_max.append(v)
                s_clean.append(f"{int(v) if v.is_integer() else v}-{int(v) if v.is_integer() else v}")
                s_valid.append(1)
        else:
            # Reject Excel date-mangled values like '10-Oct', 'Oct-20', '9-Dec'
            s_min.append(np.nan)
            s_max.append(np.nan)
            s_clean.append("Not Specified")
            s_valid.append(0)

    return (
        pd.Series(s_min, index=series.index, dtype="float64"),
        pd.Series(s_max, index=series.index, dtype="float64"),
        pd.Series(s_clean, index=series.index, dtype="object"),
        pd.Series(s_valid, index=series.index, dtype="int8"),
    )


def flag_iqr_outliers(series: pd.Series) -> pd.Series:
    """Return int8 indicator (1 = IQR outlier, 0 = within bounds or NaN)."""
    valid = series.dropna()
    if valid.empty:
        return pd.Series(0, index=series.index, dtype="int8")
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((series < lo) | (series > hi)).fillna(False).astype("int8")


def run_cleaning_pipeline(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = CLEANED_DATA_PATH,
) -> pd.DataFrame:
    """Run full cleaning pipeline and save cleaned CSV."""
    df = pd.read_csv(input_path)
    raw_shape = df.shape

    # 1. Standardize column names to snake_case
    df.columns = [to_snake_case(c) for c in df.columns]

    # 2. Clean text/string formatting (HTML entities, NBSP, whitespace)
    text_and_cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in text_and_cat_cols:
        df[col] = df[col].map(clean_text_value)

    # Convert pure sentinel strings in benefits/requirements/description to NaN
    for col in ["description", "requirements", "benefits"]:
        mask_sentinel = df[col].dropna().str.lower().isin(SENTINEL_STRINGS)
        if mask_sentinel.any():
            df.loc[mask_sentinel[mask_sentinel].index, col] = np.nan

    # 3. Standardize department casing (preserving acronyms)
    df["department"] = df["department"].map(standardize_department)

    # 4. Preserve high-signal missingness indicators BEFORE imputation
    df["has_company_profile"] = df["company_profile"].notna().astype("int8")

    # 5. Parse location into structured columns
    loc_parsed = parse_location(df["location"])
    df["location"] = loc_parsed["location"]
    df["country_code"] = loc_parsed["country_code"]
    df["state_region"] = loc_parsed["state_region"]
    df["city"] = loc_parsed["city"]

    # 6. Parse salary_range into numeric salary_min / salary_max & validity/outlier flags
    salary_min, salary_max, salary_clean, has_valid_salary = parse_salary_range(df["salary_range"])
    df["salary_range"] = salary_clean
    df["salary_min"] = salary_min
    df["salary_max"] = salary_max
    df["has_valid_salary"] = has_valid_salary

    # 7. Impute missing values cleanly per column group
    categorical_fill_unknown = [
        "department",
        "employment_type",
        "required_experience",
        "required_education",
        "industry",
        "function",
    ]
    for col in categorical_fill_unknown:
        df[col] = df[col].fillna("Unknown")

    free_text_fill = ["company_profile", "description", "requirements", "benefits"]
    for col in free_text_fill:
        df[col] = df[col].fillna("Not Provided")

    # 8. Remove exact duplicates AFTER all normalization/standardization (excluding job_id)
    content_cols = [c for c in df.columns if c != "job_id"]
    exact_dupes_removed = int(df.duplicated(subset=content_cols, keep="first").sum())
    df = df.drop_duplicates(subset=content_cols, keep="first").reset_index(drop=True)

    # Compute IQR outlier flags on deduplicated valid salaries
    df["salary_outlier_flag"] = (
        (flag_iqr_outliers(df["salary_min"]) == 1) | (flag_iqr_outliers(df["salary_max"]) == 1)
    ).astype("int8")

    # 9. Enforce clean numeric dtypes
    df["job_id"] = df["job_id"].astype("int32")
    for bin_col in ["telecommuting", "has_company_logo", "has_questions", "fraudulent"]:
        df[bin_col] = df[bin_col].astype("int8")

    # 10. Order columns logically
    ordered_cols = [
        "job_id",
        "title",
        "location",
        "country_code",
        "state_region",
        "city",
        "department",
        "salary_range",
        "salary_min",
        "salary_max",
        "has_valid_salary",
        "salary_outlier_flag",
        "company_profile",
        "has_company_profile",
        "description",
        "requirements",
        "benefits",
        "telecommuting",
        "has_company_logo",
        "has_questions",
        "employment_type",
        "required_experience",
        "required_education",
        "industry",
        "function",
        "fraudulent",
    ]
    df = df[ordered_cols]

    # 11. Export cleaned dataset
    os.makedirs(output_path.parent, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print("=" * 72)
    print("CLEANING PIPELINE SUMMARY")
    print("=" * 72)
    print(f"Raw dataset shape:          {raw_shape}")
    print(f"Exact duplicates removed:   {exact_dupes_removed}")
    print(f"Cleaned dataset shape:      {df.shape}")
    print(f"Exported to:                {output_path}")
    print(f"Exported file size:         {output_path.stat().st_size / (1024**2):.2f} MB")
    print("\nFraud class distribution (cleaned):")
    vc = df["fraudulent"].value_counts()
    for k, v in vc.items():
        print(f"  fraudulent={k}: {v:,} ({v/len(df)*100:.2f}%)")
    print("\nPost-cleaning null counts (non-zero only):")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0].to_string() if (nulls > 0).any() else "  None (0 nulls)")
    print(f"\nDepartment unique count after casing standardization: {df['department'].nunique()}")
    print(f"Valid parsed numeric salaries: {df['has_valid_salary'].sum():,} "
          f"(IQR outliers flagged: {df['salary_outlier_flag'].sum():,})")
    print(f"Unique countries parsed: {df['country_code'].nunique()}")
    return df


if __name__ == "__main__":
    run_cleaning_pipeline()
