import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 48)

ALPHA = 0.05
NULL_LIKE = {"", "na", "n/a", "null", "none", "nan", "not specified", "not available", "-"}
CITY_ALIASES = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "secunderabad": "Hyderabad",
    "gurgaon": "Gurugram",
    "bombay": "Mumbai",
    "madras": "Chennai",
    "calcutta": "Kolkata",
}
SKILL_TERMS = [
    "Python", "Java", "JavaScript", "SQL", "MySQL", "Oracle", "PHP", ".NET", "ASP.NET", "C++", "C#",
    "HTML", "CSS", "jQuery", "Angular", "React", "Node.js", "Android", "iOS", "AWS", "Linux",
    "Selenium", "Manual Testing", "Hadoop", "Machine Learning", "SEO", "Excel", "MS Office",
    "Tally", "SAP", "Salesforce", "Photoshop", "AutoCAD", "Communication", "English", "Hibernate", "WordPress",
]
PAY_RANGE = re.compile(r"^(\d[\d,]*)\s*-\s*(\d[\d,]*)\s*P\.?\s?A", re.IGNORECASE)


def banner(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def sub_banner(title):
    print("\n--- " + title + " ---")


def format_p(p_value):
    if p_value < 1e-300:
        return "< 1e-300"
    if p_value < 0.001:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def p_text(p_value):
    if p_value < 1e-300:
        return "p < 1e-300"
    return f"p = {format_p(p_value)}"


def text_bars(counts, width=40, top=10):
    counts = counts.head(top)
    peak = counts.max()
    total_width = max(len(str(label)) for label in counts.index)
    for label, value in counts.items():
        bar = "#" * max(1, int(width * value / peak))
        print(f"{str(label)[:48]:<{min(total_width, 48)}} | {bar} {value:,}")


def clean_text(series):
    cleaned = series.astype("object").where(series.notna(), None)
    cleaned = cleaned.map(lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v)
    is_null_like = cleaned.map(lambda v: isinstance(v, str) and v.lower() in NULL_LIKE)
    return cleaned.mask(is_null_like, np.nan)


def primary_city(value):
    if not isinstance(value, str):
        return np.nan
    first = value.split(",")[0].split("/")[0].strip().lower()
    if not first:
        return np.nan
    return CITY_ALIASES.get(first, first.title())


def normalize_title(value):
    if not isinstance(value, str):
        return np.nan
    title = re.sub(r"\(.*?\)", " ", value.lower())
    title = title.split(" - ")[0]
    title = re.sub(r"\b(walk-?in|urgent|hiring|opening|openings|required|requirement|wanted|for)\b", " ", title)
    title = re.sub(r"[^a-z/& ]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title.title() if title else np.nan


def ug_label(value):
    if not isinstance(value, str):
        return np.nan
    match = re.search(r"UG:\s*(.*?)(?:\s+PG:|\s+Doctorate:|$)", value)
    if not match:
        return "No UG listed"
    return match.group(1).split(" - ")[0].split(",")[0].strip() or "No UG listed"


def pg_listed(value):
    if not isinstance(value, str):
        return np.nan
    match = re.search(r"PG:\s*(.*?)(?:\s+Doctorate:|$)", value)
    if not match:
        return False
    return "not required" not in match.group(1).lower()


def pay_category(raw, is_valid):
    if not isinstance(raw, str):
        return "Missing"
    lowered = raw.lower()
    if "not disclosed" in lowered:
        return "Not disclosed"
    if re.search(r"pay band|pay scale|grade pay|agp", lowered):
        return "Government pay band"
    if is_valid:
        return "Annual range (P.A)"
    if re.search(r"best in|negotiable|competitive|as per|industry standard|market|attractive|lucrative|incentive", lowered):
        return "Non-numeric text"
    return "Other format"


def skill_pattern(term):
    return rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"


def iqr_report(series, name):
    clean = series.dropna()
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    spread = q3 - q1
    lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
    flagged = ((series < lower) | (series > upper)).fillna(False)
    row = {
        "column": name, "n": len(clean), "q1": q1, "median": clean.median(), "q3": q3,
        "lower_fence": lower, "upper_fence": upper, "outliers": int(flagged.sum()),
        "outlier_pct": round(100 * flagged.sum() / max(len(clean), 1), 2),
        "skewness": round(clean.skew(), 2),
    }
    return row, flagged


def load_raw(path):
    banner("PHASE 1: DATA INSPECTION")
    raw = pd.read_csv(path)
    print(f"Source file      : {path}")
    print(f"Records          : {raw.shape[0]:,}")
    print(f"Attributes       : {raw.shape[1]}")
    sub_banner("Column names and data types")
    print(raw.dtypes.to_string())
    sub_banner("Sample records (5)")
    print(raw[["company", "experience", "industry", "joblocation_address", "jobtitle", "payrate", "postdate"]].head(5).to_string())
    sub_banner("Missing values")
    missing = pd.DataFrame({
        "missing": raw.isna().sum(),
        "missing_pct": (raw.isna().mean() * 100).round(2),
        "unique": raw.nunique(),
    }).sort_values("missing_pct", ascending=False)
    print(missing.to_string())
    sub_banner("Duplicate checks")
    print(f"Exact duplicate rows                 : {raw.duplicated().sum()}")
    print(f"Duplicate rows ignoring uniq_id      : {raw.drop(columns='uniq_id').duplicated().sum()}")
    print(f"Rows sharing a jobid with another row: {raw['jobid'].duplicated(keep=False).sum()}")
    print(f"Excess rows from repeated jobid      : {raw['jobid'].duplicated().sum()}")
    return raw


def examine_repeated_jobids(df):
    sub_banner("Repeated jobid examination (not auto-removed)")
    repeated = df[df["jobid"].duplicated(keep=False)]
    key_fields = ["company", "jobtitle", "joblocation_address", "postdate"]
    group_sizes = repeated.groupby("jobid").size()
    same_key = repeated.groupby("jobid")[key_fields].nunique(dropna=False).max(axis=1).eq(1)
    print(f"jobids repeated                          : {len(group_sizes)}")
    print(f"Rows involved                            : {len(repeated)}")
    print(f"Groups identical on {key_fields}: {int(same_key.sum())}")
    print(f"Groups differing on at least one field   : {int((~same_key).sum())}")
    print("Decision: rows kept and flagged (jobid_repeated) because uniq_id differs and the portal")
    print("documentation does not state that jobid is a unique listing key.")


def clean_data(raw):
    banner("PHASE 2: CLEANING AND PREPROCESSING")
    df = raw.copy()
    text_columns = ["company", "education", "experience", "industry", "jobdescription",
                    "joblocation_address", "jobtitle", "payrate", "skills", "site_name"]
    before_nulls = int(df[text_columns].isna().sum().sum())
    for column in text_columns:
        df[column] = clean_text(df[column])
    after_nulls = int(df[text_columns].isna().sum().sum())
    print(f"Whitespace normalized and null-like strings converted: {after_nulls - before_nulls} new missing values")

    examine_repeated_jobids(df)

    df["jobdescription"] = df["jobdescription"].str.replace(
        r"^Job Description\s*Send me Jobs like this\s*-?\s*", "", regex=True)
    df["industry"] = df["industry"].str.replace(r"\s*/\s*", " / ", regex=True)
    df["functional_area"] = df["skills"]
    df["primary_city"] = df["joblocation_address"].map(primary_city)
    df["location_count"] = df["joblocation_address"].map(
        lambda v: len([p for p in v.split(",") if p.strip()]) if isinstance(v, str) else np.nan)
    df["title_clean"] = df["jobtitle"].map(normalize_title)

    df["postdate"] = pd.to_datetime(df["postdate"], errors="coerce", utc=True).dt.tz_localize(None)

    experience = df["experience"].str.extract(r"(\d+)\s*-\s*(\d+)").astype(float)
    df["exp_min"], df["exp_max"] = experience[0], experience[1]
    invalid_range = df["exp_max"] < df["exp_min"]
    df.loc[invalid_range, ["exp_min", "exp_max"]] = np.nan
    df["exp_mid"] = (df["exp_min"] + df["exp_max"]) / 2

    df["numberofpositions"] = pd.to_numeric(df["numberofpositions"], errors="coerce")
    df.loc[df["numberofpositions"] <= 0, "numberofpositions"] = np.nan

    pay_parts = df["payrate"].fillna("").str.extract(PAY_RANGE)
    df["pay_min"] = pd.to_numeric(pay_parts[0].str.replace(",", "", regex=False), errors="coerce")
    df["pay_max"] = pd.to_numeric(pay_parts[1].str.replace(",", "", regex=False), errors="coerce")
    valid_pay = (df["pay_min"] > 0) & (df["pay_max"] >= df["pay_min"]) & (df["pay_max"] <= 1e8)
    df.loc[~valid_pay, ["pay_min", "pay_max"]] = np.nan
    df["pay_mid"] = (df["pay_min"] + df["pay_max"]) / 2
    df["has_salary"] = valid_pay
    df["pay_category"] = [pay_category(raw_value, flag) for raw_value, flag in zip(df["payrate"], valid_pay)]

    df["ug_requirement"] = df["education"].map(ug_label)
    df["pg_listed"] = df["education"].map(pg_listed)
    return df


def report_outliers(df):
    sub_banner("Outlier detection (IQR method; flagged, not deleted)")
    rows = []
    for column, label in [("pay_mid", "pay_mid (annual range midpoint)"),
                          ("numberofpositions", "numberofpositions"),
                          ("exp_mid", "exp_mid (years)"),
                          ("desc_length", "desc_length (characters)")]:
        row, flagged = iqr_report(df[column], label)
        df[f"{column}_outlier"] = flagged
        rows.append(row)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nTreatment: pay, positions and description length are right-skewed, so extreme values are")
    print("plausible (senior roles, bulk hiring drives, long descriptions). They are flagged and retained;")
    print("rank-based statistics are used so a few large values cannot dominate results.")


def engineer_features(df):
    banner("PHASE 3: FEATURE ENGINEERING")
    df["posting_year"] = df["postdate"].dt.year
    df["posting_month"] = df["postdate"].dt.month
    df["posting_quarter"] = df["postdate"].dt.to_period("Q").astype("string")
    df["posting_year_month"] = df["postdate"].dt.to_period("M").astype("string")
    df["exp_category"] = pd.cut(
        df["exp_min"], bins=[-1, 0, 2, 5, 10, 100],
        labels=["0 yrs minimum", "1-2 yrs minimum", "3-5 yrs minimum", "6-10 yrs minimum", "10+ yrs minimum"])
    df["desc_length"] = df["jobdescription"].str.len()
    df["jobid_repeated"] = df["jobid"].duplicated(keep=False)

    patterns = {term: skill_pattern(term) for term in SKILL_TERMS}
    description = df["jobdescription"].fillna("").str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
    for term, pattern in patterns.items():
        df[f"skill_{term}"] = description.str.contains(pattern, case=False, regex=True)
    skill_columns = [f"skill_{term}" for term in SKILL_TERMS]
    df["n_skills_extracted"] = df[skill_columns].sum(axis=1)

    features = pd.DataFrame([
        ("posting_year / month / quarter", "postdate", "Time trend analysis"),
        ("exp_min, exp_max, exp_mid, exp_category", "experience range text", "Experience demand and salary correlation (category uses minimum years)"),
        ("primary_city, location_count", "joblocation_address", "First listed city after alias merging; count of listed cities"),
        ("title_clean", "jobtitle", "Lower-cased, brackets and filler words removed for grouping"),
        ("ug_requirement, pg_listed", "education", "First UG option listed; whether a PG option appears without a 'Not Required' waiver"),
        ("pay_min, pay_max, pay_mid, has_salary", "payrate", "Only strict 'X - Y P.A' strings parsed"),
        ("pay_category", "payrate", "Not disclosed / non-numeric / government band / annual range / other"),
        ("desc_length", "jobdescription", "Character count after boilerplate removal"),
        ("skill_* flags, n_skills_extracted", "jobdescription", "Keyword match with word boundaries over a fixed vocabulary"),
        ("jobid_repeated", "jobid", "Flags jobids that appear more than once"),
    ], columns=["feature", "source", "how_and_why"])
    print(features.to_string(index=False))
    print("\nLeakage note: no target variable is modelled, so no leakage risk arises; all features use only")
    print("information present in each listing.")
    return df


def validate(raw, df):
    banner("PHASE 4: VALIDATION")
    print(f"Original shape : {raw.shape}")
    print(f"Cleaned shape  : {df.shape}  (rows unchanged; no records dropped, new columns added)")
    print(f"Exact duplicates remaining: {df.drop(columns=['uniq_id']).duplicated().sum()}")
    sub_banner("Remaining missing values in analysis columns")
    columns = ["company", "education", "experience", "industry", "jobdescription", "primary_city",
               "title_clean", "postdate", "exp_mid", "pay_mid", "numberofpositions", "functional_area"]
    remaining = pd.DataFrame({"missing": df[columns].isna().sum(),
                              "missing_pct": (df[columns].isna().mean() * 100).round(2)})
    print(remaining.to_string())
    sub_banner("Converted data types")
    print(df[["postdate", "exp_min", "exp_max", "exp_mid", "numberofpositions", "pay_min", "pay_max",
              "pay_mid", "desc_length"]].dtypes.to_string())
    sub_banner("Value validation")
    print(f"exp_min <= exp_max for all parsed rows : {bool((df['exp_min'].dropna() <= df['exp_max'].dropna()).all())}")
    print(f"pay_min <= pay_max for all parsed rows : {bool((df['pay_min'].dropna() <= df['pay_max'].dropna()).all())}")
    print(f"Post date range                        : {df['postdate'].min()} to {df['postdate'].max()}")
    print(f"Parsed experience coverage             : {df['exp_mid'].notna().mean():.2%}")
    print(f"Parsed numeric salary coverage         : {df['has_salary'].mean():.2%}")


def run_eda(df):
    banner("PHASE 5: EXPLORATORY DATA ANALYSIS")
    total = len(df)

    sub_banner("Top 10 industries")
    text_bars(df["industry"].value_counts())
    shares = df["industry"].value_counts(normalize=True)
    print(f"Distinct industries: {df['industry'].nunique()} | top-1 share: {shares.iloc[0]:.1%} | "
          f"top-3 share: {shares.iloc[:3].sum():.1%} | HHI: {(shares ** 2).sum():.3f}")

    sub_banner("Top 10 normalized job titles")
    text_bars(df["title_clean"].value_counts())
    print(f"Distinct raw titles: {df['jobtitle'].nunique():,} | distinct normalized titles: {df['title_clean'].nunique():,}")

    sub_banner("Top 10 primary locations")
    text_bars(df["primary_city"].value_counts())
    city_shares = df["primary_city"].value_counts(normalize=True)
    print(f"Top-5 cities share of listings with a location: {city_shares.iloc[:5].sum():.1%}")
    print(f"Listings naming more than one city: {(df['location_count'] > 1).mean():.1%}")

    sub_banner("Functional area (the dataset's 'skills' column)")
    text_bars(df["functional_area"].value_counts())

    sub_banner("Top 15 skills extracted from job descriptions (share of listings)")
    skill_share = df[[f"skill_{t}" for t in SKILL_TERMS]].mean().sort_values(ascending=False)
    skill_share.index = [name.replace("skill_", "") for name in skill_share.index]
    for name, share in skill_share.head(15).items():
        print(f"{name:<18} | {'#' * int(share * 100 / 2)} {share:.1%}")
    print(f"Listings with at least one extracted skill: {(df['n_skills_extracted'] > 0).mean():.1%}")

    sub_banner("Experience (minimum years required)")
    experience_counts = df["exp_category"].value_counts().reindex(df["exp_category"].cat.categories)
    text_bars(experience_counts)
    print(df["exp_mid"].describe().round(2).to_string())

    sub_banner("Undergraduate requirement (first option listed), top 8")
    text_bars(df["ug_requirement"].value_counts(), top=8)
    print(f"Listings with a PG option listed without a waiver: {df['pg_listed'].dropna().mean():.1%}")

    sub_banner("Salary information availability")
    text_bars(df["pay_category"].value_counts(), top=6)

    sub_banner("Annual pay range midpoint (only strict 'X - Y P.A' rows)")
    print(df["pay_mid"].describe().round(0).to_string())

    sub_banner("Number of positions (populated rows only)")
    print(df["numberofpositions"].describe().round(2).to_string())
    print(f"Populated: {df['numberofpositions'].notna().mean():.1%} of listings")

    sub_banner("Job description length (characters)")
    print(df["desc_length"].describe().round(0).to_string())

    sub_banner("Listings per posting month")
    monthly = df["posting_year_month"].value_counts().sort_index()
    for month, value in monthly.items():
        print(f"{month} | {'#' * int(40 * value / monthly.max())} {value:,}")
    print("Caution: coverage is uneven (some months have very few listings), so month-to-month changes")
    print("reflect how the sample was collected as much as hiring activity.")

    sub_banner("Industry x primary city (top 5 x top 5, counts)")
    top_industries = df["industry"].value_counts().head(5).index
    top_cities = df["primary_city"].value_counts().head(5).index
    subset = df[df["industry"].isin(top_industries) & df["primary_city"].isin(top_cities)]
    table = pd.crosstab(subset["industry"].str[:30], subset["primary_city"])
    print(table.to_string())

    sub_banner("Experience category x industry (row %, top 5 industries)")
    subset = df[df["industry"].isin(top_industries)]
    table = pd.crosstab(subset["industry"].str[:30], subset["exp_category"], normalize="index") * 100
    print(table.round(1).to_string())

    sub_banner("Median annual pay midpoint by functional area (n >= 30 valid pay rows)")
    grouped = df.dropna(subset=["pay_mid"]).groupby("functional_area")["pay_mid"].agg(["count", "median"])
    print(grouped[grouped["count"] >= 30].sort_values("median", ascending=False).round(0).head(10).to_string())

    sub_banner("Median pay midpoint by experience category (valid pay rows)")
    by_experience = df.dropna(subset=["pay_mid", "exp_category"]).groupby("exp_category")["pay_mid"].agg(["count", "median"])
    print(by_experience.round(0).to_string())

    sub_banner("Top 5 functional areas: most common extracted skills (share of listings in area)")
    for area in df["functional_area"].value_counts().head(5).index:
        area_rows = df[df["functional_area"] == area]
        shares = area_rows[[f"skill_{t}" for t in SKILL_TERMS]].mean().sort_values(ascending=False).head(5)
        label = ", ".join(f"{n.replace('skill_', '')} {v:.0%}" for n, v in shares.items())
        print(f"{area[:36]:<36} (n={len(area_rows):,}): {label}")


def run_statistics(df):
    banner("PHASE 6: STATISTICAL ANALYSIS (alpha = 0.05)")
    results = []

    sub_banner("Test 1: Spearman correlation, experience vs annual pay midpoint")
    pay_rows = df.dropna(subset=["pay_mid", "exp_mid"])
    rho, p_value = stats.spearmanr(pay_rows["exp_mid"], pay_rows["pay_mid"])
    print("Research question : Do listings requiring more experience advertise higher pay?")
    print("H0                : no monotonic association between experience and pay midpoint")
    print("H1                : a monotonic association exists")
    print(f"Why Spearman      : pay skewness = {pay_rows['pay_mid'].skew():.2f}, experience is a bounded range;")
    print("                    rank correlation does not assume normality or linearity")
    print(f"Sample size       : {len(pay_rows):,} listings with both values parsed")
    print(f"rho = {rho:.3f}, {p_text(p_value)}")
    robust = pay_rows[~pay_rows["pay_mid_outlier"]]
    robust_rho, robust_p = stats.spearmanr(robust["exp_mid"], robust["pay_mid"])
    print(f"Sensitivity       : excluding IQR pay outliers (n={len(robust):,}) rho = {robust_rho:.3f}, {p_text(robust_p)}")
    verdict = "reject H0" if p_value < ALPHA else "fail to reject H0"
    strength = "weak" if abs(rho) < 0.3 else "moderate" if abs(rho) < 0.6 else "strong"
    print(f"Decision          : {verdict}; association is {strength} and {'positive' if rho > 0 else 'negative'}.")
    print("Limitation        : only listings that disclosed a P.A range are included; association is not causation.")
    results.append(("Spearman experience vs pay", len(pay_rows), rho, p_value))

    sub_banner("Test 2: Mann-Whitney U, annual pay midpoint between two functional areas")
    area_counts = pay_rows["functional_area"].value_counts()
    eligible = area_counts[area_counts >= 30].index[:2]
    title_counts = pay_rows["title_clean"].value_counts()
    print(f"Role-level comparison feasibility: largest normalized title has {int(title_counts.iloc[0])} pay-valid listings;")
    print("job-title pairs are too small for a reliable test, so functional areas are compared instead.")
    if len(eligible) == 2:
        group_a = pay_rows.loc[pay_rows["functional_area"] == eligible[0], "pay_mid"]
        group_b = pay_rows.loc[pay_rows["functional_area"] == eligible[1], "pay_mid"]
        u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
        common_language = u_stat / (len(group_a) * len(group_b))
        print(f"Groups            : A = {eligible[0]} (n={len(group_a)}), B = {eligible[1]} (n={len(group_b)})")
        print("H0                : pay distributions of A and B are equal")
        print("H1                : pay distributions differ")
        print("Why Mann-Whitney  : independent groups, skewed pay, no normality assumption")
        print(f"Medians           : A = {group_a.median():,.0f}, B = {group_b.median():,.0f}")
        print(f"U = {u_stat:,.1f}, {p_text(p_value)}")
        print(f"Effect size       : P(A > B) + 0.5*P(tie) = {common_language:.3f}")
        print(f"Decision          : {'reject H0' if p_value < ALPHA else 'fail to reject H0 (no significant difference shown)'}")
        results.append((f"Mann-Whitney {eligible[0]} vs {eligible[1]}", len(group_a) + len(group_b), u_stat, p_value))
    else:
        print("Fewer than two groups with at least 30 pay-valid listings; test skipped.")

    sub_banner("Test 3: Chi-square test of independence, industry vs primary city (top 8 x top 8)")
    top_industries = df["industry"].value_counts().head(8).index
    top_cities = df["primary_city"].value_counts().head(8).index
    subset = df[df["industry"].isin(top_industries) & df["primary_city"].isin(top_cities)]
    contingency = pd.crosstab(subset["industry"], subset["primary_city"])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    cramers_v = np.sqrt(chi2 / (contingency.values.sum() * (min(contingency.shape) - 1)))
    print("Research question : Is the industry mix of listings the same across major cities?")
    print("H0                : industry and city are independent")
    print("H1                : industry and city are associated")
    print(f"Sample size       : {contingency.values.sum():,} listings (top 8 industries in top 8 cities only)")
    print(f"Table shape       : {contingency.shape[0]} x {contingency.shape[1]}")
    print(f"chi2 = {chi2:,.1f}, dof = {dof}, {p_text(p_value)}")
    print(f"Assumption check  : min expected count = {expected.min():.2f}; cells with expected < 5: "
          f"{(expected < 5).mean():.1%} (rule of thumb: at most 20%)")
    magnitude = "negligible" if cramers_v < 0.1 else "weak" if cramers_v < 0.3 else "moderate" if cramers_v < 0.5 else "strong"
    print(f"Cramer's V        : {cramers_v:.3f} ({magnitude} association)")
    print(f"Decision          : {'reject H0' if p_value < ALPHA else 'fail to reject H0'}")
    ratio = pd.DataFrame(contingency.values / expected, index=contingency.index, columns=contingency.columns)
    top_pairs = ratio.stack().sort_values(ascending=False).head(5)
    print("Most over-represented industry-city pairs (observed / expected):")
    for (industry, city), value in top_pairs.items():
        print(f"  {industry[:40]:<40} in {city:<12} {value:.2f}x  (observed {contingency.loc[industry, city]})")
    print("Limitation        : listings from one portal sample; association does not imply cause.")
    results.append(("Chi-square industry vs city", int(contingency.values.sum()), chi2, p_value))

    sub_banner("Statistical results summary")
    summary = pd.DataFrame(results, columns=["analysis", "n", "estimate", "p_value"])
    summary["significant_at_0.05"] = summary["p_value"] < ALPHA
    summary["p_value"] = summary["p_value"].map(lambda v: "< 1e-300" if v < 1e-300 else format_p(v))
    print(summary.round(3).to_string(index=False))


def business_insights(df):
    banner("PHASE 7: BUSINESS INSIGHTS AND RECOMMENDATIONS")
    industries = df["industry"].value_counts()
    cities = df["primary_city"].value_counts()
    titles = df["title_clean"].value_counts()
    areas = df["functional_area"].value_counts()
    skills = df[[f"skill_{t}" for t in SKILL_TERMS]].mean().sort_values(ascending=False)
    skill_names = [name.replace("skill_", "") for name in skills.index[:5]]
    experience = df["exp_category"].value_counts()
    ug = df["ug_requirement"].value_counts()
    total = len(df)
    print(f"1. Industry concentration : {industries.index[0]} accounts for {industries.iloc[0] / total:.1%} of listings.")
    print(f"2. Geography              : {cities.index[0]} leads with {cities.iloc[0] / cities.sum():.1%} of listings that name a city; "
          f"top 3 are {', '.join(cities.index[:3])}.")
    print(f"3. Roles                  : most frequent normalized title is '{titles.index[0]}' ({titles.iloc[0]} listings, "
          f"{titles.iloc[0] / total:.1%}); titles are highly fragmented.")
    print(f"4. Functional areas       : {areas.index[0]} ({areas.iloc[0] / areas.sum():.1%}), then {areas.index[1]} and {areas.index[2]}.")
    print(f"5. Skills in descriptions : {', '.join(skill_names)} appear most often (keyword extraction, vocabulary-limited).")
    print(f"6. Experience             : most common minimum-experience band is {experience.index[0]} ({experience.iloc[0] / experience.sum():.1%}).")
    print(f"7. Education              : most common first UG option is '{ug.index[0]}' ({ug.iloc[0] / ug.sum():.1%}).")
    print(f"8. Salary transparency    : only {df['has_salary'].mean():.1%} of listings give a comparable annual range; "
          f"{(df['pay_category'] == 'Not disclosed').mean():.1%} say Not disclosed.")

    sub_banner("Recommendations (each tied to an observed finding)")
    print(f"Job seekers  : Software roles dominate listings, so prioritize {', '.join(skill_names[:3])} and verify against")
    print(f"               live postings; look at {', '.join(cities.index[:3])} for the highest listing volume.")
    print("Institutions : Align electives and projects with the most frequent extracted skills and the dominant")
    print("               industry; teach salary-negotiation literacy given how rarely pay is disclosed.")
    print("Recruiters   : Listings are clustered in a few cities; non-disclosure of pay is the norm, so disclosed")
    print("               ranges can be a differentiator. Standardize job titles to improve searchability.")


def ethics_and_limitations():
    banner("PHASE 8: ETHICS, BIAS AND LIMITATIONS")
    notes = [
        "Scope        : Results describe this 22,000-listing sample from one portal, not the Indian labour market.",
        "Time         : Posting dates span roughly 2015-2017 with uneven monthly coverage; patterns may be outdated.",
        "Salary       : Only strict 'X - Y P.A' strings are parsed; currency is not documented in the data and is",
        "               treated as an annual figure per the P.A suffix. Disclosing listings may differ from the rest.",
        "Skills       : The 'skills' column holds one functional-area label per row; keyword extraction from",
        "               descriptions covers only a fixed vocabulary and may miss or over-count terms.",
        "Location     : Only the first listed city is used as primary; multi-city listings are under-represented.",
        "Bias         : Sampling, geographic and industry-representation bias (IT-Software heavy) and missing-data",
        "               bias (education 9% missing, positions ~80% missing) affect generalization.",
        "Privacy      : The dataset holds company and listing data, no applicant data; keep it for the stated",
        "               academic purpose, restrict access, and do not attempt to identify individuals.",
        "Fairness     : Do not infer protected characteristics from descriptions or use these patterns to rank people;",
        "               keep a human in the loop for any hiring decision and document methods for transparency.",
    ]
    print("\n".join(notes))


def main():
    parser = argparse.ArgumentParser(description="Naukri.com job market analysis (terminal output only)")
    parser.add_argument("--input", default="naukri_com-job_sample_copy.csv")
    parser.add_argument("--output", default="cleaned_dataset.csv")
    parser.add_argument("--no-save", action="store_true")
    arguments = parser.parse_args()

    input_path = Path(arguments.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    raw = load_raw(input_path)
    df = clean_data(raw)
    df = engineer_features(df)
    banner("OUTLIER ANALYSIS")
    report_outliers(df)
    validate(raw, df)
    run_eda(df)
    run_statistics(df)
    business_insights(df)
    ethics_and_limitations()

    if not arguments.no_save:
        df.to_csv(arguments.output, index=False)
        print(f"\nCleaned dataset with engineered features saved to {arguments.output} ({df.shape[0]:,} rows, {df.shape[1]} columns)")
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
