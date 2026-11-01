import argparse
import re
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)

ALPHA = 0.05
RANDOM_SEED = 42
MIN_GROUP_SIZE = 30
MIN_CORRELATION_SIZE = 10
MAX_ANNUAL_PAY = 1e8

EXPECTED_COLUMNS = [
    "jobid", "company", "education", "experience", "industry", "jobdescription",
    "joblocation_address", "jobtitle", "numberofpositions", "payrate", "postdate", "skills", "site_name",
]
TEXT_COLUMNS = [
    "company", "education", "experience", "industry", "jobdescription", "joblocation_address",
    "jobtitle", "payrate", "skills", "site_name",
]
NULL_LIKE = {"", "na", "n/a", "null", "none", "nan", "-", "not specified", "not available", "not mentioned"}

CITY_ALIASES = {
    "bangalore": "Bengaluru", "bengaluru": "Bengaluru", "secunderabad": "Hyderabad", "gurgaon": "Gurugram",
    "bombay": "Mumbai", "madras": "Chennai", "calcutta": "Kolkata", "cochin": "Kochi", "ernakulam": "Kochi",
    "vizag": "Visakhapatnam", "baroda": "Vadodara", "trivandrum": "Thiruvananthapuram",
    "bhubaneshwar": "Bhubaneswar", "ncr": "Delhi NCR", "delhi ncr": "Delhi NCR", "new delhi": "Delhi",
}
NON_CITY_TOKENS = {
    "india", "anywhere in india", "all india", "karnataka", "maharashtra", "tamilnadu", "tamil nadu",
    "kerala", "gujarat", "andhra pradesh", "telangana", "uttar pradesh", "west bengal", "rajasthan",
    "punjab", "haryana", "madhya pradesh", "bihar", "odisha", "orissa", "jharkhand", "uttarakhand",
    "assam", "chhattisgarh", "himachal pradesh", "jammu and kashmir", "goa state",
}

ROLE_RULES = [
    ("Teaching / Academic", r"professor|lecturer|teacher|faculty|tutor|principal|\bpgt\b|\btgt\b"),
    ("QA / Testing", r"tester|testing|\bqa\b|quality assurance|test engineer|selenium|\bsdet\b"),
    ("Data / Analytics", r"data scientist|data analyst|analytics|business intelligence|data engineer|machine learning|hadoop|big data|\bbi\b"),
    ("Customer Support / BPO", r"customer (?:care|support|service)|\bbpo\b|call cent|voice process|process executive|\bcce\b|technical support|helpdesk|help desk|chat support|\bcsa\b"),
    ("Software Development", r"developer|programmer|software|java|\.net|php|android|\bios\b|python|full[\s-]?stack|front[\s-]?end|back[\s-]?end|architect|devops|\bsap\b|oracle|technical lead|tech lead|engineer.*(?:application|system)"),
    ("Project / Program Management", r"project manager|program manager|programme manager|delivery manager|scrum|project lead"),
    ("Sales / Business Development", r"sales|business development|\bbde\b|relationship manager|account manager|territory|field executive|area manager"),
    ("Marketing", r"marketing|\bseo\b|digital|brand|social media|content writer|copywriter|\bsem\b"),
    ("HR / Recruitment", r"recruiter|recruitment|\bhr\b|human resource|talent|payroll"),
    ("Finance / Accounts", r"account(?:s|ant)|finance|financial|audit|taxation|\btax\b|\bca\b|banking|loan|credit|insurance|treasury"),
    ("Design / Creative", r"designer|graphic|ui/?ux|animator|video editor|creative"),
    ("Admin / Data Entry", r"admin|executive assistant|secretary|receptionist|data entry|back office|office assistant"),
    ("Medical / Healthcare", r"doctor|nurse|physician|surgeon|pharmac|clinical|medical|dentist|therapist|paramedic|\bmbbs\b"),
    ("Operations / Engineering", r"production|operations|mechanical|electrical|civil|maintenance|plant|manufactur|supply chain|logistics|quality|procurement|engineer"),
]
ROLE_PATTERNS = [(name, re.compile(pattern)) for name, pattern in ROLE_RULES]

GENERAL_SKILLS = ["Communication", "English", "Excel", "MS Office"]
TECHNICAL_SKILLS = [
    "Python", "Java", "JavaScript", "SQL", "MySQL", "Oracle", "PHP", ".NET", "ASP.NET", "C++", "C#", "HTML",
    "CSS", "jQuery", "Angular", "React", "Node.js", "Android", "iOS", "AWS", "Linux", "Selenium",
    "Manual Testing", "Hadoop", "Machine Learning", "SEO", "Tally", "SAP", "Salesforce", "Photoshop",
    "AutoCAD", "Hibernate", "WordPress",
]
SKILL_TERMS = TECHNICAL_SKILLS + GENERAL_SKILLS
SKILL_OVERRIDES = {
    "Java": r"java(?!\s*script)",
    "JavaScript": r"java\s?script",
    "Communication": r"communication\s+skills?|(?:verbal|written|oral|excellent|good|strong|effective)\s+communication",
    "English": r"(?:fluen\w*|command|proficien\w*|spoken|written|verbal|excellent|good|strong)\s+(?:in\s+|of\s+|over\s+)?english|english\s+(?:communication|speaking|language|proficiency|typing|skills?)",
    "Excel": r"(?:ms|microsoft|advanced)[\s-]+excel|excel\s+(?:skills?|sheets?|spreadsheets?|formulas?|macros?|knowledge|proficien\w*)|vlookup|pivot\s+tables?",
    "MS Office": r"ms[\s-]?office|microsoft\s+office",
    "React": r"react\.?js|reactjs|react\s+native|react\s+(?:developer|framework|library)",
    "Angular": r"angular(?:\.?js)?",
    "HTML": r"html5?",
    "CSS": r"css3?",
    "Node.js": r"node\.?js",
    "Manual Testing": r"manual\s+testing",
    "Machine Learning": r"machine\s+learning",
}


def boundary_pattern(term):
    core = SKILL_OVERRIDES.get(term, re.escape(term))
    return re.compile(rf"(?<![A-Za-z0-9])(?:{core})(?![A-Za-z0-9])", re.IGNORECASE)


SKILL_PATTERNS = {term: boundary_pattern(term) for term in SKILL_TERMS}
URL_PATTERN = r"https?://\S+|www\.\S+"

AMOUNT = r"(\d[\d,]*(?:\.\d+)?)"
UNIT = r"(lacs?|lakhs?)?"
ANNUAL_TOKEN = r"(p\.?\s?a|per\s+annum|per\s+year|annually|lpa)(?![a-z])"
MONTHLY_TOKEN = r"(p\.?\s?m|per\s+month|monthly)(?![a-z])"
PREFIX = r"^(?:rs\.?\s*|inr\s*)?"
PAY_RANGE_PATTERN = re.compile(rf"{PREFIX}{AMOUNT}\s*{UNIT}\s*(?:-|to)\s*{AMOUNT}\s*{UNIT}\s*(?:{ANNUAL_TOKEN}|{MONTHLY_TOKEN})", re.IGNORECASE)
PAY_SINGLE_PATTERN = re.compile(rf"{PREFIX}{AMOUNT}\s*{UNIT}\s*(?:{ANNUAL_TOKEN}|{MONTHLY_TOKEN})", re.IGNORECASE)


def banner(title):
    print("\n" + "-" * 100)
    print(title)


def sub_banner(title):
    print("\n--- " + title + " ---")


def wrap_print(text, indent=""):
    print(textwrap.fill(text, width=108, initial_indent=indent, subsequent_indent=indent + "  "))


def format_p(p_value):
    if p_value < 1e-300:
        return "< 1e-300"
    if p_value < 0.001:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def p_text(p_value):
    return f"p {format_p(p_value)}" if p_value < 1e-300 else f"p = {format_p(p_value)}"


def run_section(label, function, *args):
    try:
        return function(*args)
    except Exception as error:
        print(f"\n[Section skipped: {label}] {type(error).__name__}: {error}")
        return None


def text_bars(counts, width=40, top=10):
    counts = counts.head(top)
    if counts.empty:
        print("  (no data available)")
        return
    label_width = min(max(len(str(label)) for label in counts.index), 48)
    for label, value in counts.items():
        print(f"{str(label)[:48]:<{label_width}} | {value:,}")


def share_bars(shares, top=15):
    shares = shares.head(top)
    for name, share in shares.items():
        print(f"{name:<18} | {share:.1%}")
        
def effect_label(value, thresholds=(0.1, 0.3, 0.5)):
    magnitude = abs(value)
    if magnitude < thresholds[0]:
        return "negligible"
    if magnitude < thresholds[1]:
        return "small/weak"
    if magnitude < thresholds[2]:
        return "moderate"
    return "large/strong"


def clean_text(series):
    cleaned = series.astype("object").where(series.notna(), None)
    cleaned = cleaned.map(lambda v: re.sub(r"\s+", " ", v).strip().strip('"') if isinstance(v, str) else v)
    is_null_like = cleaned.map(lambda v: isinstance(v, str) and v.lower() in NULL_LIKE)
    return cleaned.mask(is_null_like, np.nan).astype("object")


def split_cities(value):
    if not isinstance(value, str):
        return []
    cities = []
    for token in re.split(r"[,/;]", value):
        name = re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", token)).strip().lower()
        name = re.sub(r"\s+other$", "", name)
        if not name or name in NON_CITY_TOKENS:
            continue
        name = CITY_ALIASES.get(name, name.title())
        if name not in cities:
            cities.append(name)
    if "Delhi NCR" in cities and "Delhi" in cities:
        cities.remove("Delhi NCR")
    return cities


def clean_title(value):
    if not isinstance(value, str):
        return np.nan
    title = re.sub(r"\(.*?\)|\[.*?\]", " ", value.lower())
    title = re.sub(r"\b(?:walk[\s-]?in(?:\s+interview)?|urgent(?:ly)?|hiring(?:\s+for)?|openings?|requirements?|required|wanted|looking for|vacanc(?:y|ies)|immediate)\b", " ", title)
    for segment in title.split(" - "):
        segment = re.sub(r"\d+", " ", segment)
        segment = re.sub(r"[^a-z/&+#. ]", " ", segment)
        segment = re.sub(r"\s+", " ", segment).strip(" ./&")
        if len(re.sub(r"[^a-z]", "", segment)) >= 3:
            return segment.title()
    return np.nan


def classify_role(value):
    if not isinstance(value, str):
        return np.nan
    lowered = value.lower()
    for name, pattern in ROLE_PATTERNS:
        if pattern.search(lowered):
            return name
    return "Other"


def parse_experience(series):
    text = series.fillna("").astype(str).str.lower()
    ranges = text.str.extract(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)").astype(float)
    singles = text.str.extract(r"^\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:yrs?|years?)").astype(float)[0]
    fresher = text.str.contains(r"\bfresher", regex=True)
    exp_min = ranges[0].fillna(singles).mask(fresher & ranges[0].isna() & singles.isna(), 0.0)
    exp_max = ranges[1].fillna(singles).mask(fresher & ranges[1].isna() & singles.isna(), 0.0)
    invalid = exp_max < exp_min
    exp_min, exp_max = exp_min.mask(invalid), exp_max.mask(invalid)
    exp_format = np.select(
        [series.isna(), ranges[0].notna() & ~invalid, singles.notna(), fresher, invalid],
        ["missing", "range", "single value", "fresher", "invalid range"], default="unparsed")
    return exp_min, exp_max, pd.Series(exp_format, index=series.index)


def to_amount(number, unit, in_lakhs):
    value = float(number.replace(",", ""))
    return value * 1e5 if (unit and unit.lower().startswith(("lac", "lakh"))) or in_lakhs else value


def parse_pay(raw):
    empty = (np.nan, np.nan, np.nan, np.nan)
    if not isinstance(raw, str):
        return ("Missing",) + empty[:3] + (np.nan,)
    lowered = raw.lower().strip()
    if "not disclosed" in lowered:
        return ("Not disclosed",) + empty[:3] + (np.nan,)
    if re.search(r"pay band|pay scale|scale of pay|grade pay|\bagp\b", lowered):
        return ("Government pay band / scale",) + empty[:3] + (np.nan,)
    if re.search(r"^less than|&\s*above|and above", lowered):
        return ("Open-ended band (excluded)",) + empty[:3] + (np.nan,)
    for pattern, kind in ((PAY_RANGE_PATTERN, "range"), (PAY_SINGLE_PATTERN, "single")):
        match = pattern.match(lowered)
        if not match:
            continue
        groups = match.groups()
        if kind == "range":
            first, unit_a, second, unit_b, annual_token, monthly_token = groups[:6]
            unit = unit_b or unit_a
            in_lakhs = bool(annual_token and annual_token.lower() == "lpa")
            low, high = to_amount(first, unit, in_lakhs), to_amount(second, unit, in_lakhs)
        else:
            first, unit_a, annual_token, monthly_token = groups[:4]
            in_lakhs = bool(annual_token and annual_token.lower() == "lpa")
            low = high = to_amount(first, unit_a, in_lakhs)
        period = "monthly" if monthly_token else "annual"
        factor = 12 if period == "monthly" else 1
        low, high = low * factor, high * factor
        if low <= 0 or high < low or high > MAX_ANNUAL_PAY:
            return ("Unrealistic numeric value (excluded)", np.nan, np.nan, np.nan, period)
        label = f"{period.title()} {kind}"
        return (label, low, high, (low + high) / 2, period)
    if re.search(r"\d", lowered):
        return ("Other numeric format (not comparable)",) + empty[:3] + (np.nan,)
    return ("Non-numeric text",) + empty[:3] + (np.nan,)


def ug_label(value):
    if not isinstance(value, str):
        return np.nan
    match = re.search(r"UG:\s*(.*?)(?:\s+PG:|\s+Doctorate:|$)", value)
    if match:
        return match.group(1).split(" - ")[0].split(",")[0].strip() or "UG listed (blank)"
    if re.search(r"PG:|Doctorate:", value):
        return "No UG listed (PG/Doctorate only)"
    return "Unrecognised format"


def segment_listed(value, label, next_labels):
    if not isinstance(value, str):
        return np.nan
    stop = "|".join(rf"\s+{n}:" for n in next_labels)
    match = re.search(rf"{label}:\s*(.*?)(?:{stop}|$)" if stop else rf"{label}:\s*(.*)$", value)
    if not match:
        return False
    return "not required" not in match.group(1).lower()


def iqr_report(series, name):
    clean = series.dropna()
    if clean.empty:
        return {"column": name, "n": 0}, pd.Series(False, index=series.index)
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    spread = q3 - q1
    lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
    flagged = ((series < lower) | (series > upper)).fillna(False)
    return {
        "column": name, "n": len(clean), "q1": q1, "median": clean.median(), "q3": q3,
        "lower_fence": lower, "upper_fence": upper, "outliers": int(flagged.sum()),
        "outlier_pct": round(100 * flagged.sum() / len(clean), 2), "skewness": round(clean.skew(), 2),
    }, flagged


def find_input(explicit):
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"Input file not found: {path}")
        return path
    candidates = sorted(p for folder in (Path("."), Path("data")) if folder.exists()
                        for p in folder.glob("*.csv") if "naukri" in p.name.lower())
    if not candidates:
        raise SystemExit("No Naukri CSV found in the current folder or ./data. Pass the file with --input PATH.")
    return candidates[0]


def read_csv_safely(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def load_raw(path, reports):
    banner("PHASE 1: DATA INSPECTION")
    raw = read_csv_safely(path)
    print(f"Source file : {path}")
    print(f"Records     : {raw.shape[0]:,}")
    print(f"Attributes  : {raw.shape[1]}")
    absent = [column for column in EXPECTED_COLUMNS if column not in raw.columns]
    extra = [column for column in raw.columns if column not in EXPECTED_COLUMNS]
    if absent:
        print(f"WARNING: expected columns absent from file, filled with missing values: {absent}")
        for column in absent:
            raw[column] = pd.Series(np.nan, index=raw.index, dtype="object")
    if extra:
        print(f"Additional columns present (kept, not analysed): {extra}")
    sub_banner("Column names and data types")
    print(raw.dtypes.to_string())
    sub_banner("Sample records (5)")
    sample_columns = [c for c in ["company", "experience", "industry", "joblocation_address", "jobtitle", "payrate", "postdate"] if c in raw.columns]
    print(raw[sample_columns].head(5).to_string())
    sub_banner("Missing values")
    missing = pd.DataFrame({
        "missing": raw.isna().sum(), "missing_pct": (raw.isna().mean() * 100).round(2), "unique": raw.nunique(),
    }).sort_values("missing_pct", ascending=False)
    print(missing.to_string())
    reports["missing_value_report"] = missing.reset_index().rename(columns={"index": "column"})
    sub_banner("Duplicate checks (original data)")
    identifier = [c for c in ["uniq_id"] if c in raw.columns]
    exact = int(raw.duplicated().sum())
    content = int(raw.drop(columns=identifier).duplicated().sum())
    repeated_ids = int(raw["jobid"].dropna().duplicated().sum())
    print(f"Exact duplicate rows                 : {exact}")
    print(f"Duplicate rows ignoring identifiers  : {content}")
    print(f"Excess rows from repeated jobid      : {repeated_ids}")
    reports["duplicate_report"] = pd.DataFrame([
        {"check": "exact duplicate rows", "count": exact},
        {"check": "duplicate rows ignoring uniq_id", "count": content},
        {"check": "excess rows from repeated jobid", "count": repeated_ids},
    ])
    return raw.drop_duplicates()


def examine_repeated_jobids(df):
    sub_banner("Repeated jobid examination (kept and flagged, not removed)")
    ids = df["jobid"].dropna()
    repeated = df[df["jobid"].notna() & df["jobid"].duplicated(keep=False)]
    if repeated.empty:
        print("No repeated jobids.")
        return
    key_fields = [c for c in ["company", "jobtitle", "joblocation_address", "postdate"] if c in df.columns]
    same_key = repeated.groupby("jobid")[key_fields].nunique(dropna=False).max(axis=1).eq(1)
    print(f"jobids repeated                      : {repeated['jobid'].nunique()} ({len(repeated)} rows)")
    print(f"Groups identical on {key_fields}: {int(same_key.sum())}")
    print(f"Groups differing on at least one key : {int((~same_key).sum())}")
    print("A repeated jobid is not proof of a duplicate listing: the same id can be re-posted with a different")
    print("date, location or company. Such rows stay in the data and carry jobid_repeated = True.")


def clean_data(raw):
    banner("PHASE 2: CLEANING AND PREPROCESSING")
    df = raw.copy()
    for column in TEXT_COLUMNS:
        df[column] = clean_text(df[column])
    print("Whitespace collapsed, wrapping quotes removed, null-like strings (NA, NULL, None, Not Specified,")
    print("Not Mentioned, '-', empty) converted to missing.")

    before = len(df)
    content_columns = [c for c in df.columns if c != "uniq_id"]
    df = df.drop_duplicates(subset=content_columns).reset_index(drop=True)
    print(f"Duplicates created by cleaning and removed: {before - len(df)}")
    examine_repeated_jobids(df)

    df["jobdescription"] = df["jobdescription"].str.replace(
        r"^Job Description\s*Send me Jobs like this\s*-?\s*", "", regex=True)
    df["functional_area"] = df["skills"]
    city_lists = df["joblocation_address"].map(split_cities)
    df["city_list"] = city_lists
    df["primary_city"] = city_lists.map(lambda cities: cities[0] if cities else np.nan)
    df["location_count"] = city_lists.map(len).where(city_lists.map(len) > 0)
    df["cities_listed"] = city_lists.map("|".join)
    df["title_clean"] = df["jobtitle"].map(clean_title)
    df["role_family"] = df["jobtitle"].map(classify_role)

    df["postdate"] = pd.to_datetime(df["postdate"], errors="coerce", utc=True).dt.tz_localize(None)
    df["exp_min"], df["exp_max"], df["exp_format"] = parse_experience(df["experience"])
    df["exp_mid"] = (df["exp_min"] + df["exp_max"]) / 2

    df["numberofpositions"] = pd.to_numeric(df["numberofpositions"], errors="coerce")
    df.loc[df["numberofpositions"] <= 0, "numberofpositions"] = np.nan

    parsed_pay = pd.DataFrame(
        [parse_pay(value) for value in df["payrate"]],
        columns=["pay_category", "pay_min_annual", "pay_max_annual", "pay_mid_annual", "pay_source_period"],
        index=df.index)
    df = pd.concat([df, parsed_pay], axis=1)
    df["has_salary"] = df["pay_mid_annual"].notna()

    df["ug_requirement"] = df["education"].map(ug_label)
    df["pg_listed"] = df["education"].map(lambda v: segment_listed(v, "PG", ["Doctorate"]))
    df["doctorate_listed"] = df["education"].map(lambda v: segment_listed(v, "Doctorate", []))
    return df


def engineer_features(df):
    banner("PHASE 3: FEATURE ENGINEERING")
    df["posting_year"] = df["postdate"].dt.year
    df["posting_month"] = df["postdate"].dt.month
    df["posting_quarter"] = df["postdate"].dt.to_period("Q").astype("string")
    df["posting_year_month"] = df["postdate"].dt.to_period("M").astype("string")
    df["exp_category"] = pd.cut(
        df["exp_min"], bins=[-1, 0, 2, 5, 10, 1000],
        labels=["0 yrs minimum", "1-2 yrs minimum", "3-5 yrs minimum", "6-10 yrs minimum", "10+ yrs minimum"])
    df["desc_length"] = df["jobdescription"].str.len()
    df["jobid_repeated"] = df["jobid"].notna() & df["jobid"].duplicated(keep=False)

    description = df["jobdescription"].fillna("").str.replace(URL_PATTERN, " ", regex=True)
    df["description_text"] = description
    for term, pattern in SKILL_PATTERNS.items():
        df[f"skill_{term}"] = description.map(lambda text, p=pattern: bool(p.search(text)))
    df["n_technical_skills"] = df[[f"skill_{t}" for t in TECHNICAL_SKILLS]].sum(axis=1)
    df["n_skills_extracted"] = df[[f"skill_{t}" for t in SKILL_TERMS]].sum(axis=1)

    features = pd.DataFrame([
        ("posting_year / month / quarter / year_month", "postdate", "Time-trend analysis"),
        ("exp_min, exp_max, exp_mid, exp_format, exp_category", "experience", "Range, single value or fresher parsed; category uses minimum years"),
        ("city_list, primary_city, location_count", "joblocation_address", "Cities de-aliased, states removed, duplicates merged; first city is primary"),
        ("title_clean, role_family", "jobtitle", "Text before ' - ' with noise words removed; keyword-rule role grouping"),
        ("ug_requirement, pg_listed, doctorate_listed", "education", "First UG option; PG/Doctorate listed without a 'Not Required' option"),
        ("pay_category, pay_*_annual, pay_source_period", "payrate", "Annual or monthly ranges and single values; monthly x 12; lakh units expanded"),
        ("desc_length", "jobdescription", "Characters after boilerplate removal"),
        ("skill_* flags, n_technical_skills, n_skills_extracted", "jobdescription", "Context-aware keyword match over a fixed vocabulary, URLs removed"),
        ("jobid_repeated", "jobid", "Flags ids that appear more than once"),
    ], columns=["feature", "source", "how_and_why"])
    print(features.to_string(index=False))
    print("\nNo target variable is modelled, so leakage is not a risk; every feature uses only that listing's own fields.")
    return df


def report_outliers(df, reports):
    sub_banner("Outlier detection (IQR method; flagged, not deleted)")
    rows = []
    for column, label in [("pay_mid_annual", "pay_mid_annual"), ("numberofpositions", "numberofpositions"),
                          ("exp_mid", "exp_mid (years)"), ("desc_length", "desc_length (characters)")]:
        row, flagged = iqr_report(df[column], label)
        df[f"{column}_outlier"] = flagged
        rows.append(row)
    table = pd.DataFrame(rows)
    print(table.round(2).to_string(index=False))
    reports["outlier_report"] = table
    print("\nTreatment: these variables are right-skewed and extreme values are plausible (senior roles, bulk hiring,")
    print("long descriptions). They are flagged and retained; rank-based tests limit their influence and the")
    print("correlation is re-run without pay outliers as a sensitivity check.")


def parsing_report(df):
    sub_banner("Parsing success by field")
    rows = [
        ("experience", df["experience"].notna().sum(), df["exp_mid"].notna().sum()),
        ("payrate -> numeric annual", df["payrate"].notna().sum(), df["has_salary"].sum()),
        ("joblocation_address -> city", df["joblocation_address"].notna().sum(), df["primary_city"].notna().sum()),
        ("education -> UG label", df["education"].notna().sum(), (~df["ug_requirement"].isin(["Unrecognised format"]) & df["ug_requirement"].notna()).sum()),
        ("jobtitle -> role family (not Other)", df["jobtitle"].notna().sum(), (df["role_family"].notna() & (df["role_family"] != "Other")).sum()),
    ]
    table = pd.DataFrame(rows, columns=["field", "non_missing", "parsed"])
    table["parsed_pct"] = (100 * table["parsed"] / table["non_missing"].replace(0, np.nan)).round(1)
    print(table.to_string(index=False))
    sub_banner("Experience values that could not be parsed")
    print(df.loc[df["exp_format"].isin(["unparsed", "invalid range"]), "experience"].value_counts().head(5).to_string() or "none")
    sub_banner("Salary text by category")
    print(df["pay_category"].value_counts().to_string())
    sub_banner("Most common unparsed numeric salary texts (not comparable)")
    unparsed = df.loc[df["pay_category"].str.startswith(("Other numeric", "Open-ended", "Unrealistic")), "payrate"]
    print(unparsed.str[:70].value_counts().head(5).to_string() if len(unparsed) else "none")


def validation_samples(df):
    banner("MANUAL VALIDATION SAMPLES ")
    sample = lambda frame, n: frame.sample(min(n, len(frame)), random_state=RANDOM_SEED)
    sub_banner("Experience")
    print(sample(df[df["experience"].notna()], 6)[["experience", "exp_min", "exp_max", "exp_format"]].to_string(index=False))
    sub_banner("Education")
    shown = sample(df[df["education"].notna()], 5)
    shown = shown.assign(education=shown["education"].str[:70])
    print(shown[["education", "ug_requirement", "pg_listed", "doctorate_listed"]].to_string(index=False))
    sub_banner("Job title normalisation and role family")
    shown = sample(df[df["jobtitle"].notna()], 10)
    shown = shown.assign(jobtitle=shown["jobtitle"].str[:60])
    print(shown[["jobtitle", "title_clean", "role_family"]].to_string(index=False))
    sub_banner("Skill extraction: matched context for the 4 most frequent technical skills")
    top_skills = df[[f"skill_{t}" for t in TECHNICAL_SKILLS]].mean().sort_values(ascending=False).head(4).index
    for column in top_skills:
        term = column.replace("skill_", "")
        hits = sample(df[df[column]], 2)
        for text in hits["description_text"]:
            match = SKILL_PATTERNS[term].search(text)
            print(f"{term:<10}: ...{text[max(0, match.start() - 45):match.end() + 35].replace(chr(10), ' ')}...")


def validate(raw, df):
    banner("PHASE 4: VALIDATION")
    original_columns = [c for c in raw.columns if c != "uniq_id"]
    print(f"Original shape : {raw.shape}")
    print(f"Cleaned shape  : {df.shape}  (records kept; engineered columns added)")
    print(f"Exact duplicates remaining (content columns): {int(df[original_columns].duplicated().sum())}")
    sub_banner("Remaining missing values in analysis columns")
    columns = ["company", "education", "experience", "industry", "jobdescription", "primary_city", "title_clean",
               "role_family", "postdate", "exp_mid", "pay_mid_annual", "numberofpositions", "functional_area"]
    print(pd.DataFrame({"missing": df[columns].isna().sum(), "missing_pct": (df[columns].isna().mean() * 100).round(2)}).to_string())
    sub_banner("Converted data types")
    print(df[["postdate", "exp_min", "exp_max", "exp_mid", "numberofpositions", "pay_min_annual", "pay_max_annual",
              "pay_mid_annual", "desc_length"]].dtypes.to_string())
    sub_banner("Value validation")
    print(f"exp_min <= exp_max everywhere parsed : {bool((df['exp_min'].dropna() <= df['exp_max'].dropna()).all())}")
    print(f"pay_min <= pay_max everywhere parsed : {bool((df['pay_min_annual'].dropna() <= df['pay_max_annual'].dropna()).all())}")
    print(f"Post date range                      : {df['postdate'].min()} to {df['postdate'].max()}")
    print(f"Experience parse coverage            : {df['exp_mid'].notna().mean():.2%}")
    print(f"Numeric salary coverage              : {df['has_salary'].mean():.2%}")


def eda_distributions(df):
    sub_banner("Top 10 industries")
    industries = df["industry"].value_counts()
    text_bars(industries)
    if len(industries):
        shares = industries / industries.sum()
        print(f"Distinct industries: {len(industries)} | top-1 share: {shares.iloc[0]:.1%} | top-3 share: {shares.iloc[:3].sum():.1%} | HHI: {(shares ** 2).sum():.3f}")

    sub_banner("Top 10 job titles (text before ' - ', noise words removed)")
    text_bars(df["title_clean"].value_counts())
    print(f"Distinct raw titles: {df['jobtitle'].nunique():,} | distinct cleaned titles: {df['title_clean'].nunique():,}")

    sub_banner("Role family (keyword rules, approximate)")
    text_bars(df["role_family"].value_counts(), top=14)

    sub_banner("Top 10 primary cities (first listed)")
    text_bars(df["primary_city"].value_counts())
    sub_banner("Top 10 cities by listings mentioning them (any position in the list)")
    mentions = df["city_list"].explode().dropna().value_counts()
    text_bars(mentions)
    located = df["primary_city"].notna().sum()
    if located:
        print(f"Listings with a city: {located:,} | listing more than one city: {(df['location_count'] > 1).sum() / located:.1%}")

    sub_banner("Functional area (the dataset's 'skills' column)")
    text_bars(df["functional_area"].value_counts())


def eda_skills(df):
    sub_banner("Top 15 skills (share of listings, extracted from job descriptions)")
    shares = df[[f"skill_{t}" for t in SKILL_TERMS]].mean().sort_values(ascending=False)
    shares.index = [name.replace("skill_", "") for name in shares.index]
    share_bars(shares)
    sub_banner("Top 10 technical skills only")
    share_bars(shares[[s for s in shares.index if s in TECHNICAL_SKILLS]], top=10)
    print(f"Listings with at least one extracted skill: {(df['n_skills_extracted'] > 0).mean():.1%}")

    sub_banner("Most common technical skills within the 5 largest role families (share of that family's listings)")
    families = df.loc[df["role_family"] != "Other", "role_family"].value_counts().head(5).index
    for family in families:
        rows = df[df["role_family"] == family]
        top = rows[[f"skill_{t}" for t in TECHNICAL_SKILLS]].mean().sort_values(ascending=False).head(5)
        label = ", ".join(f"{n.replace('skill_', '')} {v:.0%}" for n, v in top.items() if v > 0) or "none matched"
        print(f"{family:<30} (n={len(rows):>5,}): {label}")


def eda_experience_education(df):
    sub_banner("Experience (minimum years required)")
    text_bars(df["exp_category"].value_counts().reindex(df["exp_category"].cat.categories))
    print(df["exp_mid"].describe().round(2).to_string())

    sub_banner("Undergraduate requirement (first option listed), top 8")
    text_bars(df["ug_requirement"].value_counts(), top=8)
    pg = df["pg_listed"].dropna()
    if len(pg):
        print(f"Listings with a PG option and no 'Not Required' option: {pg.mean():.1%}")

    top_industries = df["industry"].value_counts().head(5).index
    subset = df[df["industry"].isin(top_industries)]
    sub_banner("Experience band by industry (row %, top 5 industries)")
    print((pd.crosstab(subset["industry"].str[:30], subset["exp_category"], normalize="index") * 100).round(1).to_string())
    sub_banner("Top 5 UG requirements by industry (row %, top 5 industries)")
    top_ug = df["ug_requirement"].value_counts().head(5).index
    ug_subset = subset[subset["ug_requirement"].isin(top_ug)]
    print((pd.crosstab(ug_subset["industry"].str[:30], ug_subset["ug_requirement"], normalize="index") * 100).round(1).to_string())
    sub_banner("Median minimum experience by role family")
    print(df.groupby("role_family")["exp_mid"].agg(["count", "median"]).sort_values("count", ascending=False).round(1).head(10).to_string())


def eda_salary(df):
    sub_banner("Salary information availability")
    text_bars(df["pay_category"].value_counts(), top=10)
    sub_banner("Annualised pay midpoint (numeric rows only; currency not stated in the source)")
    print(df["pay_mid_annual"].describe().round(0).to_string())
    sub_banner("Median annualised pay midpoint by experience band (n shown)")
    print(df.dropna(subset=["pay_mid_annual"]).groupby("exp_category")["pay_mid_annual"].agg(["count", "median"]).round(0).to_string())
    sub_banner("Median annualised pay midpoint by role family (n >= 30 only)")
    grouped = df.dropna(subset=["pay_mid_annual"]).groupby("role_family")["pay_mid_annual"].agg(["count", "median"])
    print(grouped[grouped["count"] >= MIN_GROUP_SIZE].sort_values("median", ascending=False).round(0).to_string())
    sub_banner("Number of positions (populated rows only)")
    print(df["numberofpositions"].describe().round(2).to_string())
    print(f"Populated: {df['numberofpositions'].notna().mean():.1%} of listings")
    sub_banner("Job description length (characters)")
    print(df["desc_length"].describe().round(0).to_string())


def eda_time(df):
    sub_banner("Listings per posting month")
    monthly = df["posting_year_month"].value_counts().sort_index()
    if monthly.empty:
        print("  (no valid posting dates)")
        return
    print(monthly.rename_axis("month").to_frame("job_listings").to_string())
    span = (df["postdate"].max().to_period("M") - df["postdate"].min().to_period("M")).n + 1
    sparse = int((monthly < 0.01 * monthly.sum()).sum())
    print(f"Months with data: {len(monthly)} of {span} in the date span | months holding < 1% of dated listings: {sparse}")
    print(f"Top 3 months hold {monthly.nlargest(3).sum() / monthly.sum():.1%} of dated listings.")
    print("Coverage is uneven, so month-to-month changes mostly reflect how the sample was collected, not hiring trends.")


def eda_bivariate(df):
    sub_banner("Industry x primary city (top 5 x top 5, counts)")
    industries = df["industry"].value_counts().head(5).index
    cities = df["primary_city"].value_counts().head(5).index
    subset = df[df["industry"].isin(industries) & df["primary_city"].isin(cities)]
    print(pd.crosstab(subset["industry"].str[:30], subset["primary_city"]).to_string())
    sub_banner("Industry concentration inside each of the top 5 cities (top industry share)")
    for city in cities:
        city_rows = df[df["primary_city"] == city]["industry"].value_counts(normalize=True)
        print(f"{city:<12} {city_rows.index[0][:38]:<38} {city_rows.iloc[0]:.1%}")


def run_eda(df):
    banner("PHASE 5: EXPLORATORY DATA ANALYSIS")
    for label, function in [("distributions", eda_distributions), ("skills", eda_skills),
                            ("experience and education", eda_experience_education), ("salary", eda_salary),
                            ("time", eda_time), ("bivariate", eda_bivariate)]:
        run_section(f"EDA {label}", function, df)


def correlation_summary(frame, x, y):
    pairs = frame[[x, y]].dropna()
    if len(pairs) < MIN_CORRELATION_SIZE or pairs[x].nunique() < 2 or pairs[y].nunique() < 2:
        return None
    rho, p_value = stats.spearmanr(pairs[x], pairs[y])
    return rho, p_value, len(pairs)


def test_spearman(df, results):
    sub_banner("Test 1: Spearman correlation, required experience vs annualised pay midpoint")
    print("Research question : Do listings that require more experience advertise higher pay?")
    print("H0 : no monotonic association    H1 : a monotonic association exists")
    outcome = correlation_summary(df, "exp_mid", "pay_mid_annual")
    if outcome is None:
        print("Skipped: too few paired observations or a constant variable.")
        return
    rho, p_value, n = outcome
    skew = df["pay_mid_annual"].skew()
    print(f"Why Spearman      : pay skewness = {skew:.2f}; rank correlation needs no normality or linearity")
    print(f"Sample size       : {n:,} listings with parsed experience and numeric pay")
    z, se = np.arctanh(min(max(rho, -0.999999), 0.999999)), 1 / np.sqrt(n - 3)
    low, high = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    print(f"rho = {rho:.3f} (95% CI {low:.3f} to {high:.3f}), {p_text(p_value)}")
    ranges_only = correlation_summary(df[df["pay_category"].str.endswith("range")], "exp_mid", "pay_mid_annual")
    trimmed = correlation_summary(df[~df["pay_mid_annual_outlier"]], "exp_mid", "pay_mid_annual")
    for label, value in (("ranges only (no single values)", ranges_only), ("excluding IQR pay outliers", trimmed)):
        if value:
            print(f"Sensitivity       : {label}: rho = {value[0]:.3f}, n = {value[2]:,}, {p_text(value[1])}")
    decision = "reject H0" if p_value < ALPHA else "fail to reject H0"
    print(f"Decision          : {decision}; association is {effect_label(rho, (0.1, 0.3, 0.5))} and {'positive' if rho > 0 else 'negative'}.")
    print("Limitation        : only listings that disclose pay are included, and they may differ from those that do not.")
    print("                    Correlation is not causation; pay here is an advertised range, not an offer.")
    results.append({"test": "Spearman experience vs pay", "n": n, "statistic": round(rho, 4), "p_value": p_value,
                    "effect_size": f"rho={rho:.3f} ({effect_label(rho)})", "decision": decision})


def test_mann_whitney(df, results):
    sub_banner("Test 2: Mann-Whitney U, annualised pay midpoint between two role families")
    pay = df.dropna(subset=["pay_mid_annual", "role_family"])
    title_sizes = pay["title_clean"].value_counts()
    largest_title = int(title_sizes.iloc[0]) if len(title_sizes) else 0
    print(f"Role-level feasibility: the largest single cleaned title has {largest_title} pay-valid listings.")
    pay = pay[pay["role_family"] != "Other"]
    sizes = pay["role_family"].value_counts()
    eligible = [name for name in sizes.index if sizes[name] >= MIN_GROUP_SIZE][:2]
    if len(eligible) < 2:
        print(f"Skipped: fewer than two role families with at least {MIN_GROUP_SIZE} pay-valid listings.")
        return
    group_a = pay.loc[pay["role_family"] == eligible[0], "pay_mid_annual"]
    group_b = pay.loc[pay["role_family"] == eligible[1], "pay_mid_annual"]
    if group_a.nunique() < 2 and group_b.nunique() < 2:
        print("Skipped: both groups are constant.")
        return
    u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    common_language = u_stat / (len(group_a) * len(group_b))
    rank_biserial = 2 * common_language - 1
    print("Research question : Do the two largest role families (with enough pay data) advertise different pay?")
    print(f"Groups            : A = {eligible[0]} (n={len(group_a)}), B = {eligible[1]} (n={len(group_b)})")
    print("H0 : the pay distributions are equal    H1 : they differ")
    print("Why Mann-Whitney  : two independent groups, skewed pay, no normality assumption")
    print(f"Medians           : A = {group_a.median():,.0f}, B = {group_b.median():,.0f}")
    print(f"U = {u_stat:,.1f}, {p_text(p_value)}")
    print(f"Effect size       : P(A > B) + 0.5 P(tie) = {common_language:.3f}; rank-biserial r = {rank_biserial:.3f} ({effect_label(rank_biserial)})")
    decision = "reject H0" if p_value < ALPHA else "fail to reject H0 (no significant difference shown)"
    print(f"Decision          : {decision}")
    print("Limitation        : role families come from keyword rules; pay is advertised, disclosed pay only.")
    results.append({"test": f"Mann-Whitney {eligible[0]} vs {eligible[1]}", "n": len(group_a) + len(group_b),
                    "statistic": round(u_stat, 1), "p_value": p_value,
                    "effect_size": f"r={rank_biserial:.3f} ({effect_label(rank_biserial)})", "decision": decision})


def test_chi_square(df, results):
    sub_banner("Test 3: Chi-square test of independence, industry vs primary city (top 8 x top 8)")
    subset = df.dropna(subset=["industry", "primary_city"])
    industries = subset["industry"].value_counts().head(8).index
    cities = subset["primary_city"].value_counts().head(8).index
    table = pd.crosstab(subset.loc[subset["industry"].isin(industries) & subset["primary_city"].isin(cities), "industry"],
                        subset.loc[subset["industry"].isin(industries) & subset["primary_city"].isin(cities), "primary_city"])
    table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if min(table.shape) < 2:
        print("Skipped: the contingency table has fewer than 2 rows or columns.")
        return
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    n = int(table.values.sum())
    rows, cols = table.shape
    plain_v = np.sqrt(chi2 / (n * (min(rows, cols) - 1)))
    phi2 = max(0.0, chi2 / n - (rows - 1) * (cols - 1) / (n - 1))
    rows_corrected = rows - (rows - 1) ** 2 / (n - 1)
    cols_corrected = cols - (cols - 1) ** 2 / (n - 1)
    corrected_v = np.sqrt(phi2 / max(min(rows_corrected, cols_corrected) - 1, 1e-9))
    low_cells = (expected < 5).mean()
    print("Research question : Is the industry mix of listings the same across the major cities?")
    print("H0 : industry and city are independent    H1 : they are associated")
    print(f"Sample size       : {n:,} listings (top 8 industries within top 8 primary cities)")
    print(f"Table shape       : {rows} x {cols}")
    print(f"chi2 = {chi2:,.1f}, dof = {dof}, {p_text(p_value)}")
    print(f"Assumption check  : min expected count = {expected.min():.2f}; cells with expected < 5: {low_cells:.1%} (guideline: at most 20%)")
    if low_cells > 0.2:
        print("                    WARNING: assumption violated; treat the p-value as approximate.")
    print(f"Cramer's V        : {plain_v:.3f} (bias-corrected {corrected_v:.3f}) -> {effect_label(corrected_v, (0.1, 0.3, 0.5))} association")
    decision = "reject H0" if p_value < ALPHA else "fail to reject H0"
    print(f"Decision          : {decision}. With thousands of rows even small differences are 'significant';")
    print("                    the effect size shows whether the difference matters in practice.")
    ratios = pd.DataFrame(table.values / expected, index=table.index, columns=table.columns).stack().sort_values(ascending=False).head(5)
    print("Most over-represented industry-city pairs (observed / expected):")
    for (industry, city), value in ratios.items():
        print(f"  {industry[:40]:<40} in {city:<12} {value:.2f}x (observed {table.loc[industry, city]})")
    print("Limitation        : one portal's sample; the primary city is only the first city listed.")
    results.append({"test": "Chi-square industry vs city", "n": n, "statistic": round(chi2, 1), "p_value": p_value,
                    "effect_size": f"V={corrected_v:.3f} ({effect_label(corrected_v)})", "decision": decision})


def run_statistics(df):
    banner(f"PHASE 6: STATISTICAL ANALYSIS (alpha = {ALPHA})")
    results = []
    for label, function in [("Spearman", test_spearman), ("Mann-Whitney", test_mann_whitney), ("Chi-square", test_chi_square)]:
        run_section(label, function, df, results)
    table = pd.DataFrame(results)
    if table.empty:
        return table
    sub_banner("Statistical results summary")
    display = table.copy()
    display["p_value"] = display["p_value"].map(format_p)
    print(display.to_string(index=False))
    return table


def business_insights(df, stat_results, reports):
    banner("PHASE 7: BUSINESS INSIGHTS AND RECOMMENDATIONS (generated from the computed results)")
    rows = []

    def add(kind, area, statement, evidence):
        rows.append({"type": kind, "area": area, "statement": statement, "evidence": evidence})

    industries = df["industry"].value_counts()
    if len(industries):
        share = industries.iloc[0] / industries.sum()
        statement = (f"Listings are dominated by {industries.index[0]}" if share >= 0.30
                     else f"{industries.index[0]} leads but listings are spread across industries")
        add("Observation", "Industry", statement, f"{industries.iloc[0]:,} of {industries.sum():,} listings ({share:.1%}); top 3 industries hold {industries.iloc[:3].sum() / industries.sum():.1%}")

    cities = df["primary_city"].value_counts()
    mentions = df["city_list"].explode().dropna().value_counts()
    if len(cities):
        add("Observation", "Location", f"Recruitment activity is concentrated in a few cities, led by {cities.index[0]}" if cities.iloc[:3].sum() / cities.sum() >= 0.5 else f"{cities.index[0]} leads without extreme concentration",
            f"{cities.iloc[0]:,} listings have it as first city ({cities.iloc[0] / cities.sum():.1%}); top 3 first cities hold {cities.iloc[:3].sum() / cities.sum():.1%}; {mentions.index[0]} appears in {mentions.iloc[0]:,} listings overall")

    families = df.loc[df["role_family"] != "Other", "role_family"].value_counts()
    if len(families):
        add("Observation", "Roles", f"{families.index[0]} is the largest identifiable role family",
            f"{families.iloc[0]:,} listings ({families.iloc[0] / df['role_family'].notna().sum():.1%} of titled listings); {(df['role_family'] == 'Other').mean():.1%} fall in Other; titles are fragmented ({df['title_clean'].nunique():,} cleaned titles)")

    technical = df[[f"skill_{t}" for t in TECHNICAL_SKILLS]].mean().sort_values(ascending=False)
    top_technical = [name.replace("skill_", "") for name in technical.index[:5] if technical[name] > 0]
    general = df[[f"skill_{t}" for t in GENERAL_SKILLS]].mean().sort_values(ascending=False)
    if top_technical:
        add("Observation", "Skills", f"Most mentioned technical skills: {', '.join(top_technical)}",
            "; ".join(f"{n.replace('skill_', '')} {technical[n]:.1%}" for n in technical.index[:5]) + " of listings (keyword mentions, not confirmed requirements)")
    if general.iloc[0] > 0:
        add("Observation", "Skills", f"Most mentioned general skill: {general.index[0].replace('skill_', '')}", f"{general.iloc[0]:.1%} of listings")

    experience = df["exp_category"].value_counts()
    if len(experience):
        add("Observation", "Experience", f"The most common minimum-experience band is {experience.index[0]}",
            f"{experience.iloc[0] / experience.sum():.1%} of listings with parsed experience; median required experience midpoint {df['exp_mid'].median():.1f} years")

    ug = df["ug_requirement"].value_counts()
    if len(ug):
        add("Observation", "Education", f"The most common first UG option is {ug.index[0]}", f"{ug.iloc[0] / ug.sum():.1%} of listings with education text")

    disclosed = df["has_salary"].mean()
    not_disclosed = (df["pay_category"] == "Not disclosed").mean()
    add("Observation", "Salary", "Salary non-disclosure is the norm" if not_disclosed > 0.5 else "Most listings disclose salary",
        f"{disclosed:.1%} give a comparable numeric pay; {not_disclosed:.1%} say 'Not disclosed'")

    spearman = stat_results[stat_results["test"].str.startswith("Spearman")] if len(stat_results) else pd.DataFrame()
    if len(spearman):
        rho, p_value, n = spearman.iloc[0]["statistic"], spearman.iloc[0]["p_value"], spearman.iloc[0]["n"]
        verdict = "increases with" if (rho > 0 and p_value < ALPHA) else "shows no clear monotonic link with"
        add("Observation", "Pay vs experience", f"Among listings that disclose pay, advertised pay {verdict} required experience",
            f"Spearman rho = {rho:.3f}, n = {n:,}, {p_text(p_value)}")

    add("Possible explanation (untested)", "Industry / location", "The strong IT-Software and Bengaluru presence may reflect the portal's user base or how the sample was drawn, not the wider labour market", "No test in this project separates sampling effects from real demand")
    add("Possible explanation (untested)", "Pay vs experience", "Employers who disclose pay may differ systematically from those who do not", "Only listings with numeric pay enter the correlation")

    if top_technical:
        add("Recommendation", "Job seekers", f"Build and evidence {', '.join(top_technical[:3])} alongside communication skills, then confirm against current live postings",
            f"Based on keyword frequencies above ({technical.iloc[0]:.1%} top technical skill)")
    if len(industries):
        add("Recommendation", "Job seekers", f"Investigate {industries.index[0]} and {', '.join(cities.index[:2])} first for volume, but also look at less crowded industries",
            f"Based on industry share {industries.iloc[0] / industries.sum():.1%} and city counts")
    if len(experience):
        add("Recommendation", "Job seekers", f"The most common requirement is {experience.index[0]} ({experience.iloc[0] / experience.sum():.1%}); entry-level (0 yrs minimum) listings are only {(df['exp_category'] == '0 yrs minimum').sum() / experience.sum():.1%}",
            "Based on parsed minimum experience")
    if top_technical:
        add("Recommendation", "Educational institutions", f"Include hands-on work in {', '.join(top_technical[:4])} in curricula and projects",
            "Based on the most frequently mentioned technical skills; validate with employer input")
    if len(ug):
        add("Recommendation", "Educational institutions", f"Most listings accept '{ug.index[0]}' ({ug.iloc[0] / ug.sum():.1%}), so emphasise demonstrable skills over degree label alone",
            "Based on first listed UG option")
    add("Recommendation", "Recruiters and employers", "Disclosing a pay range can differentiate a listing" if not_disclosed > 0.5 else "Keep disclosing pay ranges to stay competitive",
        f"{not_disclosed:.1%} of listings do not disclose pay")
    if len(cities):
        add("Recommendation", "Recruiters and employers", f"Consider the competition in {cities.index[0]}, where first-city listings are most concentrated, and standardise job titles for searchability",
            f"{cities.iloc[0] / cities.sum():.1%} of first cities; {df['title_clean'].nunique():,} cleaned title variants")

    table = pd.DataFrame(rows)
    for kind in table["type"].unique():
        sub_banner(kind)
        for _, row in table[table["type"] == kind].iterrows():
            wrap_print(f"[{row['area']}] {row['statement']}", "  ")
            wrap_print(f"Evidence: {row['evidence']}", "      ")
    reports["business_insights"] = table
    return table


def ethics_and_limitations():
    banner("PHASE 8: ETHICS, BIAS AND LIMITATIONS")
    notes = [
        "Scope: results describe this sample of listings from one portal, not the Indian labour market.",
        "Time: postings span roughly 2015-2017 with uneven monthly coverage; patterns may be outdated and are not trends.",
        "Salary: only explicit numeric annual or monthly figures are parsed (monthly x 12; lakh units expanded). Currency is not documented in the data, so no currency is assumed. Open-ended bands, pay bands, hourly or daily pay and text such as 'Best in Industry' are excluded.",
        "Skills: the 'skills' column holds one functional-area label per row. Real skills come from keyword matching on descriptions; a mention is not a confirmed requirement and unlisted skills are missed.",
        "Roles: role families come from keyword rules and titles are cleaned heuristically; both can mis-group roles and were only spot-checked.",
        "Location: the first listed city is used as primary; multi-city listings are also counted by mention.",
        "Bias: sampling, geographic and industry-representation bias (IT-Software heavy) and missing-data bias (education, positions, pay) limit generalisation.",
        "Privacy: the data are employer listings, not applicant records; use them only for the stated academic purpose, restrict access, and do not try to identify individuals.",
        "Fairness: do not infer protected characteristics from descriptions or use these patterns to rank people; keep a human in the loop for any hiring decision, and document methods for transparency.",
    ]
    for note in notes:
        wrap_print(note)




# ---------------------------------------------------------------------------
# ADDITIONS FOR THE REMAINING COURSE REQUIREMENTS
# Original cleaning, feature engineering, EDA, statistical tests,
# business insights and ethics functions are left unchanged.
# ---------------------------------------------------------------------------

def calculate_kpis(df, reports):
    """Calculate a concise set of job-market KPIs."""
    total = len(df)
    salary_count = int(df["pay_mid_annual"].notna().sum())

    kpis = pd.DataFrame([
        ["Total job listings", total],
        ["Unique companies", df["company"].nunique()],
        ["Salary disclosure rate (%)", round(salary_count / total * 100, 2) if total else np.nan],
        ["Median annualised salary", round(df["pay_mid_annual"].median(), 2)],
        ["Median experience (years)", round(df["exp_mid"].median(), 2)],
    ], columns=["KPI", "Value"])

    reports["kpi_summary"] = kpis
    reports["kpi_by_industry"] = (
        df["industry"].value_counts().rename_axis("industry")
        .reset_index(name="job_listings")
    )
    reports["kpi_by_location"] = (
        df["primary_city"].value_counts().head(15).rename_axis("city")
        .reset_index(name="job_listings")
    )
    print("\nKEY PERFORMANCE INDICATORS")
    print(kpis.to_string(index=False))


def analyse_monthly_postings(df, reports):
    """Create monthly counts and a simple, cautious three-month forecast."""
    dated = df.dropna(subset=["postdate"])
    if dated.empty:
        print("\nTime-series analysis skipped: no valid posting dates.")
        return

    monthly = dated.set_index("postdate").resample("MS").size().astype(float)
    # Missing months are unknown, not zero.
    observed = set(dated["postdate"].dt.to_period("M").astype(str))
    labels = monthly.index.to_period("M").astype(str)
    monthly.loc[[label not in observed for label in labels]] = np.nan

    table = pd.DataFrame({
        "month": monthly.index.strftime("%Y-%m"),
        "job_listings": monthly.values
    })
    table["moving_average_3m"] = table["job_listings"].rolling(3, min_periods=2).mean()
    reports["monthly_job_postings"] = table

    valid = monthly.dropna()
    if len(valid) >= 8:
        # Use actual calendar-month positions so gaps are not compressed.
        positions = np.array([monthly.index.get_loc(date) for date in valid.index], dtype=float)
        values = valid.to_numpy()
        split = len(values) - max(2, min(3, round(len(values) * 0.2)))
        train_x, train_y = positions[:split], values[:split]
        test_x, test_y = positions[split:], values[split:]

        slope, intercept = np.polyfit(train_x, train_y, 1)
        predictions = np.maximum(0, intercept + slope * test_x)
        errors = test_y - predictions
        reports["forecast_evaluation"] = pd.DataFrame([{
            "model": "Linear trend",
            "MAE": round(float(np.mean(np.abs(errors))), 2),
            "RMSE": round(float(np.sqrt(np.mean(errors ** 2))), 2),
            "test_months": len(test_y)
        }])

        # Refit on all observed months and forecast the next three calendar months.
        slope, intercept = np.polyfit(positions, values, 1)
        future_positions = np.arange(len(monthly), len(monthly) + 3)
        future_values = np.maximum(0, intercept + slope * future_positions)
        future_dates = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq="MS")
        reports["forecast_results"] = pd.DataFrame({
            "month": future_dates.strftime("%Y-%m"),
            "forecast_job_listings": np.round(future_values, 0).astype(int)
        })
        print("\nFORECAST MODEL EVALUATION")
        print(reports["forecast_evaluation"].to_string(index=False))
        print("\nTHREE-MONTH EXPLORATORY FORECAST")
        print(reports["forecast_results"].to_string(index=False))
    else:
        print("\nForecast skipped: at least 8 observed months are needed.")

    print("\nMonthly posting counts and moving averages prepared.")
    print("Note: missing months are unknown, and the historical forecast is exploratory.")


def create_charts(df, reports, output_dir):
    """Create a small set of charts that directly support the project findings."""
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    charts = [
        (df["industry"].value_counts().head(10), "Top Industries", "industry.png"),
        (df["primary_city"].value_counts().head(10), "Top Job Locations", "locations.png"),
        (df["role_family"].value_counts().head(10), "Job Listings by Role", "roles.png"),
    ]
    for values, title, filename in charts:
        if values.empty:
            continue
        ax = values.sort_values().plot(kind="barh", figsize=(9, 5))
        ax.set_title(title)
        ax.set_xlabel("Number of listings")
        ax.set_ylabel("")
        ax.figure.tight_layout()
        ax.figure.savefig(chart_dir / filename, dpi=150)
        plt.close(ax.figure)

    skill_columns = [f"skill_{term}" for term in SKILL_TERMS if f"skill_{term}" in df.columns]
    if skill_columns:
        skills = df[skill_columns].fillna(False).astype(bool).sum().sort_values(ascending=False).head(10)
        skills.index = [name.replace("skill_", "") for name in skills.index]
        ax = skills.sort_values().plot(kind="barh", figsize=(9, 5))
        ax.set_title("Most Frequently Mentioned Skills")
        ax.set_xlabel("Number of listings mentioning skill")
        ax.set_ylabel("")
        ax.figure.tight_layout()
        ax.figure.savefig(chart_dir / "skills.png", dpi=150)
        plt.close(ax.figure)

    dated = df.dropna(subset=["postdate"])
    if not dated.empty:
        monthly = dated.set_index("postdate").resample("MS").size()
        ax = monthly.plot(marker="o", figsize=(10, 5))
        ax.set_title("Monthly Job Listings")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of listings")
        ax.figure.tight_layout()
        ax.figure.savefig(chart_dir / "monthly_postings.png", dpi=150)
        plt.close(ax.figure)

    print(f"\nCharts saved in: {chart_dir.resolve()}")


def write_powerbi_guide(output_dir):
    """Provide a short guide for building the dashboard in Power BI."""
    mapping = pd.DataFrame([
        ["Overview", "KPI cards and industry/location/role charts", "kpi_summary.csv; kpi_by_industry.csv; kpi_by_location.csv"],
        ["Salary and Skills", "Salary KPIs and skill frequency", "kpi_summary.csv; cleaned_dataset.csv"],
        ["Recruitment Trends", "Monthly postings and forecast", "monthly_job_postings.csv; forecast_results.csv"],
    ], columns=["dashboard_page", "suggested_visuals", "source_files"])
    mapping.to_csv(output_dir / "powerbi_field_mapping.csv", index=False)

    (output_dir / "POWER_BI_README.txt").write_text(
        "Power BI dashboard plan\n\n"
        "1. Job Market Overview: KPI cards, industry, location and role charts.\n"
        "2. Salary and Skills: median salary, salary availability and skill frequency.\n"
        "3. Recruitment Trends: monthly postings, moving average and exploratory forecast.\n\n"
        "Import the listed CSV files into Power BI Desktop and create the visuals. "
        "This Python script prepares the data and mapping; it does not create a PBIX file.\n",
        encoding="utf-8"
    )


def save_outputs(df, reports, stat_results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    drop_columns = ["city_list", "description_text"]
    df.drop(columns=[c for c in drop_columns if c in df.columns]).to_csv(output_dir / "cleaned_dataset.csv", index=False)
    numeric = df.select_dtypes(include="number")
    numeric.describe().T.to_csv(output_dir / "descriptive_statistics.csv")
    if len(stat_results):
        stat_results.to_csv(output_dir / "statistical_results.csv", index=False)
    for name, table in reports.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    print(f"\nSaved {len(list(output_dir.glob('*.csv')))} CSV files to {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Naukri.com job market analysis (terminal output)")
    parser.add_argument("--input", default=None, help="CSV path; auto-detected if it contains 'naukri' in the name")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-save", action="store_true")
    arguments = parser.parse_args()

    reports = {}
    raw = load_raw(find_input(arguments.input), reports)
    df = engineer_features(clean_data(raw))
    banner("OUTLIERS, PARSING AND MANUAL VALIDATION")
    run_section("outliers", report_outliers, df, reports)
    run_section("parsing report", parsing_report, df)
    run_section("validation samples", validation_samples, df)
    run_section("validation", validate, raw, df)
    run_eda(df)
    stat_results = run_section("statistics", run_statistics, df)
    stat_results = stat_results if stat_results is not None else pd.DataFrame()
    run_section("business insights", business_insights, df, stat_results, reports)

    # Added only the remaining course requirements.
    output_dir = Path(arguments.output_dir)
    run_section("KPI calculations", calculate_kpis, df, reports)
    run_section("monthly analysis and forecasting", analyse_monthly_postings, df, reports)
    run_section("project charts", create_charts, df, reports, output_dir)
    write_powerbi_guide(output_dir)

    ethics_and_limitations()

    if not arguments.no_save:
        run_section("saving outputs", save_outputs, df, reports, stat_results, output_dir)
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()