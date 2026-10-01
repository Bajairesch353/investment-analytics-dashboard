"""Merchant name cleaning and spending-category classification.

Used by notebooks/06_cash_consumption_analytics.ipynb.

Classification is layered, first match wins:

1. Private overrides (``data/manual/merchant_overrides_manual.csv``,
   gitignored) -- merchant-specific corrections that should not live in
   tracked code (landlord, doctors, visa fees, ...).
2. Strong MCC -- the merchant category code attached to card payments
   by the card network (ISO 18245). Most reliable signal we have.
3. Keyword rules -- generic merchant-name patterns, mainly for direct
   debits, which carry no MCC.
4. Weak MCC -- generic codes (e.g. 5499 "misc. food stores", 5999 "misc.
   retail") only act as a fallback when no keyword rule matched.
5. Otherwise "Uncategorized".
"""

import re

import pandas as pd


def normalize_merchant_text(value):
    if pd.isna(value):
        return "UNKNOWN"

    text = str(value).upper().strip()

    replacements = {
        "Ä": "AE",
        "Ö": "OE",
        "Ü": "UE",
        "ß": "SS"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


DIRECT_DEBIT_PATTERN = re.compile(
    r"^SEPA DIRECT DEBIT TRANSFER TO (?P<creditor>.+?)(?:\s*\([^)]*\))?$"
)


def extract_merchant_text(name, description):
    """Pick the text that actually identifies the merchant.

    Direct debits show the account holder (or nothing) as ``name`` and hide
    the creditor in the description ("Sepa Direct Debit transfer to X
    (IBAN)"), so the creditor is extracted from there. Otherwise ``name``
    is used, falling back to the description when the name is empty.

    Returns (merchant_clean, merchant_source).
    """
    description_text = normalize_merchant_text(description)
    match = DIRECT_DEBIT_PATTERN.match(description_text)

    if match:
        return match.group("creditor").strip(), "description_direct_debit"

    name_text = normalize_merchant_text(name)

    if name_text in ["UNKNOWN", "NAN", "NONE", ""]:
        if description_text not in ["UNKNOWN", "NAN", "NONE", ""]:
            return description_text, "description_missing_name"
        return "UNKNOWN", "name"

    return name_text, "name"


# Ordered (regex, canonical merchant name) rules for the auto-naming pass.
# First match wins. Word boundaries keep brand names from matching inside
# other words (e.g. UBER in "BAECKEREI HUBER").
MERCHANT_AUTO_RULES = [
    (r"\bKAUFLAND\b", "KAUFLAND"),
    (r"\bREWE\b", "REWE"),
    (r"\bEDEKA\b", "EDEKA"),
    (r"\bLIDL\b", "LIDL"),
    (r"\bALDI\b", "ALDI"),
    (r"\bNETTO\b", "NETTO"),
    (r"\bDM\b", "DM"),
    (r"\bDROGERIE MARKT\b", "DM"),

    (r"\bAPPLE\b", "APPLE"),
    (r"\bAMAZON\b", "AMAZON"),
    (r"\bAMZN\b", "AMAZON"),
    (r"\bPAYPAL\b", "PAYPAL"),
    (r"\bSPOTIFY\b", "SPOTIFY"),
    (r"\bNETFLIX\b", "NETFLIX"),
    (r"\bGOOGLE\b", "GOOGLE"),
    (r"\bOPENAI\b", "OPENAI"),

    (r"\bDEUTSCHE BAHN\b", "DEUTSCHE BAHN"),
    (r"\bDB VERTRIEB\b", "DEUTSCHE BAHN"),
    (r"\bUBER\b", "UBER"),
    (r"\bRYANAIR\b", "RYANAIR"),
    (r"\bLUFTHANSA\b", "LUFTHANSA"),

    (r"\bMC ?DONALD", "MCDONALD'S"),
    (r"\bBURGER KING\b", "BURGER KING"),
    (r"\bSUBWAY\b", "SUBWAY"),
    (r"\bSTARBUCKS\b", "STARBUCKS"),
]


def infer_merchant_final_auto(value):
    text = normalize_merchant_text(value)

    for pattern, merchant in MERCHANT_AUTO_RULES:
        if re.search(pattern, text):
            return merchant

    # Generic cleanup for branch/location/company noise
    text = re.sub(r"\bINNEN\b", "", text)
    text = re.sub(r"\bFILIALEN\b", "", text)
    text = re.sub(r"\bFILIALE\b", "", text)
    text = re.sub(r"\bGMBH\b", "", text)
    text = re.sub(r"\bKG\b", "", text)
    text = re.sub(r"\bLTD\b", "", text)
    text = re.sub(r"\bSAGT DANKE\b", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    if text == "":
        return "UNKNOWN"

    return text


def infer_spending_category_auto(merchant_final_auto):
    text = normalize_merchant_text(merchant_final_auto)

    if text in ["KAUFLAND", "REWE", "EDEKA", "LIDL", "ALDI", "NETTO"]:
        return "Groceries"

    if text in ["DM"]:
        return "Drugstore"

    if text in ["APPLE", "SPOTIFY", "NETFLIX", "GOOGLE", "OPENAI"]:
        return "Subscriptions / Digital"

    if text in ["AMAZON", "PAYPAL"]:
        return "Online Shopping"

    if text in ["DEUTSCHE BAHN", "UBER", "RYANAIR", "LUFTHANSA"]:
        return "Transport / Travel"

    if text in ["MCDONALD'S", "BURGER KING", "SUBWAY", "STARBUCKS"]:
        return "Eating Out"

    return "Uncategorized"


# ISO 18245 merchant category codes -> spending category. Strong codes are
# specific enough to trust over any name-based keyword rule.
MCC_CATEGORY_MAP = {
    # Transport / travel
    4011: "Transport / Travel",
    4111: "Transport / Travel",
    4112: "Transport / Travel",
    4121: "Transport / Travel",
    4131: "Transport / Travel",
    4411: "Transport / Travel",
    4511: "Transport / Travel",
    4722: "Transport / Travel",
    4789: "Transport / Travel",
    7512: "Transport / Travel",
    7513: "Transport / Travel",
    7519: "Transport / Travel",
    4784: "Parking / Tolls",
    7523: "Parking / Tolls",
    5541: "Fuel / Car",
    5542: "Fuel / Car",
    7538: "Fuel / Car",
    7542: "Fuel / Car",
    7011: "Accommodation",
    7012: "Accommodation",
    7032: "Accommodation",
    7033: "Accommodation",

    # Food
    5411: "Groceries",
    5441: "Groceries",
    5451: "Groceries",
    5462: "Eating Out",
    5811: "Eating Out",
    5812: "Eating Out",
    5813: "Eating Out",
    5814: "Eating Out",

    # Shopping
    5611: "Clothing",
    5621: "Clothing",
    5631: "Clothing",
    5641: "Clothing",
    5655: "Clothing",
    5661: "Clothing",
    5691: "Clothing",
    5699: "Clothing",
    5111: "Household / Misc",
    5200: "Household / Misc",
    5251: "Household / Misc",
    5712: "Household / Misc",
    5719: "Household / Misc",
    5722: "Household / Misc",
    5732: "Household / Misc",
    5943: "Household / Misc",
    5945: "Household / Misc",
    5992: "Household / Misc",
    5993: "Household / Misc",
    5994: "Household / Misc",
    5942: "Books / Media",
    5192: "Books / Media",
    5977: "Drugstore",

    # Digital
    4814: "Subscriptions / Digital",
    4816: "Subscriptions / Digital",
    4899: "Subscriptions / Digital",
    5734: "Subscriptions / Digital",
    5815: "Subscriptions / Digital",
    5816: "Subscriptions / Digital",
    5817: "Subscriptions / Digital",
    5818: "Subscriptions / Digital",
    7372: "Subscriptions / Digital",

    # Health / sports / leisure
    8011: "Health / Medication",
    8021: "Health / Medication",
    8042: "Health / Medication",
    8043: "Health / Medication",
    8062: "Health / Medication",
    8099: "Health / Medication",
    5940: "Sports / Fitness",
    5941: "Sports / Fitness",
    7941: "Sports / Fitness",
    7997: "Sports / Fitness",
    5996: "Leisure / Wellness",
    7991: "Leisure / Wellness",
    7832: "Entertainment",
    7922: "Entertainment",
    7929: "Entertainment",
    7996: "Entertainment",
    7999: "Entertainment",

    # Education / admin / finance
    8211: "Education",
    8220: "Education",
    8241: "Education",
    8244: "Education",
    8249: "Education",
    8299: "Education",
    9211: "Admin / Fees",
    9222: "Admin / Fees",
    9311: "Admin / Fees",
    9399: "Admin / Fees",
    9402: "Admin / Fees",
    6300: "Insurance",
    6010: "Cash Withdrawal",
    6011: "Cash Withdrawal",
}

# Generic codes: only used when no keyword rule matched. None means the
# code carries no usable category information at all.
MCC_WEAK_CATEGORY_MAP = {
    5310: "Household / Misc",
    5311: "Household / Misc",
    5331: "Household / Misc",
    5399: "Household / Misc",
    5422: "Groceries",
    5499: "Groceries",
    5651: "Clothing",
    5912: "Health / Medication",
    5999: "Household / Misc",
    7298: "Leisure / Wellness",
    8699: "Leisure / Wellness",
    4829: None,
    7399: None,
    8999: None,
}

# MCC ranges for airline (3000-3350), car rental (3351-3500) and hotel
# (3501-3999) chain codes.
MCC_RANGE_MAP = [
    (3000, 3350, "Transport / Travel"),
    (3351, 3500, "Transport / Travel"),
    (3501, 3999, "Accommodation"),
]


def normalize_mcc(mcc_code):
    if mcc_code is None or pd.isna(mcc_code):
        return None

    try:
        return int(float(mcc_code))
    except (TypeError, ValueError):
        return None


def category_from_mcc(mcc_code):
    """Map an MCC to (category, strength).

    strength is "strong", "weak" or None (unknown / missing code). A weak
    code may still return category None when it carries no information.
    """
    mcc = normalize_mcc(mcc_code)

    if mcc is None:
        return None, None

    if mcc in MCC_CATEGORY_MAP:
        return MCC_CATEGORY_MAP[mcc], "strong"

    for low, high, category in MCC_RANGE_MAP:
        if low <= mcc <= high:
            return category, "strong"

    if mcc in MCC_WEAK_CATEGORY_MAP:
        return MCC_WEAK_CATEGORY_MAP[mcc], "weak"

    return None, None


# Ordered (category, include patterns, exclude patterns) rules on the
# normalized merchant text. First match wins. Only generic patterns belong
# here -- anything identifying a specific person, place or doctor goes into
# the gitignored override file instead.
KEYWORD_CATEGORY_RULES = [
    ("Housing / Rent", [r"IMMOBILIEN", r"\bIMMO\b", r"HAUSVERWALTUNG", r"\bMIETE\b"], []),
    ("Sports / Fitness", [
        r"RSG GROUP", r"FITNESS", r"\bGYM\b", r"MCFIT", r"JOHN REED",
        r"\bFITX\b", r"URBAN SPORTS",
    ], []),
    ("Admin / Fees", [r"VISA FEE", r"IMMIGRATION HEALTH SURCHARGE"], []),
    ("Health / Medication", [
        r"APOTHEKE", r"PHARMACY", r"FARMACIA", r"DOCTOR", r"MEDICAL",
    ], []),
    ("Education", [
        r"BRITISH COUNCIL", r"IELTS", r"UNIVERSITY", r"UNIVERSITAET",
        r"SCHOOL", r"COURSE", r"LANGUAGE TEST", r"SEMESTERBEITRAG",
        r"SEMESTER FEE", r"RUECKMELDUNG", r"STUDENT UNION FEE",
    ], []),
    ("Groceries", [
        r"\bKAUFLAND\b", r"\bREWE\b", r"\bEDEKA\b", r"\bLIDL\b", r"\bALDI\b",
        r"\bNETTO\b", r"\bNORMA\b", r"\bCONAD\b", r"\bCOOP\b", r"\bITALMARK\b",
        r"\bSUPERMERCAT[OI]\b", r"\bFLINK\b", r"\bDI PIU",
    ], []),
    ("Drugstore", [r"ROSSMANN", r"\bMUELLER\b", r"^DM\b", r"DROGERIE MARKT"], [
        r"CAFE", r"B.?.?CKEREI", r"BACKSTUBE", r"BACKWERK", r"BACKER", r"BAKERY",
    ]),
    ("Fuel / Car", [
        r"TANKSTELLE", r"\bAGIP\b", r"\bARAL\b", r"\bSHELL\b", r"\bESSO\b",
        r"SERVICE-STATION", r"\bFUEL\b", r"PETROL",
    ], []),
    ("Online Shopping", [r"AMAZON", r"AMZN", r"PAYPAL"], []),
    ("Transport / Travel", [
        r"FLIXBUS", r"FLIX SE", r"DEUTSCHE BAHN", r"DB VERTRIEB", r"^DB ",
        r"\bUBER\b", r"\bTAXI\b", r"\bTRAIN\b", r"\bBUS\b", r"AIRLINE",
        r"FLIGHT", r"RYANAIR", r"LUFTHANSA", r"VERKEHRSVERBUND", r"\bRVV\b",
        r"DEUTSCHLANDTICKET", r"TRENITALIA",
    ], []),
    ("Parking / Tolls", [
        r"ASFINAG", r"APCOA", r"PARCHEGGIO", r"PARKING", r"AUTOSTRADA",
        r"\bTOLL\b",
    ], []),
    ("Accommodation", [r"HOSTEL", r"HOTEL", r"AIRBNB", r"BOOKING"], []),
    ("Supplements / Nutrition", [r"PROTEIN", r"SUPPLEMENT", r"NUTRITION"], []),
    ("Eating Out", [
        r"MCDONALD", r"BURGER KING", r"SUBWAY", r"PIZZ", r"SUSHI",
        r"RESTAURANT", r"RISTORANTE", r"TRATTORIA", r"OSTERIA", r"BISTRO",
        r"IMBISS", r"DONER", r"DOENER", r"KEBAB", r"GOURMET", r"TAPAS",
        r"GRILL", r"^BAR ", r"\bBAR\b", r"B.?.?CKEREI", r"BACKSTUBE",
        r"BACKWERK", r"BAECKER", r"BAKERY", r"CAFE", r"CAFFE", r"CREMERIA",
        r"GELATERIA", r"MENSA", r"CAFETERIA", r"CANTEEN", r"SPEISESAAL",
        r"PIADINERIA", r"FOCACC", r"PANINI",
        r"STU\w*WERK.*AUTOMAT", r"AUTOMAT.*STU\w*WERK",
    ], []),
    ("Entertainment", [r"CINEMA", r"MOVIE", r"THEATER", r"THEATRE"], []),
    ("Leisure / Wellness", [r"THERME", r"SAUNA", r"WELLNESS"], []),
    ("Books / Media", [
        r"BUECHER", r"PUSTET", r"FELTRINELLI", r"LIBRERI", r"BOOKSHOP",
        r"BOOKSTORE",
    ], []),
    ("Household / Misc", [
        r"\bTEDI\b", r"\bACTION\b", r"PAPIER", r"PAPER", r"STATIONERY",
        r"SCHREIBWAREN",
    ], []),
    ("Clothing", [r"\bNIKE\b", r"\bLEVI", r"UNIQLO", r"H&M", r"\bZARA\b"], [
        r"GRILL", r"RESTAURANT", r"\bBAR\b", r"PIZZ", r"SUSHI",
    ]),
]


def infer_category_from_keywords(text):
    text = normalize_merchant_text(text)

    for category, include_patterns, exclude_patterns in KEYWORD_CATEGORY_RULES:
        if any(re.search(p, text) for p in include_patterns) and not any(
            re.search(p, text) for p in exclude_patterns
        ):
            return category

    return None


OVERRIDE_COLUMNS = [
    "pattern",
    "merchant_final",
    "spending_category",
    "exclude_from_consumption",
    "fixed_variable",
    "note",
]


def match_override(detail_text, overrides):
    """Return the first override row (as dict) whose regex matches.

    ``overrides`` is a DataFrame with OVERRIDE_COLUMNS; ``pattern`` is a
    case-insensitive regex on the normalized merchant + description text.
    """
    if overrides is None or len(overrides) == 0:
        return None

    text = normalize_merchant_text(detail_text)

    for _, override in overrides.iterrows():
        pattern = override.get("pattern")
        if pd.isna(pattern) or str(pattern).strip() == "":
            continue
        if re.search(str(pattern), text, flags=re.IGNORECASE):
            return override.to_dict()

    return None


def _is_set(value):
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _to_bool(value):
    return str(value).strip().lower() in ["true", "1", "yes", "y", "ja", "x"]


def classify_consumption_transaction(merchant_clean, description, mcc_code, overrides=None):
    """Classify one consumption transaction.

    Returns a dict with merchant_final, spending_category,
    exclude_from_consumption, fixed_variable_override (None unless an
    override sets it) and classification_source (override / mcc / keyword /
    mcc_weak / unclassified).
    """
    merchant_final = infer_merchant_final_auto(merchant_clean)
    exclude = False
    fixed_variable_override = None

    override = match_override(f"{merchant_clean} {description}", overrides)

    if override is not None:
        if _is_set(override.get("merchant_final")):
            merchant_final = str(override["merchant_final"]).strip()
        if _is_set(override.get("exclude_from_consumption")):
            exclude = _to_bool(override["exclude_from_consumption"])
        if _is_set(override.get("fixed_variable")):
            fixed_variable_override = str(override["fixed_variable"]).strip()
        if _is_set(override.get("spending_category")):
            return {
                "merchant_final": merchant_final,
                "spending_category": str(override["spending_category"]).strip(),
                "exclude_from_consumption": exclude,
                "fixed_variable_override": fixed_variable_override,
                "classification_source": "override",
            }

    mcc_category, mcc_strength = category_from_mcc(mcc_code)

    if mcc_strength == "strong":
        category, source = mcc_category, "mcc"
    else:
        keyword_category = infer_category_from_keywords(merchant_clean)

        if keyword_category is None:
            brand_category = infer_spending_category_auto(merchant_final)
            if brand_category != "Uncategorized":
                keyword_category = brand_category

        if keyword_category is not None:
            category, source = keyword_category, "keyword"
        elif mcc_category is not None:
            category, source = mcc_category, "mcc_weak"
        else:
            category, source = "Uncategorized", "unclassified"

    return {
        "merchant_final": merchant_final,
        "spending_category": category,
        "exclude_from_consumption": exclude,
        "fixed_variable_override": fixed_variable_override,
        "classification_source": source,
    }


def preview_unique(series, n=5):
    values = (
        series
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    return " | ".join(values[:n])


def classify_fixed_variable_from_category(
    spending_category,
    merchant_final="",
    exclude_from_consumption=False
):
    category = "" if pd.isna(spending_category) else str(spending_category)
    merchant = normalize_merchant_text(merchant_final)

    if bool(exclude_from_consumption) or category in [
        "Internal",
        "Cash Withdrawal"
    ]:
        return "Internal / exclude"

    if category in [
        "Housing / Rent",
        "Subscriptions / Digital",
        "Insurance"
    ]:
        return "Fixed / recurring"

    if category == "Sports / Fitness" and any(
        key in merchant
        for key in [
            "MCFIT",
            "JOHN REED",
            "RSG GROUP",
            "FITX",
            "URBAN SPORTS"
        ]
    ):
        return "Fixed / recurring"

    if category == "Transport / Travel" and any(
        key in merchant
        for key in [
            "VERKEHRSVERBUND",
            "RVV",
            "DEUTSCHLANDTICKET"
        ]
    ):
        return "Fixed / recurring"

    if category in [
        "Admin / Fees",
        "Accommodation",
        "Education",
        "Transport / Travel"
    ]:
        return "One-off"

    return "Variable"
