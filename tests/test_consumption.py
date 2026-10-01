import pandas as pd

from src.consumption import (
    OVERRIDE_COLUMNS,
    category_from_mcc,
    classify_consumption_transaction,
    classify_fixed_variable_from_category,
    extract_merchant_text,
    infer_category_from_keywords,
    infer_merchant_final_auto,
    infer_spending_category_auto,
    normalize_merchant_text,
    preview_unique,
)


def test_normalize_merchant_text_uppercases_and_strips():
    assert normalize_merchant_text("  some shop  ") == "SOME SHOP"


def test_normalize_merchant_text_replaces_german_umlauts():
    assert normalize_merchant_text("Bäckerei Müller") == "BAECKEREI MUELLER"
    assert normalize_merchant_text("Straße") == "STRASSE"


def test_normalize_merchant_text_collapses_whitespace():
    assert normalize_merchant_text("A   B\tC") == "A B C"


def test_normalize_merchant_text_handles_nan():
    assert normalize_merchant_text(float("nan")) == "UNKNOWN"


def test_infer_merchant_final_auto_matches_known_rule():
    assert infer_merchant_final_auto("REWE Filiale 123") == "REWE"
    assert infer_merchant_final_auto("some AMAZON MARKETPLACE payment") == "AMAZON"


def test_infer_merchant_final_auto_strips_generic_company_noise():
    result = infer_merchant_final_auto("Some Local Shop GmbH")
    assert "GMBH" not in result


def test_infer_merchant_final_auto_falls_back_to_unknown_for_empty_text():
    assert infer_merchant_final_auto("GmbH KG Ltd") == "UNKNOWN"


def test_infer_spending_category_auto_groceries():
    assert infer_spending_category_auto("REWE") == "Groceries"
    assert infer_spending_category_auto("KAUFLAND") == "Groceries"


def test_infer_spending_category_auto_subscriptions():
    assert infer_spending_category_auto("NETFLIX") == "Subscriptions / Digital"


def test_infer_spending_category_auto_unknown_merchant_is_uncategorized():
    assert infer_spending_category_auto("SOME RANDOM SHOP") == "Uncategorized"


def test_preview_unique_joins_up_to_n_values():
    series = pd.Series(["a", "b", "c", "d", "e", "f"])
    assert preview_unique(series, n=3) == "a | b | c"


def test_preview_unique_drops_na_and_dedupes():
    series = pd.Series(["a", None, "a", "b"])
    assert preview_unique(series) == "a | b"


def test_classify_fixed_variable_from_category_excluded_is_internal():
    result = classify_fixed_variable_from_category(
        "Groceries", merchant_final="REWE", exclude_from_consumption=True
    )
    assert result == "Internal / exclude"


def test_classify_fixed_variable_from_category_housing_is_fixed():
    assert classify_fixed_variable_from_category("Housing / Rent") == "Fixed / recurring"


def test_classify_fixed_variable_from_category_gym_merchant_override():
    result = classify_fixed_variable_from_category(
        "Sports / Fitness", merchant_final="MCFIT CITY CENTER"
    )
    assert result == "Fixed / recurring"


def test_classify_fixed_variable_from_category_default_is_variable():
    assert classify_fixed_variable_from_category("Groceries") == "Variable"


def test_classify_fixed_variable_from_category_admin_is_one_off():
    assert classify_fixed_variable_from_category("Admin / Fees") == "One-off"


def test_extract_merchant_text_direct_debit_uses_creditor_without_iban():
    merchant, source = extract_merchant_text(
        "Jane Doe",
        "Sepa Direct Debit transfer to Example Gym GmbH (DE00123456780000000000)",
    )
    assert merchant == "EXAMPLE GYM GMBH"
    assert source == "description_direct_debit"


def test_extract_merchant_text_missing_name_falls_back_to_description():
    merchant, source = extract_merchant_text(float("nan"), "Gelateria Mej")
    assert merchant == "GELATERIA MEJ"
    assert source == "description_missing_name"


def test_extract_merchant_text_card_payment_uses_name():
    merchant, source = extract_merchant_text("Rewe Markt", "Rewe Markt")
    assert merchant == "REWE MARKT"
    assert source == "name"


def test_category_from_mcc_strong_weak_and_unknown():
    assert category_from_mcc(5411.0) == ("Groceries", "strong")
    assert category_from_mcc("5812") == ("Eating Out", "strong")
    assert category_from_mcc(5499) == ("Groceries", "weak")
    assert category_from_mcc(8999) == (None, "weak")
    assert category_from_mcc(float("nan")) == (None, None)
    assert category_from_mcc(1234) == (None, None)


def test_category_from_mcc_hotel_chain_range():
    assert category_from_mcc(3700) == ("Accommodation", "strong")


def test_infer_category_from_keywords_respects_exclusions():
    assert infer_category_from_keywords("ZARA GRILL") == "Eating Out"
    assert infer_category_from_keywords("ZARA") == "Clothing"
    assert infer_category_from_keywords("DM-DROGERIE MARKT") == "Drugstore"
    assert infer_category_from_keywords("SOMETHING UNKNOWN") is None


def _overrides(rows):
    return pd.DataFrame(rows, columns=OVERRIDE_COLUMNS)


def test_classify_strong_mcc_beats_keyword():
    # Name says "SPA" / "COOP"-like noise, the card network says parking.
    result = classify_consumption_transaction("ABACO PARKING HOTEL", "", 7523)
    assert result["spending_category"] == "Parking / Tolls"
    assert result["classification_source"] == "mcc"


def test_classify_keyword_beats_weak_mcc():
    result = classify_consumption_transaction("MYPROTEIN", "", 5499)
    assert result["spending_category"] == "Supplements / Nutrition"
    assert result["classification_source"] == "keyword"


def test_classify_weak_mcc_as_fallback():
    result = classify_consumption_transaction("NAPUL'E'", "", 5499)
    assert result["spending_category"] == "Groceries"
    assert result["classification_source"] == "mcc_weak"


def test_classify_no_mcc_uses_keyword_then_brand():
    result = classify_consumption_transaction("EXAMPLE IMMOBILIEN VERW. GMBH", "", None)
    assert result["spending_category"] == "Housing / Rent"
    brand = classify_consumption_transaction("PAYPAL EUROPE S.A.R.L.", "", None)
    assert brand["spending_category"] == "Online Shopping"
    assert brand["merchant_final"] == "PAYPAL"


def test_classify_unclassified_without_any_signal():
    result = classify_consumption_transaction("XYZ 123", "", None)
    assert result["spending_category"] == "Uncategorized"
    assert result["classification_source"] == "unclassified"


def test_classify_override_beats_everything():
    overrides = _overrides([
        ["EXAMPLE DOC", "EXAMPLE DOCTOR", "Health / Medication", "", "", ""],
    ])
    result = classify_consumption_transaction("EXAMPLE DOC LTD", "", 8699, overrides)
    assert result["merchant_final"] == "EXAMPLE DOCTOR"
    assert result["spending_category"] == "Health / Medication"
    assert result["classification_source"] == "override"


def test_classify_override_matches_description_and_can_exclude():
    overrides = _overrides([
        ["REFUND TO SELF", "", "Internal", "true", "", ""],
    ])
    result = classify_consumption_transaction("SOME NAME", "Refund to self", 5411, overrides)
    assert result["spending_category"] == "Internal"
    assert result["exclude_from_consumption"] is True


def test_classify_override_without_category_only_renames():
    overrides = _overrides([["CAFE X", "CAFE X (RENAMED)", "", "", "", ""]])
    result = classify_consumption_transaction("CAFE X 01", "", 5812, overrides)
    assert result["merchant_final"] == "CAFE X (RENAMED)"
    assert result["spending_category"] == "Eating Out"
    assert result["classification_source"] == "mcc"


def test_infer_merchant_final_auto_matches_mcdonalds_with_space():
    assert infer_merchant_final_auto("MC DONALDS 1752") == "MCDONALD'S"
    assert infer_merchant_final_auto("MCDONALDS 01723") == "MCDONALD'S"


def test_infer_merchant_final_auto_respects_word_boundaries():
    assert infer_merchant_final_auto("BAECKEREI HUBER") == "BAECKEREI HUBER"
    assert infer_merchant_final_auto("UBER *TRIP") == "UBER"
    assert infer_merchant_final_auto("DM-FIL. 123") == "DM"
    assert infer_merchant_final_auto("GOOGLE*CLOUD 59B73X") == "GOOGLE"


def test_classify_override_can_set_fixed_variable_only():
    overrides = _overrides([["EXAMPLE AI", "", "", "", "One-off", "not a subscription"]])
    result = classify_consumption_transaction("EXAMPLE AI", "", 5734, overrides)
    assert result["spending_category"] == "Subscriptions / Digital"
    assert result["classification_source"] == "mcc"
    assert result["fixed_variable_override"] == "One-off"


def test_classify_without_override_has_no_fixed_variable_override():
    result = classify_consumption_transaction("REWE", "", 5411)
    assert result["fixed_variable_override"] is None
