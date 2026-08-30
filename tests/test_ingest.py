from rag_assistant.ingest import (
    _lines_from_words,
    clean_text,
    find_column_gutter,
    strip_headers_footers,
)


def _word(text, x0, x1, top):
    return {"text": text, "x0": x0, "x1": x1, "top": top}


def test_strips_hsg141_headers_on_exact_line_match_only():
    page = (
        "PART 1 PLANNING: THE PRE-CONSTRUCTION PHASE\n"
        "12\n"
        "Site rules apply here.\n"
        "Electrical Safety on Construction Sites"
    )
    cleaned = strip_headers_footers(page, "hsg141.pdf")
    assert "PART 1 PLANNING" not in cleaned
    assert "Electrical Safety on Construction Sites" not in cleaned
    assert cleaned == "Site rules apply here."


def test_back_cover_blurb_survives_partial_header_wording():
    page = (
        "Electrical safety on\n"
        "construction sites\n"
        "Electricity can kill. Every year, the use of electricity on construction "
        "sites results in accidents."
    )
    cleaned = strip_headers_footers(page, "hsg141.pdf")
    assert "Electricity can kill." in cleaned


def test_strips_hsg150_running_header_and_page_of():
    page = "Health and Safety\nExecutive\nIntro body text.\nPage 1 of 141"
    cleaned = strip_headers_footers(page, "hsg150.pdf")
    assert cleaned == "Intro body text."


def test_clean_text_joins_hyphenated_line_breaks():
    assert clean_text("construc-\ntion sites") == "construction sites"


def test_clean_text_keeps_real_numbers():
    assert "1998" in clean_text("Work Equipment Regulations 1998 (PUWER 1998).")


def test_clean_text_strips_watermark_lines():
    cleaned = clean_text("O N L I N E V E R S I O N\n3.24 Escape stairs should comply.")
    assert "O N L I N E" not in cleaned
    assert "3.24 Escape stairs should comply." in cleaned


def test_lines_are_rebuilt_in_reading_order():
    words = [
        _word("safety", 120, 160, 40.0),
        _word("Electrical", 60, 110, 40.4),
        _word("on site", 60, 100, 60.0),
    ]
    assert _lines_from_words(words) == ["Electrical safety", "on site"]


def test_full_width_text_is_not_split_into_columns():
    words = [
        _word(f"w{i}", 40 + (i % 10) * 55, 90 + (i % 10) * 55, float(i // 10) * 12)
        for i in range(60)
    ]
    assert find_column_gutter(words, 600.0) is None


def test_gutter_layout_is_detected_as_two_columns():
    words = []
    for row in range(12):
        for col in range(5):
            words.append(_word("left", 40 + col * 40, 70 + col * 40, row * 12.0))
            words.append(_word("right", 320 + col * 40, 350 + col * 40, row * 12.0))
    gutter = find_column_gutter(words, 600.0)
    assert gutter is not None
    assert 240 < gutter < 320


def test_contents_page_number_strip_is_not_treated_as_a_column():
    words = []
    for row in range(20):
        for col in range(4):
            words.append(_word("entry", 40 + col * 45, 80 + col * 45, row * 12.0))
        words.append(_word(str(row), 540, 552, row * 12.0))
    assert find_column_gutter(words, 600.0) is None
