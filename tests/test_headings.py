from rag_assistant.headings import looks_like_heading, trim_trailing_headings
from rag_assistant.hsg150 import HSG150_KNOWN_HEADINGS


def test_trim_trailing_headings_removes_next_section_title():
    text = (
        "3.28 Fixed ladders should not be provided as a means of escape.\n"
        "External walls adjacent to protected stairways"
    )
    assert trim_trailing_headings(text) == (
        "3.28 Fixed ladders should not be provided as a means of escape."
    )


def test_trim_removes_section_title_that_wrapped_across_lines():
    # Extraction drops the word "Section" and the title wraps, so the last
    # line is the bare continuation "safety".
    text = (
        "101 Make sure everybody knows and follows the rules relevant to them.\n"
        "3: Construction-phase health and\n"
        "safety"
    )
    assert trim_trailing_headings(text, HSG150_KNOWN_HEADINGS) == (
        "101 Make sure everybody knows and follows the rules relevant to them."
    )


def test_wrapped_section_continuation_is_not_a_heading_on_its_own():
    assert not looks_like_heading("safety", HSG150_KNOWN_HEADINGS)
    assert looks_like_heading(
        "3: Construction-phase health and safety", HSG150_KNOWN_HEADINGS
    )


def test_wrapping_body_text_is_not_mistaken_for_a_heading():
    # No full stop and short enough to pass the shape test; the comma and
    # the displaced footnote digits are what give these away as body text.
    assert not looks_like_heading("If isolations are undertaken by others, you should get")
    assert not looks_like_heading("Equipment Regulations 1998 (PUWER 1998). 8 8")


def test_citation_beginning_with_section_number_is_not_a_heading():
    citation = (
        "Section 59 (Drainage of building) of the Building Act 1984 allows "
        "the local authority to require the owner to render it innocuous"
    )
    assert not looks_like_heading(citation)
    assert looks_like_heading("Section 4 Fire safety")


def test_figure_captions_are_kept():
    text = (
        "234 Figure 23 shows some of the options for sloping-roof edge protection.\n"
        "Figure 23 Typical sloping-roof edge protection"
    )
    assert trim_trailing_headings(text, HSG150_KNOWN_HEADINGS) == text
