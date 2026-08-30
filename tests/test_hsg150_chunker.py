from rag_assistant.chunker import DocumentChunker, find_numbered_paragraphs
from rag_assistant.hsg150 import (
    extract_subheading_from_gap,
    label_paragraphs,
    peel_trailing_subheading,
)


def _hsg150(pages):
    return {
        "filename": "hsg150.pdf",
        "title": "Health and safety in construction (HSG150)",
        "source_url": "https://www.hse.gov.uk/pubns/books/hsg150.htm",
        "doc_type": "broad_reference",
        "pages": [{"page_number": n, "text": t} for n, t in pages],
    }


def test_figure_lines_are_not_subheadings():
    gap = "Figure 4 A welfare unit with a rest area and drying room\n"
    assert extract_subheading_from_gap(gap) == ""


def test_tower_scaffolds_is_detected_as_subheading():
    gap = "Inspections and reports\n\nTower scaffolds\n"
    assert extract_subheading_from_gap(gap) == "Tower scaffolds"


def test_peel_glued_washing_facilities_from_paragraph_tail():
    text = (
        "50 A washbasin with water, soap and towels or dryers should be "
        "located close to the toilets. Washing facilities"
    )
    cleaned, heading = peel_trailing_subheading(text)
    assert heading == "Washing facilities"
    assert cleaned.endswith("toilets.")
    assert "Washing facilities" not in cleaned


def test_short_paragraphs_merge_within_topic():
    document = _hsg150(
        [
            (
                9,
                "1: Preparing for work\n"
                "Organising the work\n"
                "29 Decide who will supervise the work – check that they are "
                "adequately trained and experienced.\n"
                "30 When taking on workers, ask about the training they have "
                "received and ask to see certificates of training achievement.\n"
                "31 Make sure that firms coming onto site provide adequate "
                "supervision for their workers.\n"
                "Notifying the site to HSE\n"
                "34 HSE should be notified in writing before construction starts "
                "if the work is expected to last longer than 30 days or involve "
                "more than 500 person days of construction work on the project.",
            )
        ]
    )
    chunks = DocumentChunker().chunk(document)
    guidance = [c for c in chunks if c.chunk_type == "topic_guidance"]
    assert any(c.paragraph_numbers == "29-31" or c.paragraph_numbers.startswith("29") for c in guidance)
    merged = next(c for c in guidance if "29" in c.paragraph_numbers)
    assert merged.topic == "Organising the work"
    assert "Notifying" not in merged.text or "34" not in merged.paragraph_numbers
    # Must not merge across the topic boundary into para 34.
    assert not any(c.paragraph_numbers.endswith("-34") for c in guidance)


def test_tower_scaffold_paragraph_carries_subtopic_metadata():
    document = _hsg150(
        [
            (
                35,
                "3: Construction-phase health and safety\n"
                "Working at height\n"
                "Tower scaffolds\n"
                "154 Tower scaffolds (also known as mobile access scaffolds) are "
                "widely used and can provide an effective and safe means of gaining "
                "access to work at height while preventing falls. However inappropriate "
                "erection and misuse of tower scaffolds are the cause of numerous "
                "accidents each year.\n"
                "155 Before selecting or specifying a tower you must be satisfied "
                "that it is the most suitable item of equipment for the job on site.",
            )
        ]
    )
    chunks = DocumentChunker().chunk(document)
    first = next(c for c in chunks if c.paragraph_numbers.startswith("154"))
    assert first.chunk_type == "topic_guidance"
    assert first.topic == "Working at height"
    assert first.subtopic == "Tower scaffolds"
    assert "Tower scaffolds" in first.context_prefix
    assert "paragraph 154" in first.context_prefix or "paragraphs 154" in first.context_prefix
    assert not first.text.startswith("HSG150")


def test_site_access_chunks_have_section_and_prefix():
    document = _hsg150(
        [
            (
                11,
                "2: Setting up the site\n"
                "Site access\n"
                "38 There should be safe access onto and around the site for people "
                "and vehicles. Plan how vehicles will be kept clear of pedestrians, "
                "especially at site entrances where it may be necessary to provide "
                "doors or gates to achieve this segregation.\n"
                "39 Your plan should include how vehicles can be kept clear of "
                "pedestrians at vehicle loading and unloading areas on the site.\n"
                "Site boundaries\n"
                "40 Construction work should be fenced off and suitably signed to "
                "protect people especially children from site dangers and theft.",
            )
        ]
    )
    chunks = DocumentChunker().chunk(document)
    access = [c for c in chunks if c.topic == "Site access"]
    assert access
    assert all("Setting up the site" in c.section for c in access)
    assert all(c.context_prefix.startswith("HSG150:") for c in access)
    boundaries = [c for c in chunks if c.topic == "Site boundaries"]
    assert boundaries
    assert boundaries[0].paragraph_numbers.startswith("40")


def test_sentence_case_subheading_on_its_own_line_is_detected():
    # HSE sets many headings in sentence case, so a Title-Case ratio test
    # misses them.
    gap = "Monitoring and reviewing\n\nHealth and safety competence\n"
    assert extract_subheading_from_gap(gap) == "Health and safety competence"


def test_next_section_title_does_not_bleed_into_the_previous_paragraph():
    document = _hsg150(
        [
            (
                20,
                "2: Setting up the site\n"
                "Site rules\n"
                "100 Clients may insist on certain safety precautions, especially "
                "where their business continues at the premises while construction "
                "work is in progress. It may assist everyone if site rules are "
                "applied.\n"
                "101 Make it clear where your site rules apply and where the client "
                "premises rules apply. Make sure everybody knows and follows the "
                "rules relevant to them.\n"
                "3: Construction-phase health and\n"
                "safety",
            )
        ]
    )
    chunks = DocumentChunker().chunk(document)
    tail = next(c for c in chunks if "101" in c.paragraph_numbers)
    assert tail.text.rstrip().endswith("rules relevant to them.")
    assert "Construction-phase" not in tail.text


def test_label_paragraphs_forward_fills_subtopic():
    text = (
        "Working at height\n"
        "Tower scaffolds\n"
        "154 First paragraph about towers here with enough words to stand alone "
        "as a meaningful unit of guidance for the site.\n"
        "155 Second paragraph still under towers with enough words to clear the "
        "minimum length used by the merge floor in the chunker."
    )
    starts = find_numbered_paragraphs(text, max_gap=20)
    labels = label_paragraphs(text, starts)
    assert labels[0].subtopic == "Tower scaffolds"
    assert labels[1].subtopic == "Tower scaffolds"
    assert labels[0].topic == "Working at height"
