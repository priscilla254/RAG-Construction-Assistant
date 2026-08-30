from rag_assistant.approved_documents import (
    assign_parts,
    looks_like_index,
    map_document,
    page_part_votes,
)


def test_watermark_letters_do_not_vote_for_a_part():
    votes = page_part_votes("O N L I N E V E R S I O N\nEscape stair guidance")
    assert votes == {}


def test_requirement_code_in_the_header_votes_for_its_part():
    assert page_part_votes("B1\nAccess to storey exits").get("B") == 1.0


def test_appendix_heading_does_not_vote_for_another_part():
    # "Appendix C: Fire doorsets" sits inside Part B and must not be read
    # as the start of Part C.
    assert "C" not in page_part_votes("Appendix C: Fire doorsets\nA fusible link")


def test_document_title_votes_where_no_code_is_present():
    votes = page_part_votes("SITE PREPARATION AND RESISTANCE TO CONTAMINANTS\nWALLS")
    assert votes.get("C") == 1.0


def test_parts_cannot_run_backwards():
    # A stray Part A vote in the middle of Part B must not split Part B.
    votes = [{"A": 1.0}, {"B": 1.0}, {"B": 1.0}, {"A": 1.0}, {"B": 1.0}, {"C": 1.0}]
    assert assign_parts(votes) == ["A", "B", "B", "B", "B", "C"]


def test_sparse_votes_are_filled_across_a_part():
    votes = [{"A": 1.0}, {}, {}, {"B": 1.0}, {}, {}]
    assert assign_parts(votes) == ["A", "A", "A", "B", "B", "B"]


def test_index_page_is_detected():
    text = "\n".join(
        [
            "Hose laying distances 15.7",
            "Provision of fire mains 15.1",
            "Firefighting stairs construction 3.82",
            "Fire hydrants provision 14.8",
            "Fire mains design 13.5",
            "Fire penetration resistance 19",
            "Fire performance classification 21",
            "Fire resistance standards 7",
        ]
    )
    assert looks_like_index(text)


def test_diagram_of_numbers_is_not_mistaken_for_an_index():
    text = "Diagram 6 Map showing wind speeds\n" + "\n".join(
        ["24.5", "23.5", "22.5", "21.5", "23 22", "24.5", "23.5", "22.5", "21.5"]
    )
    assert not looks_like_index(text)


def test_wrapped_section_heading_is_rejoined():
    pages = [
        (1, "M1\nSection 3: Horizontal\ncirculation in buildings other than dwellings\n3.1 Text."),
    ]
    assert map_document(pages)[1].section_title == (
        "Section 3: Horizontal circulation in buildings other than dwellings"
    )
