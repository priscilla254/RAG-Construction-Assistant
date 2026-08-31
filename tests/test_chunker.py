from rag_assistant.chunker import (
    Chunk,
    DocumentChunker,
    find_numbered_paragraphs,
    flatten_document,
)


def _document(pages, doc_type="procedural_guidance"):
    return {
        "filename": "hsg141.pdf",
        "title": "Electrical safety on construction sites (HSG141)",
        "source_url": "https://www.hse.gov.uk/pubns/books/hsg141.htm",
        "doc_type": doc_type,
        "pages": [
            {"page_number": number, "text": text} for number, text in pages
        ],
    }


def test_chunker_init():
    chunker = DocumentChunker(chunk_size=400, chunk_overlap=60)
    assert chunker.chunk_size == 400
    assert chunker.chunk_overlap == 60


def test_chunker_defaults_come_from_config_dataclass():
    chunker = DocumentChunker()
    assert chunker.chunk_size == 400
    assert chunker.chunk_overlap == 60


def test_voltages_are_not_mistaken_for_paragraph_numbers():
    text = (
        "1 Electricity can kill.\n"
        "110 V AC distribution points are provided.\n"
        "2 This guidance is aimed at those responsible.\n"
        "13 A socket outlets should not be used.\n"
        "3 This guidance explains what to do.\n"
    )
    assert [number for number, _ in find_numbered_paragraphs(text)] == [1, 2, 3]


def test_page_numbers_are_preserved_from_the_source_pdf():
    document = _document([(7, "1 First paragraph here."), (8, "2 Second paragraph here.")])
    chunks = DocumentChunker().chunk(document)
    assert [c.page_number for c in chunks] == [7, 8]


def test_next_subheading_does_not_bleed_onto_the_previous_paragraph():
    document = _document(
        [
            (
                7,
                "16 The distribution network operator should be contacted before "
                "any work starts near their equipment.\n"
                "Underground services\n"
                "17 Buried cables can be damaged during excavation work.",
            )
        ]
    )
    chunks = DocumentChunker().chunk(document)
    first = chunks[0]
    assert first.text.rstrip().endswith("near their equipment.")
    assert "Underground services" not in first.text


def test_case_study_is_split_into_its_own_chunk():
    document = _document(
        [
            (
                9,
                "27 On some industrial sites overhead power lines may be present.\n"
                "Case study\n"
                "A client wanted to build a structure on his land.\n"
                "28 When working in an existing building, services are present.",
            )
        ]
    )
    chunks = DocumentChunker().chunk(document)
    types = [c.chunk_type for c in chunks]
    assert types == ["procedural_guidance", "case_study", "procedural_guidance"]
    assert "A client wanted to build" in chunks[1].text
    assert "case study" in chunks[1].section


def test_chunks_carry_citation_metadata():
    document = _document([(5, "1 Electricity can kill on construction sites.")])
    chunk = DocumentChunker().chunk(document)[0]
    assert chunk.chunk_id == "hsg141_0001"
    assert chunk.source_doc == "hsg141.pdf"
    assert chunk.source_url.startswith("https://")
    assert chunk.section.startswith("Introduction")


def test_generic_chunker_windows_long_text_with_overlap():
    words = " ".join(f"w{i}" for i in range(250))
    # doc_types without a structural strategy fall through to word windows.
    document = _document([(1, words)], doc_type="unknown")
    # 250 words, 100-word windows stepping by 80: 0-99, 80-179, 160-249.
    chunks = DocumentChunker(chunk_size=100, chunk_overlap=20).chunk(document)
    assert len(chunks) == 3
    assert all(c.chunk_type == "generic" for c in chunks)
    first = chunks[0].text.split()
    second = chunks[1].text.split()
    assert first[80:] == second[:20]


def _regulatory(pages):
    return {
        "filename": "The_Merged_Approved_Documents_Oct24.pdf",
        "title": "The Merged Approved Documents",
        "source_url": "https://www.gov.uk/",
        "doc_type": "regulatory",
        "pages": [{"page_number": n, "text": t} for n, t in pages],
    }


def test_clause_chunk_carries_part_section_and_clause_number():
    document = _regulatory(
        [
            (
                1,
                "B1\nSection 3: Construction of escape stairs\n"
                "3.24 The flights and landings of escape stairs should be constructed of "
                "materials achieving class A2-s3, d2 or better in all of the following "
                "situations, unless the building is an office building of two storeys.\n"
                "3.25 Further guidance on fire performance is given in Appendix B of this "
                "approved document, which should be read alongside the tables provided.",
            )
        ]
    )
    chunks = DocumentChunker().chunk(document)
    first = chunks[0]
    assert first.chunk_type == "regulatory_clause"
    assert first.part_code == "B"
    assert first.clause_number == "3.24"
    assert first.section == "Section 3: Construction of escape stairs"
    assert first.context_prefix == (
        "Approved Document B: Fire safety \u2014 "
        "Section 3: Construction of escape stairs \u2014 clause 3.24"
    )
    assert not first.text.startswith("Approved Document")
    assert first.embedding_text.startswith(first.context_prefix)


def test_chunk_round_trips_through_dict():
    original = Chunk(
        chunk_id="hsg150_0001",
        source_doc="hsg150.pdf",
        title="Health and safety in construction",
        source_url="https://www.hse.gov.uk/pubns/books/hsg150.htm",
        doc_type="broad_reference",
        chunk_type="topic_guidance",
        section="Section 2: Setting up the site",
        page_number=20,
        text="101 Follow the rules.",
        context_prefix="HSG150 — Site rules — paragraph 101",
        topic="Site rules",
        paragraph_numbers="101",
    )
    restored = Chunk.from_dict(original.to_dict())
    assert restored == original
    assert restored.embedding_text.startswith("HSG150")


def test_short_clauses_are_merged_and_keep_both_numbers():
    document = _regulatory(
        [(1, "B1\nSection 2: Walls\n2.1 See Table 2.2.\n2.2 See Diagram 4.\n2.3 Comply.")]
    )
    chunk = DocumentChunker().chunk(document)[0]
    assert chunk.clause_number.startswith("2.1-")
    assert "clauses 2.1-" in chunk.context_prefix


def test_long_clause_is_windowed_and_only_the_first_window_keeps_the_clause_tag():
    body = " ".join(f"word{i}" for i in range(300))
    document = _regulatory([(1, f"B1\nSection 1: Scope\n1.1 The following applies. {body}")])
    chunks = DocumentChunker(chunk_size=100, chunk_overlap=20).chunk(document)
    assert len(chunks) > 1
    assert chunks[0].clause_number == "1.1"
    assert "clause 1.1" in chunks[0].context_prefix
    assert all(c.chunk_type == "reference_material" for c in chunks[1:])
    assert all(c.clause_number == "" for c in chunks[1:])


def test_runaway_clause_falls_back_to_reference_material():
    body = " ".join(f"word{i}" for i in range(3000))
    document = _regulatory([(1, f"B1\nSection 1: Scope\n1.1 Undetected numbering. {body}")])
    chunks = DocumentChunker().chunk(document)
    assert all(c.chunk_type == "reference_material" for c in chunks)
    assert all(c.clause_number == "" for c in chunks)


def test_dimensions_are_not_mistaken_for_clause_numbers():
    document = _regulatory(
        [
            (
                1,
                "B1\nSection 3: Stairs\n"
                "3.10 The going should be at least 250mm as set out below.\n"
                "1.5 m clear width is required at the landing point of the stair.\n"
                "3.11 The rise should be no more than 170mm for each step provided.",
            )
        ]
    )
    numbers = [c.clause_number for c in DocumentChunker().chunk(document)]
    assert "1.5" not in numbers


def test_diagram_dimension_does_not_hijack_the_clause_sequence():
    # "7.5m max." on a diagram leaves "7.5" starting a line. Accepting it
    # next to clause 3.27 would reject every real clause behind it.
    document = _regulatory(
        [
            (
                1,
                "B1\nSection 3: Escape routes\n"
                "3.27 Flats should be served by a common stair as described in this clause.\n"
                "7.5 Flat Flat Flat Stair lobby with no flat opening directly into it\n"
                "3.28 Escape routes over flat roofs should comply with all of the following.",
            )
        ]
    )
    numbers = [c.clause_number for c in DocumentChunker().chunk(document)]
    assert "7.5" not in numbers
    assert any("3.28" in n for n in numbers)


def test_regulation_page_is_not_glued_to_preceding_clause():
    document = _regulatory(
        [
            (
                194,
                "B1\nSection 16: Venting\n"
                "16.14 Natural smoke outlet shafts should be separated from each other "
                "using construction of class A1 rating and fire resistance at least equal "
                "to that of the storeys they serve, where the shafts are either of the following. "
                "a. From different compartments of the same basement storey. "
                "b. From different basement storeys.",
            ),
            (
                195,
                "O N L I N E V E R S I O N R38\n"
                "Regulation 38: Fire safety information\n"
                "38. (1) This regulation applies where building work consists of or includes "
                "the erection or extension of a relevant building.",
            ),
            (
                196,
                "B1\nSection 17: Fire safety information\n"
                "17.1 For building work involving the erection or extension of a relevant building, "
                "fire safety information should be given to the responsible person.",
            ),
        ]
    )
    chunks = DocumentChunker().chunk(document)
    clause_16_14 = [c for c in chunks if c.clause_number == "16.14"]
    assert len(clause_16_14) == 1
    assert "Regulation 38" not in clause_16_14[0].text
    assert not clause_16_14[0].text.startswith("the building")
    assert any(c.chunk_type == "reference_material" for c in chunks)


def test_short_clauses_are_not_merged_across_sections():
    document = _regulatory(
        [
            (
                1,
                "E\nSection 2: Walls\n"
                "2.163 There are important details concerning junctions with separating floors.",
            ),
            (
                2,
                "E\nSection 3: Floors\n"
                "Introduction\n"
                "3.1 This Section gives examples of floor types which should achieve the standards.",
            ),
        ]
    )
    chunks = DocumentChunker().chunk(document)
    numbers = [c.clause_number for c in chunks if c.chunk_type == "regulatory_clause"]
    assert "2.163-3.1" not in numbers
    assert "2.163" in numbers
    assert "3.1" in numbers


def test_index_pages_are_dropped():
    index_text = "\n".join(
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
    document = _regulatory(
        [(1, "B1\nSection 1: Scope\n1.1 A real clause with enough words to stand alone here."),
         (2, index_text)]
    )
    chunks = DocumentChunker().chunk(document)
    assert all(c.page_number == 1 for c in chunks)


def test_flatten_document_skips_empty_pages():
    doc = flatten_document(_document([(1, "First page."), (2, ""), (3, "Third page.")]))
    assert doc.page_numbers == [1, 3]
    assert doc.page_for_offset(len("First page.") + 2) == 3
