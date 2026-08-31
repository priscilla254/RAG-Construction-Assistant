from rag_assistant.approved_documents import map_document
from rag_assistant.chunker import DocumentChunker, flatten_document
from rag_assistant.publisher_matter import looks_like_publisher_matter


ORDER_PAGE = """
Published by NBS, part of RIBA Enterprises Ltd.
To order:
Tel: 0845 300 9924
Fax: 0845 300 9924
www.thebuildingregs.com
"""

HSE_BOOKS_COVER = """
HSE Books
PO Box 1999
Sudbury
Suffolk CO10 2WA
Tel: 01787 881165
Fax: 01787 313995
Further information
"""


def test_nbs_order_page_is_publisher_matter():
    assert looks_like_publisher_matter(ORDER_PAGE)


def test_hse_books_cover_is_publisher_matter():
    assert looks_like_publisher_matter(HSE_BOOKS_COVER)


def test_clause_page_is_not_publisher_matter():
    text = (
        "1.5 Have level treads on steps, ensuring that the rise and going "
        "of each step are consistent throughout a flight of steps.\n"
        "1.6 The rise and going of each step are consistent throughout a flight."
    )
    assert not looks_like_publisher_matter(text)


def test_numbered_hsg_paragraph_is_not_publisher_matter():
    text = (
        "148 Guard rails, toe boards and other similar barriers should be provided.\n"
        "149 They should include a main guard rail at least 950 mm above any edge."
    )
    assert not looks_like_publisher_matter(text)


def test_mad_chunker_drops_order_pages():
    document = {
        "filename": "The_Merged_Approved_Documents_Oct24.pdf",
        "title": "The Merged Approved Documents",
        "source_url": "",
        "doc_type": "regulatory",
        "pages": [
            {"page_number": 2, "text": ORDER_PAGE},
            {
                "page_number": 95,
                "text": (
                    "B1\nSection 2: Means of escape – dwellinghouses\n"
                    "2.1 See Diagram 2.1a. All habitable rooms should have an opening "
                    "onto a hall.\n"
                    "2.2 Where served by only one stair, habitable rooms should have "
                    "an emergency escape window."
                ),
            },
        ],
    }
    chunks = DocumentChunker().chunk(document)
    blob = " ".join(chunk.text for chunk in chunks)
    assert "0845 300 9924" not in blob
    assert "2.1 See Diagram" in blob
    assert map_document([(2, ORDER_PAGE), (95, document["pages"][1]["text"])])[2].is_index


def test_flatten_skips_publisher_pages_for_hsg():
    document = {
        "filename": "hsg150.pdf",
        "title": "Health and safety in construction",
        "source_url": "",
        "doc_type": "broad_reference",
        "pages": [
            {
                "page_number": 10,
                "text": "148 Guard rails should be provided whenever practicable.\n"
                "149 They should be strong and rigid enough to prevent people from falling.",
            },
            {"page_number": 141, "text": HSE_BOOKS_COVER},
        ],
    }
    flat = flatten_document(document)
    assert "HSE Books" not in flat.text
    assert "148 Guard rails" in flat.text
    chunks = DocumentChunker().chunk(document)
    blob = " ".join(chunk.text for chunk in chunks)
    assert "PO Box 1999" not in blob
