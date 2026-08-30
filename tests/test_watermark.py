from rag_assistant.watermark import strip_watermark_line, strip_watermark_text


def test_strip_watermark_removes_spaced_online_version():
    line = "O N L I N E V E R S I O N B1"
    assert strip_watermark_line(line) == "B1"


def test_strip_watermark_text_drops_margin_codes():
    text = "O N L I N E V E R S I O N R38\nRegulation 38: Fire safety information"
    assert "O N L I N E" not in strip_watermark_text(text)
    assert "Regulation 38" in strip_watermark_text(text)
