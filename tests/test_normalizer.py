from turkish_rag_assistant.normalizer import normalize_turkish


def test_normalize_turkish_dotted_capital_i():
    assert normalize_turkish("İSTANBUL") == "istanbul"


def test_normalize_turkish_dotless_capital_i():
    assert normalize_turkish("ISPARTA") == "ısparta"


def test_normalize_turkish_mixed_case():
    assert normalize_turkish("İzmir ve Iğdır") == "izmir ve ığdır"


def test_normalize_leaves_lowercase_unchanged():
    assert normalize_turkish("ankara") == "ankara"


def test_normalize_turkish_empty():
    assert normalize_turkish("") == ""


def test_normalize_turkish_other_letters():
    assert normalize_turkish("ÜSKÜDAR ÖĞRENCİ") == "üsküdar öğrenci"
