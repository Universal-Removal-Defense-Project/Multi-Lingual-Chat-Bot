"""Knowledge-base retrieval across scripts and accents (Issue #35)."""

from unittest.mock import patch

import pytest

import knowledge
import rag


def top_hit(docs, query):
    """Index of the best-matching doc for query, or None if nothing matches."""
    hits = rag.TfidfIndex(docs).search(query, k=1)
    return hits[0][0] if hits else None


# --- Rui's reproduction from #35 ---

def test_chinese_text_is_tokenized_and_found():
    assert rag.tokenize("驱逐出境的风险")
    assert rag.TfidfIndex(["移民法规定驱逐出境的相关流程"]).search("驱逐出境")


@pytest.mark.parametrize(
    "text, whole_word",
    [
        ("riesgo de expulsión", "expulsión"),  # Spanish
        ("expulsão", "expulsão"),              # Portuguese
        ("Ausländer", "ausländer"),            # German
        ("früher", "früher"),
        ("trục xuất", "trục"),                 # Vietnamese
        ("हिन्दी", "हिन्दी"),                  # Hindi: vowel signs stay in the word
    ],
)
def test_accented_and_indic_words_stay_whole(text, whole_word):
    assert whole_word in rag.tokenize(text)


# --- Retrieval works in every script ---

@pytest.mark.parametrize(
    "language, doc, distractor, query",
    [
        ("Chinese", "申请庇护必须在入境后一年内提出", "工作许可需要提交表格", "庇护申请"),
        ("Japanese", "難民認定の申請は入国後すぐに行ってください", "就労許可の書類について", "難民認定"),
        ("Korean", "망명 신청서를 제출하세요", "취업 허가 안내", "신청서"),
        ("Arabic", "طَلَبُ اللُّجُوءِ يَجِبُ تَقْدِيمُهُ خِلَالَ عَامٍ", "معلومات عن تصريح العمل", "طلب اللجوء"),
        ("Hindi", "शरण के लिए आवेदन एक साल के भीतर करना होगा", "काम की अनुमति के बारे में जानकारी", "शरण आवेदन"),
        ("Russian", "Заявление на убежище нужно подать в течение года", "Информация о разрешении на работу", "убежище"),
        ("Farsi", "درخواست پناهندگی باید ظرف یک سال ارائه شود", "اطلاعات مجوز کار", "پناهندگی"),
        ("Thai", "การขอลี้ภัยต้องยื่นภายในหนึ่งปี", "ข้อมูลใบอนุญาตทำงาน", "ลี้ภัย"),
        ("Vietnamese", "Đơn xin tị nạn phải nộp trong vòng một năm", "Thông tin về giấy phép lao động", "tị nạn"),
    ],
)
def test_query_finds_the_right_document_in_each_script(language, doc, distractor, query):
    assert top_hit([distractor, doc], query) == 1, language


@pytest.mark.parametrize(
    "doc, query",
    [
        ("El riesgo de expulsión es alto", "expulsion"),             # typed without accents
        ("Đơn xin tị nạn phải nộp trong vòng một năm", "ti nan"),    # Vietnamese without tones
    ],
)
def test_queries_without_accents_still_match(doc, query):
    assert top_hit(["Información sobre el trabajo", doc], query) == 1


@pytest.mark.parametrize("query", ["I-589", "I-٥٨٩", "I-۵۸۹", "Ｉ－５８９"])
def test_form_numbers_match_in_any_digit_script(query):
    assert top_hit(["Work permits use Form I-765", "Submit Form I-589 within one year"], query) == 1


# --- Normalisation details ---

def test_english_tokenization_is_unchanged():
    assert rag.tokenize("Form I-765, filing! don't a_b") == [
        "form", "i", "765", "filing", "don", "t", "a", "b",
    ]


def test_invisible_characters_inside_words_are_ignored():
    assert rag.tokenize("immi­gration") == ["immigration"]          # soft hyphen from PDFs
    assert rag.tokenize("كـتـاب") == rag.tokenize("كتاب")                # Arabic tatweel
    assert rag.tokenize("می‌خواهم") == rag.tokenize("میخواهم")      # Persian zero-width joiner


def test_symbols_and_stray_marks_produce_no_tokens():
    assert rag.tokenize("👍 ́ ... ___") == []


# --- End to end through the knowledge base ---

def test_knowledge_search_finds_chinese_pdf(tmp_path):
    db = tmp_path / "knowledge.db"
    pages = [(1, "移民法规定驱逐出境的相关流程"), (2, "工作许可的申请步骤")]
    with patch.object(knowledge, "extract_pages", lambda data: pages):
        knowledge.add_pdf("guide-zh.pdf", b"%PDF-fake", path=db)
    hits = knowledge.search("驱逐出境", k=1, path=db)
    assert hits and hits[0]["page"] == 1
