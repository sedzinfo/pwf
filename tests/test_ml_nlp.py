import numpy as np
import pytest

from conftest import assert_matches_reference
from pwf import ml_nlp

# Penn Treebank tags, the tag set of both Apache OpenNLP (rwf) and NLTK (pwf)
PENN_TAGS = {"CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT",
             "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP",
             "VBZ", "WDT", "WP", "WP$", "WRB", ".", ",", ":", "(", ")", "``", "''", "$", "#", "-LRB-", "-RRB-"}


@pytest.fixture(scope="module")
def nlp_texts():
    from conftest import _load_reference
    return _load_reference("nlp_texts")["text"].tolist()


def test_clear_text(nlp_texts, reference):
    assert ml_nlp.clear_text(nlp_texts) == reference("clear_text", keep_default_na=False)["text"].tolist()


def test_clear_stopwords(nlp_texts, reference):
    assert ml_nlp.clear_stopwords(nlp_texts) == reference("clear_stopwords", keep_default_na=False)["text"].tolist()


def test_clear_text_accepts_one_string():
    assert ml_nlp.clear_text("Hello, World 42!") == ["hello world"]


def test_compute_text_similarity(nlp_texts, reference):
    expected = reference("compute_text_similarity")
    words = [t.split(" ") for t in nlp_texts]
    for _, row in expected.iterrows():
        result = ml_nlp.compute_text_similarity(words[int(row["i"]) - 1], words[int(row["j"]) - 1])
        assert_matches_reference(result, expected.loc[[row.name]].drop(columns=["i", "j"]).reset_index(drop=True))


def test_compute_tversky_index(reference):
    cases = [(["a", "b", "c", "d"], ["b", "c", "d", "e"], 0.5, 0.5), (["a", "b", "c", "d"], ["b", "c", "d", "e"], 1, 1),
             (["a", "b", "c", "d"], ["b", "c", "d", "e"], 0.9, 0.1), (["a", "a", "b"], ["b", "b", "c"], 0.5, 0.5),
             ([1, 2, 3], ["2", "3", "4"], 0.5, 0.5), (["a", "b"], ["c", "d"], 0.5, 0.5)]
    result = [ml_nlp.compute_tversky_index(x, y, alpha=a, beta=b) for x, y, a, b in cases]
    np.testing.assert_allclose(result, reference("compute_tversky_index")["value"], rtol=1e-12)


def test_stat_word_char(nlp_texts, reference):
    try:
        result = ml_nlp.stat_word_char(nlp_texts)
    except FileNotFoundError:
        pytest.skip("no hunspell en_US dictionary on this machine")
    assert_matches_reference(result, reference("stat_word_char"))


def test_tag_pos(nlp_texts, reference):
    """NLTK and OpenNLP use the same tag set but different models: same tokens, mostly the same tags."""
    result = ml_nlp.tag_pos(nlp_texts[2:5])
    expected = reference("tag_pos")["tag"].tolist()
    assert len(result["POStags"]) == len(expected)
    assert set(result["POStags"]) <= PENN_TAGS
    agreement = np.mean([a == b for a, b in zip(result["POStags"], expected)])
    assert agreement > 0.9
    assert result["POStagged"].split(" ")[0] == "All/" + result["POStags"][0]
