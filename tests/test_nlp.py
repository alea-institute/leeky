"""
Unit tests for the leeky.nlp module
"""
from leeky.nlp import (
    get_ws_token_boundaries,
    get_ws_tokens,
    get_spacy_tokens,
    get_spacy_stems,
    compare_token_sequences,
    compare_token_sets_jaccard,
    compare_tokens_spacy_similarity,
    compare_tokens_spacy_points,
)

# setup some fixture strings
text_1 = "This is a test."
text_2 = "This is a test. This is another test."
text_3 = "This is a test of the emergency broadcast system."
text_4 = "There is a testable emergency broadcasting system."
text_5 = "This is a test of the tokenization system."
text_6 = "Non, je ne regrette rien"
text_7 = "Purple terrier runs."
text_8 = "Maroon terrier runs."
text_9 = "Maroon poodle runs."

# setup some fixture lists
text_list_1 = [text_1, text_2, text_3, text_4, text_5]

# test the get_ws_token_boundaries method
def test_get_ws_token_boundaries():
    assert get_ws_token_boundaries(text_1) == [0, 4, 7, 9, 15]
    assert get_ws_token_boundaries(text_2) == [0, 4, 7, 9, 15, 20, 23, 31, 37]


def test_get_ws_tokens():
    assert get_ws_tokens(text_1) == ["This", "is", "a", "test."]
    assert get_ws_tokens(text_2) == [
        "This",
        "is",
        "a",
        "test.",
        "This",
        "is",
        "another",
        "test.",
    ]


def test_get_ws_tokens_lowercase():
    assert get_ws_tokens(text_1, lowercase=True) == ["this", "is", "a", "test."]
    assert get_ws_tokens(text_2, lowercase=True) == [
        "this",
        "is",
        "a",
        "test.",
        "this",
        "is",
        "another",
        "test.",
    ]


def test_get_spacy_tokens():
    assert get_spacy_tokens(text_1) == ["This", "is", "a", "test", "."]
    assert get_spacy_tokens(text_2) == [
        "This",
        "is",
        "a",
        "test",
        ".",
        "This",
        "is",
        "another",
        "test",
        ".",
    ]


def test_get_spacy_tokens_lowercase():
    assert get_spacy_tokens(text_1, lowercase=True) == ["this", "is", "a", "test", "."]
    assert get_spacy_tokens(text_2, lowercase=True) == [
        "this",
        "is",
        "a",
        "test",
        ".",
        "this",
        "is",
        "another",
        "test",
        ".",
    ]


def test_get_spacy_stems():
    assert get_spacy_stems(text_1) == ["this", "be", "a", "test", "."]
    assert get_spacy_stems(text_2) == [
        "this",
        "be",
        "a",
        "test",
        ".",
        "this",
        "be",
        "another",
        "test",
        ".",
    ]


def test_compare_token_sequences():
    assert compare_token_sequences(text_1, text_2) == 1.0
    assert compare_token_sequences(text_2, text_1) == 1.0
    assert (
        compare_token_sequences(text_1, text_3) == 0.75
    )  # 3/4 without spacy due to "test."
    assert compare_token_sequences(text_1, text_3, spacy=True) == 0.8  # 4/5 with spacy
    assert compare_token_sequences(text_1, text_4) == 0.5
    assert compare_token_sequences(text_1, text_4, spacy=True) == 0.4


def test_compare_token_sets_jaccard():
    assert compare_token_sets_jaccard(text_1, text_2) == 0.8  # another => 4/5
    assert compare_token_sets_jaccard(text_1, text_3) == 0.3  # test. => 3/5


def test_compare_tokens_spacy_similarity():
    assert compare_tokens_spacy_similarity(text_1, text_2) >= 0.9  # another => 4/5
    assert compare_tokens_spacy_similarity(text_1, text_3) >= 0.75  # test. => 4/5
    assert compare_tokens_spacy_similarity(text_1, text_6) <= 0.1  # fr vs. en

def test_compare_tokens_spacy_points():
    assert compare_tokens_spacy_points(text_1, text_2) >= 0.9  # another => 4/5
    assert compare_tokens_spacy_points(text_7, text_8) == 0.75
    assert compare_tokens_spacy_points(text_7, text_9) == 0.75

