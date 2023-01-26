"""
This module provides simple NLP methods, mostly wrapping spacy,
handle the following tasks:
  * Get token boundaries and tokens in text (non-spacy)
  * Get tokens in text (spacy)
  * Get stems/lemmas in text (spacy)
  * Compare the token or stem/lemma sets
"""

# packages
import numpy
import spacy

# setup model
# NB: make sure you have the model installed: $ python3 -m spacy download en_core_web_lg
# NB: we can _sm with tensors or _lg to get better "word vectors" for similarity
nlp = spacy.load("en_core_web_lg")


def get_ws_token_boundaries(text: str) -> list[int]:
    """
    This method returns the token splits for a given text based on very simple
    non-consecutive whitespace tokenization.
    """
    if len(text) == 0:
        return []

    # get whitespace positions
    whitespace_index = [i for i, c in enumerate(text) if c.isspace()]

    # remove consecutive whitespace positions
    whitespace_index = [
        whitespace_index[i]
        for i in range(len(whitespace_index))
        if i == 0 or whitespace_index[i] != whitespace_index[i - 1] + 1
    ]

    # return the whitespace positions with initial position
    return [0] + whitespace_index + [len(text)]


def get_ws_tokens(text: str, lowercase: bool = False) -> list[str]:
    """
    This method returns the tokens for a given text based on very simple
    non-consecutive whitespace tokenization.
    """
    # get the token splits
    token_splits = get_ws_token_boundaries(text)

    # get the tokens
    tokens = [
        text[token_splits[i] : token_splits[i + 1]].strip().lower()
        if lowercase
        else text[token_splits[i] : token_splits[i + 1]].strip()
        for i in range(len(token_splits) - 1)
    ]

    # return the tokens
    return tokens


def get_spacy_tokens(text: str, lowercase: bool = False) -> list[str]:
    """
    This method returns the tokens for a given text based on spacy's tokenization.
    """
    # get the tokens
    tokens = [token.text.lower() if lowercase else token.text for token in nlp(text) if token.text.strip() != ""]

    # return the tokens
    return tokens


def get_spacy_stems(text: str, lowercase: bool = False) -> list[str]:
    """
    This method returns the stems for a given text based on spacy's tokenization.
    """
    # get the stems
    stems = [token.lemma_.lower() if lowercase else token.lemma_ for token in nlp(text) if token.lemma_.strip() != ""]

    # return the stems
    return stems


def compare_token_sequences(
    text_a: str,
    text_b: str,
    lowercase: bool = False,
    spacy: bool = False,
    stem: bool = False,
) -> float:
    """
    This method returns the similarity between the token sequences of two texts by
    comparing the sequence elements directly.

    Args:
        text_a: The first text.
        text_b: The second text.
        lowercase: Whether to lowercase the tokens.
        spacy: Whether to use spacy's tokenization.
        stem: Whether to use spacy's lemmatization; does not have an effect if spacy is False.

    Returns:
        The proportion of the text tokens/stems that are identical in sequence.
    """

    # get tokens
    if spacy:
        if stem:
            tokens_a = get_spacy_stems(text_a, lowercase)
            tokens_b = get_spacy_stems(text_b, lowercase)
        else:
            tokens_a = get_spacy_tokens(text_a, lowercase)
            tokens_b = get_spacy_tokens(text_b, lowercase)
    else:
        tokens_a = get_ws_tokens(text_a, lowercase)
        tokens_b = get_ws_tokens(text_b, lowercase)

    # get the proportion of tokens in the shorter text that are identical in longer text
    token_count = min(len(tokens_a), len(tokens_b))
    identical_token_count = sum(tokens_a[i] == tokens_b[i] for i in range(token_count))
    return identical_token_count / float(token_count)


def compare_token_sets_jaccard(
    text_a: str,
    text_b: str,
    lowercase: bool = False,
    spacy: bool = False,
    stem: bool = False,
) -> float:
    """
    This method returns the Jaccard similarity between the token sets of two texts.

    Args:
        text_a: The first text.
        text_b: The second text.
        lowercase: Whether to lowercase the tokens.
        spacy: Whether to use spacy's tokenization.
        stem: Whether to use spacy's lemmatization; does not have an effect if spacy is False.

    Returns:
        The Jaccard similarity between the token sets of the two texts.
    """
    # get the tokens
    if spacy:
        if stem:
            tokens_a = get_spacy_stems(text_a, lowercase)
            tokens_b = get_spacy_stems(text_b, lowercase)
        else:
            tokens_a = get_spacy_tokens(text_a, lowercase)
            tokens_b = get_spacy_tokens(text_b, lowercase)
    else:
        tokens_a = get_ws_tokens(text_a, lowercase)
        tokens_b = get_ws_tokens(text_b, lowercase)

    # get the Jaccard similarity
    jaccard_similarity = len(set(tokens_a).intersection(set(tokens_b))) / len(
        set(tokens_a).union(set(tokens_b))
    )

    # return the Jaccard similarity
    return jaccard_similarity


def compare_tokens_spacy_similarity(
    text_a: str,
    text_b: str,
) -> float:
    """
    This method returns the similarity between the tokens of two texts by comparing
    the tokens directly with spacy's tensor/embedding similarity.

    NB: _sm and _lg/_trf behave differently.

    Args:
        text_a: The first text.
        text_b: The second text.

    Returns:
        The proportion of the text tokens that are identical.
    """

    # get the nlp docs
    doc_a = nlp(text_a)
    doc_b = nlp(text_b)

    # return the similarity docs from the smaller doc
    if len(doc_a) < len(doc_b):
        return doc_a.similarity(doc_b)
    else:
        return doc_b.similarity(doc_a)


def compare_tokens_spacy_points(
    text_a: str,
    text_b: str,
    epsilon: float = 0.25,
) -> float:
    """
    This method returns the similarity between the tokens of two texts by calculating
    the number of tokens that are within \eps in the embedding space.


    Args:
        text_a: The first text.
        text_b: The second text.
        epsilon: The epsilon for the embedding space.

    Returns:
        The proportion of the text tokens that are approximately identical in embedding space.
    """

    # get the nlp docs
    doc_a = nlp(text_a)
    doc_b = nlp(text_b)

    # get unique set of tokens
    tokens_a = list(set([token.text for token in doc_a]))
    tokens_b = list(set([token.text for token in doc_b]))

    # token points
    token_points = {}

    # get the token points for a
    for token in doc_a:
        token_points[token.text] = token.vector

    # get the token points for b
    for token in doc_b:
        token_points[token.text] = token.vector

    # get the overlapping token pairs
    num_overlapping_tokens = 0
    denominator = len(set(tokens_a).union(set(tokens_b)))
    for i in range(len(tokens_a)):
        for j in range(len(tokens_b)):
            # skip zero vectors
            if (numpy.linalg.norm(token_points[tokens_a[i]]) == 0
                or numpy.linalg.norm(token_points[tokens_b[j]]) == 0):
                continue

            # get cosine similarity with numpy
            similarity = numpy.dot(
                token_points[tokens_a[i]], token_points[tokens_b[j]]
            ) / (
                numpy.linalg.norm(token_points[tokens_a[i]])
                * numpy.linalg.norm(token_points[tokens_b[j]])
            )

            # if the distance is less than epsilon, add to the count
            if 1.0 - similarity < epsilon:
                num_overlapping_tokens += 1

    # return the proportion of overlapping tokens
    return num_overlapping_tokens / float(denominator)
