"""
This module implements Method C: Semantic Recital.

Provide the model with an initial sequence of `N` tokens from the source material; this
sequence is typically the initial tokens of a segment of text, but can also be sampled
from a random position within the source.

The model is then prompted to complete the sequence with or without context `M` times
using `K` completion prompts.  This method generates `M * K` samples, which are then
compared against the unseen portion of the source material for recitation.

Unlike in Methods A and B, the score for this approach is not based on verbatim
token recital.  Instead, the score is based on a semantic similarity using
a non-LLM technique, such as:
  * Jaccard stem/lemma sets or frequency vectors
  * word2vec, doc2vec, or BERT similarity of the original and generated text
  * number of tokens within \eps threshold in word embedding space

"""

# imports
from enum import Enum
import logging

# leeky
import leeky.nlp
from leeky.engines.base_engine import BaseEngine
from leeky.methods.recital import RecitalTester, RECITAL_COMPLETION_PROMPTS

# set up logging
logger = logging.getLogger(__name__)


# create a similarity enum for types
class SimilarityType(Enum):
    """Enum for similarity types:
    JACCARD_TOKEN: compare_token_sets_jaccard(spacy=True)
    JACCARD_STEM: compare_token_sets_jaccard(spacy=True, stem=True)
    SPACY_SIMILARITY: compare_tokens_spacy_similarity
    SPACY_SIMILARITY_POINTS: compare_tokens_spacy_points
    """

    JACCARD_TOKEN = 1
    JACCARD_STEM = 2
    SPACY_SIMILARITY = 3
    SPACY_SIMILARITY_POINTS = 4


class SemanticRecitalTester(RecitalTester):
    """Class for testing recital using a semantic similarity measure."""

    def __init__(
        self,
        completion_engine: BaseEngine,
        similarity_method: SimilarityType = SimilarityType.SPACY_SIMILARITY,
        similarity_method_args: dict | None = None,
        min_tokens: int = 5,
        max_tokens: int = 50,
        completion_prompts: list = RECITAL_COMPLETION_PROMPTS,
        seed: int | None = None,
    ) -> None:
        """Initialize the tester."""

        # initialize the parent class
        super().__init__(
            completion_engine=completion_engine,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            completion_prompts=completion_prompts,
            seed=seed,
        )

        # set the similarity method
        self.similarity_method = similarity_method

        # set the similarity method args to pass to the similarity method
        # e.g., epsilon for compare_tokens_spacy_points
        if similarity_method_args is None:
            similarity_method_args = {}
        self.similarity_method_args = similarity_method_args

    def test(
        self,
        text: str,
        num_samples: int = 5,
    ) -> dict:
        """Test a model's ability to recite text.

        Args:
            text (str): The text to recite.
            num_samples (int): The number of samples to generate.

        Returns:
            dict: A dictionary with all results and score.
        """

        # find the whitespace token positions as a zero-dep alternative to tokenization
        text_token_splits = leeky.nlp.get_ws_token_boundaries(text)

        # initialize result data
        results = {
            "text": text,
            "score": None,
            "samples": [],
        }

        for _ in range(num_samples):
            # get prompt
            prompt_method = self.rng.choice(self.completion_prompts)

            # get initial text to test
            num_initial_tokens = self.rng.randint(
                self.min_tokens, min(self.max_tokens, len(text_token_splits) - 1)
            )
            initial_text = text[: text_token_splits[num_initial_tokens]]
            terminal_text = text[text_token_splits[num_initial_tokens] :]
            terminal_text_token_splits = leeky.nlp.get_ws_token_boundaries(
                terminal_text
            )

            # send the text to the completion engine
            sample = {
                "prompt_method": str(prompt_method.__name__),
                "num_initial_tokens": num_initial_tokens,
                "initial_text": initial_text,
                "terminal_text": terminal_text,
                "text_completion": None,
                "score": None,
            }
            try:
                # get the completion
                completion_list = self.completion_engine.get_completions(
                    prompt_method(initial_text),
                )
                sample["text_completion"] = (
                    completion_list[0] if len(completion_list) > 0 else None
                )

                # if we have a sample, score it
                if sample["text_completion"] is not None:
                    # switch on the similarity method
                    if self.similarity_method == SimilarityType.JACCARD_TOKEN:
                        sample["score"] = leeky.nlp.compare_token_sets_jaccard(
                            sample["text_completion"],
                            terminal_text,
                            spacy=True,
                            **self.similarity_method_args,
                        )
                    elif self.similarity_method == SimilarityType.JACCARD_STEM:
                        sample["score"] = leeky.nlp.compare_token_sets_jaccard(
                            sample["text_completion"],
                            terminal_text,
                            spacy=True,
                            stem=True,
                            **self.similarity_method_args,
                        )
                    elif self.similarity_method == SimilarityType.SPACY_SIMILARITY:
                        sample["score"] = leeky.nlp.compare_tokens_spacy_similarity(
                            sample["text_completion"],
                            terminal_text,
                        )
                    elif (
                        self.similarity_method == SimilarityType.SPACY_SIMILARITY_POINTS
                    ):
                        sample["score"] = leeky.nlp.compare_tokens_spacy_points(
                            sample["text_completion"],
                            terminal_text,
                            **self.similarity_method_args,
                        )
                    else:
                        raise ValueError(
                            f"Invalid similarity method: {self.similarity_method}"
                        )
            except Exception as e:
                logger.error(f"Error getting completions: {e}")
                continue
            finally:
                results["samples"].append(sample)

        # aggregate non-None scores and compute the average
        scores = [
            sample["score"]
            for sample in results["samples"]
            if sample["score"] is not None
        ]
        results["score"] = sum(scores) / float(len(scores)) if len(scores) > 0 else None

        return results
