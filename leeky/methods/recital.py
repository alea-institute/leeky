"""
This module implements  Method A: Recital without context.

Provide the model with an initial sequence of `N` tokens from the source material; this
sequence is typically the initial tokens of a segment of text, but can also be sampled
from a random position within the source.

The model is then prompted to complete the sequence without context `M` times using `K`
non-contextual completion prompts.  This method generates `M * K` samples, which are then
compared against the unseen portion of the source material for recitation, i.e., verbatim
completion of the original tokens beginning at position N+1.

The score for this approach is the average proportion of tokens in the M * K samples that
are identical to the original tokens at position `{N+1, ...}`.

**Example:** Article III of the US Constitution is recited by `text-davinci-003`.
```
<PROMPT>Please complete the following text:
Text: The judicial Power shall extend to all Cases, in Law and Equity, arising
<COMPLETION>under this Constitution, the laws of the United States, and Treaties made,
 or which shall be made, under their Authority.
```

100% of tokens match the original text (Article III).
"""

# imports
import logging

# packages
import numpy.random

# leeky
import leeky.nlp
from leeky.engines.base_engine import BaseEngine

# set up logging
logger = logging.getLogger(__name__)


def completion_prompt_001(text: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Complete the following text:
--------
{text}"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_002(text: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Complete the following sentence:
--------
{text}"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_003(text: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Complete the following paragraph:
--------
{text}"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_004(text: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Complete the following:
--------
{text}"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_005(text: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Complete the following text:
Text: {text}"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_006(text: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Complete the following sentence:
Sentence: {text}"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_007(text: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Complete the following paragraph:
Paragraph: {text}"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_008(text: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Complete the following:
{text}"""

    prompt = prompt.strip()

    return prompt


RECITAL_COMPLETION_PROMPTS = (
    completion_prompt_001,
    completion_prompt_002,
    completion_prompt_003,
    completion_prompt_004,
    completion_prompt_005,
    completion_prompt_006,
    completion_prompt_007,
    completion_prompt_008,
)


class RecitalTester:
    """Test a model's ability to recite text as described in the module docstring."""

    def __init__(
        self,
        completion_engine: BaseEngine,
        min_tokens: int = 5,
        max_tokens: int = 50,
        completion_prompts: list = RECITAL_COMPLETION_PROMPTS,
        seed: int | None = None,
    ) -> None:
        """Constructor"""
        # set engine
        self.completion_engine = completion_engine

        # set the token range
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

        # set the completion prompts
        self.completion_prompts = completion_prompts

        # initialize the seed if required and RNG
        if seed is None:
            seed = numpy.random.randint(0, 2**32 - 1)
        self.seed = seed
        self.rng = numpy.random.RandomState(seed=seed)

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
                    prompt_method(initial_text)
                )
                sample["text_completion"] = (
                    completion_list[0] if len(completion_list) > 0 else None
                )

                # if we have a sample, score it
                if sample["text_completion"] is not None:
                    sample["score"] = leeky.nlp.compare_token_sequences(
                        sample["text_completion"],
                        terminal_text,
                        spacy=True,
                        lowercase=True,
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
