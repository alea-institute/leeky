"""
This module implements  Method B: Contextual Recital.

As in Method A: Recital testing without context, but the model is prompted to complete the
sequence with explicit knowledge of the source in question.

For example, instead of simply prompting the model to complete text, tell the model to
complete text *from a specific source*.

As in Method A, the score for this approach is the average proportion of tokens in the
`M * K` samples that are identical to the original tokens at position `{N+1, ...}`.

**Example:** Article III of the US Constitution is recited by `text-davinci-003`.

```
<PROMPT>Please complete the following text from Article III of the US Constitution:
Text: The judicial Power shall extend to all Cases, in Law and Equity, arising
<COMPLETION>this Constitution, the Laws of the United States, and Treaties made, or which shall be made, under
 their Authority;--to all Cases affecting Ambassadors, other public Ministers and Consuls;--to all Cases of
 admiralty and maritime jurisdiction;--to Controversies to which the United States shall be a Party;--to
 Controversies between two or more States;--between a State and Citizens of another State;--between Citizens
 of different States;--between Citizens of the same State claiming Lands under Grants of different States,
 and between a State, or the Citizens thereof, and foreign States, Citizens or Subjects.[...]
```

100% of tokens match the original text (Article III).
"""


# imports
import logging

# leeky
import leeky.nlp
from leeky.engines.base_engine import BaseEngine
from leeky.methods.recital import RecitalTester

# set up logging
logger = logging.getLogger(__name__)


def completion_prompt_001(question_text: str, source: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Complete the following text from {source}:
--------
{question_text}"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_002(question_text: str, source: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Complete the following sentence from {source}:
--------
{question_text}"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_003(question_text: str, source: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Complete the following paragraph from {source}:
--------
{question_text}"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_004(question_text: str, source: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Complete the following from {source}:
--------
{question_text}"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_005(question_text: str, source: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Complete the following text:
Source: {source}
Text: {question_text}"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_006(question_text: str, source: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Complete the following sentence:
Source: {source}
Sentence: {question_text}"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_007(question_text: str, source: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Complete the following paragraph:
Source: {source}
Paragraph: {question_text}"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_008(question_text: str, source: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Complete the following:
Source: {source}
{question_text}"""

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


class ContextualRecitalTester(RecitalTester):
    """Test a model's ability to recite text in context as described in the module docstring.

    From an OOP perspective, this is a subclass of `RecitalTester` that overrides the
    `completion_prompt` list and how the completion prompt is called to support passing source
    as well as text.
    """

    def __init__(
        self,
        completion_engine: BaseEngine,
        source: str,
        min_tokens: int = 5,
        max_tokens: int = 50,
        completion_prompts: list = RECITAL_COMPLETION_PROMPTS,
        seed: int | None = None,
    ) -> None:
        """Initialize the tester."""

        super().__init__(
            completion_engine=completion_engine,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            completion_prompts=completion_prompts,
            seed=seed,
        )

        # set source
        self.source = source

    def get_source(self) -> str:
        """Get the source."""

        return self.source

    def set_source(self, source: str) -> None:
        """Set the source."""

        self.source = source

    def test(
        self,
        text: str,
        num_samples: int = 5,
    ) -> dict:
        """Test a model's ability to recite text.

        Args:
            text (str): The text to recite.
            source (str): The source of the text.
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
                    prompt_method(initial_text, self.source),
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
