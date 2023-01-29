"""
This module implements  Method E: Source Recall

Provide the model with a sequence of `N` tokens from the source material, which can be
either a subset or the complete text.

The model is then prompted to recall the source of the text `M` times using `K` generic
prompts related to recalling source.  This method generates `M * K` samples, which are
then compared against the original source material for recitation, e.g., the original
document name, URL, or other valid identifier.

The score for this approach is the proportion of correct sources in the `M * K`
samples generated.

**Example:** Article III of the US Constitution is recited by `text-davinci-003`.

```
<PROMPT>What is the source of the following text?
----
Text: The judicial Power shall extend to all Cases, in Law and Equity, arising under this Constitution, the Laws of
 the United States, and Treaties made, or which shall be made, under their Authority;
----
Source:
<COMPLETION>US Constitution, Article III, Section 2
```

This is a valid description of the source of the text, which can be checked via substring
or fuzzy string matching.

TODO: Decide where the "Is the following text from XYZ?" prompt should go.
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

    prompt = f"""What is the source of the following text?
----
Text: {text}
----
Source:"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_002(text: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""What document is the following text from?
----
Text: {text}
----
Document:"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_003(text: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""Where is the following text from?
----
Text: {text}
----
Source:"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_004(text: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""What is the citation for the following text?
----
Text: {text}
----
Citation:"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_005(text: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""What is the reference for the following text?
----
Text: {text}
----
Reference:"""

    prompt = prompt.strip()

    return prompt


def completion_prompt_006(text: str) -> str:
    """Generate a simple prompt to complete the following question WITHOUT context."""

    prompt = f"""What is the original source for the following text?
----
Text: {text}
----
Source:"""

    prompt = prompt.strip()

    return prompt


SOURCE_COMPLETION_PROMPTS = (
    completion_prompt_001,
    completion_prompt_002,
    completion_prompt_003,
    completion_prompt_004,
    completion_prompt_005,
)


class SourceRecallTester:
    """Just ask the model where it thinks the document or text came from."""

    def __init__(
        self,
        completion_engine: BaseEngine,
        min_token_proportion: float = 0.5,
        max_token_proportion: float = 1.0,
        completion_prompts: list = SOURCE_COMPLETION_PROMPTS,
        seed: int | None = None,
    ):
        """Initialize the tester

        Args:
            completion_engine (BaseEngine): The completion engine to use.
            min_token_proportion (float): The minimum proportion of tokens to use from the source text.
            max_token_proportion (float): The maximum proportion of tokens to use from the source text.
            completion_prompts (list): The list of completion prompts to use.
            seed (int | None): The random seed to use.
        """

        # set engine
        self.completion_engine = completion_engine

        # set the token range
        self.min_token_proportion = min_token_proportion
        self.max_token_proportion = max_token_proportion

        # set the completion prompts
        self.completion_prompts = completion_prompts

        # initialize the seed if required and RNG
        if seed is None:
            seed = numpy.random.randint(0, 2**32 - 1, dtype=numpy.int64)
        self.seed = seed
        self.rng = numpy.random.RandomState(seed=seed)

    def test(
        self,
        text: str,
        match_list: list[str],
        num_samples: int = 5,
    ) -> dict:
        """Test where the model thinks the text came from.

        Args:
            text (str): The text to recite.
            match_list (list[str]): The list of valid substring matches.
            num_samples (int): The number of samples to generate.

        Returns:
            dict: A dictionary with all results and score.
        """

        # find the whitespace token positions as a zero-dep alternative to tokenization
        text_token_splits = leeky.nlp.get_ws_token_boundaries(text)
        num_tokens = len(text_token_splits)

        # initialize result data
        results = {
            "text": text,
            "score": None,
            "samples": [],
        }

        for _ in range(num_samples):
            # get prompt
            prompt_method = self.rng.choice(self.completion_prompts)

            # get the number of tokens to send
            num_tokens_sample = self.rng.randint(
                int(self.min_token_proportion * num_tokens),
                int(self.max_token_proportion * num_tokens),
            )

            # get the initial position
            initial_position = self.rng.randint(0, num_tokens - num_tokens_sample)

            # get the tokens to test
            sample_text = text[
                text_token_splits[initial_position] : text_token_splits[
                    initial_position + num_tokens_sample
                ]
            ]

            # send the text to the completion engine
            sample = {
                "prompt_method": str(prompt_method.__name__),
                "num_tokens_sample": num_tokens_sample,
                "sample_text": sample_text,
                "text_completion": None,
                "score": None,
            }
            try:
                # get the completion
                completion_list = self.completion_engine.get_completions(
                    prompt_method(sample_text)
                )
                sample["text_completion"] = (
                    completion_list[0] if len(completion_list) > 0 else None
                )

                # if we have a sample, score it
                if sample["text_completion"] is not None:
                    for match in match_list:
                        if (
                            match.lower().strip()
                            in sample["text_completion"].lower().strip()
                        ):
                            sample["score"] = 1.0
                            break
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


if __name__ == "__main__":
    from leeky.engines.openai_engine import OpenAIEngine

    oai = OpenAIEngine(model="text-davinci-003", parameters={"temperature": 0.5})
    t = SourceRecallTester(oai)

    r = t.test(
        """We the People of the United States, in Order to form a more perfect Union, establish Justice, insure 
        domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of
        Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of
        America.""".strip(),
        ["Constitution"],
        num_samples=5,
    )
    print(r["score"])

    r = t.test(
        """We the People of the United States, in Order to form a more perfect Union, establish Justice, insure 
        domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of
        Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of
        America.""".strip(),
        ["Declaration of Independence"],
        num_samples=5,
    )
    print(r["score"])

    r = t.test(
        """We the Persons of the Moon, so that we might franchise a McDonalds on the Sinus Concordiae, seek
        to establish a more effective space launch system, insure lunar vibes, and provide for the common
        supply of McRibs, do therefore totally and completely reject the idea that cows cannot be sent
        into space.""".strip(),
        ["Constitution"],
        num_samples=5,
    )
    print(r["score"])
