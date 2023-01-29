"""
This module implements the Bloom engine for text completion
using the HuggingFace transformers library.
"""

# imports
import logging
from itertools import product, combinations_with_replacement

# packages
import numpy.random
import torch
from transformers import AutoTokenizer, BloomForCausalLM

# leeky imports
from leeky.engines.base_engine import BaseEngine

# set up logging
logger = logging.getLogger(__name__)

# model ID
BLOOM_MODEL_ID = "bigscience/bloom-3b"

# set default sampling parameters
BLOOM_VALID_PARAMETERS = {
    "temperature": [0.0, 0.5, 1.0],
    "max_length": [16, 32, 64, 128, 256],
}


class BloomEngine(BaseEngine):
    """
    Bloom text completion engine implementing the BaseEngine interface.
    """

    def __init__(
        self,
        model_id: str = BLOOM_MODEL_ID,
        model_max_length: int = 256,
        parameters: dict = None,
        seed: int | None = None,
        device: str | None = None,
    ):
        """
        Constructor for Bloom class.
        """
        # set the parameters
        if parameters is None:
            parameters = {}
        self.parameters = parameters

        # setup rng
        if seed is None:
            seed = numpy.random.randint(0, 2**32 - 1, dtype=numpy.int64)
        self.rng = numpy.random.RandomState(seed)

        # setup device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # setup the model and tokenizer for reuse across calls
        self.model = BloomForCausalLM.from_pretrained(model_id)
        try:
            self.model = self.model.to(self.device)
        except:
            logger.warning(f"Failed to move model to device {self.device}.")
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            model_max_length=model_max_length,
        )

    def get_name(self) -> str:
        """
        This function returns the name of the engine.
        """
        return f"transformers:{BLOOM_MODEL_ID}"

    def get_current_parameters(self) -> dict:
        """
        This function returns the current parameters for the engine.
        """
        return self.parameters

    def get_valid_parameters(self) -> dict:
        """
        This function returns the valid parameters for the engine.
        """
        # get all combinations of values from DEFAULT_OPENAI_VALID_PARAMETERS with itertools/functools
        for parameter_combination in product(*BLOOM_VALID_PARAMETERS.values()):
            # convert to dict
            parameter_dict = dict(
                zip(BLOOM_VALID_PARAMETERS.keys(), parameter_combination)
            )
            # yield the parameter dict
            yield parameter_dict

    def get_random_parameters(self) -> dict:
        """
        This method returns a random set of parameters.

        Returns:
            dict: A random set of parameters.
        """
        # get a random combination of values from DEFAULT_OPENAI_VALID_PARAMETERS with itertools/functools
        parameter_combination = self.rng.choice(
            list(
                combinations_with_replacement(
                    BLOOM_VALID_PARAMETERS.values(), len(BLOOM_VALID_PARAMETERS)
                )
            )
        )

        # convert to dict
        parameter_dict = dict(zip(BLOOM_VALID_PARAMETERS.keys(), parameter_combination))

        # return the parameter dict
        return parameter_dict

    def set_parameters(self, parameters: dict) -> None:
        """
        This method sets the parameters of the engine.

        N.B.: This does NOT update parameters.  To do so, create a copy with `get_current_parameters` and
        set from the updated copy.

        Args:
            parameters (dict): The parameters to set.
        """
        # set the parameters
        self.parameters = parameters

    def get_completions(self, prompt: str, n: int = 1) -> list[str]:
        """
        Run the `.generate()` method of the Bloom model to produce completions
        from the given prompt.
        """

        # get the tokenized prompt with the attention mask/pad token
        tokenized_prompt = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            truncation=True,
        )

        # send it to the device/GPU
        try:
            tokenized_prompt = tokenized_prompt.to(self.device)
        except:
            pass

        # generate the completions with the matching attention mask and pad token
        completions = self.model.generate(
            tokenized_prompt,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=n,
            **self.parameters,
        )

        # decode the tokens back to text
        completions = [
            self.tokenizer.decode(completion, skip_special_tokens=True)
            for completion in completions
        ]

        # strip the prompt off the beginning
        completions = [completion[len(prompt) :] for completion in completions]

        # return the completions
        return completions
