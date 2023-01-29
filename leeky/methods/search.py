"""
This module implements Method F: Search Engine.

Provide a substring of the source material to a search engine like Google Search, Bing,
or Archive.org.

Unlike the other methods, this method does not return a score in [0.0, 1.0].  Instead,
this method returns the number of results from the search engine.

Each search engine needs to be configured to operate, as we avoid using techniques
like screen-scraping.  See the documentation from Google, Microsoft, and Archive.org
to learn more about how to create your own API keys.

TODO: Refine this method to optionally allow the user to subsample token sequences
from the source and then principle
"""

# imports
import logging
import time
from pathlib import Path

# packages
import httpx
import numpy.random
from playwright.sync_api import Page, sync_playwright

# project imports
import leeky.nlp

# set up logging
logger = logging.getLogger(__name__)

# load the google api key from the `.googlesearch_key` file if present
GOOGLE_API_KEY = None
GOOGLE_API_KEY_PATH = Path(__file__).parent / ".googlesearch_key"
if GOOGLE_API_KEY_PATH.exists():
    GOOGLE_API_KEY = GOOGLE_API_KEY_PATH.read_text().strip()

# default sleep
DEFAULT_SEARCH_SLEEP = 1.0


class SearchTester:
    """
    This module implements Method F: Search Engine, which takes a substring of the
    source material and searches for exact matches on search engines.

    Currently supported search engines:
     * [x] Google Search
     * [ ] Bing
     * [ ] Archive.org
    """

    def __init__(
        self,
        min_token_proportion: float = 0.5,
        max_token_proportion: float = 1.0,
        google_api_key: str | None = None,
        google_search_engine_id: str | None = None,
        bing_api_key: str | None = None,
        archive_api_key: str | None = None,
        seed: int | None = None,
    ):
        """
        Initialize the tester, which is mostly just setting optional keys

        Args:
            min_token_proportion: The minimum proportion of tokens to use from the source text.
            max_token_proportion: The maximum proportion of tokens to use from the source text.
            google_api_key: Google API key
            google_search_engine_id: Google Search Engine ID
            bing_api_key: Bing API key
            archive_api_key: Archive.org API key
        """
        # set the token proportions
        self.min_token_proportion = min_token_proportion
        self.max_token_proportion = max_token_proportion

        # google setup
        if google_api_key is None:
            google_api_key = GOOGLE_API_KEY
        self.google_api_key = google_api_key
        self.google_search_engine_id = google_search_engine_id

        # bing setup
        self.bing_api_key = bing_api_key

        # archive.org setup
        self.archive_api_key = archive_api_key

        # initialize the seed if required and RNG
        if seed is None:
            seed = numpy.random.randint(0, 2**32 - 1, dtype=numpy.int64)
        self.seed = seed
        self.rng = numpy.random.RandomState(seed=seed)

    def search_google_api(self, text: str) -> dict:
        """
        Search for the given text using Google Search.

        Args:
            text: Text to test

        Returns:
            List of results
        """
        # make request with **exact** query matching, like "{text}" (note the quotes)
        request_params = {
            "key": self.google_api_key,
            "q": f'"{text}"',
        }

        if self.google_search_engine_id is not None:
            request_params["cx"] = self.google_search_engine_id

        response = httpx.get(
            "https://customsearch.googleapis.com/customsearch/v1",
            params=request_params,
        )

        # return results
        return response.json()

    @staticmethod
    def search_google_playwright(text: str, headless: bool = False) -> dict:
        """Use playwright instead of API if you can't get your
        quota increased."""

        # create a playwright browser instance and navigate to google.com
        # setup the device as an iPhone 11 Pro
        with sync_playwright() as p:
            # setup browser and context
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context()

            # setup page
            page: Page = context.new_page()
            page.goto("https://google.com")

            # select the q input box and type the text
            page.click("input[name=q]")
            page.fill("input[name='q']", f'"{text}"')

            # hit enter from inside the `q` input
            page.keyboard.press("Enter")

            # wait for the results to load #main
            page.wait_for_selector("#main")
            time.sleep(1)

            # get the number out and convert to int
            num_results = 0
            try:
                # get the whole body text
                body_element = page.query_selector("body")
                if "No results found for " in body_element.text_content():
                    num_results = 0
                else:
                    result_stat_element = page.query_selector("div#result-stats")
                    if result_stat_element is not None:
                        result_stat_text = result_stat_element.text_content()
                        num_results = int(
                            result_stat_text.split(" ")[1].replace(",", "")
                        )
            except Exception as e:
                logger.error(f"Failed to parse total results: {e}")
                total_results = 0

            time.sleep(DEFAULT_SEARCH_SLEEP)

            # close the browser
            browser.close()

            results = {
                "text": text,
                "score": num_results,
                "samples": [],
            }

            # return the results
            return results

    def test(self, text: str, num_samples: int = 5) -> dict:
        """
        Test the given text using Google Search.

        Args:
            text: Text to test
            num_samples: Number of samples to take from the source text

        Returns:
            Result dictionary
        """

        # find the whitespace token positions as a zero-dep alternative to tokenization
        text_token_splits = leeky.nlp.get_ws_token_boundaries(text)
        num_tokens = len(text_token_splits)

        results = {
            "text": text,
            "score": None,
            "samples": [],
        }

        for _ in range(num_samples):
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

            # setup sample
            sample = {
                "sample_text": sample_text,
                "results": [],
                "score": None,
            }

            try:
                # search google
                api_results = self.search_google_playwright(sample_text)
                time.sleep(DEFAULT_SEARCH_SLEEP)

                # get the number of results
                sample["score"] = api_results["score"]

            except Exception as e:
                logger.exception(e)
                sample["score"] = None
                sample["results"] = []
            finally:
                results["samples"].append(sample)

        # calculate average non-None score
        scores = [
            sample["score"]
            for sample in results["samples"]
            if sample["score"] is not None
        ]
        if len(scores) > 0:
            results["score"] = sum(scores) / float(len(scores))

        return results


if __name__ == "__main__":
    # test the google search
    # this search engine ID is public and can be shared between people with API keys.
    # to setup your own, see https://developers.google.com/custom-search/v1/overview
    st = SearchTester()

    text = """It is a truth universally acknowledged, that a single man in possession 
            of a good fortune must be in want of a wife."""
    r = st.test(text, num_samples=1)
    print(r["score"])

    text = "It is a truth universally acknowledged, that a tomato unattended must be in want of a hornworm."
    r = st.test(text, num_samples=1)
    print(r["score"])
