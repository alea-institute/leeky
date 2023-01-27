"""
This module implements Method F: Search Engine.

Provide a substring of the source material to a search engine like Google Search, Bing,
or Archive.org.

Unlike the other methods, this method does not return a score in [0.0, 1.0].  Instead,
this method returns the number of results from the search engine.

Each search engine needs to be configured to operate, as we avoid using techniques
like screen-scraping.  See the documentation from Google, Microsoft, and Archive.org
to learn more about how to create your own API keys.
"""

# imports
from pathlib import Path

# packages
import httpx

# load the google api key from the `.googlesearch_key` file if present
GOOGLE_API_KEY = None
GOOGLE_API_KEY_PATH = Path(__file__).parent / ".googlesearch_key"
if GOOGLE_API_KEY_PATH.exists():
    GOOGLE_API_KEY = GOOGLE_API_KEY_PATH.read_text().strip()


class SearchTester:
    """
    This module implements Method F: Search Engine, which takes a substring of the
    source material and searches for exact matches on search engines.

    Currently supported search engines:
     * [x] Google Search
     * [ ] Bing
     * [ ] Archive.org
    """

    def __init__(self,
                 google_api_key: str | None = None,
                 google_search_engine_id: str | None = None,
                 bing_api_key: str | None = None,
                 archive_api_key: str | None = None
                 ):
        """
        Initialize the tester, which is mostly just setting optional keys

        Args:
            google_api_key: Google API key
            google_search_engine_id: Google Search Engine ID
            bing_api_key: Bing API key
            archive_api_key: Archive.org API key
        """
        # google setup
        if google_api_key is None:
            google_api_key = GOOGLE_API_KEY
        self.google_api_key = google_api_key
        self.google_search_engine_id = google_search_engine_id

        # bing setup
        self.bing_api_key = bing_api_key

        # archive.org setup
        self.archive_api_key = archive_api_key

    def search_google(self, text: str) -> list[dict]:
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
            "https://www.googleapis.com/customsearch/v1",
            params=request_params,
        )

        # return results
        return response.json()

    def test(self, text: str) -> int:
        """
        Test the given text using Google Search.

        Args:
            text: Text to test

        Returns:
            Number of results
        """
        # search google
        results = self.search_google(text)

        # return number of results
        return results["searchInformation"]["totalResults"]

if __name__ == "__main__":
    # test the google search
    st = SearchTester(google_search_engine_id="your-cx-id")

    text = """It is a truth universally acknowledged, that a single man in possession 
            of a good fortune must be in want of a wife."""
    print(st.test(text))

    text = "It is a truth universally acknowledged, that a tomato unattended must be in want of a hornworm."
    print(st.test(text))
