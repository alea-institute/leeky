"""
This script demonstrates the use of the Recital, Contextual Recital, Semantic Recital,
Source Veracity, Source Recall, and Search Engine methods.
"""

# imports
import json
from pathlib import Path

# packages
from rich.console import Console
from rich.table import Table

# leeky imports
from leeky.engines.openai_engine import OpenAIEngine
from leeky.methods.recital import RecitalTester
from leeky.methods.contextual_recital import ContextualRecitalTester
from leeky.methods.semantic_recital import SemanticRecitalTester, SimilarityType
from leeky.methods.source_veracity import SourceVeracityTester
from leeky.methods.source_recall import SourceRecallTester
from leeky.methods.search import SearchTester


# set parameters for testing
OPENAI_API_KEY = None

# this is public and can be used by anyone with valid API keys
# if you want to create your own, see: https://programmablesearchengine.google.com/
GOOGLE_SEARCH_ENGINE_ID = "97f6bca7b22b74752"


if __name__ == "__main__":
    # initialize console with wide display
    console = Console(width=200)

    # initialize default engine, which we'll assume is OpenAI text-davinci-001 for this testing
    engine = OpenAIEngine(api_key=OPENAI_API_KEY, model="text-davinci-003")

    # set number of samples to generate
    num_samples = 5

    # setup text examples
    text_1 = """"We the People of the United States, in Order to form a more perfect Union, establish Justice, insure 
        domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of
        Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of
        America."""

    text_2 = """We the Persons of the Moon, so that we might franchise a McDonalds on the Sinus Concordiae, seek
        to establish a more effective space launch system, insure lunar vibes, and provide for the common
        supply of McRibs, do therefore totally and completely reject the idea that cows cannot be sent
        into space."""

    text_3 = """The Racketeer Influenced and Corrupt Organizations (RICO) Act is a United States federal law that provides
for extended criminal penalties and a civil cause of action for acts performed as part of an ongoing criminal
organization."""

    text_4 = """The Robot Interaction and Cool Organizations (RICO) Act is a United States federal law that provides
for extended robot video games as part of cool employer benefit programs."""

    # setup recital testers
    recital_tester = RecitalTester(completion_engine=engine)
    contextual_recital_tester = ContextualRecitalTester(
        completion_engine=engine, source="United States Constitution"
    )
    semantic_recital_tester = SemanticRecitalTester(
        completion_engine=engine,
        similarity_method=SimilarityType.SPACY_SIMILARITY_POINTS,
    )

    source_veracity_tester = SourceVeracityTester(completion_engine=engine)
    source_recall_tester = SourceRecallTester(completion_engine=engine)
    search_tester = SearchTester()

    # that's it - the rest of this is just formatting for pretty output.

    # build the dictionary of results for both text_1 and text_2 for each tester
    results = {
        "text_1": {
            "recital": recital_tester.test(text_1, num_samples=num_samples),
            "contextual_recital": contextual_recital_tester.test(
                text_1, num_samples=num_samples
            ),
            "semantic_recital": semantic_recital_tester.test(
                text_1, num_samples=num_samples
            ),
            "source_veracity": source_veracity_tester.test(
                text_1, num_samples=num_samples
            ),
            "source_recall": source_recall_tester.test(
                text_1, match_list=["Constitution", "Preamble"], num_samples=num_samples
            ),
            "search": search_tester.test(text_1, num_samples=1),
        },
        "text_2": {
            "recital": recital_tester.test(text_2, num_samples=num_samples),
            "contextual_recital": contextual_recital_tester.test(
                text_2, num_samples=num_samples
            ),
            "semantic_recital": semantic_recital_tester.test(
                text_2, num_samples=num_samples
            ),
            "source_veracity": source_veracity_tester.test(
                text_2, num_samples=num_samples
            ),
            "source_recall": source_recall_tester.test(
                text_2, match_list=["Constitution", "Preamble"], num_samples=num_samples
            ),
            "search": search_tester.test(text_2, num_samples=1),
        },
    }

    # you can change the source for the contextual recital tester
    contextual_recital_tester.set_source("Wikipedia")

    # add results for text_3 and text_4
    results.update(
        {
            "text_3": {
                "recital": recital_tester.test(text_3, num_samples=num_samples),
                "contextual_recital": contextual_recital_tester.test(
                    text_3, num_samples=num_samples
                ),
                "semantic_recital": semantic_recital_tester.test(
                    text_3, num_samples=num_samples
                ),
                "source_veracity": source_veracity_tester.test(
                    text_3, num_samples=num_samples
                ),
                "source_recall": source_recall_tester.test(
                    text_3, match_list=["Wikipedia", "RICO"], num_samples=num_samples
                ),
                "search": search_tester.test(text_3, num_samples=1),
            },
            "text_4": {
                "recital": recital_tester.test(text_4, num_samples=num_samples),
                "contextual_recital": contextual_recital_tester.test(
                    text_4, num_samples=num_samples
                ),
                "semantic_recital": semantic_recital_tester.test(
                    text_4, num_samples=num_samples
                ),
                "source_veracity": source_veracity_tester.test(
                    text_4, num_samples=num_samples
                ),
                "source_recall": source_recall_tester.test(
                    text_4, match_list=["Wikipedia", "RICO"], num_samples=num_samples
                ),
                "search": search_tester.test(text_4, num_samples=1),
            },
        }
    )

    # get the score for each tester for each text
    scores = {
        "text_1": {
            "recital": results["text_1"]["recital"]["score"],
            "contextual_recital": results["text_1"]["contextual_recital"]["score"],
            "semantic_recital": results["text_1"]["semantic_recital"]["score"],
            "source_veracity": results["text_1"]["source_veracity"]["score"],
            "source_recall": results["text_1"]["source_recall"]["score"],
            "search": results["text_1"]["search"]["score"],
        },
        "text_2": {
            "recital": results["text_2"]["recital"]["score"],
            "contextual_recital": results["text_2"]["contextual_recital"]["score"],
            "semantic_recital": results["text_2"]["semantic_recital"]["score"],
            "source_veracity": results["text_2"]["source_veracity"]["score"],
            "source_recall": results["text_2"]["source_recall"]["score"],
            "search": results["text_2"]["search"]["score"],
        },
        "text_3": {
            "recital": results["text_3"]["recital"]["score"],
            "contextual_recital": results["text_3"]["contextual_recital"]["score"],
            "semantic_recital": results["text_3"]["semantic_recital"]["score"],
            "source_veracity": results["text_3"]["source_veracity"]["score"],
            "source_recall": results["text_3"]["source_recall"]["score"],
            "search": results["text_3"]["search"]["score"],
        },
        "text_4": {
            "recital": results["text_4"]["recital"]["score"],
            "contextual_recital": results["text_4"]["contextual_recital"]["score"],
            "semantic_recital": results["text_4"]["semantic_recital"]["score"],
            "source_veracity": results["text_4"]["source_veracity"]["score"],
            "source_recall": results["text_4"]["source_recall"]["score"],
            "search": results["text_4"]["search"]["score"],
        },
    }

    # display it in a rich table with lines between rows
    table = Table(
        title="Recital Scores", show_lines=True, header_style="bold magenta", width=200
    )

    table.add_column("Text", justify="center", style="cyan")

    # show the scores from each testing method
    table.add_column("Recital", justify="center")
    table.add_column("Contextual Recital", justify="center")
    table.add_column("Semantic Recital", justify="center")
    table.add_column("Source Veracity", justify="center")
    table.add_column("Source Recall", justify="center")
    table.add_column("Search", justify="center")

    # add the rows
    table.add_row(
        "Preamble, US Constitution",
        str(scores["text_1"]["recital"]),
        str(scores["text_1"]["contextual_recital"]),
        str(scores["text_1"]["semantic_recital"]),
        str(scores["text_1"]["source_veracity"]),
        str(scores["text_1"]["source_recall"]),
        str(scores["text_1"]["search"]),
    )
    table.add_row(
        "Text in style of Preamble, novel",
        str(scores["text_2"]["recital"]),
        str(scores["text_2"]["contextual_recital"]),
        str(scores["text_2"]["semantic_recital"]),
        str(scores["text_2"]["source_veracity"]),
        str(scores["text_2"]["source_recall"]),
        str(scores["text_2"]["search"]),
    )
    table.add_row(
        "RICO Act, Wikipedia",
        str(scores["text_3"]["recital"]),
        str(scores["text_3"]["contextual_recital"]),
        str(scores["text_3"]["semantic_recital"]),
        str(scores["text_3"]["source_veracity"]),
        str(scores["text_3"]["source_recall"]),
        str(scores["text_3"]["search"]),
    )
    table.add_row(
        "Fake RICO Act, novel",
        str(scores["text_4"]["recital"]),
        str(scores["text_4"]["contextual_recital"]),
        str(scores["text_4"]["semantic_recital"]),
        str(scores["text_4"]["source_veracity"]),
        str(scores["text_4"]["source_recall"]),
        str(scores["text_4"]["search"]),
    )

    # show the table
    console.print(table)

    # save the results to a json file
    with open(Path(__file__).parent / "recital_results.json", "w") as f:
        json.dump(results, f, indent=4)
