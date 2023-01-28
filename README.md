## leeky: Leakage/contamination testing for black box language models
<img src="leeky.png" alt="DALL-E generated logo of a leek in a bucket that is NOT leaking." width="100" />

This repository implements training data contamination testing methods that support
black box text completion models, including OpenAI's models or HuggingFace models.


## Example Results
The following table is generated from the `examples/example.py` script.

It shows that the methods in this repository can effectively discriminate between
information that is in the training data and information that is not in the training
data. 

Further research on behavior in common models and for k-eidetic samples is ongoing.

| Text                                            | Recital               | Contextual Recital | Semantic Recital       | Source Veracity | Source Recall | Search   |
|--------------------------------------------------|------------------------|--------------------|------------------------|------------------|----------------|----------|
| Preamble, US Constitution                       | 0.91                   | 1.00                | 0.53                    | 1.00              | 1.00           | 87100.00 |
| Text in style of Preamble, novel                | 0.01                   | 0.10                | 0.10                    | 0.00              | None         | 0.00     |
| RICO Act, Wikipedia                             | 1.00                   | 0.95                | 0.46                    | 1.00              | 1.00           | 4738.00  |
| Fake RICO Act, novel                           | 0.07                   | 0.03                | 0.10                    | 0.00              | None         | 0.00     |


The following methods are currently implemented:

### A. Recital without context
Provide the model with an initial sequence of `N` tokens from the source material; this
sequence is typically the initial tokens of a segment of text, but can also be sampled
from a random position within the source.

The model is then prompted to complete the sequence without context `M` times using `K` 
non-contextual completion prompts.  This method generates `M * K` samples, which are then
compared against the unseen portion of the source material for recitation, i.e., verbatim 
completion of the original tokens beginning at position N+1.

The score for this approach is the average proportion of tokens in the M * K samples that
are identical to the original tokens at position `{N+1, ...}`.

**Code**:
```python
from leeky.engines.openai_engine import OpenAIEngine
from leeky.methods.recital import RecitalTester

engine = OpenAIEngine(api_key=OPENAI_API_KEY, model="text-davinci-003")

recital_tester = RecitalTester(completion_engine=engine)
result = recital_tester.test(text_1, num_samples=5)
```


**Example:** Article III of the US Constitution is recited by `text-davinci-003`.
```
<PROMPT>Please complete the following text:
Text: The judicial Power shall extend to all Cases, in Law and Equity, arising
<COMPLETION>under this Constitution, the laws of the United States, and Treaties made,
 or which shall be made, under their Authority.
```

100% of tokens match the original text (Article III).

### B. Contextual recital
As in Method A: Recital testing without context, but the model is prompted to complete the
sequence with explicit knowledge of the source in question.

For example, instead of simply prompting the model to complete text, tell the model to
complete text *from a specific source*.

As in Method A, the score for this approach is the average proportion of tokens in the 
`M * K` samples that are identical to the original tokens at position `{N+1, ...}`.


**Code**:
```python
from leeky.engines.openai_engine import OpenAIEngine
from leeky.methods.contextual_recital import ContextualRecitalTester

engine = OpenAIEngine(api_key=OPENAI_API_KEY, model="text-davinci-003")

contextual_recital_tester = ContextualRecitalTester(completion_engine=engine, source="Wikipedia")
result = contextual_recital_tester.test(text_1, num_samples=5)
```

**Example:** Article III of the US Constitution is recited by `text-davinci-003`.

```
<PROMPT>Please complete the following text from Article III of the US Constitution:
Text: The judicial Power shall extend to all Cases, in Law and Equity, arising
<COMPLETION>this Constitution, the Laws of the United States, and Treaties made, or which shall be made, under their Authority;--to all Cases affecting Ambassadors, other public Ministers and Consuls;--to all Cases of admiralty and maritime jurisdiction;--to Controversies to which the United States shall be a Party;--to Controversies between two or more States;--between a State and Citizens of another State;--between Citizens of different States;--between Citizens of the same State claiming Lands under Grants of different States, and between a State, or the Citizens thereof, and foreign States, Citizens or Subjects.[...]
```

100% of tokens match the original text (Article III).

### C. Semantic recital
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

**Code**:
```python
from leeky.engines.openai_engine import OpenAIEngine
from leeky.methods.semantic_recital import SemanticRecitalTester, SimilarityType

engine = OpenAIEngine(api_key=OPENAI_API_KEY, model="text-davinci-003")

semantic_recital_tester = SemanticRecitalTester(
    completion_engine=engine,
    similarity_method=SimilarityType.SPACY_SIMILARITY_POINTS
)
result = semantic_recital_tester.test(text_1, num_samples=5)
```

### D. Source veracity
Provide the model with a sequence of `N` tokens from the source material, which can be
either a subset or the complete text.  

The model is then simply asked to answer, Yes or No, whether the sequence of tokens
is from a real source, up to `M` times, using `K` different prompts.  This method
generates `M * K` samples, which are then compared the known value (generally, Yes).

```
Is this text from a real document?  Answer yes or no.
----
Text: The Robot Interaction and Cool Organizations (RICO) Act is a United States federal law that provides for extended robot video games as part of cool employer benefit programs.
----
Answer: No
```

```
Is this text from a real document?  Answer yes or no.
----
Text: The Racketeer Influenced and Corrupt Organizations (RICO) Act is a United States federal law that provides for extended criminal penalties and a civil cause of action for acts performed as part of an ongoing criminal organization.
----
Answer: Yes
```

**Code**:
```python
from leeky.engines.openai_engine import OpenAIEngine
from leeky.methods.source_veracity import SourceVeracityTester

engine = OpenAIEngine(api_key=OPENAI_API_KEY, model="text-davinci-003")

source_veracity_tester = SourceVeracityTester(completion_engine=engine)
result = source_veracity_tester.test(text_1, num_samples=5)
```

### E. Source recall
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
Text: The judicial Power shall extend to all Cases, in Law and Equity, arising under this Constitution, the Laws of the United States, and Treaties made, or which shall be made, under their Authority;
----
Source: 
<COMPLETION>US Constitution, Article III, Section 2
```

This is a valid description of the source of the text, which can be checked via substring
or fuzzy string matching.


**Code**:
```python
from leeky.engines.openai_engine import OpenAIEngine
from leeky.methods.source_recall import SourceRecallTester

engine = OpenAIEngine(api_key=OPENAI_API_KEY, model="text-davinci-003")

source_recall_tester = SourceRecallTester(completion_engine=engine)
result = source_recall_tester.test(text_1, match_list=["Constitution", "Preamble"], num_samples=5)
```


### F. Search Engines
Provide a random substring of the source material to a search engine like Google Search, Bing,
or Archive.org.  Generate `M` samples like this.

Unlike the other methods, this method does not return a score in [0.0, 1.0].  Instead,
this method returns the average number of results from the search engine across the
`M` samples.

```python

from leeky.methods.search import SearchTester

search_tester = SearchTester()
result = search_tester.test(text_1, num_samples=5)
```
