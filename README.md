## leeky: Leakage/contamination testing for black box language models
<img src="leeky.png" alt="drawing" width="100" />

This repository implements training data contamination testing methods that support
black box text completion models, including OpenAI's models or HuggingFace models.

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
