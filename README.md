# Stressful

Library for Russian stress placement.

## Installation

Install from the Git repository with pip:

```sh
pip install "git+https://github.com/esbudylin/stressful"
```

or with uv:

```sh
uv add "git+https://github.com/esbudylin/stressful"
```

## Usage

`Accentuator()` loads the accent dictionaries and the ML models on
construction. It exposes three methods:

- `mark_stresses(line, stress_mark) -> str` - returns the line with a
  stress mark inserted after each stressed vowel. By default, '+' will
  be used as a stress mark.
- `accentuate(line) -> list[bool]` - returns a binary mask with an
  element per syllable (`True` where the syllable is stressed).
- `accentuate_detailed(line) -> list[WordAccentuation]` - returns one
  record per word with the per-syllable stress probabilities and the
  source they came from.

Examples:
```python
from stressful import Accentuator

accentuator = Accentuator()
line = "Я весь день пролежал на ладони у снегопада."

# put stress marks on the original text
stressed = accentuator.mark_stresses(line)
assert(stressed == "Я весь день пролежа+л на ладо+ни у снегопа+да.")

# optionally specify a stress mark
with_apostrophe = accentuator.mark_stresses(line, stress_mark="'")
assert(with_apostrophe == "Я весь день пролежа'л на ладо'ни у снегопа'да.")

# alternatively, extract a boolean list showing stressed syllables
mask = accentuator.accentuate("сорока-воровка")

assert(len(mask) == 6)  # each element represents a syllable
assert(mask == [False, True, False, False, True, False])
```

## What's inside

Stressful combines dictionary lookups with neural-network
predictions. It follows the combined dictionary–neural approach
previously described in a
[paper](https://doi.org/10.31912/pvrli-2022.3.11) and implemented in
the [`ru-accent-poet`](https://github.com/yuliya1324/ru_accent)
library.

Compared to ru-accent-poet, Stressful offers a new implementation of
accent dictionary lookups that significantly improves
performance. Per-word lookups are about 150 times faster with
Stressful, compared to ru-accent-poet implementation.

For its neural-network backend, Stressful uses Silero Stress
models. The models are converted and shipped in the ONNX format for
ease of installation.

The following table shows Stressful's accuracy compared with other
libraries.

|accentuator                                               |accuracy|
|----------------------------------------------------------|-------:|
|[stressrnn](https://github.com/dbklim/StressRNN)          |  0.8395|
|[ruaccent](https://github.com/Den4ikAI/ruaccent)          |  0.9351|
|[silero-stress](https://github.com/snakers4/silero-stress)|  0.9629|
|**stressful**                                             |  0.9670|

<details>
<summary>Validation dataset</summary>

The validation dataset was built from the poetic corpus of the Russian
National Corpus. The dataset included 11110 lines of accentual verse
with hand-marked poetical accents. The accentual verse was chosen
specifically because its poetic stress aligns with linguistic stress.
The word accuracy metric is based on prediction accuracy for words
with multiple syllables (58605 words in total). The dataset is not
included in the repository due to copyright constraints.
</details>

## Tools

Run unit tests
```sh
make test
```

Re-export Silero models to ONNX
```sh
make export-silero-onnx
```

Benchmark accent dictionary lookups against ru-accent-poet
```sh
make benchmark-dict-lookups
```
