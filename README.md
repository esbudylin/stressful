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
construction. It exposes two methods:

- `mark_stresses(line, stress_mark) -> str` - returns the line with
  stress mark inserted after each stressed vowel. By default, '+' will
  be used as a stress mark.
- `accentuate(line) -> list[bool]` - returns binary mask with an
  element per syllable.
  
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

assert(len(mask) == 6) # each element represents a syllable
assert([False, True, False, False, True, False])
```

## What's inside

Stressful combines dictionary lookups with neural-network predictions
for stress placement. It follows the combined dictionary–neural
approach previously described in a
[paper](https://doi.org/10.31912/pvrli-2022.3.11) and implemented in
the [`ru_accent`](https://github.com/yuliya1324/ru_accent) library.

However, Stressful offers a new implementation of accent dictionary
lookups that significantly improves performance. (TODO: add benchmarks)

For its neural-network backend the library uses Silero Stress models
converted to ONNX format for ease of distribution. (TODO: add model comparison)

## Tools

Run unit tests
```sh
make test
```

Re-export Silero models to ONNX
```sh
make export-silero-onnx
```
