# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "stressful",
#     "ruaccent",
#     "silero-stress",
#     "stressrnn @ git+https://github.com/dbklim/StressRNN",
#     "torch",
#     "transformers<5",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# stressful = { path = "..", editable = true }
# torch = [{ index = "pytorch-cpu" }]
# ///
"""Compare accentuation models against a dataset produced by velimir.

The dataset is a CSV with the columns ``path, text, mask``:

- ``text`` is the line with stress marks removed (the model input);
- ``mask`` is the reference stress mask from the corpus, one bit per vowel.

Run against a dataset built by velimir:

    uv run scripts/compare_accentuators.py --dataset ../velimir/data/accent_dataset.csv
"""

import argparse
import csv
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from stressful import Accentuator
from stressful.accentuator import is_vowel, vowel_count

Predictor = Callable[[str], list[bool]]


@dataclass(slots=True)
class DatasetLine:
    path: str
    text: str
    mask: list[bool]


@dataclass(slots=True)
class PreparedLine:
    text: str
    mask: list[bool]
    words: list[str]
    analysed: int


def bits_to_mask(bits: str) -> list[bool]:
    return [bit == "1" for bit in bits]


def read_dataset(path: str) -> list[DatasetLine]:
    lines = []

    with open(path, "r", encoding="utf8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            lines.append(
                DatasetLine(
                    path=row["path"],
                    text=row["text"],
                    mask=bits_to_mask(row["mask"]),
                )
            )

    return lines


def marked_to_mask(
    clean: str,
    marked: str,
    mark_before_vowel: bool = True,
) -> list[bool]:
    mask = []
    stress_next = False

    for char in marked:
        if char == "+":
            if mark_before_vowel:
                stress_next = True
            elif mask:
                mask[-1] = True
        elif is_vowel(char):
            mask.append(stress_next or char in "ёЁ")
            stress_next = False

    expected = vowel_count(clean)

    if len(mask) != expected:
        raise ValueError(f"Vowel count mismatch: expected {expected}, got {len(mask)}")

    return mask


ACCENTUATOR_NAMES = (
    "stressful",
    "silero-onnx",
    "ruaccent",
    "silero-stress",
    "stressrnn",
)

# Predictors that only rely on the stressful library.
STRESSFUL_ACCENTUATORS = ("stressful", "silero-onnx")


def words_for_accentuation(text: str) -> list[str]:
    stripped = re.sub(r"[^А-яЁё\s-]+", "", text)
    return [word for word in stripped.split() if vowel_count(word)]


def neural_mask(accentuator: Accentuator, text: str) -> list[bool]:
    """Neural-only mask: the accentor's probabilities without the dictionary."""
    words = words_for_accentuation(text)

    return [
        probability > 0.5
        for probabilities, _ in accentuator._silero.accent_word_probabilities(
            text, words
        )
        for probability in probabilities
    ]


def build_predictors(only: set[str] | None = None) -> dict[str, Predictor]:
    def requested(name: str) -> bool:
        return only is None or name in only

    accentuator = Accentuator()

    predictors: dict[str, Predictor] = {}

    if requested("stressful"):
        predictors["stressful"] = accentuator.accentuate

    if requested("silero-onnx"):
        predictors["silero-onnx"] = lambda text: neural_mask(accentuator, text)

    if requested("ruaccent"):
        try:
            from ruaccent import RUAccent
        except ImportError:
            logging.warning("ruaccent is not installed, skipping it")
        else:
            ruaccent = RUAccent()
            ruaccent.load(omograph_model_size="turbo3.1", use_dictionary=True)

            predictors["ruaccent"] = lambda text: marked_to_mask(
                text, ruaccent.process_all(text)
            )

    if requested("silero-stress"):
        try:
            from silero_stress import load_accentor
        except ImportError:
            logging.warning("silero-stress is not installed, skipping it")
        else:
            silero = load_accentor()

            predictors["silero-stress"] = lambda text: marked_to_mask(
                text, silero(text)
            )

    if requested("stressrnn"):
        try:
            from stressrnn import StressRNN
        except ImportError:
            logging.warning("stressrnn is not installed, skipping it")
        else:
            stress_rnn = StressRNN()

            predictors["stressrnn"] = lambda text: marked_to_mask(
                text,
                stress_rnn.put_stress(text),
                mark_before_vowel=False,
            )

    return predictors


def reference_analysed(words: list[str], reference: list[bool]) -> int:
    """Count analysed words: polysyllabic words carrying a corpus stress.

    Depends only on the reference annotation, so the result is the same
    for every model.
    """
    analysed = 0
    position = 0

    for word in words:
        word_vowels = vowel_count(word)
        word_ref = reference[position : position + word_vowels]
        position += word_vowels

        if word_vowels > 1 and any(word_ref):
            analysed += 1

    return analysed


def word_diff_stats(
    predicted: list[bool],
    reference: list[bool],
    words: list[str],
) -> tuple[list[int], int]:
    """Return (error word indexes, correct words)."""
    error_indexes = []
    correct = 0
    position = 0

    for index, word in enumerate(words):
        word_vowels = vowel_count(word)
        word_pred = predicted[position : position + word_vowels]
        word_ref = reference[position : position + word_vowels]
        position += word_vowels

        if word_vowels > 1 and any(word_ref):
            if any(a != b for a, b in zip(word_pred, word_ref)):
                error_indexes.append(index)
            else:
                correct += 1

    return error_indexes, correct


def prepare_dataset(lines: list[DatasetLine]) -> list[PreparedLine]:
    prepared = []

    for line in lines:
        words = [word for word in line.text.split() if vowel_count(word)]

        if not words:
            continue

        prepared.append(
            PreparedLine(
                text=line.text,
                mask=line.mask,
                words=words,
                analysed=reference_analysed(words, line.mask),
            )
        )

    return prepared


def evaluate(
    lines: list[PreparedLine],
    predict: Predictor,
    name: str,
) -> dict:
    total_words = sum(len(line.words) for line in lines)
    total_analysed = sum(line.analysed for line in lines)
    total_correct = 0
    errors = 0
    diffed_words: Counter[str] = Counter()
    start = time.perf_counter()

    for line in lines:
        try:
            predicted = predict(line.text)

            if len(predicted) != len(line.mask):
                raise ValueError(
                    f"Prediction length {len(predicted)} != reference {len(line.mask)}"
                )

            error_indexes, correct = word_diff_stats(
                predicted,
                line.mask,
                line.words,
            )
        except Exception as exception:
            logging.error("[%s] error while processing line %s", name, line.text)
            logging.exception(exception)
            errors += 1
            continue

        for index in error_indexes:
            if index < len(line.words):
                diffed_words[re.sub(r"[^А-яЁё]", "", line.words[index]).lower()] += 1

        total_correct += correct

    seconds = time.perf_counter() - start

    return dict(
        name=name,
        lines=len(lines),
        words=total_words,
        analysed=total_analysed,
        correct=total_correct,
        word_accuracy=total_correct / total_analysed if total_analysed else 0.0,
        errors=errors,
        seconds=seconds,
        diffed_words=diffed_words,
    )


def render_table(results: list[dict]) -> str:
    rows = [
        "| accentuator | lines | words | analysed | correct | "
        "word_acc | errors | time (s) |",
        "|-------------|------:|------:|---------:|--------:|"
        "---------:|-------:|---------:|",
    ]

    for result in results:
        rows.append(
            f"| {result['name']} "
            f"| {result['lines']} "
            f"| {result['words']} "
            f"| {result['analysed']} "
            f"| {result['correct']} "
            f"| {result['word_accuracy']:.4f} "
            f"| {result['errors']} "
            f"| {result['seconds']:.2f} |"
        )

    return "\n".join(rows)


def main(
    dataset_path: str,
    limit: int | None = None,
    output: str | None = None,
    only: set[str] | None = None,
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    lines = read_dataset(dataset_path)

    if limit:
        lines = lines[:limit]

    prepared = prepare_dataset(lines)

    logging.info("Loaded %d lines from %s", len(prepared), dataset_path)

    results = []

    for name, predict in build_predictors(only).items():
        logging.info("Evaluating %s", name)
        results.append(evaluate(prepared, predict, name))

    table = render_table(results)
    print(table)

    if output:
        with open(output, "w", encoding="utf8") as output_file:
            output_file.write(table + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare accentuation models on a velimir accent dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the accent_dataset.csv built by velimir",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N dataset lines",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the comparison table to a markdown file",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated accentuators to evaluate, e.g. "
        "'stressful,silero-onnx' for stressful only. "
        f"Known: {', '.join(ACCENTUATOR_NAMES)}. Default: all.",
    )
    parser.add_argument(
        "--only-stressful",
        action="store_true",
        help="Shorthand for --only stressful,silero-onnx",
    )
    args = parser.parse_args()

    if args.only_stressful:
        only = set(STRESSFUL_ACCENTUATORS)
    elif args.only:
        only = {name.strip() for name in args.only.split(",") if name.strip()}
    else:
        only = None

    if only is not None:
        unknown = only - set(ACCENTUATOR_NAMES)
        if unknown:
            parser.error(f"unknown accentuators: {', '.join(sorted(unknown))}")

    main(args.dataset, limit=args.limit, output=args.output, only=only)
