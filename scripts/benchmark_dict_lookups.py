# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "stressful",
#     "ru-accent-poet",
# ]
#
# # russtress (and its TensorFlow dependency) is not needed for the
# # dictionary lookup, which lives in the self-contained rules.py.
# [tool.uv]
# override-dependencies = ["russtress; python_version < '0'"]
#
# [tool.uv.sources]
# stressful = { path = "..", editable = true }
# ///
"""Benchmark accent-dictionary lookups: stressful vs ru-accent-poet.

``stressful`` reads the same Zaliznyak dictionary that the ``ru_accent``
project ships (``accent.dic`` / ``accent1.dic``) but stores it in a
structured form and resolves words by set membership instead of running a
regular expression per candidate entry.  This script measures the
difference on the dictionary-only code path; no neural network is used.

``ru-accent-poet`` depends on ``russtress``, which in turn depends on
TensorFlow, but the dictionary lookup lives in a self-contained
``rules.py``.  The script loads that file directly so importing the
package (and its TensorFlow model) is avoided.

Run with the dictionary's own surface forms:

    uv run scripts/benchmark_dict_lookups.py

or against a text corpus:

    uv run scripts/benchmark_dict_lookups.py --words corpus.txt --repeat 5
"""

import argparse
import importlib.util
import logging
import random
import re
import time
from pathlib import Path
from timeit import repeat

from stressful.accentuator import (
    accent_word_by_dict,
    build_accent_dict,
    find_accent_entry,
    read_accent_dicts,
)
from stressful.settings import ACCENT_DICT_PATHS

CYRILLIC_RE = re.compile(r"[А-яЁё]")


def load_ru_accent_rules():
    """Load ru_accent_poet/rules.py without importing the package __init__."""
    spec = importlib.util.find_spec("ru_accent_poet")

    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("ru-accent-poet is not installed")

    rules_path = Path(next(iter(spec.submodule_search_locations))) / "rules.py"
    module_spec = importlib.util.spec_from_file_location("ru_accent_rules", rules_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module


def iter_surface_forms(paths):
    for path in paths:
        with open(path, encoding="utf8") as file_read:
            for line in file_read:
                if line.startswith("#"):
                    continue

                split = line.split()

                if not split:
                    continue

                word = split[0]
                base, lpar, rest = word.partition("(")

                if lpar and rest.endswith(")"):
                    endings = rest[:-1].split("|")
                else:
                    endings = [""]

                for ending in endings:
                    yield (base + ending).lower()


def collect_words(dict_paths, words_path, limit):
    if words_path:
        text = Path(words_path).read_text(encoding="utf8")
        words = [w for w in re.split(r"\s+", text) if CYRILLIC_RE.search(w)]
    else:
        words = list(set(iter_surface_forms(dict_paths)))

    if limit and limit < len(words):
        words = random.Random(0).sample(words, limit)

    return words


def build_stressful_accentuator():
    accent_dict = build_accent_dict(read_accent_dicts(ACCENT_DICT_PATHS))

    def accentuate(word):
        entry = find_accent_entry(word, accent_dict)

        if not entry:
            return None

        return accent_word_by_dict(word, entry)

    return accentuate


def measure(fn, words, repeats):
    def run():
        for word in words:
            fn(word)

    return min(repeat(run, repeat=repeats, number=1))


def evaluate(name, build, adapt, words, repeats):
    build_start = time.perf_counter()
    fn = adapt(build())
    build_time = time.perf_counter() - build_start

    duration = measure(fn, words, repeats)

    return dict(
        name=name,
        build_time=build_time,
        count=len(words),
        total=duration,
        per_word=duration / len(words) if words else 0.0,
    )


def render_table(results):
    baseline = results[0]["per_word"] or 1.0
    rows = [
        "| implementation | load (s) | words | total (s) | "
        "per-word (µs) | speedup |",
        "|----------------|---------:|------:|----------:|"
        "--------------:|--------:|",
    ]

    for result in results:
        rows.append(
            f"| {result['name']} "
            f"| {result['build_time']:.3f} "
            f"| {result['count']} "
            f"| {result['total']:.3f} "
            f"| {result['per_word'] * 1e6:.3f} "
            f"| {result['per_word'] / baseline:.2f}x |"
        )

    return "\n".join(rows)


def main(words_path, limit, repeats, output):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    words = collect_words(ACCENT_DICT_PATHS, words_path, limit)
    logging.info("Benchmarking %d words, %d repeats", len(words), repeats)

    results = [
        evaluate(
            "stressful",
            build_stressful_accentuator,
            lambda accentuate: accentuate,
            words,
            repeats,
        ),
        evaluate(
            "ru-accent-poet",
            load_ru_accent_rules,
            lambda rules: rules.accentw,
            words,
            repeats,
        ),
    ]

    table = render_table(results)
    print(table)

    if output:
        with open(output, "w", encoding="utf8") as output_file:
            output_file.write(table + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark accent-dictionary lookups."
    )
    parser.add_argument(
        "--words",
        default=None,
        help="Text corpus to pull words from (defaults to dictionary surface forms)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Use only this many words (default: 10000)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of timing repeats, minimum is reported (default: 3)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the comparison table to a markdown file",
    )
    args = parser.parse_args()

    main(args.words, args.limit, args.repeat, args.output)
