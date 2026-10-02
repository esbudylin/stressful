"""
MIT License

Copyright (c) 2021 yuliya1324

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from typing import Iterable, Iterator

from .settings import ACCENT_DICT_PATHS, SILERO_MODEL_DIR, STRESS_TOKEN
from .silero import SileroAccentor

unstressed_prefixes = ["по-"]
unstressed_postfixes = ["-ка", "-нибудь"]


@dataclass
class AccentEntry:
    endings: set[str]
    capitalized: bool
    accents: list[int]
    secondary_accents: list[int]
    yo: list[int]
    no_accent: bool


def parse_word_entry(word: str) -> tuple[str, set[str]]:
    base = word
    endings = [""]

    lpar = word.find("(")

    if lpar != -1:
        base = word[:lpar]
        rpar = word.find(")", lpar)

        if rpar == -1 or rpar + 1 != len(word):
            raise ValueError("Can't parse word entry")

        endings_str = word[lpar + 1 : rpar]
        endings = endings_str.split("|")

    return base, set(endings)


def parse_accent_entry(accent: str) -> dict:
    secondary_accent_mark = "`"
    yo_mark = '"'

    accents = []
    secondary_accents = []
    yo = []
    no_accent = False
    capitalized = accent.endswith("!")

    for accent_entry in accent.rstrip("!").split(","):
        if not accent_entry:
            continue

        sep = 0
        for c in accent_entry:
            if not c.isdigit():
                break
            sep += 1

        if not sep:
            raise ValueError("Accent position is not specified")

        accent_pos = int(accent_entry[:sep])
        symbol = accent_entry[sep:]

        if len(symbol) > 1:
            raise ValueError(f"Too many symbols: {accent_entry}")

        if accent_pos == 0:
            no_accent = True
        elif symbol == secondary_accent_mark:
            secondary_accents.append(accent_pos)
        elif symbol == yo_mark:
            yo.append(accent_pos)
        elif symbol == "":
            accents.append(accent_pos)
        else:
            raise ValueError(f"Invalid accent symbol: {symbol!r}")

    return dict(
        accents=accents,
        secondary_accents=secondary_accents,
        yo=yo,
        capitalized=capitalized,
        no_accent=no_accent,
    )


def parse_dict_entry(word: str, accent: str) -> tuple[str, AccentEntry]:
    base, endings = parse_word_entry(word)

    entry = AccentEntry(
        endings=endings,
        **parse_accent_entry(accent),
    )

    return base, entry


def read_accent_dicts(filenames: Iterable[str]) -> Iterator[str]:
    for filename in filenames:
        with open(filename, encoding="utf8") as file_read:
            yield from file_read


def build_accent_dict(
    rows: Iterator[str],
) -> defaultdict[str, list[AccentEntry]]:
    result: defaultdict[str, list[AccentEntry]] = defaultdict(list)

    for row in rows:
        if row.startswith("#"):
            continue

        if split := row.split():
            word, accent = split
            try:
                base, entry = parse_dict_entry(word, accent)
                result[base].append(entry)
            except Exception as e:
                logging.warning(
                    "Error while parsing dict entry %s %s: %s",
                    word,
                    accent,
                    e,
                )

    return result


def is_vowel(char):
    vowels = "аеиоуыэюяёАЕИОУЫЭЮЯЁ"

    return char in vowels


def vowel_count(word):
    return sum(map(is_vowel, word))


def accent_line(
    line: str,
    accent_dict: defaultdict[str, list[AccentEntry]],
    silero_accentor: SileroAccentor,
) -> list[bool]:
    line_stripped = re.sub(r"[^А-яЁё\s-]+", "", line)
    words = list(filter(vowel_count, line_stripped.split()))
    words_nacc = None

    def extract_words_nacc():
        res = extract_neuro_accents(line, silero_accentor)
        if len(words) != len(res):
            raise ValueError(
                f"""Number of words with vowels ({len(words)})
                does not match number of neuro-accented words ({len(res)})"""
            )
        return res

    res = []

    for j, word in enumerate(words):
        if is_word_without_accent(word):
            word_mask = base_mask(word)
        else:
            accent_entry = find_accent_entry(word, accent_dict)

            if not accent_entry or should_use_neuro_accent(word, accent_entry):
                if words_nacc is None:
                    words_nacc = extract_words_nacc()

                word_mask = words_nacc[j]
            else:
                word_mask = accent_word_by_dict(word, accent_entry)

        res += apply_special_rules(word, word_mask)

    return res


def apply_special_rules(word: str, mask: list[bool]) -> list[bool]:
    if not mask:
        return mask

    for prefix in unstressed_prefixes:
        if word.startswith(prefix):
            vowels = vowel_count(prefix)
            mask[:vowels] = [False] * vowels
            break

    for postfix in unstressed_postfixes:
        if word.endswith(postfix):
            vowels = vowel_count(postfix)
            mask[-vowels:] = [False] * vowels
            break

    return mask


def accent_word_by_dict(word: str, accent_entry: AccentEntry) -> list[bool]:
    mask = base_mask(word)

    all_accents = (
        accent_entry.accents + accent_entry.secondary_accents + accent_entry.yo
    )

    for accent in all_accents:
        accent_pos = min(len(mask) - 1, accent - 1)
        mask[accent_pos] = True

    return mask


def should_use_neuro_accent(word: str, accent: AccentEntry) -> bool:
    """
    Слово берется из строки, размеченной нейросетевым
    акцентуатором, если

    1) словарный акцентуатор поставил в этом слове ударение в двух
    местах, или не поставил совсем и при этом в нем нет буквы ё,

    2) словарный акцентуатор поставил и ударение, и
    букву ё (кроме слов через дефис: например, тёмно-си'ний )

    См. https://trudy.ruslang.ru/ru/archive/2022-3/181-190
    """
    if accent.no_accent:
        return False

    if len(accent.accents) > 1 or len(accent.secondary_accents) > 1:
        return True

    if accent.accents and accent.yo:
        return "-" not in word

    return False


def is_word_without_accent(word: str) -> bool:
    """
    Слово остается без ударения, если
    1) в нем нет гласных,
    2) оно односложное (оно содержит одну гласную)
    """
    vowels = vowel_count(word)

    return vowels < 2


def normalize(s):
    if re.match("^[А-Я]", s):
        caps = True
    else:
        caps = False
    return s.lower(), caps


def base_mask(word):
    return [False for c in word if is_vowel(c)]


def find_accent_entry(
    word: str,
    accent_dict: defaultdict[str, list[AccentEntry]],
) -> AccentEntry | None:
    key, capitalized = normalize(word)
    found_entry = None

    for i in range(len(key), -1, -1):
        entries = accent_dict[key[0:i]]
        ending = key[i:]

        for entry in entries:
            if ending in entry.endings:
                if not capitalized and entry.capitalized:
                    continue
                found_entry = entry
                break
        if found_entry:
            break

    if not found_entry:
        return None

    return found_entry


def extract_accent_mask(text: str) -> list[bool]:
    result = []

    def is_accent_mark(char):
        return char and char == STRESS_TOKEN

    pending_accent = False

    for i, char in enumerate(text):
        if is_accent_mark(char):
            pending_accent = True

        if is_vowel(char):
            result.append(pending_accent)
            pending_accent = False

    return result


def extract_neuro_accents(
    line: str,
    silero_accentor: SileroAccentor,
) -> list[list[bool]]:
    words = silero_accentor(line).split()

    res = []
    for word in words:
        if mask := extract_accent_mask(word):
            res.append(mask)

    return res


class Accentuator:
    def __init__(self):
        self.accent_dict = build_accent_dict(read_accent_dicts(ACCENT_DICT_PATHS))
        self.silero = SileroAccentor(SILERO_MODEL_DIR)

    def accentuate(self, line: str) -> list[bool]:
        return accent_line(line, self.accent_dict, self.silero)

    def mark_stresses(
        self,
        line: str,
        stress_mark: str | None = None,
    ) -> str:
        stress_mark = stress_mark or STRESS_TOKEN
        mask = self.accentuate(line)
        vowels = count()
        res = []

        for c in line:
            res.append(c)

            if not is_vowel(c):
                continue

            if mask[next(vowels)]:
                res.append(stress_mark)

        return "".join(res)
