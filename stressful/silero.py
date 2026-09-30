"""
MIT License

Copyright (c) 2020-present Silero Team

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

import json
import os
import re
import unicodedata
from collections import OrderedDict

import numpy as np
import onnxruntime as ort

from .settings import STRESS_TOKEN

VOWELS = "аоуыэиеяёю"


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _word_ngrams(text: str) -> list[str]:
    text = "<" + text + ">"
    grams = []
    for n in range(1, len(text) + 1):
        for i in range(len(text) - n + 1):
            grams.append(text[i : i + n])
    return grams


def _decapitalize_stress(stressed_word: str) -> str:
    for idx, char in enumerate(stressed_word):
        if char.isupper():
            return stressed_word[:idx] + "+" + stressed_word[idx:].lower()
    return stressed_word


_LETTER_RE = re.compile(r"[А-Яа-яёЁ]")


def _letter_positions(word: str) -> list[int]:
    """Positions of Cyrillic letters in ``word``.

    Mirrors the tokenizer cleaning (lowercase, drop everything that is not a
    Cyrillic letter), so positions stored for a cleaned word can be mapped
    back onto the raw token, which may still contain punctuation or marks.
    """
    return [i for i, char in enumerate(word.lower()) if _LETTER_RE.fullmatch(char)]


class _BasicTokenizer:
    def __init__(self, never_split=None):
        self.never_split = set(never_split) if never_split else set()

    def tokenize(self, text, never_split=None):
        never_split = (
            self.never_split.union(set(never_split))
            if never_split
            else self.never_split
        )
        text = self._clean_text(text)

        orig_tokens = text.strip().split()
        split_tokens = []
        for token in orig_tokens:
            if token in never_split:
                split_tokens.append(token)
            else:
                split_tokens.extend(self._run_split_on_punc(token, never_split))

        return " ".join(split_tokens).split()

    @staticmethod
    def _clean_text(text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, token, never_split=None):
        never_split = never_split or set()
        if token in never_split:
            return [token]

        chars = list(token)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]


def _is_control(char):
    if char in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(char).startswith("C")


def _is_whitespace(char):
    if char in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(char) == "Zs"


def _is_punctuation(char):
    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(char).startswith("P")


class _WordpieceTokenizer:
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        output_tokens = []
        for token in text.split():
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1

                if cur_substr is None:
                    is_bad = True
                    break

                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens


class BertTokenizer:
    def __init__(self, config: dict, vocab: OrderedDict):
        self.vocab = vocab
        self.cls_token = config["cls_token"]
        self.sep_token = config["sep_token"]
        self.pad_token = config["pad_token"]
        self.unk_token = config["unk_token"]
        self.mask_token = config["mask_token"]
        self.never_split = set(config.get("never_split", []))

        self.cls_token_id = self.vocab[self.cls_token]
        self.sep_token_id = self.vocab[self.sep_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.mask_token_id = self.vocab[self.mask_token]
        self.homo_start_id = self.vocab["[HOMO]"]
        self.homo_end_id = self.vocab["[/HOMO]"]

        self.basic_tokenizer = _BasicTokenizer(never_split=self.never_split)
        self.wordpiece_tokenizer = _WordpieceTokenizer(
            vocab=self.vocab, unk_token=self.unk_token
        )

    def tokenize(self, text):
        tokens = self.basic_tokenizer.tokenize(text, never_split=self.never_split)
        return self.wordpiece_tokenizer.tokenize(" ".join(tokens))

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.unk_token_id)
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def encode(self, text):
        tokens = self.tokenize(text)
        tokens = [self.cls_token] + tokens + [self.sep_token]
        return self.convert_tokens_to_ids(tokens)


class SileroAccentor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir

        providers = ["CPUExecutionProvider"]
        self.homo_session = ort.InferenceSession(
            os.path.join(model_dir, "homo.onnx"), providers=providers
        )
        self.stress_session = ort.InferenceSession(
            os.path.join(model_dir, "stress_clf.onnx"), providers=providers
        )
        self.yo_session = ort.InferenceSession(
            os.path.join(model_dir, "yo_clf.onnx"), providers=providers
        )

        self.embedding_weight = np.load(os.path.join(model_dir, "embedding.npy"))
        with open(os.path.join(model_dir, "ngram_dict.json")) as f:
            self.ngram_dict = json.load(f)
        with open(os.path.join(model_dir, "exceptions.json")) as f:
            self.exceptions = {
                word: tuple(value) for word, value in json.load(f).items()
            }
        with open(os.path.join(model_dir, "homodict.json")) as f:
            self.homodict = json.load(f)
        self.yohomodict = {
            word: variants
            for word, variants in self.homodict.items()
            if any("ё" in variant for variant in variants)
        }

        with open(os.path.join(model_dir, "phrases.json")) as f:
            phrases = json.load(f)
        self.compiled_phrases = {
            word: re.compile(pattern, re.IGNORECASE)
            for word, pattern in phrases.items()
        }

        with open(os.path.join(model_dir, "bert_config.json")) as f:
            bert_config = json.load(f)
        vocab = self._load_vocab(os.path.join(model_dir, "bert_vocab.txt"))
        self.tokenizer = BertTokenizer(bert_config, vocab)

        self.vowels = VOWELS
        self.pattern = re.compile(r"(?=.*[а-яё])[а-яё+]+", re.IGNORECASE)

        self._re_remove_extra = re.compile(r"[^a-zA-Zа-яА-ЯёЁ0-9\s.!?,\-]")
        self._re_spaces = re.compile(r"\s+")
        self._re_double_dash = re.compile(r"-{2,}")
        self._re_repeat_punct = re.compile(r"([.!?])\1+")
        self._re_repeat_comma = re.compile(r",{2,}")
        self._re_space_before_punct = re.compile(r"\s+([.,!?])")
        self._re_punct_add_space = re.compile(r"([.,!?])(?=\S)")
        self.window_size = 300

    @staticmethod
    def _load_vocab(path: str) -> OrderedDict:
        vocab = OrderedDict()
        with open(path, encoding="utf-8") as reader:
            for index, token in enumerate(reader):
                vocab[token.rstrip("\n")] = index
        return vocab

    def __call__(
        self,
        sentence: str,
        put_stress: bool = True,
        put_stress_homo: bool = True,
        put_yo: bool = True,
        put_yo_homo: bool = True,
        stress_single_vowel: bool = True,
        words_to_ignore=None,
    ) -> str:
        solved = self._solve_homographs(
            sentence,
            put_stress=put_stress_homo,
            put_yo=put_yo_homo,
            stress_single_vowel=stress_single_vowel,
            words_to_ignore=words_to_ignore,
        )

        return self._accentuate(
            solved,
            put_stress=put_stress,
            put_yo=put_yo,
            stress_single_vowel=stress_single_vowel,
            skip_stress_words=None if put_stress_homo else self.homodict,
            skip_yo_words=None if put_yo_homo else self.yohomodict,
            words_to_ignore=words_to_ignore,
        )

    def _embed(self, words: list[str]) -> np.ndarray:
        result = np.empty(
            (len(words), self.embedding_weight.shape[1]), dtype=np.float32
        )

        for i, word in enumerate(words):
            indexes = [
                self.ngram_dict[gram]
                for gram in _word_ngrams(word)
                if gram in self.ngram_dict
            ]
            if not indexes:
                indexes = [self.ngram_dict["UNK"]]
            result[i] = self.embedding_weight[indexes].mean(axis=0)

        return result

    def _get_model_preds(self, words):
        embeddings = self._embed(words)

        stress_logits = self.stress_session.run(None, {"emb": embeddings})[0]
        yo_logits = self.yo_session.run(None, {"emb": embeddings})[0]

        stress_probs = _softmax(stress_logits, axis=1)
        stress_preds = np.argmax(stress_probs, axis=1)
        yo_probs = _softmax(yo_logits, axis=1)
        yo_preds = np.argmax(yo_probs, axis=1)

        return stress_probs, stress_preds, yo_probs, yo_preds

    def _accentuate(
        self,
        sentence,
        put_stress=True,
        put_yo=True,
        stress_single_vowel=False,
        skip_stress_words=None,
        skip_yo_words=None,
        words_to_ignore=None,
    ):
        skip_stress_words = skip_stress_words if skip_stress_words is not None else []
        skip_yo_words = skip_yo_words if skip_yo_words is not None else []

        if not (put_stress or put_yo):
            return sentence

        raw_tokens, clean_tokens, prediction_mask = self._tokenize(
            sentence, words_to_ignore
        )

        stress_probs, stress_preds, yo_probs, yo_preds = self._get_model_preds(
            clean_tokens
        )

        accented_sentence = []
        for word_idx, (raw_word, clean_word, need_processing) in enumerate(
            zip(raw_tokens, clean_tokens, prediction_mask)
        ):
            raw_word_lower = raw_word.lower()

            if not need_processing:
                accented_sentence.append(raw_word)
                continue

            have_stress = STRESS_TOKEN in raw_word_lower
            have_yo = "ё" in raw_word_lower
            if have_stress is True and have_yo is True:
                accented_sentence.append(raw_word)
                continue
            if have_stress is False and have_yo is True and put_stress:
                if (
                    sum(c in self.vowels for c in raw_word_lower) == 1
                    and not stress_single_vowel
                ) or clean_word.replace("ё", "е") in skip_stress_words:
                    accented_sentence.append(raw_word)
                    continue
                user_yo_positions = [
                    i for i, x in enumerate(raw_word_lower) if x == "ё"
                ]
                for i, yo_pos in enumerate(user_yo_positions):
                    raw_word = (
                        raw_word[: yo_pos + i] + STRESS_TOKEN + raw_word[(yo_pos + i) :]
                    )
                accented_sentence.append(raw_word)
                continue

            if clean_word in self.exceptions:
                accented_sentence.append(
                    self._accentuate_exception(
                        clean_word=clean_word,
                        raw_word=raw_word,
                        have_stress=have_stress,
                    )
                )
                continue

            stressed_vowel_ids = [stress_preds[word_idx]]
            passed_stress_trs = stress_probs[word_idx][stressed_vowel_ids[0]] > (
                0.5 if put_stress else 1
            )
            set_stress = (
                passed_stress_trs
                and not have_stress
                and (clean_word.replace("ё", "е") not in skip_stress_words)
            )

            yo_vowel_ids = [yo_preds[word_idx]] if yo_preds is not None else [-10]
            passed_yo_trs = yo_preds is not None and yo_probs[word_idx][
                yo_vowel_ids[0]
            ] > (0.5 if put_yo else 1)
            set_yo = passed_yo_trs and (
                clean_word.replace("ё", "е") not in skip_yo_words
            )

            if have_stress:
                stressed_vowel_ids = [
                    sum(map(stressed_part.count, self.vowels))
                    for stressed_part in raw_word_lower.split(STRESS_TOKEN)
                ]

            stress_positions, yo_positions, num_vowels, first_vowel_pos = (
                self._get_positions(raw_word_lower, stressed_vowel_ids, yo_vowel_ids)
            )
            if num_vowels == 0:
                accented_sentence.append(raw_word)
                continue

            for yo_pos in yo_positions:
                if yo_pos in stress_positions and set_yo:
                    if raw_word_lower[yo_pos] == "е":
                        raw_word = (
                            raw_word[:yo_pos]
                            + ("ё" if raw_word[yo_pos].islower() else "Ё")
                            + raw_word[(yo_pos + 1) :]
                        )

            if num_vowels == 1:
                stress_positions = [first_vowel_pos]
                set_stress = stress_single_vowel and put_stress

            if not have_stress and set_stress:
                for i, stress_pos in enumerate(stress_positions):
                    raw_word = (
                        raw_word[: (stress_pos + i)]
                        + STRESS_TOKEN
                        + raw_word[(stress_pos + i) :]
                    )

            accented_sentence.append(raw_word)

        return self._fuse_words_to_sentence(accented_sentence)

    def _get_positions(self, word, stressed_vowel_ids, yo_vowel_ids):
        vowel_ids = [i for i, c in enumerate(word) if c in self.vowels]
        ye_ids = [i for i, c in enumerate(word) if c == "е"]

        stress_positions = [
            vowel_ids[idx]
            for idx in stressed_vowel_ids
            if (idx < len(vowel_ids)) and (len(vowel_ids) > 0)
        ]
        yo_positions = [
            ye_ids[idx - 1]
            for idx in yo_vowel_ids
            if (idx > 0) and (idx - 1 < len(ye_ids)) and (len(ye_ids) > 0)
        ]

        num_vowels = len(vowel_ids)
        first_vowel_pos = vowel_ids[0] if len(vowel_ids) > 0 else -1
        return stress_positions, yo_positions, num_vowels, first_vowel_pos

    def _accentuate_exception(self, clean_word, raw_word, have_stress):
        exc_stress = self.exceptions[clean_word][0]
        exc_yo = self.exceptions[clean_word][1]

        # Indices in exceptions.json are relative to the cleaned word
        # (lowercased, punctuation stripped). Map them onto raw_word, which
        # may still contain punctuation, so slicing happens at the right
        # characters.
        positions = _letter_positions(raw_word)
        exc_stress = positions[exc_stress]
        exc_yo = positions[exc_yo] if exc_yo != -1 else -1

        if have_stress:
            user_stress_token_positions = [
                i for i, c in enumerate(raw_word) if c == STRESS_TOKEN
            ]
            accentuated_word = raw_word.replace(STRESS_TOKEN, "")
            if exc_yo != -1 and (exc_yo + 1 in user_stress_token_positions):
                accentuated_word = (
                    accentuated_word[:exc_yo]
                    + ("ё" if accentuated_word[exc_yo].islower() else "Ё")
                    + accentuated_word[(exc_yo + 1) :]
                )
            for stress_token_pos in user_stress_token_positions:
                accentuated_word = (
                    accentuated_word[:stress_token_pos]
                    + STRESS_TOKEN
                    + accentuated_word[stress_token_pos:]
                )
        elif not have_stress:
            if exc_yo != -1:
                raw_word = (
                    raw_word[:exc_yo]
                    + ("ё" if raw_word[exc_yo].islower() else "Ё")
                    + raw_word[(exc_yo + 1) :]
                )
            accentuated_word = (
                raw_word[:exc_stress] + STRESS_TOKEN + raw_word[exc_stress:]
            )

        return accentuated_word

    def _tokenize(self, sentence, words_to_ignore=None):
        words_to_ignore = words_to_ignore if words_to_ignore is not None else []

        tokens = []
        model_inputs = []
        prediction_mask = []

        for word in re.split(r"([\s.,!?;:<>=()/\\]+)", sentence):
            parts = word.split("-")

            if len(parts) == 1:
                cur_tokens = parts
                cur_prediction_mask = [True]
            else:
                cur_tokens = [part + "-" for part in parts[:-1]] + [parts[-1]]
                cur_prediction_mask = [True for p in parts[:-1]] + [parts[-1] != "то"]

            cur_model_inputs = [
                re.sub(r"[^А-Яа-яёЁ]", "", token.lower()) for token in cur_tokens
            ]
            cur_prediction_mask = [
                ((len(x) > 0) and (x not in words_to_ignore)) & mask
                for x, mask in zip(cur_model_inputs, cur_prediction_mask)
            ]

            tokens.extend(cur_tokens)
            model_inputs.extend(cur_model_inputs)
            prediction_mask.extend(cur_prediction_mask)

        return tokens, model_inputs, prediction_mask

    @staticmethod
    def _fuse_words_to_sentence(words):
        return "".join(words).replace("-", "-")

    def _solve_homographs(
        self,
        sentence,
        put_stress=True,
        put_yo=True,
        stress_single_vowel=True,
        words_to_ignore=None,
    ):
        if not (put_stress or put_yo):
            return sentence

        tagged = self._find_and_tag_homos(sentence, words_to_ignore=words_to_ignore)

        batch_starts = []
        batch_ends = []
        batch_sents = []

        starts = []
        ends = []
        words = []
        regex_preds = []
        is_neural_preds = []

        for start, end, word, word_lower, raw_mark, raw_clean in tagged:
            if raw_mark is None:
                continue

            regex_pred = None
            if word_lower in self.compiled_phrases:
                regex_pred = self._predict_with_pattern(
                    self.compiled_phrases[word_lower], raw_mark
                )

            if regex_pred is not None:
                regex_preds.append(regex_pred)
            else:
                if word_lower not in self.homodict:
                    continue
                bert_ids = self.tokenizer.encode(raw_mark)
                homo_start = bert_ids.index(self.tokenizer.homo_start_id)
                homo_end = bert_ids.index(self.tokenizer.homo_end_id)
                batch_starts.append(homo_start)
                batch_ends.append(homo_end)
                batch_sents.append(bert_ids)

            starts.append(start)
            ends.append(end)
            words.append(word)
            is_neural_preds.append(regex_pred is None)

        if len(words) == 0:
            return sentence

        if len(batch_sents) > 0:
            max_len = max(len(ids) for ids in batch_sents)
            input_ids = np.full(
                (len(batch_sents), max_len),
                self.tokenizer.pad_token_id,
                dtype=np.int64,
            )
            for i, ids in enumerate(batch_sents):
                input_ids[i, : len(ids)] = ids

            logits = self.homo_session.run(
                None,
                {
                    "input_ids": input_ids,
                    "homo_start": np.array(batch_starts, dtype=np.int64),
                    "homo_end": np.array(batch_ends, dtype=np.int64),
                },
            )[0]
            homosolver_preds = np.round(_sigmoid(logits))

        regex_idx = 0
        homosolver_idx = 0
        word_preds = []
        for word, is_neural in zip(words, is_neural_preds):
            if is_neural:
                pred = int(homosolver_preds[homosolver_idx][0])
                word_pred = sorted(self.homodict[word.lower()])[pred]
                homosolver_idx += 1
            else:
                word_pred = regex_preds[regex_idx]
                regex_idx += 1
            word_preds.append(word_pred)

        stressed_sent = sentence
        offset = 0
        for start, end, word, word_pred in zip(starts, ends, words, word_preds):
            start = start + offset
            end = end + offset
            word_pred = word_pred if put_yo else word_pred.replace("ё", "е")
            n_vowels = sum(c.lower() in self.vowels for c in word_pred)
            stress_idx = word_pred.index("+")
            word_pred = word_pred.replace("+", "")
            word_pred = "".join(
                [
                    c2.lower() if c1.islower() else c2.upper()
                    for c1, c2 in zip(word, word_pred)
                ]
            )
            if (n_vowels > 1 or stress_single_vowel) and put_stress:
                word_pred = word_pred[:stress_idx] + "+" + word_pred[stress_idx:]
                offset += 1
            stressed_sent = stressed_sent[:start] + word_pred + stressed_sent[end:]

        return stressed_sent

    @staticmethod
    def _predict_with_pattern(pattern, text):
        match = pattern.search(text)
        if match:
            for group_name, matched_text in match.groupdict().items():
                if matched_text is not None:
                    return _decapitalize_stress(group_name)
        return None

    def _find_and_tag_homos(self, sentence, words_to_ignore=None):
        words_to_ignore = words_to_ignore if words_to_ignore is not None else []
        tagged = []

        for match in self.pattern.finditer(sentence):
            start, end = match.span()
            word = match.group()
            word_lower = word.lower()
            is_homograph = (
                (word_lower in self.homodict) or (word_lower in self.compiled_phrases)
            ) and (word_lower not in words_to_ignore)

            if is_homograph:
                raw_start_text = self._clean_text(sentence[:start], is_start=True)[
                    (-self.window_size // 2) :
                ]
                raw_end_text = self._clean_text(sentence[end:], is_start=False)[
                    : (self.window_size // 2)
                ]
                raw_mark = (
                    raw_start_text
                    + " [HOMO] "
                    + word_lower
                    + " [/HOMO] "
                    + raw_end_text
                ).strip()
                raw_clean = (
                    raw_start_text + " " + word_lower + " " + raw_end_text
                ).strip()
            else:
                raw_mark = None
                raw_clean = None

            tagged.append((start, end, word, word_lower, raw_mark, raw_clean))

        return tagged

    def _clean_text(self, text, is_start=True) -> str:
        if not text:
            return ""

        text = self._re_remove_extra.sub("", text)
        text = self._re_spaces.sub(" ", text)
        text = self._re_double_dash.sub(" - ", text)
        text = self._re_repeat_punct.sub(r"\1", text)
        text = self._re_repeat_comma.sub(",", text)
        text = self._re_space_before_punct.sub(r"\1", text)
        text = self._re_repeat_punct.sub(r"\1", text)
        text = self._re_repeat_comma.sub(",", text)
        text = self._re_punct_add_space.sub(r"\1 ", text)
        text = self._re_spaces.sub(" ", text).strip()

        if is_start:
            text = text.lstrip(" .,!?-")
            if text:
                text = text[0].upper() + text[1:].lower()
        else:
            if text:
                text = text[0] + text[1:].lower()
            if text and text[-1] not in ".!?":
                text += "."
        return text
