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

        self.embedding_weight = np.load(os.path.join(model_dir, "embedding.npy"))
        with open(os.path.join(model_dir, "ngram_dict.json")) as f:
            self.ngram_dict = json.load(f)
        with open(os.path.join(model_dir, "exceptions.json")) as f:
            self.exceptions = {
                word: tuple(value) for word, value in json.load(f).items()
            }
        with open(os.path.join(model_dir, "homodict.json")) as f:
            self.homodict = json.load(f)

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

        stress_probs = _softmax(stress_logits, axis=1)
        stress_preds = np.argmax(stress_probs, axis=1)

        return stress_probs, stress_preds

    def stress_distributions(self, words: list[str]) -> list[list[float]]:
        """Per-word stress probabilities over the word's vowel positions.

        The stress model is word-level and does not use sentential context,
        so each word is scored independently. The raw softmax output spans a
        fixed number of class positions; only the positions corresponding to
        the word's vowels are kept, and the values are not renormalized. Words
        with more vowels than the model has output classes are zero-padded to
        the word's vowel count.
        """
        clean_tokens = []
        spans = []

        for word in words:
            _, clean, _ = self._tokenize(word)
            start = len(clean_tokens)
            clean_tokens.extend(clean)
            spans.append((start, len(clean_tokens)))

        if not clean_tokens:
            return [[] for _ in words]

        stress_probs, _ = self._get_model_preds(clean_tokens)

        result = []

        for word, (start, end) in zip(words, spans):
            probs = []

            for i in range(start, end):
                n_vowels = sum(c in self.vowels for c in clean_tokens[i])
                part = stress_probs[i][:n_vowels].tolist()

                if len(part) < n_vowels:
                    part.extend([0.0] * (n_vowels - len(part)))

                probs.extend(part)

            result.append(probs)

        return result

    def word_parts(self, word: str) -> list[tuple[str, bool]]:
        """Split a word into accentuation parts.

        Parts are separated by hyphens; each is accented independently.
        Returns ``(clean_part, needs_processing)`` pairs, where
        ``needs_processing`` is ``False`` for enclitic parts such as the final
        ``-то``.
        """
        _, clean, mask = self._tokenize(word)
        return list(zip(clean, mask))


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


    def _resolve_homographs(self, sentence, words_to_ignore=None):
        """Pick a reading for each homograph in the sentence.

        Returns a list of ``(start, end, word, word_pred, confidence)`` in
        occurrence order, where ``confidence`` is the sigmoid score of the
        chosen reading (1.0 for deterministic phrase-based predictions).
        """
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
            return []

        homosolver_probs = None
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
            homosolver_probs = _sigmoid(logits)

        regex_idx = 0
        homosolver_idx = 0
        resolved = []

        for start, end, word, is_neural in zip(starts, ends, words, is_neural_preds):
            if is_neural:
                confidence = float(homosolver_probs[homosolver_idx][0])
                candidates = sorted(self.homodict[word.lower()])
                pred = int(np.round(confidence))
                word_pred = candidates[pred]
                homosolver_idx += 1
            else:
                word_pred = regex_preds[regex_idx]
                candidates = [word_pred]
                confidence = 1.0
                regex_idx += 1

            resolved.append((start, end, word, word_pred, confidence, candidates))

        return resolved

    @staticmethod
    def _stress_vowel_index(variant: str) -> int:
        index = 0
        for char in variant:
            if char == "+":
                return index
            if char in VOWELS:
                index += 1
        return index

    def _candidate_distribution(
        self,
        candidates: list[str],
        confidence: float,
    ) -> list[float]:
        num_vowels = sum(c in VOWELS for c in candidates[0])
        probs = [0.0] * num_vowels

        if len(candidates) == 1:
            probs[self._stress_vowel_index(candidates[0])] = 1.0
        else:
            probs[self._stress_vowel_index(candidates[0])] += 1.0 - confidence
            probs[self._stress_vowel_index(candidates[1])] += confidence

        return probs

    def homograph_distributions(
        self,
        sentence: str,
        words: list[str],
        words_to_ignore=None,
    ) -> dict[int, list[float]]:
        """Per-syllable stress distribution of each resolved homograph.

        Returns ``{word_index: probabilities}``, keyed by the index of the
        word in ``words``. Both ``words`` and the resolved homographs are in
        text order, so occurrences are matched by walking a cursor forward.
        Words not present in ``words`` as a whole token (e.g. a hyphen part)
        are skipped. ``probabilities`` is spread over the stressed vowels of
        the candidate readings; deterministic phrase predictions yield a
        one-hot vector.
        """
        result = {}
        cursor = 0

        for _, _, word, _, confidence, candidates in self._resolve_homographs(
            sentence, words_to_ignore
        ):
            target = word.lower()

            for index in range(cursor, len(words)):
                if words[index].lower() == target:
                    result[index] = self._candidate_distribution(
                        candidates, confidence
                    )
                    cursor = index + 1
                    break

        return result

    @staticmethod
    def _letter_to_vowel_index(word: str, letter_index: int) -> int:
        return sum(c in VOWELS for c in word[:letter_index])

    def exception_distribution(self, word: str) -> list[float] | None:
        """Per-syllable stress distribution for an exception word.

        Returns ``None`` if the word is not an exception. Exception words carry
        deterministic stress (and optionally ё), so the distribution is one-hot.
        """
        clean = re.sub(r"[^А-Яа-яёЁ]", "", word.lower())

        if clean not in self.exceptions:
            return None

        num_vowels = sum(c in VOWELS for c in clean)
        probs = [0.0] * num_vowels

        stress_pos, yo_pos = self.exceptions[clean]
        probs[self._letter_to_vowel_index(clean, stress_pos)] = 1.0

        if yo_pos != -1:
            probs[self._letter_to_vowel_index(clean, yo_pos)] = 1.0

        return probs

    def accent_word_probabilities(
        self,
        line: str,
        words: list[str],
    ) -> list[tuple[list[float], bool]]:
        """Per-word stress probabilities, resolving hyphen parts.

        Each hyphen-separated part is an independent accentuation unit and may
        be an enclitic (no stress), a homograph, an exception, or a model
        prediction. Returns ``(probabilities, is_homograph)`` per word, with
        the probabilities concatenated over the word's parts.
        """
        parts = [
            (index, text, needs_processing)
            for index, word in enumerate(words)
            for text, needs_processing in self.word_parts(word)
            if sum(c in self.vowels for c in text)
        ]

        if not parts:
            return [([], False) for _ in words]

        texts = [text for _, text, _ in parts]
        distributions = self.stress_distributions(texts)
        homographs = self.homograph_distributions(line, texts)

        probabilities = [[] for _ in words]
        is_homograph = [False] * len(words)

        for i, (index, text, needs_processing) in enumerate(parts):
            num_vowels = sum(c in self.vowels for c in text)

            if not needs_processing:
                probs = [0.0] * num_vowels
            elif i in homographs:
                probs = homographs[i]
                is_homograph[index] = True
            else:
                exception_probs = self.exception_distribution(text)

                if exception_probs is not None:
                    probs = exception_probs
                else:
                    probs = distributions[i]

            probabilities[index].extend(probs)

        return [
            (probabilities[index], is_homograph[index]) for index in range(len(words))
        ]


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
