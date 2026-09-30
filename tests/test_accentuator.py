import unittest

from parameterized import parameterized

from stressful import Accentuator
from stressful.accentuator import vowel_count


def mask(bits: str) -> list[bool]:
    return [bit == "1" for bit in bits]


class TestAccentuator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accentuator = Accentuator()

    @parameterized.expand(
        [
            ("Еще вкруг солнцев не вращались", "010100010"),
            ("Ваше Величество, мы прибыли ко дворцу", "1001000100001"),
            ("йошкин кот", "100"),
        ]
    )
    def test_accent_line(self, line, with_accents):
        self.assertEqual(self.accentuator.accentuate(line), mask(with_accents))

    @parameterized.expand(
        [
            ("легкий", "10"),
            ("темно-синий", "1010"),
            ("ёлка", "10"),
            ("еще", "01"),
            ("Еще", "01"),
            ("какой-нибудь", "0100"),
            ("что-то", "10"),
            ("какие-нибудь", "01000"),
            ("по-русски", "010"),
            ("по-волчьи", "010"),
            ("давай-ка", "010"),
            ("", ""),
        ]
    )
    def test_accent_word(self, word, with_accents):
        self.assertEqual(
            self.accentuator.accentuate(word),
            mask(with_accents),
            f"failed for {word}",
        )

    @parameterized.expand(
        [
            ("«Силен", "01"),
            ("«Лету»", "10"),
            ("«Рассек", "01"),
            ("(Маневры)", "010"),
            ("«Полет", "01"),
            ("«Мытье»", "01"),
            ("«Броней", "01"),
            ("«Чета", "10"),
            ("«Ежа».", "10"),
            ("«Силен ты, богат и славен,", "01001010"),
            (
                "«Чета! иди за мной, -- сказал отец судьбины. --",
                "1001000101010",
            ),
        ]
    )
    def test_exception_word_with_punctuation(self, line, with_accents):
        # Words present in exceptions.json used to shift stress/ё indices by
        # the leading punctuation, adding a phantom vowel to the mask.
        result = self.accentuator.accentuate(line)

        self.assertEqual(result, mask(with_accents))
        self.assertEqual(len(result), vowel_count(line))

    @parameterized.expand(
        [
            ("Силен", "«Силен"),
            ("Лету", "«Лету»"),
            ("Рассек", "«Рассек"),
            ("Маневры", "(Маневры)"),
            ("Полет", "«Полет"),
            ("Мытье", "«Мытье»"),
            ("Броней", "«Броней"),
            ("Чета", "«Чета"),
            ("Ежа", "«Ежа»."),
        ]
    )
    def test_punctuation_does_not_change_mask(self, bare, with_punctuation):
        self.assertEqual(
            self.accentuator.accentuate(bare),
            self.accentuator.accentuate(with_punctuation),
        )

    def test_mark_stresses(self):
        self.assertEqual(
            self.accentuator.mark_stresses("Это инструмент для разметки ударений"),
            "Э+то инструме+нт для разме+тки ударе+ний",
        )

        self.assertEqual(
            self.accentuator.mark_stresses("Вырыта дактилем яма глубокая", "!"),
            "Вы!рыта да!ктилем я!ма глубо!кая",
        )

        self.assertEqual(
            self.accentuator.mark_stresses("Ёк макарёк!"),
            "Ёк макарё+к!",
        )
