import unittest

from parameterized import parameterized

from stressful import Accentuator


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
