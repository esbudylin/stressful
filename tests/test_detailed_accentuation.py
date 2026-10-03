import unittest

from parameterized import parameterized

from stressful import AccentSource, Accentuator, WordAccentuation


class TestAccentuateDetailed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accentuator = Accentuator()

    @parameterized.expand(
        [
            (
                "кот",
                [
                    WordAccentuation("кот", AccentSource.NONE, [0.0]),
                ],
            ),
            (
                "еще",
                [
                    WordAccentuation("еще", AccentSource.DICT, [0.0, 1.0]),
                ],
            ),
            (
                "темно-синий",
                [
                    WordAccentuation(
                        "темно-синий",
                        AccentSource.DICT,
                        [1.0, 0.0, 1.0, 0.0],
                    ),
                ],
            ),
            (
                "по-русски",
                [
                    WordAccentuation(
                        "по-русски",
                        AccentSource.DICT,
                        [0.0, 1.0, 0.0],
                    ),
                ],
            ),
            (
                "какой-нибудь",
                [
                    WordAccentuation(
                        "какой-нибудь",
                        AccentSource.DICT,
                        [0.0, 1.0, 0.0, 0.0],
                    ),
                ],
            ),
        ]
    )
    def test_deterministic_probabilities(self, line, expected):
        self.assertEqual(self.accentuator.accentuate_detailed(line), expected)

    @parameterized.expand(
        [
            (
                "перед",
                [
                    ("перед", AccentSource.NEURAL, False),
                ],
            ),
            (
                "чашка-то",
                [
                    ("чашка-то", AccentSource.NEURAL, False),
                ],
            ),
            (
                "давай-ка",
                [
                    ("давай-ка", AccentSource.NEURAL, False),
                ],
            ),
            (
                "потом",
                [
                    ("потом", AccentSource.NEURAL, True),
                ],
            ),
            (
                "йошкин",
                [
                    ("йошкин", AccentSource.NEURAL, False),
                ],
            ),
            (
                "Он знаком больше с армяком;",
                [
                    ("Он", AccentSource.NONE, False),
                    ("знаком", AccentSource.NEURAL, True),
                    ("больше", AccentSource.DICT, False),
                    ("армяком", AccentSource.DICT, False),
                ],
            ),
            (
                "Сердца бедного занывающую грусть.",
                [
                    ("Сердца", AccentSource.NEURAL, True),
                    ("бедного", AccentSource.DICT, False),
                    ("занывающую", AccentSource.NEURAL, False),
                    ("грусть", AccentSource.NONE, False),
                ],
            ),
        ]
    )
    def test_classification(self, line, expected):
        result = self.accentuator.accentuate_detailed(line)
        classified = [(wa.word, wa.source, wa.is_homograph) for wa in result]

        self.assertEqual(classified, expected)


if __name__ == "__main__":
    unittest.main()
