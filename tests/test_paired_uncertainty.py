import unittest

from paper.paired_uncertainty import holm_adjust, paired_effect


def episodes(values):
    return [{"task": "task", "episode_index": i, "success": x} for i, x in enumerate(values)]


class PairedUncertaintyTests(unittest.TestCase):
    def test_identical_arms_preserve_pairing_and_warn_about_degenerate_interval(self):
        rows = episodes([True, False] * 8)
        result = paired_effect(rows, rows, replicates=200)
        self.assertEqual(result["pointwise_ci_pp"], [0, 0])
        self.assertEqual(result["exact_mcnemar_two_sided_p"], 1)
        self.assertTrue(result["zero_discordance_warning"])

    def test_discordance_sign_exact_test_and_reproducibility(self):
        left, right = episodes([False] * 16), episodes([True] * 8 + [False] * 8)
        result = paired_effect(left, right, replicates=1000)
        self.assertEqual(result["difference_pp"], 50)
        self.assertEqual(result["exact_mcnemar_two_sided_p"], 2 / 2**8)
        self.assertEqual(result, paired_effect(left, right, replicates=1000))
        inverse = paired_effect(right, left, replicates=1000)
        self.assertEqual(inverse["difference_pp"], -50)
        self.assertEqual(inverse["exact_mcnemar_two_sided_p"], result["exact_mcnemar_two_sided_p"])

    def test_tasks_are_fixed_strata(self):
        left, right = episodes([False] * 8), episodes([True] * 8)
        left += [{"task": "other", "episode_index": i, "success": True} for i in range(8)]
        right += [{"task": "other", "episode_index": i, "success": False} for i in range(8)]
        result = paired_effect(left, right, replicates=200)
        self.assertEqual(result["pointwise_ci_pp"], [0, 0])
        self.assertEqual(result["discordant_episodes"], 16)
        self.assertFalse(result["zero_discordance_warning"])

    def test_bad_pairing_and_non_boolean_outcomes_rejected(self):
        left = episodes([True, False])
        with self.assertRaisesRegex(ValueError, "do not pair"):
            paired_effect(left, list(reversed(left)))
        with self.assertRaisesRegex(ValueError, "Boolean"):
            paired_effect(left, episodes([1, 0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            paired_effect(left * 2, left * 2)

    def test_holm_is_monotone_in_sorted_raw_p(self):
        self.assertEqual(holm_adjust({"a": .001, "b": .02, "c": .021, "d": .5}),
                         {"a": .004, "b": .06, "c": .06, "d": .5})


if __name__ == "__main__":
    unittest.main()
