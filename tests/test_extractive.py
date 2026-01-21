"""
Unit tests for the from-scratch extractive summarizer
(summarizer/extractive.py). These tests do not require an API key,
network access, or a running Flask server — they exercise only pure
functions from the standard library.

Run with:
    python -m pytest tests/ -v
or:
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summarizer import extractive


SAMPLE_TEXT = """
The city council voted on Tuesday to approve funding for a new public
library branch in the downtown district. Officials said the project,
first proposed three years ago, will cost an estimated 4.2 million
dollars and take about eighteen months to complete. Supporters argue
that the new branch will improve access to books and digital resources
for residents in a part of the city currently underserved by public
facilities. The library will include a dedicated children's reading
room, a small business resource center, and free community meeting
spaces available for local groups. Critics of the plan, including two
council members who voted against it, questioned whether the city
could afford the project given ongoing budget pressures elsewhere.
Construction is expected to begin early next year, with an opening
planned before the following winter. City planners say the branch is
part of a broader ten-year strategy to modernize public infrastructure
across several underserved neighborhoods.
"""


class TestCleanText(unittest.TestCase):
    def test_collapses_whitespace(self):
        messy = "Hello    world.\n\nThis  is\ttext."
        cleaned = extractive.clean_text(messy)
        self.assertNotIn("  ", cleaned)
        self.assertNotIn("\n", cleaned)
        self.assertNotIn("\t", cleaned)

    def test_empty_input(self):
        self.assertEqual(extractive.clean_text(""), "")
        self.assertEqual(extractive.clean_text("   \n\n  "), "")


class TestSplitSentences(unittest.TestCase):
    def test_basic_split(self):
        text = "This is one sentence. This is another sentence. And a third."
        sentences = extractive.split_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertTrue(sentences[0].startswith("This is one"))
        self.assertTrue(sentences[-1].startswith("And a third"))

    def test_handles_abbreviations_without_over_splitting(self):
        text = "Dr. Smith met Mr. Jones at 3 p.m. yesterday. They discussed the report."
        sentences = extractive.split_sentences(text)
        # Should split into exactly 2 sentences, not on "Dr." or "Mr."
        self.assertEqual(len(sentences), 2)

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(extractive.split_sentences(""), [])
        self.assertEqual(extractive.split_sentences("   "), [])

    def test_sample_text_splits_into_multiple_sentences(self):
        sentences = extractive.split_sentences(SAMPLE_TEXT)
        self.assertGreaterEqual(len(sentences), 5)


class TestTokenizeWords(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        words = extractive.tokenize_words("Hello, World! It's a Test-Case.")
        self.assertIn("hello", words)
        self.assertIn("world", words)
        self.assertIn("it's", words)
        self.assertIn("test-case", words)
        self.assertNotIn("Hello", words)


class TestScoreSentencesByFrequency(unittest.TestCase):
    def test_scores_length_matches_input(self):
        sentences = extractive.split_sentences(SAMPLE_TEXT)
        scores = extractive.score_sentences_by_frequency(sentences)
        self.assertEqual(len(scores), len(sentences))

    def test_all_scores_are_non_negative(self):
        sentences = extractive.split_sentences(SAMPLE_TEXT)
        scores = extractive.score_sentences_by_frequency(sentences)
        self.assertTrue(all(s >= 0 for s in scores))

    def test_sentence_repeating_key_terms_scores_higher_than_generic_one(self):
        sentences = [
            "The library will open next year in the downtown district.",
            "Meanwhile, a cat sat quietly by the window all afternoon.",
            "The new library branch downtown will serve the whole district.",
        ]
        scores = extractive.score_sentences_by_frequency(sentences)
        # Sentences 0 and 2 share the document's dominant vocabulary
        # (library/downtown/district); sentence 1 is an outlier and
        # should score lower.
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[2], scores[1])


class TestScoreSentencesByTextrank(unittest.TestCase):
    def test_scores_length_matches_input(self):
        sentences = extractive.split_sentences(SAMPLE_TEXT)
        scores = extractive.score_sentences_by_textrank(sentences)
        self.assertEqual(len(scores), len(sentences))

    def test_single_sentence_gets_full_score(self):
        scores = extractive.score_sentences_by_textrank(["Only one sentence here."])
        self.assertEqual(scores, [1.0])

    def test_empty_input(self):
        self.assertEqual(extractive.score_sentences_by_textrank([]), [])

    def test_scores_are_normalized_to_at_most_one(self):
        sentences = extractive.split_sentences(SAMPLE_TEXT)
        scores = extractive.score_sentences_by_textrank(sentences)
        self.assertTrue(all(0 <= s <= 1.0001 for s in scores))


class TestSummarizeText(unittest.TestCase):
    def test_empty_text_returns_empty_summary(self):
        self.assertEqual(extractive.summarize_text(""), "")

    def test_short_text_returned_as_is(self):
        text = "Short sentence one. Short sentence two."
        summary = extractive.summarize_text(text, length="short")
        self.assertEqual(summary, text)

    def test_summary_is_shorter_than_original(self):
        summary = extractive.summarize_text(SAMPLE_TEXT, length="short")
        original_words = len(extractive.tokenize_words(SAMPLE_TEXT))
        summary_words = len(extractive.tokenize_words(summary))
        self.assertLess(summary_words, original_words)
        self.assertGreater(summary_words, 0)

    def test_length_presets_produce_increasing_sentence_counts(self):
        short = extractive.summarize_text(SAMPLE_TEXT, length="short")
        medium = extractive.summarize_text(SAMPLE_TEXT, length="medium")
        long = extractive.summarize_text(SAMPLE_TEXT, length="long")

        n_short = len(extractive.split_sentences(short))
        n_medium = len(extractive.split_sentences(medium))
        n_long = len(extractive.split_sentences(long))

        self.assertLessEqual(n_short, n_medium)
        self.assertLessEqual(n_medium, n_long)

    def test_explicit_sentence_count(self):
        summary = extractive.summarize_text(SAMPLE_TEXT, length="2")
        self.assertEqual(len(extractive.split_sentences(summary)), 2)

    def test_summary_sentences_are_from_original_text(self):
        summary = extractive.summarize_text(SAMPLE_TEXT, length="medium")
        original_sentences = set(extractive.split_sentences(SAMPLE_TEXT))
        for sentence in extractive.split_sentences(summary):
            self.assertIn(sentence, original_sentences)

    def test_summary_preserves_original_sentence_order(self):
        summary = extractive.summarize_text(SAMPLE_TEXT, length="long")
        original_sentences = extractive.split_sentences(SAMPLE_TEXT)
        summary_sentences = extractive.split_sentences(summary)
        # The relative order of summary sentences must match their order
        # of appearance in the source document.
        original_positions = [original_sentences.index(s) for s in summary_sentences]
        self.assertEqual(original_positions, sorted(original_positions))

    def test_unknown_length_falls_back_to_medium_without_error(self):
        summary = extractive.summarize_text(SAMPLE_TEXT, length="extra-large")
        self.assertTrue(len(summary) > 0)


class TestSummaryStats(unittest.TestCase):
    def test_reduction_percentage_is_positive_for_a_real_summary(self):
        summary = extractive.summarize_text(SAMPLE_TEXT, length="short")
        stats = extractive.summary_stats(SAMPLE_TEXT, summary)
        self.assertGreater(stats["reduction_pct"], 0)
        self.assertGreater(stats["original_words"], stats["summary_words"])

    def test_stats_keys_present(self):
        stats = extractive.summary_stats(SAMPLE_TEXT, "A summary.")
        for key in (
            "original_words", "summary_words",
            "original_sentences", "summary_sentences", "reduction_pct",
        ):
            self.assertIn(key, stats)


if __name__ == "__main__":
    unittest.main()
