import unittest
from aivisibility.eval_judge import LLMJudge
from aivisibility.ml.drift import calculate_psi, detect_visibility_anomalies, wasserstein_distance_1d
from aivisibility.ml.ner import (
    jaro_winkler_similarity,
    levenshtein_distance,
    levenshtein_similarity,
    match_entity,
    strip_accents,
    token_sort_similarity,
)
from aivisibility.ml.semantic import (
    TfidfScorer,
    dense_cosine_similarity,
    extract_ngrams,
    semantic_faithfulness,
    tokenize,
)


class TestSemanticScoring(unittest.TestCase):
    def test_tokenization_and_ngrams(self):
        text = "Iron Meridian is a high-octane sci-fi thriller."
        toks = tokenize(text)
        self.assertIn("iron", toks)
        self.assertIn("meridian", toks)
        self.assertIn("thriller", toks)

        bigrams = extract_ngrams(toks, 2)
        self.assertIn("iron_meridian", bigrams)

    def test_tfidf_and_cosine_similarity(self):
        doc1 = "The movie features deep space exploration and astronaut survival."
        doc2 = "Astronaut survival in deep space exploration is featured."
        doc3 = "A romantic comedy about two roommates in New York City."

        scorer = TfidfScorer(ngram_range=(1, 2))
        scorer.fit([doc1, doc2, doc3])

        sim_high = scorer.cosine_similarity(doc1, doc2)
        sim_low = scorer.cosine_similarity(doc1, doc3)

        self.assertTrue(sim_high > 0.35)
        self.assertTrue(sim_low < 0.05)

    def test_semantic_faithfulness(self):
        ref = "A decorated detective hunts a rogue synthetic android across a futuristic desert colony."
        cand_good = "The film follows a detective pursuing an android in a futuristic desert."
        cand_bad = "A cooking show about baking wedding cakes in Italy."

        res_good = semantic_faithfulness(ref, cand_good)
        res_bad = semantic_faithfulness(ref, cand_bad)

        self.assertTrue(res_good["overall_faithfulness"] > 0.3)
        self.assertEqual(res_bad["overall_faithfulness"], 0.0)

    def test_dense_cosine_similarity(self):
        v1 = [1.0, 0.0, 1.0]
        v2 = [1.0, 0.0, 1.0]
        v3 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(dense_cosine_similarity(v1, v2), 1.0)
        self.assertAlmostEqual(dense_cosine_similarity(v1, v3), 0.0)


class TestFuzzyEntityResolution(unittest.TestCase):
    def test_accent_stripping(self):
        self.assertEqual(strip_accents("Timothée Chalamet"), "Timothee Chalamet")

    def test_levenshtein_and_jaro_winkler(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertTrue(levenshtein_similarity("Cillian", "Killian") > 0.7)

        jw = jaro_winkler_similarity("Martha", "Marhta")
        self.assertTrue(jw > 0.9)

    def test_token_sort_and_entity_matching(self):
        ground_truth = ["Robert Downey Jr.", "Scarlett Johansson", "Cillian Murphy"]

        matched, name, score = match_entity("Downey Jr., Robert", ground_truth)
        self.assertTrue(matched)
        self.assertEqual(name, "Robert Downey Jr.")

        matched, name, score = match_entity("Killian Murphy", ground_truth)
        self.assertTrue(matched)
        self.assertEqual(name, "Cillian Murphy")

        matched, name, score = match_entity("Brad Pitt", ground_truth)
        self.assertFalse(matched)
        self.assertIsNone(name)


class TestDriftDetection(unittest.TestCase):
    def test_psi_calculation(self):
        dist_stable_a = [0.25, 0.25, 0.25, 0.25]
        dist_stable_b = [0.24, 0.26, 0.25, 0.25]
        psi_low = calculate_psi(dist_stable_a, dist_stable_b)
        self.assertTrue(psi_low < 0.1)

        dist_drifted = [0.05, 0.05, 0.10, 0.80]
        psi_high = calculate_psi(dist_stable_a, dist_drifted)
        self.assertTrue(psi_high >= 0.25)

    def test_wasserstein_distance(self):
        u = [0.1, 0.2, 0.3]
        v = [0.1, 0.2, 0.3]
        self.assertEqual(wasserstein_distance_1d(u, v), 0.0)

        w = [0.8, 0.9, 1.0]
        self.assertTrue(wasserstein_distance_1d(u, w) > 0.5)

    def test_anomaly_detector(self):
        baseline = {"TitleA": 0.8, "TitleB": 0.8, "TitleC": 0.8, "TitleD": 0.8}
        current = {"TitleA": 0.8, "TitleB": 0.8, "TitleC": 0.1, "TitleD": 0.8}  # TitleC degraded heavily
        anomalies = detect_visibility_anomalies(baseline, current, threshold_z=1.5)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["title"], "TitleC")
        self.assertEqual(anomalies[0]["direction"], "degradation")


class TestLLMJudge(unittest.TestCase):
    def test_judge_evaluation_and_pairwise(self):
        def mock_judge_model(prompt, system=None, max_tokens=0):
            if "Compare the following two model answers" in prompt:
                return '{"winner": "A", "confidence": 0.95, "rationale": "Answer A contains accurate release dates."}'
            return '{"score": 5, "reasoning": "Accurate cast and release date matching ground truth.", "has_hallucinations": false, "fabricated_claims": []}'

        judge = LLMJudge(mock_judge_model, model_name="mock-judge")
        gt = {"release_date": ["2027-03-05"], "cast": ["Ann Lee", "Bo Chan"]}

        res = judge.evaluate_consistency("Alpha Rising", gt, "Alpha Rising opens March 5, 2027 starring Ann Lee.")
        self.assertEqual(res["score"], 5)
        self.assertFalse(res["has_hallucinations"])

        cmp = judge.pairwise_compare("Alpha Rising", gt, "Answer A text", "Answer B text")
        self.assertEqual(cmp["winner"], "A")
        self.assertEqual(cmp["confidence"], 0.95)


if __name__ == "__main__":
    unittest.main()
