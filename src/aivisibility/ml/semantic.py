"""Semantic similarity, premise faithfulness scoring, and vector representations (zero external dependencies).
Provides sublinear TF-IDF + n-gram vectorization with cosine similarity, and supports pluggable dense embeddings."""
import math
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

WORD_RE = re.compile(r"\b[a-zA-Z0-9_]{2,}\b")


def tokenize(text: str) -> List[str]:
    """Extract lowercased alphanumeric tokens of length >= 2."""
    return WORD_RE.findall(text.lower())


def extract_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """Generate n-grams from a token sequence."""
    if len(tokens) < n:
        return []
    return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class TfidfScorer:
    """Dependency-free TF-IDF and n-gram cosine similarity vectorizer.

    Uses sublinear term frequency (1 + log(tf)) and smoothed inverse document frequency
    log((1 + N) / (1 + df)) + 1 with Euclidean (L2) unit vector normalization.
    """

    def __init__(self, ngram_range: Tuple[int, int] = (1, 2)):
        self.ngram_range = ngram_range
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count: int = 0

    def _extract_features(self, text: str) -> List[str]:
        toks = tokenize(text)
        features: List[str] = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            if n == 1:
                features.extend(toks)
            else:
                features.extend(extract_ngrams(toks, n))
        return features

    def fit(self, corpus: Sequence[str]) -> "TfidfScorer":
        """Learn vocabulary and IDF from training documents."""
        self.doc_count = len(corpus)
        if self.doc_count == 0:
            return self

        df: Counter = Counter()
        for doc in corpus:
            features = set(self._extract_features(doc))
            df.update(features)

        self.vocabulary = {feat: idx for idx, feat in enumerate(sorted(df.keys()))}
        self.idf = {
            feat: math.log((1.0 + self.doc_count) / (1.0 + freq)) + 1.0
            for feat, freq in df.items()
        }
        return self

    def transform(self, text: str) -> Dict[int, float]:
        """Convert a document into a sparse L2-normalized TF-IDF vector."""
        features = self._extract_features(text)
        if not features:
            return {}

        tf = Counter(features)
        raw_vec: Dict[int, float] = {}
        sum_sq = 0.0

        for feat, count in tf.items():
            if feat in self.vocabulary:
                idx = self.vocabulary[feat]
                # Sublinear TF: 1 + log(tf)
                w = (1.0 + math.log(count)) * self.idf.get(feat, 1.0)
                raw_vec[idx] = w
                sum_sq += w * w

        norm = math.sqrt(sum_sq)
        if norm == 0.0:
            return {}

        return {idx: val / norm for idx, val in raw_vec.items()}

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Compute the cosine similarity between two texts in [0.0, 1.0]."""
        vec_a = self.transform(text_a)
        vec_b = self.transform(text_b)
        if not vec_a or not vec_b:
            return 0.0

        # Dot product of normalized vectors
        shared_keys = set(vec_a.keys()) & set(vec_b.keys())
        return sum(vec_a[k] * vec_b[k] for k in shared_keys)


_EMBEDDING_PROVIDER: Optional[Callable[[str], List[float]]] = None


def set_dense_embedding_provider(fn: Optional[Callable[[str], List[float]]]) -> None:
    """Inject a custom dense embedding callable: fn(text) -> list of floats."""
    global _EMBEDDING_PROVIDER
    _EMBEDDING_PROVIDER = fn


def dense_cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two dense numeric vectors."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def semantic_faithfulness(reference: str, candidate: str) -> Dict[str, float]:
    """Calculates factual and semantic alignment between a ground-truth premise and generated answer.

    Returns:
        cosine_similarity: Lexical unigram TF-IDF similarity (0.0 to 1.0)
        token_recall: Fraction of reference keywords captured in candidate
        token_precision: Fraction of candidate words appearing in reference context
        jaccard_similarity: Intersection over union of vocabulary
        overall_faithfulness: Composite weighted index (0.0 to 1.0)
    """
    ref_toks = set(tokenize(reference))
    cand_toks = set(tokenize(candidate))

    if not ref_toks or not cand_toks:
        return {
            "cosine_similarity": 0.0,
            "token_recall": 0.0,
            "token_precision": 0.0,
            "jaccard_similarity": 0.0,
            "overall_faithfulness": 0.0,
        }

    shared = ref_toks & cand_toks
    recall = len(shared) / len(ref_toks)
    precision = len(shared) / len(cand_toks)
    jaccard = len(shared) / len(ref_toks | cand_toks)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    scorer = TfidfScorer(ngram_range=(1, 1))
    scorer.fit([reference, candidate])
    cos = scorer.cosine_similarity(reference, candidate)

    # Optional dense embedding score if a dense provider is active
    if _EMBEDDING_PROVIDER is not None:
        try:
            dense_a = _EMBEDDING_PROVIDER(reference)
            dense_b = _EMBEDDING_PROVIDER(candidate)
            dense_sim = dense_cosine_similarity(dense_a, dense_b)
            cos = 0.5 * cos + 0.5 * dense_sim
        except Exception:
            pass

    # Composite faithfulness score (weighted harmonic combination)
    composite = 0.5 * f1 + 0.3 * recall + 0.2 * cos

    return {
        "cosine_similarity": round(cos, 4),
        "token_recall": round(recall, 4),
        "token_precision": round(precision, 4),
        "jaccard_similarity": round(jaccard, 4),
        "overall_faithfulness": round(composite, 4),
    }
