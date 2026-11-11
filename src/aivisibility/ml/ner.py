"""Fuzzy entity resolution, Jaro-Winkler & Levenshtein metrics, and entity linking (dependency-free).
Prevents false-positive hallucination flags when models use nicknames, alternate spellings, or stripped diacritics."""
import unicodedata
from typing import Optional, Sequence, Tuple


def strip_accents(text: str) -> str:
    """Normalize diacritics and accents: 'Timothée' -> 'Timothee'."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the minimum edit distance between two strings (insert, delete, replace)."""
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)

    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, v0

    return v0[len(s2)]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized similarity in [0.0, 1.0] based on Levenshtein distance."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return 1.0 - (dist / max_len)


def jaro_similarity(s1: str, s2: str) -> float:
    """Compute the Jaro similarity between two character sequences."""
    len1, len2 = len(s1), len(s2)
    if len1 == 0 and len2 == 0:
        return 1.0
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    transpositions = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    transpositions //= 2
    return (
        (matches / len1) + (matches / len2) + ((matches - transpositions) / matches)
    ) / 3.0


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1, max_prefix: int = 4) -> float:
    """Compute Jaro-Winkler similarity giving weight to common prefixes. Ideal for names."""
    j = jaro_similarity(s1, s2)
    # Determine common prefix length up to max_prefix
    prefix_len = 0
    for c1, c2 in zip(s1[:max_prefix], s2[:max_prefix]):
        if c1 == c2:
            prefix_len += 1
        else:
            break
    return j + (prefix_len * p * (1.0 - j))


def token_sort_similarity(s1: str, s2: str) -> float:
    """Tokenize, sort words alphabetically, and calculate Jaro-Winkler.
    Handles 'Downey, Robert' vs 'Robert Downey'.
    """
    tok1 = " ".join(sorted(strip_accents(s1).lower().split()))
    tok2 = " ".join(sorted(strip_accents(s2).lower().split()))
    return jaro_winkler_similarity(tok1, tok2)


def match_entity(
    candidate: str,
    ground_truth_list: Sequence[str],
    threshold: float = 0.85
) -> Tuple[bool, Optional[str], float]:
    """Check if a candidate entity name matches any ground-truth entity using fuzzy resolution.

    Returns:
        (is_match, matched_truth_name, highest_score)
    """
    clean_cand = strip_accents(candidate).strip().lower()
    best_match = None
    best_score = 0.0

    for truth in ground_truth_list:
        clean_truth = strip_accents(truth).strip().lower()
        if clean_cand == clean_truth:
            return True, truth, 1.0

        # Substring exact check
        if clean_cand in clean_truth or clean_truth in clean_cand:
            best_match, best_score = truth, 0.95
            continue

        # Jaro-Winkler and Token-Sort scoring
        jw = jaro_winkler_similarity(clean_cand, clean_truth)
        ts = token_sort_similarity(clean_cand, clean_truth)
        score = max(jw, ts)

        if score > best_score:
            best_score = score
            best_match = truth

    matched = best_score >= threshold
    return matched, (best_match if matched else None), round(best_score, 4)
