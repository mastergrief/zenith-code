"""
Auto-CALM Creativity — divergent verification for brainstorming.

When the model generates multiple ideas/options, this module verifies:
1. Diversity: are the ideas actually different from each other?
2. Novelty: do they bring genuinely distinct approaches?
3. Coverage: do they span different categories/dimensions?

Prevents the common failure mode of "5 ideas that are really 1 idea
rephrased 5 ways."

Usage:
    from calm.creativity import CreativityVerifier
    cv = CreativityVerifier()
    result = cv.verify_ideas(["idea1", "idea2", "idea3"])
    print(result.diversity_score)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class CreativityResult:
    """Result of creativity verification."""
    ideas: List[str] = field(default_factory=list)
    diversity_score: float = 0.0   # 0-1, how different the ideas are
    unique_keywords: int = 0       # distinct significant words across ideas
    shared_keywords: int = 0       # words appearing in multiple ideas
    clusters: int = 0              # number of distinct idea clusters
    redundant_pairs: List[tuple] = field(default_factory=list)  # near-duplicate pairs
    summary: str = ""

    @property
    def is_diverse(self) -> bool:
        """Whether the ideas are sufficiently diverse (>0.5)."""
        return self.diversity_score > 0.5


# Common stop words to ignore in similarity comparison
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "every", "all", "any", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "just", "also", "then", "that", "this", "these",
    "those", "it", "its", "they", "them", "their", "we", "our", "you",
    "your", "he", "she", "him", "her", "his", "use", "using", "used",
}


class CreativityVerifier:
    """Verifies diversity and novelty of generated ideas."""

    def _tokenize(self, text: str) -> Set[str]:
        """Extract significant words from text, with prefix normalization."""
        words = re.findall(r'[a-z]+', text.lower())
        stems = set()
        for w in words:
            if w in _STOP_WORDS or len(w) <= 2:
                continue
            # Use first 4 chars as a stem — catches cache/caching,
            # improve/improvement, fast/faster, perform/performance.
            # 4 is the sweet spot: specific enough to avoid false merges,
            # short enough to catch morphological variants.
            stems.add(w[:4] if len(w) > 4 else w)
        return stems

    def _similarity(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Jaccard similarity between two word sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def verify_ideas(self, ideas: List[str], threshold: float = 0.5) -> CreativityResult:
        """Verify a list of ideas for diversity and novelty."""
        result = CreativityResult(ideas=ideas)

        if len(ideas) < 2:
            result.diversity_score = 1.0
            result.clusters = len(ideas)
            result.summary = "need 2+ ideas to measure diversity"
            return result

        # Tokenize all ideas
        token_sets = [self._tokenize(idea) for idea in ideas]
        all_tokens = set()
        for ts in token_sets:
            all_tokens |= ts

        result.unique_keywords = len(all_tokens)

        # Count words appearing in multiple ideas
        word_freq = Counter()
        for ts in token_sets:
            for w in ts:
                word_freq[w] += 1
        result.shared_keywords = sum(1 for w, c in word_freq.items() if c > 1)

        # Pairwise similarity
        similarities = []
        for i in range(len(ideas)):
            for j in range(i + 1, len(ideas)):
                sim = self._similarity(token_sets[i], token_sets[j])
                similarities.append(sim)
                if sim > threshold:
                    result.redundant_pairs.append((i, j, sim))

        # Diversity score: 1 - average similarity
        avg_sim = sum(similarities) / len(similarities) if similarities else 0
        result.diversity_score = round(1 - avg_sim, 3)

        # Cluster ideas by similarity (simple single-linkage)
        clusters = list(range(len(ideas)))
        for i in range(len(ideas)):
            for j in range(i + 1, len(ideas)):
                sim = self._similarity(token_sets[i], token_sets[j])
                if sim > threshold:
                    # Merge clusters
                    old_cluster = clusters[j]
                    new_cluster = clusters[i]
                    for k in range(len(clusters)):
                        if clusters[k] == old_cluster:
                            clusters[k] = new_cluster
        result.clusters = len(set(clusters))

        # Summary
        if result.diversity_score > 0.9:
            quality = "highly diverse"
        elif result.diversity_score > 0.75:
            quality = "moderately diverse"
        elif result.diversity_score > 0.6:
            quality = "somewhat similar"
        else:
            quality = "mostly redundant"

        result.summary = (
            f"{len(ideas)} ideas, {quality} (score={result.diversity_score:.2f}), "
            f"{result.clusters} clusters, {result.unique_keywords} unique keywords"
        )
        if result.redundant_pairs:
            pairs = [f"ideas {i+1}&{j+1} ({s:.0%} similar)" for i, j, s in result.redundant_pairs]
            result.summary += f", redundant: {', '.join(pairs)}"

        return result

    def suggest_diversification(self, ideas: List[str]) -> List[str]:
        """Suggest dimensions to explore for more diverse ideation."""
        token_sets = [self._tokenize(idea) for idea in ideas]
        all_tokens = set()
        for ts in token_sets:
            all_tokens |= ts

        # Detect which dimensions are NOT covered
        dimensions = {
            "technical": {"algorithm", "data", "structure", "system", "architecture", "performance", "cache", "database", "api"},
            "user": {"user", "experience", "interface", "design", "feedback", "usability", "accessibility"},
            "process": {"workflow", "automation", "pipeline", "integration", "deployment", "testing", "monitoring"},
            "business": {"cost", "revenue", "market", "customer", "growth", "strategy", "competitive"},
            "risk": {"security", "reliability", "backup", "recovery", "compliance", "privacy", "audit"},
        }

        covered = set()
        uncovered = []
        for dim, keywords in dimensions.items():
            if all_tokens & keywords:
                covered.add(dim)
            else:
                uncovered.append(dim)

        suggestions = []
        if uncovered:
            suggestions.append(f"Explore uncovered dimensions: {', '.join(uncovered)}")
        if len(set(len(ts) for ts in token_sets)) == 1:
            suggestions.append("Vary the depth/detail level across ideas")
        if all(len(ts) < 5 for ts in token_sets):
            suggestions.append("Ideas are very brief — develop them further")

        return suggestions
