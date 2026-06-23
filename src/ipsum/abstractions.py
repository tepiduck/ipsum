"""Inspectable abstraction store — admit, score, and EVICT abstractions.

This is where ipsum departs from the literature. DreamCoder and Voyager only
ever ADD abstractions, and assume exact/immediate outcomes. ipsum must:

  - admit under UNCERTAINTY (Bayesian model selection, not exact MDL)
  - track each abstraction's usefulness with decay
  - EVICT stale abstractions as the domain drifts

See DESIGN.md sections 3.1 and 3.3.

The first implementation is deliberately small: candidates are file sets proposed
from co-change statistics, and admission is a held-out likelihood gain check
against a complexity cost. That is enough to run Card A on synth before adding
real-data noise.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class Abstraction:
    """An explicit, human-readable abstraction.

    ``name`` and ``payload`` are deliberately inspectable — the bet is on
    abstractions you can read, not weights you can't.
    """

    name: str
    payload: object
    complexity: float  # c(a): the description cost the abstraction must earn back
    usefulness: float = 0.0  # decaying trace of held-out predictive-LL contribution


@dataclass
class AbstractionStore:
    decay: float = 0.99
    admission_threshold: float = 0.0
    min_support: int = 4
    cochange_threshold: float = 0.22
    max_candidates: int = 128
    complexity_per_file: float = 0.0015
    _items: dict[str, Abstraction] = field(default_factory=dict)
    _observed_commits: list[frozenset[object]] = field(default_factory=list)

    def __iter__(self):
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def observe_commit(self, changed_files) -> None:
        """Add an observed commit for co-change candidate proposal."""
        changed = frozenset(changed_files)
        if changed:
            self._observed_commits.append(changed)

    def candidates(self) -> list[Abstraction]:
        """Propose file-set abstractions from recent co-change patterns.

        This intentionally uses no oracle information: it only sees changed file
        sets recorded through ``observe_commit``. Edges connect files that often
        co-change relative to their individual support; connected components are
        the proposed inspectable abstractions.
        """
        file_counts: Counter[object] = Counter()
        pair_counts: Counter[tuple[object, object]] = Counter()
        for commit in self._observed_commits:
            files = sorted(commit, key=str)
            file_counts.update(files)
            pair_counts.update(combinations(files, 2))

        graph: dict[object, set[object]] = defaultdict(set)
        edge_scores: list[tuple[float, int, tuple[object, object]]] = []
        for pair, count in pair_counts.items():
            if count < self.min_support:
                continue
            denom = min(file_counts[pair[0]], file_counts[pair[1]])
            score = count / denom if denom else 0.0
            if score >= self.cochange_threshold:
                graph[pair[0]].add(pair[1])
                graph[pair[1]].add(pair[0])
                edge_scores.append((score, count, pair))

        proposed: dict[tuple[object, ...], Abstraction] = {}
        for component in self._components(graph):
            self._add_candidate(proposed, component)

        # Include the strongest local neighborhoods so admission can reject
        # over-broad components instead of being forced into one granularity.
        for file_id, neighbors in graph.items():
            self._add_candidate(proposed, {file_id, *neighbors})

        # Pair candidates are useful controls: "admit everything" accumulates
        # many plausible but redundant objects, while admission must earn each.
        for score, count, pair in sorted(edge_scores, reverse=True):
            if len(proposed) >= self.max_candidates:
                break
            self._add_candidate(proposed, set(pair), suffix=f"p{count}_{score:.2f}")

        return sorted(
            proposed.values(),
            key=lambda a: (-len(self._files(a)), a.name),
        )[: self.max_candidates]

    def admit(self, a: Abstraction, ll_gain: float) -> bool:
        """Admit iff held-out predictive-LL gain exceeds complexity cost.

        Bayesian model selection replaces DreamCoder's exact MDL criterion:
        admit iff ``ll_gain - a.complexity > threshold`` with confidence.
        """
        net_gain = ll_gain - a.complexity
        if net_gain <= self.admission_threshold:
            return False

        stored = Abstraction(
            name=a.name,
            payload=a.payload,
            complexity=a.complexity,
            usefulness=net_gain,
        )
        self._items[a.name] = stored
        return True

    def reinforce(self, name: str, ll_gain: float) -> None:
        """Update an abstraction's usefulness trace from a credited outcome."""
        if name not in self._items:
            raise KeyError(f"unknown abstraction: {name}")
        item = self._items[name]
        item.usefulness = self.decay * item.usefulness + ll_gain

    def decay_and_evict(self) -> list[str]:
        """Decay all usefulness traces; evict those that fall below their cost.

        Returns the names evicted. This is the anti-staleness mechanism that
        neither DreamCoder nor Voyager has.
        """
        evicted: list[str] = []
        for name, item in list(self._items.items()):
            item.usefulness *= self.decay
            if item.usefulness < item.complexity:
                evicted.append(name)
                del self._items[name]
        return evicted

    def _add_candidate(
        self,
        proposed: dict[tuple[object, ...], Abstraction],
        files: set[object],
        suffix: str = "",
    ) -> None:
        if len(files) < 2:
            return
        key = tuple(sorted(files, key=str))
        if key in proposed:
            return
        digest = "_".join(_slug(str(file_id)) for file_id in key[:4])
        if len(key) > 4:
            digest = f"{digest}_n{len(key)}"
        name = f"files_{digest}{'_' + suffix if suffix else ''}"
        proposed[key] = Abstraction(
            name=name,
            payload={"files": list(key)},
            complexity=self.complexity_per_file * len(key),
        )

    @staticmethod
    def _components(graph: dict[object, set[object]]) -> list[set[object]]:
        remaining = set(graph)
        components: list[set[object]] = []
        while remaining:
            root = remaining.pop()
            component = {root}
            stack = [root]
            while stack:
                node = stack.pop()
                for neighbor in graph[node]:
                    if neighbor not in component:
                        component.add(neighbor)
                        remaining.discard(neighbor)
                        stack.append(neighbor)
            components.append(component)
        return components

    @staticmethod
    def _files(a: Abstraction) -> frozenset[object]:
        payload = a.payload
        if not isinstance(payload, dict) or "files" not in payload:
            return frozenset()
        return frozenset(payload["files"])


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "file"
