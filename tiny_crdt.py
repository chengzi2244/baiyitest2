"""Tiny CRDT toolkit: G-Counter, PN-Counter, G-Set, and LWW-Register.

Conflict-free Replicated Data Types converge under eventual delivery:
concurrent updates merge without coordination. Each type exposes mutate
helpers plus a commutative, associative, idempotent ``merge``.
"""

from __future__ import annotations

from typing import Dict, Generic, Iterable, Optional, Set, TypeVar

T = TypeVar("T")


class GCounter:
    """Grow-only counter: each replica tracks its own non-decreasing tally."""

    def __init__(self, replica_id: str, counts: Optional[Dict[str, int]] = None) -> None:
        self.replica_id = replica_id
        self.counts: Dict[str, int] = dict(counts or {})
        self.counts.setdefault(replica_id, 0)

    def inc(self, n: int = 1) -> None:
        """Increment this replica's contribution by ``n`` (must be >= 1)."""
        if n < 1:
            raise ValueError("GCounter only grows; n must be >= 1")
        self.counts[self.replica_id] = self.counts.get(self.replica_id, 0) + n

    def value(self) -> int:
        """Return the sum of all known per-replica counts."""
        return sum(self.counts.values())

    def merge(self, other: "GCounter") -> "GCounter":
        """Pointwise max of per-replica counts; returns a new GCounter."""
        keys = set(self.counts) | set(other.counts)
        merged = {k: max(self.counts.get(k, 0), other.counts.get(k, 0)) for k in keys}
        return GCounter(self.replica_id, merged)

    def __repr__(self) -> str:
        return f"GCounter({self.replica_id!r}, {self.counts!r}, value={self.value()})"


class PNCounter:
    """Positive-Negative counter composed of two G-Counters (P and N)."""

    def __init__(
        self, replica_id: str, p: Optional[GCounter] = None, n: Optional[GCounter] = None
    ) -> None:
        self.replica_id = replica_id
        self.p = p or GCounter(replica_id)
        self.n = n or GCounter(replica_id)

    def inc(self, n: int = 1) -> None:
        """Increase the observed value by ``n``."""
        self.p.inc(n)

    def dec(self, n: int = 1) -> None:
        """Decrease the observed value by ``n`` via the negative G-Counter."""
        self.n.inc(n)

    def value(self) -> int:
        """Return ``P.value() - N.value()``."""
        return self.p.value() - self.n.value()

    def merge(self, other: "PNCounter") -> "PNCounter":
        """Merge by merging the underlying positive and negative G-Counters."""
        return PNCounter(self.replica_id, self.p.merge(other.p), self.n.merge(other.n))

    def __repr__(self) -> str:
        return f"PNCounter({self.replica_id!r}, value={self.value()})"


class GSet(Generic[T]):
    """Grow-only set: elements may be added but never removed."""

    def __init__(self, elements: Optional[Iterable[T]] = None) -> None:
        self.elements: Set[T] = set(elements or [])

    def add(self, element: T) -> None:
        """Insert ``element`` into the local set."""
        self.elements.add(element)

    def contains(self, element: T) -> bool:
        """Return True if ``element`` is present after all merges so far."""
        return element in self.elements

    def merge(self, other: "GSet[T]") -> "GSet[T]":
        """Union of element sets; returns a new GSet."""
        return GSet(self.elements | other.elements)

    def __repr__(self) -> str:
        return f"GSet({sorted(self.elements)!r})"  # type: ignore[type-var]


class LWWRegister(Generic[T]):
    """Last-Writer-Wins register: highest (timestamp, writer) pair wins."""

    def __init__(
        self,
        replica_id: str,
        value: Optional[T] = None,
        timestamp: int = 0,
        writer: str = "",
    ) -> None:
        self.replica_id = replica_id
        self.value = value
        self.timestamp = timestamp
        self.writer = writer or replica_id

    def set(self, value: T, timestamp: int) -> None:
        """Assign ``value`` when ``timestamp`` dominates the current entry."""
        if (timestamp, self.replica_id) >= (self.timestamp, self.writer):
            self.value, self.timestamp, self.writer = value, timestamp, self.replica_id

    def merge(self, other: "LWWRegister[T]") -> "LWWRegister[T]":
        """Keep the entry with the greater (timestamp, writer) pair."""
        src = other if (other.timestamp, other.writer) >= (self.timestamp, self.writer) else self
        return LWWRegister(self.replica_id, src.value, src.timestamp, src.writer)

    def __repr__(self) -> str:
        return (
            f"LWWRegister({self.replica_id!r}, value={self.value!r}, "
            f"ts={self.timestamp}, writer={self.writer!r})"
        )


def _demo() -> None:
    """Print merge results for concurrent updates on each CRDT type."""
    a, b = GCounter("alice"), GCounter("bob")
    a.inc(3)
    b.inc(5)
    g = a.merge(b)
    print("GCounter merge:", g, "->", g.value())

    p1, p2 = PNCounter("alice"), PNCounter("bob")
    p1.inc(10)
    p1.dec(2)
    p2.inc(4)
    p2.dec(7)
    pn = p1.merge(p2)
    print("PNCounter merge:", pn, "->", pn.value())

    s1: GSet[str] = GSet(["red"])
    s2: GSet[str] = GSet(["blue"])
    s1.add("green")
    s2.add("yellow")
    gs = s1.merge(s2)
    print("GSet merge:", gs, "contains green?", gs.contains("green"))

    r1: LWWRegister[str] = LWWRegister("alice")
    r2: LWWRegister[str] = LWWRegister("bob")
    r1.set("hello", timestamp=1)
    r2.set("world", timestamp=2)
    print("LWWRegister merge:", r1.merge(r2))


if __name__ == "__main__":
    _demo()
