"""Tiny state-based CRDTs with a commutative, associative, idempotent merge.

Replicas exchange whole states. ``merge`` returns a new state and does not
mutate either input, so the same deliveries converge in any order.

* ``GCounter`` — grow-only counter. Merge keeps the max of each replica slot.
* ``PNCounter`` — signed counter. Each half is its own :class:`GCounter`.
* ``GSet`` — grow-only set of hashable elements. Merge is union.
* ``LWWRegister`` — greatest ``(timestamp, writer)`` wins a concurrent write.
  ``repr(value)`` breaks a tie only when both states share one stamp.

A replica id is a non-empty string and the single writer of its own slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["GCounter", "PNCounter", "GSet", "LWWRegister"]


def _replica(value: str) -> str:
    """Return ``value`` when it is a non-empty string."""
    if not isinstance(value, str) or value == "":
        raise ValueError("replica id must be a non-empty string")
    return value


def _nonneg(value: int, label: str) -> int:
    """Return ``value`` when it is an int greater than or equal to zero."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an int")
    if value < 0:
        raise ValueError(f"{label} must be >= 0")
    return value


def _ordered(counts: dict[str, int]) -> dict[str, int]:
    """Copy a slot map with sorted keys, for stable demo output."""
    return dict(sorted(counts.items()))


@dataclass
class GCounter:
    """Grow-only counter. The value is the sum of per-replica slots.

    A replica adds only to its own slot. Slots never shrink, so the
    pointwise maximum joins two copies without losing an increment.
    """

    replica: str
    counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.replica = _replica(self.replica)
        self.counts = {
            _replica(key): _nonneg(slot, "slot") for key, slot in self.counts.items()
        }

    def increment(self, amount: int = 1) -> None:
        """Add a non-negative ``amount`` to this replica's slot."""
        self.counts[self.replica] = self.counts.get(self.replica, 0) + _nonneg(
            amount, "amount"
        )

    def value(self) -> int:
        """Return the sum of every slot."""
        return sum(self.counts.values())

    def merge(self, other: GCounter) -> GCounter:
        """Return the pointwise maximum, tagged with this replica id."""
        if not isinstance(other, GCounter):
            raise TypeError("other must be a GCounter")
        keys = self.counts.keys() | other.counts.keys()
        counts = {
            key: max(self.counts.get(key, 0), other.counts.get(key, 0)) for key in keys
        }
        return GCounter(self.replica, counts)


@dataclass
class PNCounter:
    """Signed counter: increments minus decrements, each a :class:`GCounter`.

    A decrement is its own growing total, so it cannot cancel an increment
    this replica has not observed.
    """

    replica: str
    increments: GCounter = field(init=False)
    decrements: GCounter = field(init=False)

    def __post_init__(self) -> None:
        self.replica = _replica(self.replica)
        self.increments = GCounter(self.replica)
        self.decrements = GCounter(self.replica)

    def increment(self, amount: int = 1) -> None:
        """Record ``amount`` increments for this replica."""
        self.increments.increment(amount)

    def decrement(self, amount: int = 1) -> None:
        """Record ``amount`` decrements for this replica."""
        self.decrements.increment(amount)

    def value(self) -> int:
        """Return increments minus decrements."""
        return self.increments.value() - self.decrements.value()

    def merge(self, other: PNCounter) -> PNCounter:
        """Merge each half. Neither input is modified."""
        if not isinstance(other, PNCounter):
            raise TypeError("other must be a PNCounter")
        merged = PNCounter(self.replica)
        merged.increments = self.increments.merge(other.increments)
        merged.decrements = self.decrements.merge(other.decrements)
        return merged


@dataclass
class GSet:
    """Grow-only set. ``add`` inserts a hashable element and merge is union."""

    replica: str
    elements: set[Any] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.replica = _replica(self.replica)
        self.elements = set(self.elements)

    def add(self, element: Any) -> None:
        """Insert ``element``. Adding it again leaves the set unchanged."""
        self.elements.add(element)

    def __contains__(self, element: Any) -> bool:
        """Return whether ``element`` has already been added."""
        return element in self.elements

    def merge(self, other: GSet) -> GSet:
        """Return the union of both sets. Inputs are not mutated."""
        if not isinstance(other, GSet):
            raise TypeError("other must be a GSet")
        return GSet(self.replica, self.elements | other.elements)


@dataclass
class LWWRegister:
    """Last-writer-wins register ordered by ``(timestamp, writer)``.

    A fresh register holds ``None`` at stamp ``(0, "")``. ``assign`` stores
    a value only when ``(timestamp, replica)`` is strictly greater, so a
    later timestamp replaces the payload and an older one does not. Merge
    keeps the higher rank ``(timestamp, writer, repr(value))``.
    """

    replica: str
    value: Any = None
    stamp: tuple[int, str] = (0, "")

    def __post_init__(self) -> None:
        self.replica = _replica(self.replica)
        if not isinstance(self.stamp, tuple) or len(self.stamp) != 2:
            raise TypeError("stamp must be a (timestamp, writer) pair")
        timestamp, writer = self.stamp
        if writer != "":
            writer = _replica(writer)
        self.stamp = (_nonneg(timestamp, "timestamp"), writer)

    def assign(self, value: Any, timestamp: int) -> None:
        """Store ``value`` when this replica's new stamp strictly wins."""
        candidate = (_nonneg(timestamp, "timestamp"), self.replica)
        if candidate > self.stamp:
            self.value = value
            self.stamp = candidate

    def merge(self, other: LWWRegister) -> LWWRegister:
        """Keep the higher-ranked write and tag it with this replica id."""
        if not isinstance(other, LWWRegister):
            raise TypeError("other must be an LWWRegister")
        own = (*self.stamp, repr(self.value))
        theirs = (*other.stamp, repr(other.value))
        winner = self if own >= theirs else other
        return LWWRegister(self.replica, winner.value, winner.stamp)


def demo() -> None:
    """Print a concurrent merge for every type, then check the join laws."""
    print("=== GCounter ===")
    alice, bob, carol = GCounter("alice"), GCounter("bob"), GCounter("carol")
    alice.increment(2)
    bob.increment(3)
    carol.increment(4)
    ab, ba = alice.merge(bob), bob.merge(alice)
    three = ab.merge(carol)
    print(f"  alice +2 -> {alice.value()} {_ordered(alice.counts)}")
    print(f"  bob   +3 -> {bob.value()} {_ordered(bob.counts)}")
    print(f"  alice.merge(bob) -> {ab.value()} {_ordered(ab.counts)}")
    print(f"  bob.merge(alice) -> {ba.value()} {_ordered(ba.counts)}")
    print(f"  merge carol +4 -> {three.value()} {_ordered(three.counts)}")
    alice_next, bob_next = alice.merge(bob), bob.merge(alice)
    alice_next.increment(1)
    bob_next.increment(1)
    resynced = alice_next.merge(bob_next)
    print(f"  +1 each, merge -> {resynced.value()} {_ordered(resynced.counts)}")
    print(f"  originals stay alice {alice.value()} bob {bob.value()}")

    print("=== PNCounter ===")
    up, down = PNCounter("alice"), PNCounter("bob")
    up.increment(5)
    down.decrement(2)
    pn_ab, pn_ba = up.merge(down), down.merge(up)
    print(f"  alice +5 -> {up.value()} inc {_ordered(up.increments.counts)}")
    print(f"  bob   -2 -> {down.value()} dec {_ordered(down.decrements.counts)}")
    print(
        "  alice.merge(bob) -> "
        f"{pn_ab.value()} inc {_ordered(pn_ab.increments.counts)} "
        f"dec {_ordered(pn_ab.decrements.counts)}"
    )
    print(
        "  bob.merge(alice) -> "
        f"{pn_ba.value()} inc {_ordered(pn_ba.increments.counts)} "
        f"dec {_ordered(pn_ba.decrements.counts)}"
    )
    lone_up, lone_down = PNCounter("alice"), PNCounter("bob")
    lone_up.increment(1)
    lone_down.decrement(1)
    unseen = lone_up.merge(lone_down)
    print(
        "  +1 merged with an unseen -1 -> "
        f"{unseen.value()} inc {_ordered(unseen.increments.counts)} "
        f"dec {_ordered(unseen.decrements.counts)}"
    )
    try:
        up.increment(-1)
    except ValueError as exc:
        print(f"  negative increment rejected ({exc})")

    print("=== GSet ===")
    fruit_a, fruit_b = GSet("alice"), GSet("bob")
    for name in ("apple", "pear"):
        fruit_a.add(name)
    for name in ("pear", "fig"):
        fruit_b.add(name)
    set_ab, set_ba = fruit_a.merge(fruit_b), fruit_b.merge(fruit_a)
    print(f"  alice {sorted(fruit_a.elements)}")
    print(f"  bob   {sorted(fruit_b.elements)}")
    print(f"  alice.merge(bob) {sorted(set_ab.elements)}")
    print(f"  bob.merge(alice) {sorted(set_ba.elements)}")
    print(f"  union again {sorted(set_ab.merge(fruit_b).elements)}")
    print(f"  fig in original alice? {'fig' in fruit_a}")

    print("=== LWWRegister ===")
    red, blue = LWWRegister("alice"), LWWRegister("bob")
    red.assign("red", 1)
    blue.assign("blue", 2)
    by_time, by_time_back = red.merge(blue), blue.merge(red)
    print(f"  red@1 merge blue@2 -> {by_time.value!r} stamp {by_time.stamp}")
    print(f"  blue@2 merge red@1 -> {by_time_back.value!r} stamp {by_time_back.stamp}")
    green, yellow = LWWRegister("alice"), LWWRegister("bob")
    green.assign("green", 5)
    yellow.assign("yellow", 5)
    tie_ab, tie_ba = green.merge(yellow), yellow.merge(green)
    print(f"  green@5 merge yellow@5 -> {tie_ab.value!r} stamp {tie_ab.stamp}")
    print(f"  yellow@5 merge green@5 -> {tie_ba.value!r} stamp {tie_ba.stamp}")
    held = LWWRegister("alice")
    held.assign("green", 5)
    held.assign("red", 1)
    held.assign("green", 5)
    print(f"  stale and repeated stamps stay {held.value!r} stamp {held.stamp}")
    print(f"  original alice register still {red.value!r} stamp {red.stamp}")

    print("=== laws ===")
    assert ab.counts == ba.counts == {"alice": 2, "bob": 3}
    assert three.counts == alice.merge(bob.merge(carol)).counts
    assert three.counts == {"alice": 2, "bob": 3, "carol": 4}
    assert ab.merge(ab).counts == ab.merge(bob).counts == ab.counts
    assert resynced.counts == bob_next.merge(alice_next).counts == {"alice": 3, "bob": 4}
    assert alice.counts == {"alice": 2} and bob.counts == {"bob": 3}
    assert pn_ab.value() == pn_ba.value() == 3
    assert pn_ab.increments.counts == pn_ba.increments.counts == {"alice": 5}
    assert pn_ab.decrements.counts == pn_ba.decrements.counts == {"bob": 2}
    assert unseen.increments.counts == {"alice": 1}
    assert unseen.decrements.counts == {"bob": 1} and unseen.value() == 0
    third = PNCounter("carol")
    third.increment(2)
    third.decrement(4)
    left_pn = up.merge(down).merge(third)
    right_pn = up.merge(down.merge(third))
    assert left_pn.value() == right_pn.value() == 1
    assert left_pn.increments.counts == right_pn.increments.counts
    assert left_pn.decrements.counts == right_pn.decrements.counts
    assert left_pn.merge(left_pn).value() == left_pn.value()
    assert set_ab.elements == set_ba.elements == {"apple", "fig", "pear"}
    assert set_ab.merge(set_ba).elements == set_ab.elements
    assert fruit_a.elements == {"apple", "pear"}
    assert by_time.value == by_time_back.value == "blue"
    assert by_time.stamp == by_time_back.stamp == (2, "bob")
    assert tie_ab.value == tie_ba.value == "yellow"
    assert tie_ab.stamp == tie_ba.stamp == (5, "bob")
    assert green.merge(yellow).merge(green).value == "yellow"
    assert held.value == "green" and held.stamp == (5, "alice")
    assert red.value == "red" and red.stamp == (1, "alice")
    minted = LWWRegister("zoe")
    minted.assign("navy", 0)
    assert minted.merge(LWWRegister("zoe")).value == "navy"
    pear = LWWRegister("ada", "pear", (3, "ada"))
    apple = LWWRegister("bea", "apple", (3, "ada"))
    assert pear.merge(apple).value == apple.merge(pear).value == "pear"
    try:
        GCounter("")
    except ValueError as exc:
        print(f"  empty replica id rejected ({exc})")
    else:
        raise AssertionError("empty replica id was accepted")
    print(
        "  three-way PNCounter -> "
        f"{left_pn.value()} inc {_ordered(left_pn.increments.counts)} "
        f"dec {_ordered(left_pn.decrements.counts)}"
    )
    print("  commutative, associative, idempotent: ok")


if __name__ == "__main__":
    demo()
