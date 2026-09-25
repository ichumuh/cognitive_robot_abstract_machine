from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from typing_extensions import Callable, Iterator, List

from cramph.candidate_generator import CandidateGenerator

# %% a generator that keeps the proposals passing a filter


@dataclass(eq=False, repr=False)
class FilteringCandidateGenerator(CandidateGenerator[int, str]):
    """
    Proposes a fixed list of numbers and accepts those a filter lets through, turning
    each into its text.
    """

    numbers: List[int] = field(kw_only=True)
    """
    The numbers the source proposes, in order.
    """

    is_accepted: Callable[[int], bool] = field(kw_only=True, default=lambda _: True)
    """
    Decides which proposals are viable.
    """

    times_opened: int = field(default=0, init=False)
    """
    How often the source was opened.
    """

    was_closed: bool = field(default=False, init=False)
    """
    Whether the source was closed while it was still suspended.
    """

    def _generate_proposals(self) -> Iterator[int]:
        self.times_opened += 1
        return self._propose()

    def _propose(self) -> Iterator[int]:
        try:
            yield from self.numbers
        except GeneratorExit:
            self.was_closed = True
            raise

    def _is_viable(self, proposal: int) -> bool:
        return self.is_accepted(proposal)

    def _create_candidate(self, proposal: int) -> str:
        return str(proposal)


NUMBERS = [1, 2, 3, 4]


def is_even(number: int) -> bool:
    return number % 2 == 0


@pytest.fixture()
def generator() -> FilteringCandidateGenerator:
    return FilteringCandidateGenerator(numbers=NUMBERS, is_accepted=is_even)


# %% advancing to the next viable proposal


def test_advance_makes_first_viable_proposal_the_current_candidate(generator):
    assert generator.advance()
    assert generator.current_candidate == str(NUMBERS[1])


def test_advance_resumes_after_the_previous_candidate(generator):
    generator.advance()
    assert generator.advance()
    assert generator.current_candidate == str(NUMBERS[3])


def test_advance_reports_nothing_viable_left(generator):
    generator.advance()
    generator.advance()
    assert not generator.advance()


def test_advance_keeps_current_candidate_when_nothing_viable_is_left(generator):
    generator.advance()
    generator.advance()
    last_candidate = generator.current_candidate
    generator.advance()
    assert generator.current_candidate == last_candidate


def test_advance_opens_source_again_after_exhaustion(generator):
    while generator.advance():
        pass
    assert generator.advance()
    assert generator.times_opened == 2
    assert generator.current_candidate == str(NUMBERS[1])


def test_advance_opens_source_only_once_while_proposals_remain(generator):
    generator.advance()
    generator.advance()
    assert generator.times_opened == 1


# %% releasing the source


def test_stop_generating_closes_suspended_source(generator):
    generator.advance()
    generator.stop_generating()
    assert generator.was_closed


def test_stop_generating_makes_next_advance_open_source_again(generator):
    generator.advance()
    generator.stop_generating()
    generator.advance()
    assert generator.times_opened == 2


def test_stop_generating_without_open_source_does_nothing(generator):
    generator.stop_generating()
    assert not generator.was_closed
