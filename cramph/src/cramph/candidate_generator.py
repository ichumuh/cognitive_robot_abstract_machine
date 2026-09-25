from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Generic, Iterator, Optional, TypeVar

Proposal = TypeVar("Proposal")
"""
What a candidate generator's source yields.
"""

Candidate = TypeVar("Candidate")
"""
What an accepted proposal becomes.
"""

# %% generating candidates one at a time


@dataclass(eq=False, repr=False)
class CandidateGenerator(ABC, Generic[Proposal, Candidate]):
    """
    Produces candidates on demand from a lazy source of proposals, keeping only the
    proposals that are viable.

    Nothing is generated before :meth:`advance` is called, so a proposal is judged
    against the state of the world at that moment rather than when the generator was
    built. The source is left suspended between candidates, so a later :meth:`advance`
    resumes the search where it stopped.
    """

    current_candidate: Optional[Candidate] = field(default=None, init=False, repr=False)
    """
    The candidate the latest successful :meth:`advance` created.
    """

    _proposals: Optional[Iterator[Proposal]] = field(
        default=None, init=False, repr=False
    )
    """
    The source of proposals, open from the first pull until it is exhausted or released
    by :meth:`stop_generating`.
    """

    def advance(self) -> bool:
        """
        Makes the next viable proposal the current candidate.

        Proposals that are not viable are discarded without creating a candidate.

        :return: True if a new candidate was created, False if the source ran out of
            viable proposals.
        """
        proposal = self._pull_next_proposal()
        while proposal is not None:
            if self._is_viable(proposal):
                self.current_candidate = self._create_candidate(proposal)
                return True
            proposal = self._pull_next_proposal()
        return False

    def stop_generating(self) -> None:
        """
        Releases the source once no further candidate will be requested from it.

        A suspended generator keeps every value its frame holds alive, so closing it
        frees whatever the source only holds to judge proposals by. The next
        :meth:`advance` opens the source anew.
        """
        if self._proposals is None:
            return
        self._proposals.close()
        self._proposals = None

    def _pull_next_proposal(self) -> Optional[Proposal]:
        """
        Takes the next proposal from the source, opening it if it is not open yet.

        :return: The next proposal, or None if the source is exhausted, which also
            closes it.
        """
        if self._proposals is None:
            self._proposals = self._generate_proposals()
        proposal = next(self._proposals, None)
        if proposal is None:
            self._proposals = None
        return proposal

    @abstractmethod
    def _generate_proposals(self) -> Iterator[Proposal]:
        """
        :return: A new source of proposals, in the order they are to be tried.
        """

    @abstractmethod
    def _is_viable(self, proposal: Proposal) -> bool:
        """
        :param proposal: The proposal to judge.
        :return: Whether `proposal` may become a candidate.
        """

    @abstractmethod
    def _create_candidate(self, proposal: Proposal) -> Candidate:
        """
        :param proposal: A viable proposal.
        :return: The candidate `proposal` becomes.
        """
