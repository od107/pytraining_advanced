from __future__ import annotations

import random as rng
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

Payoff = Tuple[int, int]


class Action(Enum):
    COOPERATE = "cooperate"
    DEFECT = "defect"

    def compute_payoff(self, other_action: Action) -> Payoff:
        if self == Action.COOPERATE and other_action == Action.COOPERATE:
            return 3, 3
        if self == Action.DEFECT and other_action == Action.COOPERATE:
            return 5, 0
        if self == Action.COOPERATE and other_action == Action.DEFECT:
            return 0, 5
        return 1, 1


class StrategyReport:
    def __init__(self) -> None:
        self.total_gain: int = 0
        self.action_histogram = defaultdict(int)

    def notify(self, action: Action, payoff: float) -> None:
        self.action_histogram[action] += 1
        self.total_gain += payoff

    @property
    def n_rounds(self) -> int:
        return sum(self.action_histogram.values())

    @property
    def average_gain(self) -> float:
        return self.total_gain / self.n_rounds


@dataclass
class Report:
    p1: Strategy
    strategy_report_p1: StrategyReport
    p2: Strategy
    strategy_report_p2: StrategyReport

    @property
    def winner(self) -> Optional[Strategy]:
        gain_1 = self.strategy_report_p1.total_gain
        gain_2 = self.strategy_report_p2.total_gain
        if gain_1 == gain_2:
            return None
        if gain_1 > gain_2:
            return self.p1
        else:
            return self.p2

    def __str__(self) -> str:
        winner = self.winner
        win_str = ""
        if winner is self.p1:
            win_str = "(P1)"
        if winner is self.p2:
            win_str = "(P2)"

        return f"""
Winner: {winner!r} {win_str}
Number of rounds: {self.strategy_report_p1.n_rounds}
Average gain of P1: {self.strategy_report_p1.average_gain}
Average gain of P2: {self.strategy_report_p2.average_gain}
Action histogram of P1: {self.strategy_report_p1.action_histogram}
Action histogram of P2: {self.strategy_report_p2.action_histogram}
"""


class Strategy(metaclass=ABCMeta):
    @abstractmethod
    def take_turn(self) -> Action:
        raise NotImplementedError()

    def register_opponent_action(self, action: Action) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"


class IPDEngine:
    MAX_N_TURN = 10000

    def __init__(
        self,
        player_1: Strategy,
        player_2: Strategy,
    ) -> None:
        self._p1 = player_1
        self._p2 = player_2
        self._report_1 = StrategyReport()
        self._report_2 = StrategyReport()

    def play_round(self) -> None:
        act_p1 = self._p1.take_turn()
        act_p2 = self._p2.take_turn()

        payoff_1, payoff_2 = act_p1.compute_payoff(act_p2)

        self._report_1.notify(act_p1, payoff_1)
        self._report_2.notify(act_p2, payoff_2)

        self._p1.register_opponent_action(act_p2)
        self._p2.register_opponent_action(act_p1)

    def play_game(self, n_turns: Optional[int] = None) -> Report:
        if n_turns is None:
            n_turns = rng.randint(1, self.MAX_N_TURN)

        for _ in range(n_turns):
            self.play_round()

        return Report(self._p1, self._report_1, self._p2, self._report_2)


class UniformStrategy(Strategy):
    def take_turn(self) -> Action:
        return rng.choice(Action)


class CooperativeStrategy(Strategy):
    def take_turn(self) -> Action:
        return Action.COOPERATE


class DefectStrategy(Strategy):
    def take_turn(self) -> Action:
        return Action.DEFECT


class ReplayStrategy(Strategy):
    def __init__(self, first_action: Action) -> None:
        super().__init__()
        self._first_action = first_action
        self._last_opponent_action: Optional[Action] = None

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self._first_action!r})"

    def take_turn(self) -> Action:
        if self._last_opponent_action is None:
            return self._first_action
        return self._last_opponent_action

    def register_opponent_action(self, action: Action) -> None:
        self._last_opponent_action = action


class RandomStrategy(Strategy):
    def __init__(self, p_cooperate: float = 0.5) -> None:
        super().__init__()
        self._p_cooperate = p_cooperate

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self._p_cooperate!r})"

    def take_turn(self) -> Action:
        if rng.random() < self._p_cooperate:
            return Action.COOPERATE
        else:
            return Action.DEFECT


# =============================
report = IPDEngine(
    ReplayStrategy(Action.DEFECT),
    CooperativeStrategy(),
).play_game(100)

print(report)
