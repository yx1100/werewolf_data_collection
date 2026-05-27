"""Game state and player state data structures."""

from dataclasses import dataclass, field
from enum import Enum


class GamePhase(Enum):
    SETUP = "setup"
    NIGHT = "night"
    DAY_ANNOUNCE = "day_announce"
    DAY_DISCUSSION = "day_discussion"
    DAY_FREE_DISCUSSION = "day_free_discussion"
    VOTING = "voting"
    DAY_RESULT = "day_result"
    GAME_OVER = "game_over"


@dataclass
class PlayerState:
    player_id: int
    role: str = ""
    personality: str = ""
    is_alive: bool = True
    can_shoot: bool = False  # hunter flag
    poisoned: bool = False    # killed by witch poison → hunter can't shoot


@dataclass
class GameState:
    game_id: str
    phase: GamePhase = GamePhase.SETUP
    round: int = 1
    players: dict[int, PlayerState] = field(default_factory=dict)
    alive_players: list[int] = field(default_factory=list)
    # Current night state
    werewolf_kill_target: int | None = None
    witch_antidote_target: int | None = None  # None = not used, player_id = used
    witch_poison_target: int | None = None
    seer_check_target: int | None = None
    seer_check_result: str | None = None
    witch_antidote_used: bool = False
    witch_poison_used: bool = False
    witch_has_antidote: bool = True
    witch_has_poison: bool = True
    # Voting
    votes: dict[int, int] = field(default_factory=dict)  # voter_id → target_id
    # History for collector
    phase_records: list[dict] = field(default_factory=list)
    # Current phase temporary data
    current_werewolf_chat: list[dict] = field(default_factory=list)
    current_speeches: list[dict] = field(default_factory=list)
    current_actions: list[dict] = field(default_factory=list)
    current_events: list[dict] = field(default_factory=list)
    # Result
    winner: str | None = None
    # Event queue for SSE
    events: list[dict] = field(default_factory=list)

    def push_event(self, event: dict) -> None:
        self.events.append(event)

    def pop_events(self) -> list[dict]:
        events = self.events.copy()
        self.events.clear()
        return events

    def get_player(self, player_id: int) -> PlayerState:
        return self.players[player_id]

    def get_alive_werewolves(self) -> list[int]:
        return [p for p in self.alive_players
                if self.players[p].role == "werewolf"]

    def get_alive_good(self) -> list[int]:
        return [p for p in self.alive_players
                if self.players[p].role != "werewolf"]
