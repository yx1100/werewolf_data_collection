"""Game rule validation and win condition checking."""

from .state import GameState
from .roles import Role


def check_win_condition(state: GameState) -> str | None:
    """Check if game is over. Returns 'werewolves', 'good', or None."""
    alive_werewolves = state.get_alive_werewolves()
    alive_good = state.get_alive_good()

    if len(alive_werewolves) == 0:
        return "good"

    if len(alive_werewolves) >= len(alive_good):
        return "werewolves"

    return None


def validate_night_action(state: GameState, player_id: int,
                          action_type: str, target_id: int | None) -> str | None:
    """Validate a night action. Returns error message or None if valid."""
    player = state.get_player(player_id)

    if action_type == "kill":
        if player.role != Role.WEREWOLF.value:
            return "只有狼人可以击杀"
        if target_id is not None and target_id not in state.alive_players:
            return "目标玩家已死亡"
        # Can't kill werewolf teammates
        if target_id is not None:
            target = state.get_player(target_id)
            if target.role == Role.WEREWOLF.value:
                return "不能击杀狼人同伴"

    elif action_type == "check":
        if player.role != Role.SEER.value:
            return "只有预言家可以查验"
        if target_id is not None and target_id not in state.alive_players:
            return "目标玩家已死亡"

    elif action_type == "use_antidote":
        if player.role != Role.WITCH.value:
            return "只有女巫可以使用解药"
        if state.witch_antidote_used or not state.witch_has_antidote:
            return "解药已使用"

    elif action_type == "use_poison":
        if player.role != Role.WITCH.value:
            return "只有女巫可以使用毒药"
        if state.witch_poison_used or not state.witch_has_poison:
            return "毒药已使用"
        if target_id is not None and target_id not in state.alive_players:
            return "目标玩家已死亡"

    elif action_type == "pass":
        pass  # always valid
    else:
        return f"未知行动类型: {action_type}"

    return None


def can_hunter_shoot(player_state) -> bool:
    """Hunter can shoot when killed by vote or werewolves, NOT when poisoned."""
    if player_state.role != Role.HUNTER.value:
        return False
    if player_state.poisoned:
        return False
    return True
