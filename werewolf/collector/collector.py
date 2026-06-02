"""Record full game data and write to JSON file."""

import json
import os
from datetime import datetime
from pathlib import Path
import logging

from werewolf.engine.state import GameState

logger = logging.getLogger(__name__)


class GameCollector:
    """Collects all game data during play and writes to disk on completion."""

    def __init__(self, data_dir: str = "data", api_provider: str = "",
                 model: str = ""):
        self.data_dir = Path(data_dir)
        self.api_provider = api_provider
        self.model = model
        self.output_path: Path | None = None
        self.duration_seconds: float | None = None

    def finalize(self, state: GameState, duration_seconds: float | None = None) -> str:
        """Write complete game data to JSON. Returns the file path."""
        self.duration_seconds = duration_seconds
        game_data = self._build_game_data(state)
        output_path = self._get_output_path(state.game_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(game_data, f, ensure_ascii=False, indent=2)

        logger.info("Game data saved to %s", output_path)
        return str(output_path)

    def _build_game_data(self, state: GameState) -> dict:
        """Build the complete game data dict."""
        players_config = []
        for pid in sorted(state.players):
            ps = state.players[pid]
            players_config.append({
                "id": pid,
                "role": ps.role,
                "personality": ps.personality,
            })

        # Format duration for display
        duration_display = None
        if self.duration_seconds is not None:
            total_seconds = int(self.duration_seconds)
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            duration_display = f"{minutes}分{seconds}秒"

        return {
            "game_id": state.game_id,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "api_provider": self.api_provider,
                "model": self.model,
                "players": players_config,
            },
            "phases": state.phase_records,
            "result": {
                "winner": state.winner,
                "surviving_players": sorted(state.alive_players),
                "roles_revealed": {
                    str(pid): ps.role
                    for pid, ps in state.players.items()
                },
                "duration_seconds": self.duration_seconds,
                "duration_display": duration_display,
            },
        }

    def _get_output_path(self, game_id: str) -> Path:
        """Generate output path: data/YYYY-MM-DD/game_NNN/game_data.json"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.data_dir / today / game_id / "game_data.json"
