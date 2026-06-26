#!/usr/bin/env python3
"""Integration test: run a full game with a mock LLM client."""

import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from werewolf.engine.game import Game
from werewolf.engine.roles import Role
from werewolf.agents.agent import create_agent_factory
from werewolf.collector.collector import GameCollector


# ── Mock LLM Client ─────────────────────────────────────────────

class MockClient:
    """Mock LLM client that returns deterministic-ish responses for testing."""

    def __init__(self, config=None):
        self.config = config or {}
        self.provider = "mock"

    def get_provider_name(self) -> str:
        return "mock"

    async def chat(self, messages, response_format=None, temperature=None):
        # Parse the user message to figure out what phase we're in
        user_msg = ""
        for m in messages:
            if m["role"] == "user":
                user_msg = m["content"]
                break

        # Generate appropriate mock response based on phase keywords
        thought = "Mock agent reasoning."
        speech = ""
        action = {}

        if "投票选择今晚要击杀" in user_msg or "vote" in user_msg.lower():
            target = self._pick_random_target(user_msg, "kill")
            action = {"type": "kill_vote", "target": target}
            thought = f"我觉得{target}号玩家可以杀。"
        elif "查验" in user_msg and "预言家" in str(messages):
            target = self._pick_random_target(user_msg, "check")
            action = {"type": "check", "target": target}
            thought = f"我查验{target}号。"
        elif "解药" in user_msg:
            action = {"type": "use_antidote"}
            thought = "用解药救人。"
        elif "毒药" in user_msg:
            target = self._pick_random_target(user_msg, "poison")
            action = {"type": "pass"}  # mock witch usually doesn't poison
            thought = "暂时不用毒药。"
        elif "发言" in user_msg or "讨论" in user_msg:
            target = self._pick_random_target(user_msg, "discuss")
            speech = f"我怀疑{target}号玩家，他的行为很可疑。"
            thought = f"{target}号可能是狼人。"
        elif "投票放逐" in user_msg:
            target = self._pick_random_target(user_msg, "vote")
            action = {"type": "vote", "target": target}
            thought = f"投票{target}号。"
        elif "遗言" in user_msg:
            speech = "我没什么好说的，游戏继续吧。"
            thought = "sad to go."
        elif "猎人" in user_msg and "开枪" in user_msg:
            target = self._pick_random_target(user_msg, "hunter")
            action = {"type": "shoot", "target": target}
            thought = f"带走{target}号。"
        elif "狼人" in user_msg and "同伴" in user_msg:
            target = self._pick_random_target(user_msg, "wolf_chat")
            speech = f"我觉得{target}号是个不错的选择。"
            thought = f"建议杀{target}号。"
        elif "自由讨论" in user_msg:
            speech = f"我认为我们需要更仔细地分析票型。"
            thought = "分析中..."

        return self._make_response(thought, speech, action)

    def _pick_random_target(self, msg: str, phase: str) -> int:
        """Extract alive players from context and pick one."""
        import re
        # Try to find valid targets first
        match = re.search(r'可选目标[：:]\s*([\d、, ]+)', msg)
        if match:
            ids = re.findall(r'\d+', match.group(1))
            ids = [int(i) for i in ids if int(i) > 0]
            if ids:
                return random.choice(ids)

        # Fall back to alive players
        match = re.search(r'存活玩家[：:]\s*([\d、, ]+)', msg)
        if match:
            ids = re.findall(r'\d+', match.group(1))
            ids = [int(i) for i in ids if int(i) > 0]
            if ids:
                speaker_match = re.search(r'你坐在(\d+)号位', msg)
                speaker = int(speaker_match.group(1)) if speaker_match else None
                candidates = [i for i in ids if i != speaker]
                if candidates:
                    if phase in ("kill", "wolf_chat"):
                        candidates = sorted(candidates, reverse=True)
                    return random.choice(candidates[:max(1, len(candidates)//2)])

        return 2  # safe fallback

    def _make_response(self, thought, speech, action):
        content = json.dumps({"thought": thought, "speech": speech, "action": action}, ensure_ascii=False)
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"total_tokens": 100},
        }

    @staticmethod
    def _client():
        return None


# ── Test ────────────────────────────────────────────────────────

async def test_game():
    print("=" * 60)
    print("Running mock game test...")
    print("=" * 60)

    # Load personalities
    import yaml
    from pathlib import Path
    traits_path = Path(__file__).parent / "werewolf" / "personalities" / "traits.yaml"
    with open(traits_path, encoding="utf-8") as f:
        personalities = yaml.safe_load(f)["personalities"]

    # Assign roles
    roles = ["werewolf"] * 3 + ["seer", "witch", "hunter"] + ["villager"] * 3
    random.shuffle(roles)

    shuffled_personalities = random.sample(personalities, 9)
    player_assignments = []
    for i, (role, personality) in enumerate(zip(roles, shuffled_personalities), 1):
        player_assignments.append({
            "player_id": i,
            "role": role,
            "personality": personality,
        })

    # Print role assignments (for debugging)
    print("\nRole assignments:")
    for pa in player_assignments:
        role_emoji = {"werewolf": "🐺", "seer": "🔮", "witch": "🧪", "hunter": "🔫", "villager": "👤"}
        personality_name = pa['personality']['name'] if isinstance(pa['personality'], dict) else pa['personality']
        print(f"  Player {pa['player_id']}: {role_emoji.get(pa['role'], '?')} {pa['role']} ({personality_name})")
    print()

    # Create mock client and agent factory
    mock_client = MockClient()
    agent_factory = create_agent_factory(mock_client)

    # Create collector
    test_data_dir = "data_test"
    collector = GameCollector(
        data_dir=test_data_dir,
        api_provider="mock",
        model="mock",
    )

    # Create game
    game = Game(
        game_id="test_mock_001",
        player_assignments=player_assignments,
        agent_factory=agent_factory,
        config={
            "phase_delay_seconds": 0,
            "free_discussion_rounds": 1,
            "witch_self_save_first_night": True,
        },
        collector=collector,
    )

    # Run game
    state = await game.run()

    # Verify results
    print(f"\nGame over! Winner: {state.winner}")
    print(f"Rounds played: {state.round}")
    print(f"Alive players: {state.alive_players}")
    print(f"Phase records: {len(state.phase_records)}")

    # Verify data file exists
    import glob
    data_files = glob.glob(f"{test_data_dir}/**/test_mock_001/game_data.json", recursive=True)
    if data_files:
        print(f"\n✅ Data file created: {data_files[0]}")
        with open(data_files[0], encoding="utf-8") as f:
            data = json.load(f)

        # Validate structure
        assert "game_id" in data, "Missing game_id"
        assert "phases" in data, "Missing phases"
        assert "result" in data, "Missing result"
        assert "config" in data, "Missing config"
        assert len(data["config"]["players"]) == 9, "Should have 9 players"
        assert data["result"]["winner"] is not None, "Should have a winner"

        print(f"✅ Game data valid: {len(data['phases'])} phases recorded")
        print(f"   Winner: {data['result']['winner']}")
        print(f"   Surviving: {data['result']['surviving_players']}")

        # Check phase sequence
        phase_names = [p["phase"] for p in data["phases"]]
        print(f"   Phase sequence: {' → '.join(phase_names)}")

        # Clean up test data
        import shutil
        shutil.rmtree(test_data_dir)
        print(f"\n✅ All tests passed!")
    else:
        print("\n❌ Data file not found!")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_game())
