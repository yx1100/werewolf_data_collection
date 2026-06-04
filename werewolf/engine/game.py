"""Game orchestrator — drives the state machine, coordinates agents and phases."""

import asyncio
import logging
import time
from typing import Any

from .state import GameState, GamePhase, PlayerState
from .roles import Role, ROLE_TEAM, Role as RoleEnum
from .rules import check_win_condition, validate_night_action, can_hunter_shoot

# Role name mapping for display
_ROLE_ZH = {"werewolf": "狼人", "seer": "预言家", "witch": "女巫", "hunter": "猎人", "villager": "平民"}

logger = logging.getLogger(__name__)


class Game:
    """Orchestrates a single Werewolf game."""

    def __init__(self, game_id: str, player_assignments: list[dict],
                 agent_factory, config: dict, collector=None):
        self.game_id = game_id
        self.config = config
        self.agent_factory = agent_factory
        self.collector = collector
        self.phase_delay = config.get("phase_delay_seconds", 3)
        self.free_discussion_rounds = config.get("free_discussion_rounds", 2)
        self.witch_self_save_first_night = config.get(
            "witch_self_save_first_night", True)

        # Game timing
        self.start_time = time.time()
        self.end_time: float | None = None
        self.duration_seconds: float | None = None

        # Pause / Stop controls
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # initially not paused
        self._stopped = False

        # Build agents
        self.agents: dict[int, Any] = {}
        for pa in player_assignments:
            agent = agent_factory(
                player_id=pa["player_id"],
                role=pa["role"],
                personality=pa["personality"],
            )
            self.agents[pa["player_id"]] = agent

        # Init game state
        self.state = GameState(game_id=game_id)
        for pa in player_assignments:
            personality_name = pa["personality"]["name"] if isinstance(pa["personality"], dict) else pa["personality"]
            ps = PlayerState(
                player_id=pa["player_id"],
                role=pa["role"],
                personality=personality_name,
            )
            self.state.players[pa["player_id"]] = ps
        self.state.alive_players = sorted([p["player_id"] for p in player_assignments])
        self.state.round = 1
        self.state.phase = GamePhase.NIGHT

    async def run(self) -> GameState:
        """Run the full game loop. Returns final GameState."""
        logger.info(f"Game {self.game_id} started. "
                     f"Roles: {self._role_summary()}")

        while True:
            await self._check_pause_and_stop()

            winner = check_win_condition(self.state)
            if winner:
                # If game ended overnight, announce deaths first
                if (self.state.phase == GamePhase.DAY_ANNOUNCE
                        and self.state.current_events):
                    await self._do_phase_day_announce()
                self.state.winner = winner
                await self._do_phase_game_over()
                break

            if self.state.phase == GamePhase.NIGHT:
                await self._do_phase_night()
                self.state.phase = GamePhase.DAY_ANNOUNCE

            elif self.state.phase == GamePhase.DAY_ANNOUNCE:
                await self._do_phase_day_announce()
                # Check win after deaths
                winner = check_win_condition(self.state)
                if winner:
                    self.state.winner = winner
                    await self._do_phase_game_over()
                    break
                self.state.phase = GamePhase.DAY_DISCUSSION

            elif self.state.phase == GamePhase.DAY_DISCUSSION:
                await self._do_phase_day_discussion()
                self.state.phase = GamePhase.DAY_FREE_DISCUSSION

            elif self.state.phase == GamePhase.DAY_FREE_DISCUSSION:
                await self._do_phase_free_discussion()
                self.state.phase = GamePhase.VOTING

            elif self.state.phase == GamePhase.VOTING:
                await self._do_phase_voting()
                # If tie → PK phase, otherwise → DAY_RESULT
                if self.state.pk_candidates:
                    self.state.phase = GamePhase.PK_DISCUSSION
                else:
                    self.state.phase = GamePhase.DAY_RESULT

            elif self.state.phase == GamePhase.PK_DISCUSSION:
                await self._do_phase_pk_discussion()
                self.state.phase = GamePhase.PK_VOTING

            elif self.state.phase == GamePhase.PK_VOTING:
                await self._do_phase_pk_voting()
                self.state.phase = GamePhase.DAY_RESULT

            elif self.state.phase == GamePhase.DAY_RESULT:
                await self._do_phase_day_result()
                # Check win after elimination
                winner = check_win_condition(self.state)
                if winner:
                    self.state.winner = winner
                    await self._do_phase_game_over()
                    break
                self.state.round += 1
                self.state.phase = GamePhase.NIGHT

            elif self.state.phase == GamePhase.GAME_OVER:
                break

            await self._delay()

        # Record game end time and duration
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time

        # Finalize: collect data
        if self.collector:
            self.collector.finalize(self.state, self.duration_seconds)

        logger.info(f"Game {self.game_id} over. Winner: {self.state.winner}. "
                    f"Duration: {self.duration_seconds:.1f}s")
        return self.state

    # ── Night Phase ──────────────────────────────────────────────

    async def _do_phase_night(self):
        """Execute night phase: werewolf chat + kill, seer check, witch actions."""
        self.state.current_werewolf_chat = []
        self.state.current_actions = []
        self.state.current_events = []
        # Reset night action targets (prevent stale data from previous nights)
        self.state.werewolf_kill_target = None
        self.state.witch_antidote_target = None
        self.state.witch_poison_target = None
        self.state.seer_check_target = None
        self.state.seer_check_result = None
        self.state.push_event({
            "type": "phase_change",
            "phase": "夜晚阶段",
            "round": self.state.round,
            "message": f"🌙 第 {self.state.round} 夜降临，请所有玩家闭眼。",
        })

        alive_werewolves = [p for p in self.state.alive_players
                            if self.state.players[p].role == "werewolf"]
        alive_seer = [p for p in self.state.alive_players
                      if self.state.players[p].role == "seer"]
        alive_witch = [p for p in self.state.alive_players
                       if self.state.players[p].role == "witch"]

        # 1. Werewolves discuss and decide kill target
        if alive_werewolves:
            await self._check_pause_and_stop()
            await self._werewolf_night_phase(alive_werewolves)
            kill_target = self.state.werewolf_kill_target
        else:
            kill_target = None

        # 2. Seer checks
        if alive_seer:
            await self._check_pause_and_stop()
            seer_id = alive_seer[0]
            await self._seer_night_action(seer_id)

        # 3. Witch uses potions (knows who was killed)
        if alive_witch:
            await self._check_pause_and_stop()
            witch_id = alive_witch[0]
            await self._witch_night_action(witch_id, kill_target)

        # 4. Resolve night actions
        self._resolve_night_actions(kill_target)

        # Record the phase
        record = {
            "phase": "NIGHT",
            "round": self.state.round,
            "werewolf_chat": list(self.state.current_werewolf_chat),
            "actions": list(self.state.current_actions),
        }
        self.state.phase_records.append(record)

    async def _werewolf_night_phase(self, werewolf_ids: list[int]):
        """Werewolves chat with each other, then vote on kill target."""
        # Step 1: Werewolf discussion (each wolf can speak multiple times in a chat loop)
        chat_history: list[dict] = []
        chat_turn = 0

        # Werewolves discuss — give them a few rounds of chat
        chat_rounds = 3
        for _ in range(chat_rounds):
            for wolf_id in werewolf_ids:
                await self._check_pause_and_stop()
                agent = self.agents[wolf_id]
                context = {
                    "phase": "NIGHT_WEREWOLF_CHAT",
                    "round": self.state.round,
                    "alive_players": self.state.alive_players,
                    "werewolf_teammates": [w for w in werewolf_ids if w != wolf_id],
                    "chat_history": chat_history,
                    "public_history": self._public_history(),
                }
                output = await agent.decide(context)
                if output and output.speech:
                    chat_turn += 1
                    msg = {
                        "player_id": wolf_id,
                        "message": output.speech,
                        "thought": output.thought or "",
                        "turn": chat_turn,
                        "visibility": "private",
                    }
                    chat_history.append(msg)
                    self.state.push_event({
                        "type": "werewolf_chat",
                        "player_id": wolf_id,
                        "message": output.speech,
                        "thought": output.thought or "",
                    })

        self.state.current_werewolf_chat = chat_history

        # Step 2: Each werewolf votes on kill target
        kill_votes: dict[int, int] = {}  # target → count
        for wolf_id in werewolf_ids:
            await self._check_pause_and_stop()
            agent = self.agents[wolf_id]
            context = {
                "phase": "NIGHT_WEREWOLF_KILL",
                "round": self.state.round,
                "alive_players": self.state.alive_players,
                "werewolf_teammates": [w for w in werewolf_ids if w != wolf_id],
                "chat_history": chat_history,
                "public_history": self._public_history(),
            }
            output = await agent.decide(context)
            if output and output.action:
                target = output.action.get("target")
                if target and target in self.state.alive_players:
                    kill_votes[target] = kill_votes.get(target, 0) + 1
                    self.state.current_actions.append({
                        "player_id": wolf_id,
                        "role": "werewolf",
                        "action": "kill_vote",
                        "target": target,
                        "thought": output.thought or "",
                        "visibility": "private",
                    })

        # Majority vote for kill (need > half of wolves to agree)
        if kill_votes:
            target = max(kill_votes, key=kill_votes.get)
            if kill_votes[target] > len(werewolf_ids) // 2:  # e.g. 3 wolves → >1 (need 2+); 1 wolf → >0 (need 1+)
                self.state.werewolf_kill_target = target
            else:
                # No majority — no kill this night
                self.state.werewolf_kill_target = None
            self.state.current_actions.append({
                "player_id": 0,
                "role": "werewolf",
                "action": "kill",
                "target": self.state.werewolf_kill_target,
                "vote_summary": kill_votes,
                "visibility": "private",
            })

            # Remind each werewolf of the kill decision (prevents "空刀" hallucination)
            if self.state.werewolf_kill_target is not None:
                for wolf_id in werewolf_ids:
                    self.agents[wolf_id].add_private_info(
                        f"今晚狼人投票结果：决定击杀{self.state.werewolf_kill_target}"
                        f"号玩家（{kill_votes[self.state.werewolf_kill_target]}票）。")

    async def _seer_night_action(self, seer_id: int):
        """Seer checks one player's identity."""
        agent = self.agents[seer_id]
        context = {
            "phase": "NIGHT_SEER",
            "round": self.state.round,
            "alive_players": self.state.alive_players,
            "valid_targets": [p for p in self.state.alive_players if p != seer_id],
            "public_history": self._public_history(),
        }
        output = await agent.decide(context)
        if output and output.action:
            target = output.action.get("target")
            if target and target in self.state.alive_players:
                self.state.seer_check_target = target
                target_role = self.state.players[target].role
                is_werewolf = target_role == "werewolf"
                self.state.seer_check_result = "wolf" if is_werewolf else "good"

                result_msg = "狼人" if is_werewolf else "好人"
                self.state.current_actions.append({
                    "player_id": seer_id,
                    "role": "seer",
                    "action": "check",
                    "target": target,
                    "result": result_msg,
                    "result_raw": self.state.seer_check_result,
                    "thought": output.thought or "",
                    "visibility": "private",
                })

                # Tell the seer the result via a private notification
                agent.add_private_info(
                    f"查验结果：{target}号玩家是{result_msg}。")

    async def _witch_night_action(self, witch_id: int, kill_target: int | None):
        """Witch decides whether to use antidote and/or poison."""
        agent = self.agents[witch_id]

        # Antidote decision
        if (self.state.witch_has_antidote
                and not self.state.witch_antidote_used
                and kill_target is not None):
            can_self_save = (self.state.round == 1
                             and self.witch_self_save_first_night
                             and kill_target == witch_id)
            context = {
                "phase": "NIGHT_WITCH_ANTIDOTE",
                "round": self.state.round,
                "alive_players": self.state.alive_players,
                "killed_player": kill_target,
                "can_self_save": can_self_save,
                "witch_has_antidote": self.state.witch_has_antidote,
                "witch_has_poison": self.state.witch_has_poison,
                "witch_antidote_used": self.state.witch_antidote_used,
                "witch_poison_used": self.state.witch_poison_used,
                "public_history": self._public_history(),
            }
            output = await agent.decide(context)
            if output and output.action:
                action_type = output.action.get("type", "pass")
                if action_type == "use_antidote":
                    self.state.witch_antidote_target = kill_target
                    self.state.witch_antidote_used = True
                    self.state.witch_has_antidote = False
                    self.state.current_actions.append({
                        "player_id": witch_id,
                        "role": "witch",
                        "action": "use_antidote",
                        "target": kill_target,
                        "thought": output.thought or "",
                        "visibility": "private",
                    })
                    agent.add_private_info("你已使用解药，不再拥有解药。")
                else:
                    self.state.witch_antidote_target = None
                    self.state.current_actions.append({
                        "player_id": witch_id,
                        "role": "witch",
                        "action": "pass_antidote",
                        "thought": output.thought or "",
                        "visibility": "private",
                    })
                    agent.add_private_info("你没有使用解药。")

        # Poison decision
        if self.state.witch_has_poison and not self.state.witch_poison_used:
            context = {
                "phase": "NIGHT_WITCH_POISON",
                "round": self.state.round,
                "alive_players": self.state.alive_players,
                "valid_targets": [p for p in self.state.alive_players if p != witch_id],
                "killed_player": kill_target if (self.state.witch_has_antidote and not self.state.witch_antidote_used) else None,
                "witch_has_antidote": self.state.witch_has_antidote,
                "witch_has_poison": self.state.witch_has_poison,
                "witch_antidote_used": self.state.witch_antidote_used,
                "witch_poison_used": self.state.witch_poison_used,
                "public_history": self._public_history(),
            }
            output = await agent.decide(context)
            if output and output.action:
                action_type = output.action.get("type", "pass")
                target = output.action.get("target")
                if action_type == "use_poison" and target and target in self.state.alive_players:
                    self.state.witch_poison_target = target
                    self.state.witch_poison_used = True
                    self.state.witch_has_poison = False
                    self.state.players[target].poisoned = True
                    self.state.current_actions.append({
                        "player_id": witch_id,
                        "role": "witch",
                        "action": "use_poison",
                        "target": target,
                        "thought": output.thought or "",
                        "visibility": "private",
                    })
                    agent.add_private_info("你已使用毒药，不再拥有毒药。")
                else:
                    self.state.current_actions.append({
                        "player_id": witch_id,
                        "role": "witch",
                        "action": "pass_poison",
                        "thought": output.thought or "",
                        "visibility": "private",
                    })
                    agent.add_private_info("你没有使用毒药。")

    def _build_werewolf_context(self, werewolf_ids: list[int]) -> str:
        """Build context for werewolf discussion."""
        teammates = [w for w in werewolf_ids]
        return f"你的狼人同伴是：{teammates}号玩家。请与同伴讨论今晚要击杀的目标。"

    def _public_history(self) -> list[dict]:
        """Return public events visible to all players — private thoughts stripped.

        Convention:
        - 'thought' field is always private → strip it from any entry.
        - Entries with 'visibility: private' are entirely private → skip them.
        - All other entries are public.
        """
        events = []
        for record in self.state.phase_records:
            phase = record["phase"]
            if phase == "NIGHT":
                continue  # all night entries are private
            elif phase == "DAY_ANNOUNCE":
                # Inject the day announcement message as a synthetic event
                msg = record.get("message", "")
                if msg:
                    events.append({"type": "day_announce", "message": msg})
                for e in record.get("events", []):
                    if e.get("visibility") != "private":
                        entry = dict(e)
                        entry.pop("thought", None)  # strip private field
                        events.append(entry)
            elif phase in ("DAY_DISCUSSION", "DAY_FREE_DISCUSSION"):
                for s in record.get("speeches", []):
                    entry = dict(s)
                    entry.pop("thought", None)  # strip private field
                    entry.setdefault("type", "speech")
                    events.append(entry)
            elif phase == "VOTING":
                for v in record.get("votes", []):
                    entry = dict(v)
                    entry.pop("thought", None)
                    events.append(entry)
            elif phase == "PK_DISCUSSION":
                for s in record.get("speeches", []):
                    entry = dict(s)
                    entry.pop("thought", None)
                    entry.setdefault("type", "speech")
                    events.append(entry)
            elif phase == "PK_VOTING":
                for v in record.get("votes", []):
                    entry = dict(v)
                    entry.pop("thought", None)
                    events.append(entry)
            elif phase == "DAY_RESULT":
                for e in record.get("events", []):
                    entry = dict(e)
                    entry.pop("thought", None)
                    events.append(entry)
            elif phase == "GAME_OVER":
                events.append({
                    "type": "game_over",
                    "winner": record.get("winner"),
                    "round": record.get("round"),
                })
        return events

    def _seer_history(self, seer_id: int) -> list[dict]:
        """Return seer's private check results."""
        results = []
        for record in self.state.phase_records:
            if record["phase"] == "NIGHT":
                for action in record.get("actions", []):
                    if (action.get("role") == "seer"
                            and action["player_id"] == seer_id):
                        results.append({
                            "phase": "NIGHT",
                            "round": record["round"],
                            "action": "check",
                            "target": action["target"],
                            "result": action.get("result", ""),
                        })
        return results

    def _resolve_night_actions(self, kill_target: int | None):
        """Resolve all night actions: apply kills, saves, poison."""
        deaths = []
        logger.info("Resolving night actions: kill_target=%s, antidote_target=%s, "
                     "antidote_used=%s, poison_target=%s, poison_used=%s",
                     kill_target, self.state.witch_antidote_target,
                     self.state.witch_antidote_used,
                     self.state.witch_poison_target, self.state.witch_poison_used)

        # Werewolf kill
        if kill_target is not None:
            # If witch saved this player, they survive
            saved = (self.state.witch_antidote_target == kill_target
                     and self.state.witch_antidote_used)
            logger.info("Werewolf kill: target=%d, saved=%s", kill_target, saved)
            if not saved:
                deaths.append({
                    "player_id": kill_target,
                    "cause": "werewolf_kill",
                    "poisoned": False,
                })

        # Witch poison
        if (self.state.witch_poison_target is not None
                and self.state.witch_poison_used):
            # Don't double-count if poisoned and killed
            already_dead = any(
                d["player_id"] == self.state.witch_poison_target for d in deaths)
            if not already_dead:
                deaths.append({
                    "player_id": self.state.witch_poison_target,
                    "cause": "witch_poison",
                    "poisoned": True,
                })

        # Apply deaths
        for death in deaths:
            pid = death["player_id"]
            self.state.players[pid].is_alive = False
            self.state.players[pid].poisoned = death.get("poisoned", False)
            if pid in self.state.alive_players:
                self.state.alive_players.remove(pid)
            # Notify surviving werewolves if a teammate died
            if self.state.players[pid].role == "werewolf":
                for w in self.state.get_alive_werewolves():
                    self.agents[w].add_private_info(
                        f"你的狼队友{pid}号已死亡。")

        # Notify werewolves if their kill target actually died (death is public info).
        # Note: we do NOT tell wolves about witch saving — that's witch's private info.
        # Wolves can infer it from "we voted to kill X" + "peaceful night" (public).
        alive_wolves = self.state.get_alive_werewolves()
        if kill_target is not None and alive_wolves:
            if any(d["player_id"] == kill_target for d in deaths):
                for w in alive_wolves:
                    self.agents[w].add_private_info(
                        f"你们击杀的{kill_target}号玩家死亡。")

        # Store deaths for announcement
        self.state.current_events = deaths
        logger.info("Night resolved: %d deaths, current_events=%s", len(deaths), deaths)

    # ── Day Phases ───────────────────────────────────────────────

    async def _do_phase_day_announce(self):
        """Announce night results, let night-killed hunters shoot."""
        deaths = self.state.current_events

        # Build detailed night action summary for display
        night_summary = self._build_night_summary(deaths)

        if not deaths:
            msg = "天亮了，昨晚是平安夜，没有人死亡。"
        else:
            dead_names = "、".join(str(d["player_id"]) for d in deaths)
            msg = f"天亮了，昨晚死亡玩家：{dead_names}号。"

        self.state.push_event({
            "type": "day_announce",
            "message": msg,
            "night_summary": night_summary,
            "dead_players": [d["player_id"] for d in deaths],
        })

        for death in deaths:
            pid = death["player_id"]
            # If hunter was killed (not poisoned), flag can_shoot
            player = self.state.players[pid]
            if can_hunter_shoot(player):
                player.can_shoot = True

        # Build events list for record
        events = [
            {"type": "death", "player_id": d["player_id"],
             "cause": d["cause"]}
            for d in deaths
        ]

        # Night-killed hunters: activate and shoot (before day discussion)
        for pid, ps in self.state.players.items():
            if not ps.is_alive and ps.can_shoot:
                self.state.push_event({
                    "type": "hunter_activate",
                    "message": f"猎人{pid}号昨晚死亡，可以开枪！",
                })
                await self._check_pause_and_stop()
                agent = self.agents[pid]
                output = await agent.decide(context={
                    "phase": "HUNTER_SHOOT",
                    "round": self.state.round,
                    "alive_players": self.state.alive_players,
                    "valid_targets": [p for p in self.state.alive_players if p != pid],
                    "public_history": self._public_history(),
                    "cause": "night_kill",
                })
                if output and output.action:
                    target = output.action.get("target")
                    if target and target in self.state.alive_players:
                        target_player = self.state.players[target]
                        target_player.is_alive = False
                        if target in self.state.alive_players:
                            self.state.alive_players.remove(target)
                        events.append({
                            "type": "hunter_shoot",
                            "player_id": pid,
                            "target": target,
                            "thought": output.thought or "",
                        })
                        self.state.push_event({
                            "type": "hunter_shoot",
                            "player_id": pid,
                            "target": target,
                            "message": f"🔫 猎人{pid}号开枪带走{target}号玩家！",
                        })
                ps.can_shoot = False  # clear flag after shooting
                break  # only one hunter can die per round

        record = {
            "phase": "DAY_ANNOUNCE",
            "round": self.state.round,
            "message": msg,
            "night_summary": {"data": night_summary, "visibility": "private"},
            "events": events,
        }
        self.state.phase_records.append(record)
        self.state.current_events = []

    async def _do_phase_day_discussion(self):
        """Structured round-robin discussion: each alive player speaks once."""
        self.state.current_speeches = []
        self.state.push_event({
            "type": "phase_change",
            "phase": "白天阶段：顺序发言",
            "message": f"📢 第 {self.state.round} 天讨论开始，请按顺序发言。",
        })

        speeches = []
        discussion_context: list[dict] = []  # accumulate as players speak
        speech_count = 0

        for player_id in sorted(self.state.alive_players):
            await self._check_pause_and_stop()
            agent = self.agents[player_id]
            context = {
                "phase": "DAY_DISCUSSION",
                "round": self.state.round,
                "alive_players": self.state.alive_players,
                "discussion_so_far": discussion_context,
                "public_history": self._public_history(),
                "speaker_id": player_id,
            }
            output = await agent.decide(context)
            if output:
                speech_count += 1
                # Unified entry: thought (private) + speech (public), thought first
                speeches.append({
                    "player_id": player_id,
                    "turn": speech_count,
                    "thought": output.thought or "",
                    "speech": output.speech or "",
                })
                discussion_context.append({
                    "player_id": player_id,
                    "speech": output.speech or "",
                    "thought": output.thought or "",
                    "turn": speech_count,
                })

                self.state.push_event({
                    "type": "speech",
                    "player_id": player_id,
                    "speech": output.speech or "",
                    "thought": output.thought or "",
                    "phase": "day_discussion",
                })

        self.state.current_speeches = speeches
        record = {
            "phase": "DAY_DISCUSSION",
            "round": self.state.round,
            "speeches": speeches,
        }
        self.state.phase_records.append(record)

    async def _do_phase_free_discussion(self):
        """Free discussion: N rounds where each player can speak."""
        self.state.current_speeches = []
        self.state.push_event({
            "type": "phase_change",
            "phase": "白天阶段：自由发言",
            "message": "💬 自由讨论阶段开始。",
        })

        all_speeches = []
        discussion_ctx: list[dict] = []  # includes thought for same-round context
        free_speech_count = 0

        for round_num in range(1, self.free_discussion_rounds + 1):
            for player_id in sorted(self.state.alive_players):
                await self._check_pause_and_stop()
                agent = self.agents[player_id]
                context = {
                    "phase": "DAY_FREE_DISCUSSION",
                    "round": self.state.round,
                    "free_round": round_num,
                    "alive_players": self.state.alive_players,
                    "discussion_so_far": discussion_ctx,
                    "public_history": self._public_history(),
                    "speaker_id": player_id,
                }
                output = await agent.decide(context)
                if output:
                    free_speech_count += 1
                    # Unified entry: thought (private) + speech (public), thought first
                    all_speeches.append({
                        "player_id": player_id,
                        "turn": free_speech_count,
                        "free_round": round_num,
                        "thought": output.thought or "",
                        "speech": output.speech or "",
                    })
                    discussion_ctx.append({
                        "player_id": player_id,
                        "speech": output.speech or "",
                        "thought": output.thought or "",
                        "turn": free_speech_count,
                        "free_round": round_num,
                    })

                    self.state.push_event({
                        "type": "speech",
                        "player_id": player_id,
                        "speech": output.speech or "",
                        "thought": output.thought or "",
                        "phase": "day_free_discussion",
                    })

        self.state.current_speeches = all_speeches
        record = {
            "phase": "DAY_FREE_DISCUSSION",
            "round": self.state.round,
            "speeches": all_speeches,
        }
        self.state.phase_records.append(record)

    async def _do_phase_voting(self):
        """Voting phase: each alive player votes to eliminate someone."""
        self.state.votes = {}
        self.state.push_event({
            "type": "phase_change",
            "phase": "投票阶段",
            "message": "🗳️ 投票阶段开始，请投票选出要放逐的玩家。",
        })

        votes: dict[int, int] = {}  # voter_id → target_id
        vote_entries: list[dict] = []  # unified: for both SSE and storage

        # Build discussion_so_far from the most recent discussion phase
        discussion_so_far: list[dict] = []
        for r in reversed(self.state.phase_records):
            if r.get("phase") in ("DAY_DISCUSSION", "DAY_FREE_DISCUSSION"):
                discussion_so_far = list(r.get("speeches", []))
                break

        for player_id in sorted(self.state.alive_players):
            await self._check_pause_and_stop()
            agent = self.agents[player_id]
            context = {
                "phase": "VOTING",
                "round": self.state.round,
                "alive_players": self.state.alive_players,
                "discussion_so_far": discussion_so_far,
                "public_history": self._public_history(),
                "valid_targets": [p for p in self.state.alive_players
                                  if p != player_id],
            }
            output = await agent.decide(context)
            if output and output.action:
                target = output.action.get("target")
                # Validate target is alive and not self
                if (target is not None
                        and isinstance(target, int)
                        and target in self.state.alive_players
                        and target != player_id):
                    votes[player_id] = target
                    vote_entries.append({
                        "voter_id": player_id,
                        "thought": output.thought or "",
                        "target": target,
                    })

        self.state.votes = votes

        # Tally
        tally: dict[int, int] = {}
        for target in votes.values():
            tally[target] = tally.get(target, 0) + 1

        # Find max — if tie, trigger PK phase
        max_votes = max(tally.values()) if tally else 0
        top_candidates = [p for p, c in tally.items() if c == max_votes]

        eliminated = top_candidates[0] if len(top_candidates) == 1 else None
        is_tie = len(top_candidates) > 1

        # Store PK candidates for potential tie-breaking
        self.state.pk_candidates = top_candidates if is_tie else []

        self.state.current_events = [{
            "type": "vote_result",
            "eliminated": eliminated,
            "vote_count": {str(k): v for k, v in tally.items()},
            "tie": is_tie,
        }]

        if eliminated is not None:
            self.state.push_event({
                "type": "vote_result",
                "message": f"{eliminated}号玩家（{_ROLE_ZH.get(self.state.players[eliminated].role, '?')}）被投票放逐。",
                "eliminated": eliminated,
                "votes": vote_entries,
                "tally": {str(k): v for k, v in tally.items()},
            })
        else:
            pk_names = "、".join(str(p) for p in top_candidates)
            self.state.push_event({
                "type": "vote_result",
                "message": f"投票平票（{pk_names}号），进入PK发言环节。",
                "tie": True,
                "votes": vote_entries,
                "tally": {str(k): v for k, v in tally.items()},
            })

        record = {
            "phase": "VOTING",
            "round": self.state.round,
            "votes": vote_entries,
            "result": {
                "eliminated": eliminated,
                "vote_count": {str(k): v for k, v in tally.items()},
                "tie": is_tie,
            },
        }
        self.state.phase_records.append(record)

    # ── PK (Tie-Breaking) Phases ──────────────────────────────────

    async def _do_phase_pk_discussion(self):
        """PK discussion: each tied player speaks once to defend themselves."""
        pk = self.state.pk_candidates
        pk_names = "、".join(str(p) for p in pk)
        self.state.push_event({
            "type": "phase_change",
            "phase": "PK发言阶段",
            "message": f"🎤 平票！{pk_names}号进入PK台，各发言一轮为自己辩护。",
        })

        speeches = []
        speech_count = 0
        for player_id in sorted(pk):
            await self._check_pause_and_stop()
            agent = self.agents[player_id]
            context = {
                "phase": "PK_DISCUSSION",
                "round": self.state.round,
                "alive_players": self.state.alive_players,
                "pk_candidates": pk,
                "public_history": self._public_history(),
                "speaker_id": player_id,
            }
            output = await agent.decide(context)
            if output:
                speech_count += 1
                speeches.append({
                    "player_id": player_id,
                    "turn": speech_count,
                    "thought": output.thought or "",
                    "speech": output.speech or "",
                })
                self.state.push_event({
                    "type": "speech",
                    "player_id": player_id,
                    "speech": output.speech or "",
                    "thought": output.thought or "",
                    "phase": "pk_discussion",
                })

        record = {
            "phase": "PK_DISCUSSION",
            "round": self.state.round,
            "pk_candidates": list(pk),
            "speeches": speeches,
        }
        self.state.phase_records.append(record)

    async def _do_phase_pk_voting(self):
        """PK re-vote: non-tied players vote only for tied candidates."""
        pk = self.state.pk_candidates
        pk_names = "、".join(str(p) for p in pk)
        voters = [p for p in self.state.alive_players if p not in pk]
        self.state.votes = {}
        self.state.push_event({
            "type": "phase_change",
            "phase": "PK投票阶段",
            "message": f"🗳️ PK投票：请从 {pk_names} 号中选择一人放逐。PK台玩家无投票权。",
        })

        votes: dict[int, int] = {}
        vote_entries: list[dict] = []

        for player_id in sorted(voters):
            await self._check_pause_and_stop()
            agent = self.agents[player_id]
            context = {
                "phase": "PK_VOTING",
                "round": self.state.round,
                "alive_players": self.state.alive_players,
                "pk_candidates": pk,
                "valid_targets": list(pk),
                "public_history": self._public_history(),
            }
            output = await agent.decide(context)
            if output and output.action:
                target = output.action.get("target")
                if (target is not None and isinstance(target, int)
                        and target in pk):
                    votes[player_id] = target
                    vote_entries.append({
                        "voter_id": player_id,
                        "thought": output.thought or "",
                        "target": target,
                    })

        self.state.votes = votes

        # Tally
        tally: dict[int, int] = {}
        for target in votes.values():
            tally[target] = tally.get(target, 0) + 1

        max_votes = max(tally.values()) if tally else 0
        top_candidates = [p for p, c in tally.items() if c == max_votes]
        eliminated = top_candidates[0] if len(top_candidates) == 1 else None

        self.state.current_events = [{
            "type": "vote_result",
            "eliminated": eliminated,
            "vote_count": {str(k): v for k, v in tally.items()},
            "tie": len(top_candidates) > 1,
        }]

        if eliminated is not None:
            self.state.push_event({
                "type": "vote_result",
                "message": f"{eliminated}号玩家（{_ROLE_ZH.get(self.state.players[eliminated].role, '?')}）被投票放逐。",
                "eliminated": eliminated,
                "votes": vote_entries,
                "tally": {str(k): v for k, v in tally.items()},
            })
        else:
            self.state.push_event({
                "type": "vote_result",
                "message": "PK投票再次平票，本轮无人被放逐（平安日）。",
                "tie": True,
                "votes": vote_entries,
                "tally": {str(k): v for k, v in tally.items()},
            })

        # Clear PK state
        self.state.pk_candidates = []

        record = {
            "phase": "PK_VOTING",
            "round": self.state.round,
            "pk_candidates": list(pk),
            "votes": vote_entries,
            "result": {
                "eliminated": eliminated,
                "vote_count": {str(k): v for k, v in tally.items()},
                "tie": len(top_candidates) > 1,
            },
        }
        self.state.phase_records.append(record)

    # ── Day Result ─────────────────────────────────────────────────

    async def _do_phase_day_result(self):
        """Process elimination result: last words, hunter shot."""
        vote_result = self.state.current_events[0] if self.state.current_events else {}
        eliminated = vote_result.get("eliminated")
        self.state.current_events = []

        events = []

        if eliminated is not None and eliminated in self.state.alive_players:
            player = self.state.players[eliminated]
            player.is_alive = False
            self.state.alive_players.remove(eliminated)

            events.append({
                "type": "elimination",
                "player_id": eliminated,
            })
            self.state.push_event({
                "type": "elimination",
                "player_id": eliminated,
                "message": f"{eliminated}号玩家被放逐。",
            })

            # Last words from eliminated player
            await self._check_pause_and_stop()
            agent = self.agents[eliminated]
            context = {
                "phase": "LAST_WORDS",
                "round": self.state.round,
                "alive_players": self.state.alive_players,
                "eliminated_player": eliminated,
                "public_history": self._public_history(),
            }
            output = await agent.decide(context)
            if output and output.speech:
                events.append({
                    "type": "last_words",
                    "player_id": eliminated,
                    "thought": output.thought or "",
                    "speech": output.speech,
                })
                self.state.push_event({
                    "type": "last_words",
                    "player_id": eliminated,
                    "message": f"{eliminated}号玩家的遗言：{output.speech}",
                    "thought": output.thought or "",
                })

            # Hunter shot
            if can_hunter_shoot(player):
                self.state.push_event({
                    "type": "hunter_activate",
                    "message": f"猎人{eliminated}号可以开枪！",
                })
                await self._check_pause_and_stop()
                hunter_output = await agent.decide(context={
                    "phase": "HUNTER_SHOOT",
                    "round": self.state.round,
                    "alive_players": self.state.alive_players,
                    "valid_targets": [p for p in self.state.alive_players if p != eliminated],
                    "public_history": self._public_history(),
                    "cause": "vote_out",
                })
                if hunter_output and hunter_output.action:
                    target = hunter_output.action.get("target")
                    if target and target in self.state.alive_players:
                        target_player = self.state.players[target]
                        target_player.is_alive = False
                        self.state.alive_players.remove(target)
                        events.append({
                            "type": "hunter_shoot",
                            "player_id": eliminated,
                            "target": target,
                            "thought": hunter_output.thought or "",
                        })
                        self.state.push_event({
                            "type": "hunter_shoot",
                            "player_id": eliminated,
                            "target": target,
                            "message": f"🔫 猎人{eliminated}号开枪带走{target}号玩家！",
                        })

        record = {
            "phase": "DAY_RESULT",
            "round": self.state.round,
            "events": events,
        }
        self.state.phase_records.append(record)
        self.state.current_events = events

    async def _do_phase_game_over(self):
        """Game over phase."""
        winner_zh = "狼人阵营" if self.state.winner == "werewolves" else "好人阵营"

        # Reveal all roles
        roles_revealed = {}
        for pid, ps in self.state.players.items():
            alive_mark = " (存活)" if pid in self.state.alive_players else ""
            role_name = {"werewolf": "🐺 狼人", "seer": "🔮 预言家",
                         "witch": "🧪 女巫", "hunter": "🔫 猎人",
                         "villager": "👤 平民"}.get(ps.role, ps.role)
            roles_revealed[str(pid)] = {
                "role": ps.role,
                "display": f"{role_name}{alive_mark}",
                "alive": pid in self.state.alive_players,
            }

        self.state.push_event({
            "type": "game_over",
            "winner": self.state.winner,
            "message": f"🏆 游戏结束！{winner_zh}获胜！",
            "surviving": sorted(self.state.alive_players),
            "round": self.state.round,
            "roles": roles_revealed,
        })

        self.state.push_event({
            "type": "roles_revealed",
            "roles": roles_revealed,
            "winner": self.state.winner,
            "round": self.state.round,
        })

        record = {
            "phase": "GAME_OVER",
            "round": self.state.round,
            "winner": self.state.winner,
            "visibility": "public",
        }
        self.state.phase_records.append(record)

    # ── Helpers ──────────────────────────────────────────────────

    def _role_summary(self) -> str:
        roles = {pid: ps.role for pid, ps in self.state.players.items()}
        return str(roles)

    def _build_night_summary(self, deaths: list) -> list[dict]:
        """Build a summary of night actions for display, with thoughts.

        Returns a list of dicts: {"text": "描述", "thought": "推理", "role": "seer/witch/hunter"}
        The 'thought' field is private (shown only to viewers, never to LLM agents).
        """
        summary = []
        # Read from current_actions or fall back to last NIGHT phase record
        actions = self.state.current_actions
        if not actions and self.state.phase_records:
            # Fall back to most recent NIGHT record
            for r in reversed(self.state.phase_records):
                if r.get("phase") == "NIGHT":
                    actions = r.get("actions", [])
                    break

        # Werewolf kill (no role — role is private)
        for a in actions:
            if a.get("action") == "kill" and a.get("role") == "werewolf":
                target = a.get("target")
                if target:
                    summary.append({
                        "text": f"狼人决定击杀 {target} 号玩家",
                        "role": "werewolf",
                    })
                break

        # Seer check
        for a in actions:
            if a.get("role") == "seer" and a.get("action") == "check":
                summary.append({
                    "text": f"预言家查验了 {a.get('target')} 号玩家（结果为：{a.get('result', '?')}）",
                    "thought": a.get("thought", ""),
                    "role": "seer",
                })
                break

        # Witch actions
        for a in actions:
            if a.get("role") == "witch":
                if a.get("action") == "use_antidote":
                    summary.append({
                        "text": f"女巫使用了解药，救活了 {a.get('target')} 号玩家",
                        "thought": a.get("thought", ""),
                        "role": "witch",
                    })
                elif a.get("action") == "use_poison":
                    summary.append({
                        "text": f"女巫使用了毒药，毒杀了 {a.get('target')} 号玩家",
                        "thought": a.get("thought", ""),
                        "role": "witch",
                    })
                elif a.get("action") == "pass_antidote":
                    summary.append({
                        "text": "女巫没有使用解药",
                        "thought": a.get("thought", ""),
                        "role": "witch",
                    })
                elif a.get("action") == "pass_poison":
                    summary.append({
                        "text": "女巫没有使用毒药",
                        "thought": a.get("thought", ""),
                        "role": "witch",
                    })

        # Death results (no role — role is private)
        for d in deaths:
            cause = d.get("cause", "")
            pid = d["player_id"]
            if cause == "werewolf_kill":
                summary.append({"text": f"{pid} 号玩家被狼人杀害"})
            elif cause == "witch_poison":
                summary.append({"text": f"{pid} 号玩家被女巫毒杀"})

        return summary

    async def _delay(self):
        """Phase delay that respects pause/stop."""
        try:
            await asyncio.wait_for(
                self._pause_event.wait(), timeout=self.phase_delay)
        except asyncio.TimeoutError:
            pass  # normal — delay elapsed

    async def _wait_if_paused(self):
        """Block until unpaused."""
        await self._pause_event.wait()

    async def _check_pause_and_stop(self):
        """Called before each phase. Blocks if paused, raises if stopped."""
        await self._wait_if_paused()
        if self._stopped:
            raise GameStoppedError()

    def pause(self):
        """Pause the game at the next phase boundary."""
        self._pause_event.clear()
        self.state.push_event({
            "type": "game_paused",
            "message": "游戏已暂停",
        })

    def resume(self):
        """Resume a paused game."""
        self._pause_event.set()
        self.state.push_event({
            "type": "game_resumed",
            "message": "游戏已恢复",
        })

    def stop(self):
        """Stop the game immediately."""
        self._stopped = True
        self._pause_event.set()  # unblock any waiting
        self.state.push_event({
            "type": "game_stopped",
            "message": "游戏已停止",
        })

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    @property
    def is_stopped(self) -> bool:
        return self._stopped


class GameStoppedError(Exception):
    """Raised when a game is stopped mid-execution."""
    pass
