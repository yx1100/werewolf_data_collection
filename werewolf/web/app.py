"""FastAPI application with SSE support for live game streaming."""

import asyncio
import json
import logging
import os
import random
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import yaml
from fastapi import FastAPI, Request, Form
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from werewolf.engine.game import Game, GameStoppedError
from werewolf.engine.roles import Role
from werewolf.engine.state import GamePhase
from werewolf.llm import create_llm_client
from werewolf.agents.agent import create_agent_factory
from werewolf.collector.collector import GameCollector

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# FastAPI app
app = FastAPI(title="狼人杀 LLM 数据收集平台")

# Templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# —— Serve favicon & Apple touch icon at root (browser auto-requests) ——
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(str(static_dir / "favicon.ico"))


@app.get("/apple-touch-icon.png", include_in_schema=False)
async def apple_touch_icon():
    return FileResponse(str(static_dir / "apple-touch-icon.png"))


@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
async def apple_touch_icon_precomposed():
    return FileResponse(str(static_dir / "apple-touch-icon.png"))


# In-memory game registry
active_games: dict[str, Game] = {}
game_event_queues: dict[str, list[asyncio.Queue]] = {}
completed_games: list[dict] = []

# Concurrency control
_active_games_lock = asyncio.Lock()

# Lobby SSE broadcast — one queue per connected lobby client
_lobby_queues: set[asyncio.Queue] = set()
_lobby_queues_lock = asyncio.Lock()


# ── Helpers (data loading) ────────────────────────────────────────


PHASE_NAMES_ZH = {
    GamePhase.SETUP: "准备中",
    GamePhase.NIGHT: "夜晚",
    GamePhase.DAY_ANNOUNCE: "天亮",
    GamePhase.DAY_DISCUSSION: "讨论",
    GamePhase.DAY_FREE_DISCUSSION: "自由讨论",
    GamePhase.VOTING: "投票",
    GamePhase.DAY_RESULT: "放逐",
    GamePhase.GAME_OVER: "结束",
}


def _get_active_game_info(game_id: str, game: Game) -> dict:
    """Build a lightweight status snapshot for lobby display."""
    start_ts = getattr(game, 'start_time', 0)
    if start_ts:
        start_display = datetime.fromtimestamp(start_ts).strftime("%H:%M:%S")
    else:
        start_display = ""
    return {
        "game_id": game_id,
        "phase": game.state.phase.value,
        "phase_zh": PHASE_NAMES_ZH.get(game.state.phase, "未知"),
        "round": game.state.round,
        "alive_count": len(game.state.alive_players),
        "total_players": len(game.state.players),
        "start_time": int(start_ts * 1000),
        "start_time_display": start_display,
        "is_paused": game.is_paused,
    }


async def _broadcast_lobby_event(event: dict):
    """Push an event to every connected lobby SSE client."""
    async with _lobby_queues_lock:
        dead = set()
        for q in _lobby_queues:
            try:
                q.put_nowait(dict(event))
            except asyncio.QueueFull:
                dead.add(q)
        _lobby_queues.difference_update(dead)


def _load_yaml(path: str) -> dict:
    """Load a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_env(value: str) -> str:
    """Resolve ${VAR} environment variable references in a string."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        var_name = value[2:-1]
        return os.environ.get(var_name, "")
    return value


def _get_api_config(provider: str) -> dict:
    """Load and resolve API config for a given provider."""
    config_path = (PROJECT_ROOT / "werewolf" / "config" /
                   "api_config.yaml")
    raw = _load_yaml(str(config_path))
    provider_config = raw.get(provider, {})
    resolved = {}
    for k, v in provider_config.items():
        resolved[k] = _resolve_env(v)
    return resolved


def _get_game_config() -> dict:
    """Load game config."""
    config_path = (PROJECT_ROOT / "werewolf" / "config" /
                   "game_config.yaml")
    raw = _load_yaml(str(config_path))
    return raw.get("game", {})


def _load_personalities() -> list[dict]:
    """Load personality traits."""
    path = PROJECT_ROOT / "werewolf" / "personalities" / "traits.yaml"
    raw = _load_yaml(str(path))
    return raw.get("personalities", [])


def _assign_roles(num_players: int) -> list[str]:
    """Assign roles for a 9-player game."""
    roles = (["werewolf"] * 3
             + ["seer", "witch", "hunter"]
             + ["villager"] * 3)
    random.shuffle(roles)
    return roles


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if not seconds:
        return ""
    total = int(seconds)
    mins = total // 60
    secs = total % 60
    return f"{mins}分{secs}秒"


def _get_data_dir() -> Path:
    return PROJECT_ROOT / "data"


# ── Routes ──────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page — game lobby."""
    completed = _list_completed_games()
    personalities = _load_personalities()
    # Role info for prompt display
    from werewolf.engine.roles import ROLE_NAMES_ZH, ROLE_SKILLS_ZH, TEAM_NAMES_ZH, ROLE_TEAM, Role
    roles_info = []
    for role in Role:
        team = ROLE_TEAM[role]
        roles_info.append({
            "key": role.value,
            "name": ROLE_NAMES_ZH[role],
            "team": TEAM_NAMES_ZH[team],
            "skill": ROLE_SKILLS_ZH[role],
            "count": 3 if role == Role.WEREWOLF or role == Role.VILLAGER else 1,
        })
    return templates.TemplateResponse(request, "index.html", {
        "active_games": [
            _get_active_game_info(gid, game)
            for gid, game in list(active_games.items())
        ],
        "completed_games": completed,
        "personalities": personalities,
        "roles_info": roles_info,
    })


@app.post("/game/start")
async def start_game(request: Request):
    """Start a new game with the specified API provider."""
    form_data = await request.form()
    game_id = f"game_{uuid.uuid4().hex[:8]}"

    api_provider = form_data.get("api_provider", "qwen")
    deep_thinking = form_data.get("deep_thinking", "off")
    model_name = form_data.get("model_name", "")

    # Init config
    api_config = _get_api_config(api_provider)
    game_config = _get_game_config()
    all_personalities = _load_personalities()

    # Override model name if provided (with server-side validation)
    if model_name.strip():
        from werewolf.llm import VALID_MODELS
        valid = VALID_MODELS.get(api_provider, [])
        if model_name.strip() not in valid:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=400,
                content={"error": f"模型名称 '{model_name.strip()}' 在"
                                 f" {api_provider} 下无效。"
                                 f"有效模型: {', '.join(valid)}"}
            )
        api_config["model"] = model_name.strip()

    # Deep thinking settings
    use_deep_thinking = deep_thinking == "on"
    thinking_budget = game_config.get("thinking_budget", 0)
    preserve_thinking = game_config.get("preserve_thinking", False)

    # Create LLM client
    llm_client = create_llm_client(
        api_provider, api_config,
        deep_thinking=use_deep_thinking,
        thinking_budget=thinking_budget,
        preserve_thinking=preserve_thinking,
    )

    # Assign roles
    roles = _assign_roles(9)

    # Assign personalities: use per-player selections or random
    personality_map = {p["name"]: p for p in all_personalities}
    player_assignments = []
    for i, role in enumerate(roles, 1):
        personality_key = form_data.get(f"personality_{i}", "none")
        if personality_key == "none":
            # No personality — use an empty placeholder
            personality = {"name": "", "traits": "", "speaking_style": ""}
        elif personality_key == "random" or personality_key not in personality_map:
            personality = random.choice(all_personalities)
            used_names = {pa["personality"]["name"] for pa in player_assignments}
            fallback = 0
            while personality["name"] in used_names and fallback < 50:
                personality = random.choice(all_personalities)
                fallback += 1
        else:
            personality = personality_map[personality_key]
        player_assignments.append({
            "player_id": i,
            "role": role,
            "personality": personality,
        })

    # Create agent factory
    agent_factory = create_agent_factory(llm_client)

    # Create collector
    collector = GameCollector(
        data_dir=str(_get_data_dir()),
        api_provider=api_provider,
        model=api_config.get("model", "unknown"),
    )

    # Create game
    game = Game(
        game_id=game_id,
        player_assignments=player_assignments,
        agent_factory=agent_factory,
        config=game_config,
        collector=collector,
    )

    # Register game
    async with _active_games_lock:
        active_games[game_id] = game
    game_event_queues[game_id] = []

    # Store player assignments for UI
    game._player_assignments = player_assignments

    # Run game in background
    asyncio.create_task(_run_game(game_id))

    # Return JSON (client stays on lobby page)
    return {"status": "started", "game_id": game_id}


@app.get("/game/{game_id}", response_class=HTMLResponse)
async def game_page(request: Request, game_id: str):
    """Live game view."""
    game = active_games.get(game_id)
    if not game:
        return HTMLResponse("Game not found", status_code=404)

    player_assignments = getattr(game, "_player_assignments", [])

    return templates.TemplateResponse(request, "game.html", {
        "game_id": game_id,
        "players": player_assignments,
    })


@app.get("/game/{game_id}/stream")
async def game_stream(game_id: str) -> StreamingResponse:
    """SSE endpoint for live game event streaming.

    Each connected client gets its own queue so events are broadcast
    to all viewers (fan-out), not round-robin consumed.
    """
    if game_id not in game_event_queues:
        return StreamingResponse(
            _empty_stream(),
            media_type="text/event-stream")

    async def event_generator() -> AsyncGenerator[str, None]:
        my_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

        # Register this client's queue
        async with _active_games_lock:
            if game_id in game_event_queues:
                game_event_queues[game_id].append(my_queue)
            else:
                # Game ended between check and lock acquisition
                yield f"data: {json.dumps({'type': 'game_ended', 'game_id': game_id})}\n\n"
                return

        try:
            # Send current state snapshot for reconnecting clients
            game = active_games.get(game_id)
            if game and game.state.phase.value != "setup":
                dead = [pid for pid, ps in game.state.players.items()
                        if not ps.is_alive]
                all_events = getattr(game.state, 'events', [])
                recent = all_events[-50:] if len(all_events) > 50 else all_events
                yield f"data: {json.dumps({
                    'type': 'game_state',
                    'game_id': game_id,
                    'players': getattr(game, '_player_assignments', []),
                    'phase': game.state.phase.value,
                    'round': game.state.round,
                    'dead_players': dead,
                    'recent_events': recent,
                    'game_start_time': int(getattr(game, 'start_time', 0) * 1000),
                }, ensure_ascii=False)}\n\n"

            while True:
                try:
                    # Long timeout: LLM calls with deep thinking can be slow
                    event = await asyncio.wait_for(my_queue.get(), timeout=120)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if event.get("type") in ("game_ended", "game_stopped"):
                        await asyncio.sleep(10)
                        break
                    if event.get("type") == "error":
                        await asyncio.sleep(5)
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    if game_id not in active_games:
                        yield f"data: {json.dumps({'type': 'game_ended'})}\n\n"
                        break
        finally:
            # Unregister this client's queue (but don't pop the
            # game_event_queues entry — _run_game owns that lifecycle)
            async with _active_games_lock:
                queues = game_event_queues.get(game_id)
                if queues:
                    try:
                        queues.remove(my_queue)
                    except ValueError:
                        pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    """Browse completed games."""
    completed = _list_completed_games()
    return templates.TemplateResponse(request, "history.html", {
        "games": completed,
    })


@app.get("/lobby/stream")
async def lobby_stream() -> StreamingResponse:
    """SSE endpoint for real-time lobby updates (active game list)."""

    async def event_generator() -> AsyncGenerator[str, None]:
        my_queue: asyncio.Queue = asyncio.Queue()
        async with _lobby_queues_lock:
            _lobby_queues.add(my_queue)

        try:
            # Initial snapshot — all currently active games
            async with _active_games_lock:
                games = [
                    _get_active_game_info(gid, g)
                    for gid, g in list(active_games.items())
                ]
            yield f"data: {json.dumps({'type': 'lobby_snapshot', 'games': games}, ensure_ascii=False)}\n\n"

            while True:
                try:
                    event = await asyncio.wait_for(my_queue.get(), timeout=30)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            async with _lobby_queues_lock:
                _lobby_queues.discard(my_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })


@app.post("/game/{game_id}/pause")
async def pause_game(game_id: str):
    """Pause the game."""
    game = active_games.get(game_id)
    if not game:
        return {"status": "not_found"}
    game.pause()
    return {"status": "paused"}


@app.post("/game/{game_id}/resume")
async def resume_game(game_id: str):
    """Resume a paused game."""
    game = active_games.get(game_id)
    if not game:
        return {"status": "not_found"}
    game.resume()
    return {"status": "resumed"}


@app.post("/game/{game_id}/stop")
async def stop_game(game_id: str):
    """Stop the game immediately."""
    game = active_games.get(game_id)
    if not game:
        return {"status": "not_found"}
    game.stop()
    return {"status": "stopped"}


@app.get("/game/{game_id}/data")
async def game_data(game_id: str):
    """Return the JSON data for a completed game."""
    # Check if game is active
    game = active_games.get(game_id)
    if game:
        return {"status": "active", "phase": game.state.phase.value}

    # Look for completed game data
    for game_info in _list_completed_games():
        if game_info["game_id"] == game_id:
            data_path = Path(game_info["data_path"])
            if data_path.exists():
                with open(data_path, encoding="utf-8") as f:
                    data = json.load(f)
                return Response(
                    content=json.dumps(data, ensure_ascii=False, indent=2),
                    media_type="application/json; charset=utf-8")

    return {"status": "not_found"}


@app.delete("/game/{game_id}/data")
async def delete_game_data(game_id: str):
    """Delete a completed game's data from disk."""
    for game_info in _list_completed_games():
        if game_info["game_id"] == game_id:
            game_dir = Path(game_info["data_path"]).parent
            if game_dir.exists():
                shutil.rmtree(game_dir)
            # Clean up empty date directory
            date_dir = game_dir.parent
            if date_dir.exists() and not any(date_dir.iterdir()):
                date_dir.rmdir()
            return {"status": "deleted", "game_id": game_id}
    return {"status": "not_found"}


# ── Helpers ─────────────────────────────────────────────────────

async def _run_game(game_id: str):
    """Run a game in the background, pushing events to all SSE queues."""
    game = active_games[game_id]

    def _broadcast_game_event(evt: dict):
        """Push an event to every SSE client watching this game."""
        queues = game_event_queues.get(game_id)
        if queues:
            for q in queues:
                try:
                    q.put_nowait(dict(evt))
                except asyncio.QueueFull:
                    pass

    try:
        # Patch push_event to also broadcast to SSE queues
        original_push = game.state.push_event

        def push_to_queue(event):
            original_push(event)
            try:
                _broadcast_game_event(event)
            except Exception:
                pass
            # Notify lobby on phase / pause / resume changes
            if event.get("type") in ("phase_change", "game_paused", "game_resumed"):
                asyncio.ensure_future(_broadcast_lobby_event({
                    "type": "game_update",
                    **_get_active_game_info(game_id, game),
                }))

        game.state.push_event = push_to_queue

        # Emit initial state
        start_event = {
            "type": "game_started",
            "game_id": game_id,
            "players": getattr(game, "_player_assignments", []),
            "game_start_time": int(getattr(game, 'start_time', 0) * 1000),
        }
        _broadcast_game_event(start_event)

        # Notify lobby of new game
        await _broadcast_lobby_event({
            "type": "game_started",
            **_get_active_game_info(game_id, game),
        })

        await game.run()

        if game.is_stopped:
            _broadcast_game_event({"type": "game_stopped", "game_id": game_id})
        else:
            _broadcast_game_event({"type": "game_ended", "game_id": game_id})

    except GameStoppedError:
        logger.info("Game %s stopped by user", game_id)
        _broadcast_game_event({"type": "game_stopped", "game_id": game_id})
    except Exception as e:
        logger.exception("Game %s failed: %s", game_id, e)
        _broadcast_game_event({"type": "error", "message": str(e)})
    finally:
        # Move to completed
        completed_info = None
        async with _active_games_lock:
            if game_id in active_games:
                game = active_games.pop(game_id)
                now = datetime.now()
                completed_info = {
                    "game_id": game_id,
                    "display_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "winner": game.state.winner,
                    "rounds": game.state.round,
                    "api_provider": getattr(getattr(game, 'collector', None), 'api_provider', '?'),
                    "duration_display": _format_duration(getattr(game, 'duration_seconds', 0)),
                }
                completed_games.append({
                    "game_id": game_id,
                    "completed_at": now.isoformat(),
                    "winner": game.state.winner,
                    "rounds": game.state.round,
                    "data_path": str(_get_data_dir() / now.strftime("%Y-%m-%d") / game_id / "game_data.json"),
                })

        # Clean up event queues
        game_event_queues.pop(game_id, None)

        # Notify lobby that game is gone, with completed info for table update
        await _broadcast_lobby_event({
            "type": "game_ended",
            "game_id": game_id,
            "completed": completed_info,
        })


async def _queue_put(queue: asyncio.Queue, event: dict):
    """Put an event on the queue."""
    await queue.put(event)


async def _empty_stream():
    """Empty SSE stream."""
    yield f"data: {json.dumps({'type': 'error', 'message': 'Game not found'})}\n\n"


def _list_completed_games() -> list[dict]:
    """List completed games by scanning data directory."""
    games = []
    data_dir = _get_data_dir()
    if not data_dir.exists():
        return games

    for date_dir in sorted(data_dir.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for game_dir in sorted(date_dir.iterdir(), reverse=True):
            data_file = game_dir / "game_data.json"
            if data_file.exists():
                try:
                    with open(data_file, encoding="utf-8") as f:
                        game_data = json.load(f)
                    ts = game_data.get("timestamp", "")
                    try:
                        dt = datetime.fromisoformat(ts)
                        display_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        display_time = ts.replace("T", " ")
                    # Get duration info
                    result = game_data.get("result", {})
                    duration_display = result.get("duration_display")
                    games.append({
                        "game_id": game_data.get("game_id", game_dir.name),
                        "timestamp": game_data.get("timestamp", ""),
                        "display_time": display_time,
                        "winner": result.get("winner", "?"),
                        "api_provider": game_data.get("config", {}).get("api_provider", "?"),
                        "rounds": len(game_data.get("phases", [])) // 6,
                        "duration_display": duration_display,
                        "data_path": str(data_file),
                    })
                except Exception:
                    pass
    games.sort(key=lambda g: g["timestamp"], reverse=True)
    return games
