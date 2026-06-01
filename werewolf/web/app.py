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
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from werewolf.engine.game import Game, GameStoppedError
from werewolf.engine.roles import Role
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

# In-memory game registry
active_games: dict[str, Game] = {}
game_event_queues: dict[str, asyncio.Queue] = {}
completed_games: list[dict] = []


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
    return templates.TemplateResponse("index.html", {
        "request": request,
        "active_games": list(active_games.keys()),
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
    active_games[game_id] = game
    game_event_queues[game_id] = asyncio.Queue()

    # Store player assignments for UI
    game._player_assignments = player_assignments

    # Run game in background
    asyncio.create_task(_run_game(game_id))

    # Redirect to game page
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/game/{game_id}", status_code=303)


@app.get("/game/{game_id}", response_class=HTMLResponse)
async def game_page(request: Request, game_id: str):
    """Live game view."""
    game = active_games.get(game_id)
    if not game:
        return HTMLResponse("Game not found", status_code=404)

    player_assignments = getattr(game, "_player_assignments", [])

    return templates.TemplateResponse("game.html", {
        "request": request,
        "game_id": game_id,
        "players": player_assignments,
    })


@app.get("/game/{game_id}/stream")
async def game_stream(game_id: str) -> StreamingResponse:
    """SSE endpoint for live game event streaming."""
    if game_id not in game_event_queues:
        return StreamingResponse(
            _empty_stream(),
            media_type="text/event-stream")

    async def event_generator() -> AsyncGenerator[str, None]:
        queue = game_event_queues[game_id]

        # Send current state snapshot for reconnecting clients
        game = active_games.get(game_id)
        if game and game.state.phase.value != "setup":
            dead = [pid for pid, ps in game.state.players.items() if not ps.is_alive]
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
            }, ensure_ascii=False)}\n\n"
            # Drain pending events already covered by recent_events replay
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        while True:
            try:
                # Long timeout: LLM calls with deep thinking can be slow
                event = await asyncio.wait_for(queue.get(), timeout=120)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") in ("game_ended", "game_stopped"):
                    # Give client time to close first, then exit gracefully
                    await asyncio.sleep(10)
                    break
                if event.get("type") == "error":
                    await asyncio.sleep(5)
                    break
            except asyncio.TimeoutError:
                # Keep connection alive
                yield ": keepalive\n\n"
                if game_id not in active_games:
                    yield f"data: {json.dumps({'type': 'game_ended'})}\n\n"
                    break

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
    return templates.TemplateResponse("history.html", {
        "request": request,
        "games": completed,
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
    """Run a game in the background, pushing events to the queue."""
    game = active_games[game_id]
    queue = game_event_queues[game_id]

    try:
        # Patch push_event to also send to queue
        original_push = game.state.push_event

        def push_to_queue(event):
            original_push(event)
            try:
                queue.put_nowait(dict(event))
            except asyncio.QueueFull:
                logger.warning("SSE queue full, dropping event: %s", event.get("type"))

        game.state.push_event = push_to_queue

        # Emit initial state
        await queue.put({
            "type": "game_started",
            "game_id": game_id,
            "players": getattr(game, "_player_assignments", []),
        })

        await game.run()

        if game.is_stopped:
            await queue.put({"type": "game_stopped", "game_id": game_id})
        else:
            await queue.put({"type": "game_ended", "game_id": game_id})

    except GameStoppedError:
        logger.info("Game %s stopped by user", game_id)
        await queue.put({"type": "game_stopped", "game_id": game_id})
    except Exception as e:
        logger.exception("Game %s failed: %s", game_id, e)
        await queue.put({"type": "error", "message": str(e)})
    finally:
        # Move to completed
        if game_id in active_games:
            game = active_games.pop(game_id)
            completed_games.append({
                "game_id": game_id,
                "completed_at": datetime.now().isoformat(),
                "winner": game.state.winner,
                "rounds": game.state.round,
                "data_path": str(_get_data_dir() / datetime.now().strftime("%Y-%m-%d") / game_id / "game_data.json"),
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
                    games.append({
                        "game_id": game_data.get("game_id", game_dir.name),
                        "timestamp": game_data.get("timestamp", ""),
                        "display_time": display_time,
                        "winner": game_data.get("result", {}).get("winner", "?"),
                        "api_provider": game_data.get("config", {}).get("api_provider", "?"),
                        "rounds": len(game_data.get("phases", [])) // 6,
                        "data_path": str(data_file),
                    })
                except Exception:
                    pass
    games.sort(key=lambda g: g["timestamp"], reverse=True)
    return games
