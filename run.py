#!/usr/bin/env python3
"""狼人杀 LLM 数据收集平台 — 入口

Usage:
    python run.py                          # Start web server (default)
    python run.py --provider qwen          # Default provider in web UI
    python run.py --provider deepseek      # Default provider in web UI
    python run.py --thinking               # Enable deep thinking by default
    python run.py --cli                    # Run a single game in CLI mode
    python run.py --cli --personalities "1=戏精影帝,5=理性分析师"  # CLI 指定玩家性格
    python run.py --cli --personalities random                     # 全部随机
"""

import argparse
import asyncio
import json
import logging
import os
import random
import shutil
import sys
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(override=True)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def check_env(provider: str):
    """Check that required API keys are set."""
    key_vars = {
        "qwen": "QWEN_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    var_name = key_vars.get(provider, "")
    if var_name and not os.environ.get(var_name):
        print(f"⚠️  警告: {var_name} 未设置。请先设置环境变量或 .env 文件。")
        print(f"   export {var_name}=your_api_key")
        return False
    return True


def print_banner(provider: str, thinking: bool):
    """Print startup banner."""
    thinking_status = "✅ 开启 (Chain-of-Thought)" if thinking else "❌ 关闭"
    provider_names = {
        "qwen": "Qwen (通义千问)",
        "deepseek": "DeepSeek",
    }

    print("=" * 60)
    print("🐺  狼人杀 LLM 数据收集平台")
    print("=" * 60)
    print()
    print(f"  API Provider:     {provider_names.get(provider, provider)}")
    print(f"  深度思考模式:     {thinking_status}")
    print()
    print("  环境变量:")
    print(f"    QWEN_API_KEY     = {'✅ 已设置' if os.environ.get('QWEN_API_KEY') else '❌ 未设置'}")
    print(f"    DEEPSEEK_API_KEY  = {'✅ 已设置' if os.environ.get('DEEPSEEK_API_KEY') else '❌ 未设置'}")
    print()


def parse_personality_spec(spec: str) -> dict:
    """Parse a '1=戏精影帝,5=理性分析师' spec into {player_id: personality_name}.

    Special keyword 'random' returns an empty dict → all players random.
    Invalid items (missing '=', non-numeric or out-of-range id, empty name)
    are skipped silently — those players fall back to random assignment.
    """
    spec = spec.strip()
    if spec.lower() == "random":
        return {}
    result = {}
    for item in spec.split(","):
        if "=" not in item:
            continue
        pid_str, name = item.split("=", 1)
        pid_str, name = pid_str.strip(), name.strip()
        if pid_str.isdigit() and 1 <= int(pid_str) <= 9 and name:
            result[int(pid_str)] = name
    return result


def assign_personalities(personalities: list, spec: str) -> list:
    """Return personality dicts for players 1..9, in player order.

    Players named in `spec` with a valid personality get it (duplicates allowed).
    Every other player is filled randomly, avoiding already-used names when possible.
    """
    by_name = {p["name"]: p for p in personalities}
    requested = parse_personality_spec(spec)

    explicit = {pid: by_name[name]
                for pid, name in requested.items() if name in by_name}
    used = {p["name"] for p in explicit.values()}

    assigned = []
    for pid in range(1, 10):
        if pid in explicit:
            assigned.append(explicit[pid])
            continue
        p = random.choice(personalities)
        fallback = 0
        while p["name"] in used and fallback < 50:
            p = random.choice(personalities)
            fallback += 1
        used.add(p["name"])
        assigned.append(p)
    return assigned


async def run_cli_game(provider: str, deep_thinking: bool, model_name: str = "",
                       personalities_spec: str = "", temperature: float | None = None,
                       reasoning_effort: str | None = None):
    """Run a single game from the CLI (no web UI)."""
    from werewolf.engine.game import Game
    from werewolf.llm import create_llm_client
    from werewolf.agents.agent import create_agent_factory
    from werewolf.collector.collector import GameCollector

    # Load configs
    import yaml
    from pathlib import Path
    root = Path(__file__).parent

    with open(root / "werewolf" / "config" / "api_config.yaml", encoding="utf-8") as f:
        api_raw = yaml.safe_load(f)
    api_config = {}
    for k, v in api_raw.get(provider, {}).items():
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            api_config[k] = os.environ.get(v[2:-1], "")
        else:
            api_config[k] = v

    with open(root / "werewolf" / "config" / "game_config.yaml", encoding="utf-8") as f:
        game_config = yaml.safe_load(f).get("game", {})

    with open(root / "werewolf" / "personalities" / "traits.yaml", encoding="utf-8") as f:
        personalities = yaml.safe_load(f)["personalities"]

    # Override model name if provided
    if model_name.strip():
        api_config["model"] = model_name.strip()

    # Override temperature if provided
    if temperature is not None:
        api_config["temperature"] = temperature

    # Override reasoning_effort if provided
    if reasoning_effort:
        api_config["reasoning_effort"] = reasoning_effort

    # Create LLM client
    llm_client = create_llm_client(
        provider, api_config,
        deep_thinking=deep_thinking,
        thinking_budget=game_config.get("thinking_budget", 0),
        preserve_thinking=game_config.get("preserve_thinking", False),
    )

    # Assign roles (always random); personalities follow --personalities, rest random
    roles = (["werewolf"] * 3 + ["seer", "witch", "hunter"] + ["villager"] * 3)
    random.shuffle(roles)
    assigned_personalities = assign_personalities(personalities, personalities_spec)

    player_assignments = []
    for i, (role, personality) in enumerate(
            zip(roles, assigned_personalities), 1):
        player_assignments.append({
            "player_id": i,
            "role": role,
            "personality": personality,
        })

    # Create agent factory with werewolf teammate info
    werewolf_ids = [pa["player_id"] for pa in player_assignments if pa["role"] == "werewolf"]
    teammates_map = {}
    for wid in werewolf_ids:
        teammates_map[wid] = [w for w in werewolf_ids if w != wid]
    agent_factory = create_agent_factory(llm_client, teammates_map)

    # Create collector
    collector = GameCollector(
        data_dir=str(root / "data"),
        api_provider=provider,
        model=api_config.get("model", "unknown"),
        hyperparams={
            "temperature": api_config.get("temperature"),
            "reasoning_effort": api_config.get("reasoning_effort"),
            "deep_thinking": deep_thinking,
            "thinking_budget": game_config.get("thinking_budget", 0),
            "preserve_thinking": game_config.get("preserve_thinking", False),
            "phase_delay_seconds": game_config.get("phase_delay_seconds"),
            "free_discussion_rounds": game_config.get("free_discussion_rounds"),
            "witch_self_save_first_night": game_config.get("witch_self_save_first_night"),
        },
    )

    # Create and run game
    game_id = f"game_{uuid.uuid4().hex[:8]}"
    game = Game(
        game_id=game_id,
        player_assignments=player_assignments,
        agent_factory=agent_factory,
        config=game_config,
        collector=collector,
    )

    print(f"\n游戏 ID: {game_id}")
    print("角色分配:")
    for pa in player_assignments:
        role_emoji = {"werewolf": "🐺", "seer": "🔮", "witch": "🧪",
                      "hunter": "🔫", "villager": "👤"}
        pname = pa["personality"]["name"]
        print(f"  玩家 {pa['player_id']}: {role_emoji.get(pa['role'], '?')} {pa['role']} ({pname})")
    print()

    print("游戏开始...\n")
    state = await game.run()

    print(f"\n游戏结束！获胜方: {state.winner}")
    print(f"存活玩家: {state.alive_players}")
    print(f"总轮数: {state.round}")

    # Find the data file
    from datetime import datetime
    data_dir = root / "data" / datetime.now().strftime("%Y-%m-%d") / game_id
    data_file = data_dir / "game_data.json"
    if data_file.exists():
        print(f"数据已保存: {data_file}")


def list_completed_games():
    """Scan data directory and return completed games sorted by time (newest first)."""
    from pathlib import Path
    root = Path(__file__).parent
    data_dir = root / "data"
    if not data_dir.exists():
        return []

    from datetime import datetime as dt
    games = []
    for date_dir in data_dir.iterdir():
        if not date_dir.is_dir():
            continue
        for game_dir in date_dir.iterdir():
            data_file = game_dir / "game_data.json"
            if data_file.exists():
                try:
                    with open(data_file, encoding="utf-8") as f:
                        game_data = json.load(f)
                    ts = game_data.get("timestamp", "")
                    try:
                        display_time = dt.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        display_time = ts.replace("T", " ")
                    games.append({
                        "game_id": game_data.get("game_id", game_dir.name),
                        "timestamp": game_data.get("timestamp", ""),
                        "display_time": display_time,
                        "winner": game_data.get("result", {}).get("winner", "?"),
                        "api_provider": game_data.get("config", {}).get("api_provider", "?"),
                        "rounds": max((p.get("round", 0) for p in game_data.get("phases", [])), default=0),
                        "data_path": str(data_file),
                    })
                except Exception:
                    pass
    games.sort(key=lambda g: g["timestamp"], reverse=True)
    return games


def delete_game_data(game_id: str) -> bool:
    """Delete a completed game's data directory. Returns True if deleted."""
    from pathlib import Path
    for g in list_completed_games():
        if g["game_id"] == game_id:
            game_dir = Path(g["data_path"]).parent
            if game_dir.exists():
                shutil.rmtree(game_dir)
            date_dir = game_dir.parent
            if date_dir.exists() and not any(date_dir.iterdir()):
                date_dir.rmdir()
            return True
    return False


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="狼人杀 LLM 数据收集平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py                              # 启动 Web 界面
  python run.py --provider qwen              # Web 界面默认 Qwen
  python run.py --provider deepseek --thinking  # Web 界面，开启深度思考
  python run.py --cli --provider qwen        # CLI 模式直接运行一局
  python run.py --cli --provider deepseek --thinking  # CLI + 深度思考
  python run.py --cli --personalities "1=戏精影帝,5=理性分析师"  # CLI 指定玩家性格
	python run.py --cli --personalities random                     # 全部随机
        """,
    )
    parser.add_argument(
        "--provider", default="deepseek", choices=["qwen", "deepseek"],
        help="API provider (default: qwen)")
    parser.add_argument(
        "--thinking", action="store_true",
        help="Enable deep thinking / chain-of-thought mode")
    parser.add_argument(
        "--model", default="",
        help="Override model name (default from api_config.yaml)")
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Override temperature (0.0-2.0, default from api_config.yaml)")
    parser.add_argument(
        "--reasoning-effort", default=None,
        choices=["high", "max"],
        help="Override reasoning effort for DeepSeek (high/max, only with --thinking)")
    parser.add_argument(
        "--cli", action="store_true",
        help="Run a single game in CLI mode instead of starting web server")
    parser.add_argument(
        "--personalities", default="", metavar="SPEC",
        help="仅 CLI：指定玩家性格，格式 '1=戏精影帝,5=理性分析师'（玩家号=性格名，"
             "逗号分隔）；传 'random' 全部随机；未指定名字无效的玩家随机分配")
    parser.add_argument(
        "--list", action="store_true",
        help="List all completed games and exit")
    parser.add_argument(
        "--delete", metavar="GAME_ID",
        help="Delete a completed game's data by GAME_ID")
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Web server port (default: 8000)")
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Web server host (default: 127.0.0.1)")

    args = parser.parse_args()

    # Validate --model if provided (CLI only)
    if args.model.strip():
        from werewolf.llm import VALID_MODELS
        valid = VALID_MODELS.get(args.provider, [])
        if args.model.strip() not in valid:
            print(f"错误: --model '{args.model.strip()}' 在 {args.provider} 下无效。")
            print(f"有效模型: {', '.join(valid)}")
            sys.exit(1)

    print_banner(args.provider, args.thinking)
    check_env(args.provider)

    if args.list:
        games = list_completed_games()
        if not games:
            print("暂无已完成游戏")
        else:
            print(f"\n已完成游戏 ({len(games)} 局):\n")
            print(f"{'游戏 ID':<18} {'时间':<22} {'API':<12} {'轮数':<6} {'结果'}")
            print("-" * 75)
            for g in games:
                winner_zh = "好人胜" if g["winner"] == "good" else "狼人胜"
                ts = g.get("display_time", g["timestamp"][:19] if g["timestamp"] else "?")
                print(f"{g['game_id']:<18} {ts:<22} {g['api_provider']:<12} {g['rounds']:<6} {winner_zh}")
        return

    if args.delete:
        game_id = args.delete
        print(f"确认删除游戏 {game_id} 的数据？[y/N] ", end="")
        confirm = input().strip().lower()
        if confirm in ("y", "yes"):
            if delete_game_data(game_id):
                print(f"已删除游戏 {game_id}")
            else:
                print(f"未找到游戏 {game_id}")
        else:
            print("已取消")
        return

    if args.cli:
        asyncio.run(run_cli_game(args.provider, args.thinking, args.model,
                                 args.personalities, args.temperature,
                                 args.reasoning_effort))
    else:
        import uvicorn
        from werewolf.web.app import app
        print(f"  启动 Web 界面: http://{args.host}:{args.port}")
        print()
        print("=" * 60)
        print()
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
