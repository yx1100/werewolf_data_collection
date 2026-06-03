"""Build system and user prompts for each agent based on game phase."""

from werewolf.engine.roles import Role, ROLE_SKILLS_ZH, ROLE_NAMES_ZH, TEAM_NAMES_ZH, ROLE_TEAM


def build_system_prompt(player_id: int, role: str, personality: dict) -> str:
    """Build the base system prompt with personality + role + rules."""

    role_enum = Role(role)
    team = ROLE_TEAM[role_enum]
    team_name = TEAM_NAMES_ZH[team]
    role_name = ROLE_NAMES_ZH[role_enum]
    skill_desc = ROLE_SKILLS_ZH[role_enum]

    parts = []

    # Personality (skip if no personality assigned)
    name = personality.get("name", "")
    if name:
        traits = personality.get("traits", "")
        speaking_style = personality.get("speaking_style", "")
        parts.append("[性格设定]")
        parts.append(f"你叫{name}。{traits}")
        parts.append(f"你的说话风格：{speaking_style}")
        parts.append("")

    # Game rules
    parts.append("[游戏规则]")
    parts.append(f"你是{role_name}，属于{team_name}。")
    parts.append(f"你的技能：{skill_desc}")
    parts.append("")
    parts.append("游戏规则概要：")
    parts.append("- 本局共9名玩家：3狼人、1预言家、1女巫、1猎人、3平民。")
    parts.append("- 本局仅包含以上5种角色，不存在其他角色（没有守卫、丘比特、白痴、长老等扩展角色）。请严格基于本局实际角色设定进行推理，不要引入其他角色。")
    parts.append("- 游戏分为夜晚和白天两个阶段交替进行。")
    parts.append("- 夜晚：狼人交流并选择击杀目标；预言家查验一名玩家；女巫决定是否使用解药/毒药。")
    parts.append("- 白天：公布死讯，轮流发言，自由讨论，投票放逐一名玩家。")
    parts.append("- 狼人阵营获胜条件：存活狼人数 >= 存活好人数。")
    parts.append("- 好人阵营获胜条件：所有狼人被消灭。")
    parts.append("")

    # Current context placeholder
    parts.append("[当前局势]")
    parts.append(f"你坐在{player_id}号位。")

    return "\n".join(parts)


def build_user_message(context: dict) -> str:
    """Build the user message for a specific decision point."""
    phase = context.get("phase", "")
    alive_players = context.get("alive_players", [])
    alive_str = "、".join(str(p) for p in sorted(alive_players))

    parts = []
    parts.append(f"当前存活玩家：{alive_str}。")

    # Grounding for first night: no public events have occurred yet
    discussion = context.get("discussion_so_far", [])
    chat_history = context.get("chat_history", [])
    public_history = context.get("public_history", [])
    if not discussion and not chat_history and not public_history and phase.startswith("NIGHT"):
        parts.append("注意：游戏刚开始，还没有发生过白天讨论或其他公开事件。请只根据你的角色信息和游戏策略发言，不要编造尚未发生的事件。")

    # Add discussion context if available
    discussion = context.get("discussion_so_far", [])
    if discussion:
        parts.append("\n已发言内容：")
        for s in discussion:
            pid = s.get("player_id", "?")
            sp = s.get("speech", "")
            parts.append(f"  {pid}号玩家：{sp}")

    # Add chat history (for werewolf chat)
    chat_history = context.get("chat_history", [])
    if chat_history:
        parts.append("\n狼人讨论记录：")
        for m in chat_history:
            pid = m.get("player_id", "?")
            msg = m.get("message", "")
            parts.append(f"  {pid}号狼人：{msg}")

    # Add public history summary
    public_history = context.get("public_history", [])
    if public_history:
        parts.append("\n本局已发生的关键事件：")
        for event in public_history[-20:]:  # limit to last 20 events
            parts.append(f"  - {_format_event(event)}")

    # Phase-specific instructions
    parts.append("")
    parts.append(_phase_instructions(phase, context))

    return "\n".join(parts)


def _phase_instructions(phase: str, context: dict) -> str:
    """Get instructions for the current phase."""
    alive = context.get("alive_players", [])
    valid_targets = context.get("valid_targets", alive)
    valid_str = "、".join(str(p) for p in sorted(valid_targets))

    instructions = {
        "NIGHT_WEREWOLF_CHAT": (
            f"现在是第{context.get('round', 1)}夜狼人讨论阶段。"
            + ("注意：这是第一夜，还没有进行过任何白天讨论，也没有玩家公开发言过。"
               "请根据狼队友身份和游戏策略讨论击杀目标，不要引用尚未发生的发言或事件。\n"
               if context.get('round', 1) == 1 else
               "请根据之前的游戏进展和狼队友讨论击杀目标。\n")
            + f"你的狼人同伴是：{context.get('werewolf_teammates', [])}号玩家。\n"
            "请与同伴讨论今晚要击杀的目标。请以 JSON 格式输出（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你的分析和建议", "speech": "你的发言", "action": {"type": "discuss"}}'
        ),

        "NIGHT_WEREWOLF_KILL": (
            f"请投票选择今晚要击杀的目标。可选目标：{valid_str}\n"
            "输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你的推理过程", "action": {"type": "kill_vote", "target": <目标玩家编号>}}'
        ),

        "NIGHT_SEER": (
            f"请选择你要查验的一名玩家。可选目标：{valid_str}\n"
            "输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你想查验谁，为什么", "action": {"type": "check", "target": <目标玩家编号>}}'
        ),

        "NIGHT_WITCH_ANTIDOTE": (
            f"昨晚{context.get('killed_player')}号玩家被狼人杀害。\n"
            f"你要使用解药救活TA吗？"
            + ("（注意：你本人被杀害，首夜可以自救。）" if context.get("can_self_save") else "")
            + "\n输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你的考虑", "action": {"type": "use_antidote"} 或 {"type": "pass"}}'
        ),

        "NIGHT_WITCH_POISON": (
            f"你要使用毒药毒杀一名玩家吗？可选目标：{valid_str}\n"
            "输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你的考虑", "action": {"type": "use_poison", "target": <目标>} 或 {"type": "pass"}}'
        ),

        "DAY_DISCUSSION": (
            f"现在是第{context.get('round', 1)}天依次发言阶段。轮到你了。\n"
            "请根据当前局势发表你的分析和怀疑。输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你的内心推理", "speech": "你的公开发言"}'
        ),

        "DAY_FREE_DISCUSSION": (
            f"现在是自由讨论第{context.get('free_round', 1)}轮。轮到你了。\n"
            "可以回应其他人的发言，提出新的分析，或质疑他人。输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你的内心推理", "speech": "你的公开发言"}'
        ),

        "VOTING": (
            f"请投票放逐一名玩家。可选目标：{valid_str}\n"
            "输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你投票给谁，为什么", "action": {"type": "vote", "target": <目标玩家编号>}}'
        ),

        "LAST_WORDS": (
            f"你（{context.get('eliminated_player')}号）被放逐了。\n"
            "请发表你的遗言。输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你的想法", "speech": "你的遗言"}'
        ),

        "HUNTER_SHOOT": (
            f"你是猎人！你可以开枪带走一名玩家。可选目标：{valid_str}\n"
            "输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你要带谁走，为什么", "action": {"type": "shoot", "target": <目标玩家编号>}}'
        ),
    }

    return instructions.get(phase, f"当前阶段：{phase}\n请做出你的决策。")


def _format_event(event: dict) -> str:
    """Format a history event as a readable string."""
    etype = event.get("type", "")
    if etype == "death":
        return f"{event['player_id']}号玩家死亡（{event.get('cause', '未知')}）"
    elif etype == "elimination":
        return f"{event['player_id']}号玩家被放逐（身份：{event.get('role', '未知')}）"
    elif etype == "hunter_shoot":
        return f"猎人开枪带走了{event.get('target')}号玩家"
    elif etype == "speech":
        sp = event.get("speech", "")
        return f"{event.get('player_id')}号发言：{sp[:100]}{'...' if len(sp) > 100 else ''}"
    elif etype == "vote_result":
        eliminated = event.get("eliminated")
        if eliminated:
            return f"投票结果：{eliminated}号被放逐"
        return "投票结果：平票，无人被放逐"
    elif etype == "last_words":
        sp = event.get("speech", "")
        return f"{event.get('player_id')}号遗言：{sp[:100]}{'...' if len(sp) > 100 else ''}"
    elif "voter_id" in event:
        return f"{event['voter_id']}号投票给了{event.get('target')}号"
    return str(event)
