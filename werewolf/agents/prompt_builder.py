"""Build system and user prompts for each agent based on game phase."""

from werewolf.engine.roles import Role, ROLE_SKILLS_ZH, ROLE_NAMES_ZH, TEAM_NAMES_ZH, ROLE_TEAM


def build_system_prompt(player_id: int, role: str, personality: dict,
                       teammates: list[int] | None = None) -> str:
    """Build the base system prompt with personality + role + rules.

    Args:
        teammates: For werewolves only — list of other werewolf player IDs.
                   Injected into the system prompt so the wolf never forgets its pack.
    """

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
    # Werewolf teammates — always visible (prevents hallucination in day phases)
    if role == "werewolf" and teammates:
        teammate_str = "、".join(str(t) for t in sorted(teammates))
        parts.append(f"你的狼人同伴是：{teammate_str}号玩家。请记住你的队友，在白天发言时要保护他们、避免暴露他们。")
        parts.append("")
        parts.append("[狼人战术参考]")
        parts.append("你可以选择以下战术角色，与队友在夜间讨论中协调分工：")
        parts.append("- 悍跳狼：冒充预言家，编造查验结果（发金水/查杀）来主导发言，与真预言家竞争。")
        parts.append("- 冲锋狼：坚定支持悍跳狼队友，积极发言带节奏，争取好人支持。")
        parts.append("- 倒钩狼：站边真好人阵营，踩自己的狼队友，在好人中建立高身份。")
        parts.append("- 深水狼：全程低调发言，扮演毫无存在感的平民，活到最后掌控关键一刀。")
        parts.append("核心原则：发言时伪装成好人，避免暴露信息优势；白天可与队友串联绑票。")
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

    # Speech / thought separation rule
    parts.append("")
    parts.append("[发言规则（极其重要，违反将导致游戏无效）]")
    parts.append("你的每次输出包含 thought（内心推理）和 speech（公开发言）两个字段。")
    parts.append("thought 是你私下的思考和策略分析，只有你自己能看到。")
    parts.append("speech 是你在游戏桌上公开说出的话，会被所有其他玩家听到并用于判断你的身份。")
    parts.append("绝对禁止在 speech 中包含以下内容：")
    parts.append("- 括号内的心理活动、策略备注或自言自语（如\"我是民（其实我是猎人）\"）")
    parts.append("- 对自己真实身份的暗示或明示（除非你主动选择亮明身份）")
    parts.append("- 对 thought 内容的复述或引用")
    parts.append("speech 只包含一个玩家在桌游中可以用嘴巴公开说出来的话。")
    parts.append("内心分析和策略请全部放在 thought 字段中。")

    return "\n".join(parts)


def build_user_message(context: dict) -> str:
    """Build the user message for a specific decision point."""
    phase = context.get("phase", "")
    alive_players = context.get("alive_players", [])
    alive_str = "、".join(str(p) for p in sorted(alive_players))

    parts = []
    parts.append(f"当前存活玩家：{alive_str}。")

    # Role identity anchor — factual only, no behavioural guidance
    player_role = context.get("player_role", "")
    reminder = _role_reminder(player_role, phase, context)
    if reminder:
        parts.append(reminder)

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


def _witch_potion_status(context: dict) -> str:
    """Build a human-readable potion inventory string for the witch."""
    parts = ["你的药水状态："]
    if context.get("witch_antidote_used"):
        parts.append("解药已使用；")
    else:
        parts.append("还有解药；")
    if context.get("witch_poison_used"):
        parts.append("毒药已使用。")
    else:
        parts.append("还有毒药。")
    return "".join(parts) + "\n"


_DAY_PHASES = frozenset({
    "DAY_DISCUSSION", "DAY_FREE_DISCUSSION", "VOTING",
    "PK_DISCUSSION", "PK_VOTING", "LAST_WORDS",
})


def _role_reminder(role: str, phase: str, context: dict) -> str:
    """Build a factual role identity reminder — no behavioural guidance.

    States only what the player IS (role, team, abilities, known info),
    never HOW to play.  This anchors the LLM to its true identity across
    multi-turn history, preventing self-persuasion into a fake role.
    """
    is_day = phase in _DAY_PHASES

    if role == "werewolf":
        return "[身份提示] 你是狼人，属于狼人阵营。"

    if role == "villager":
        if is_day:
            return "[身份提示] 你是平民，属于好人阵营，没有特殊技能。"
        else:
            return "[身份提示] 你是平民，属于好人阵营，没有特殊技能。今晚你没有夜间行动。"

    if role == "seer":
        if is_day:
            return ("[身份提示] 你是预言家，属于好人阵营。"
                    "你查验过的玩家结果记录在上方【你已知的信息】中。")
        else:
            return "[身份提示] 你是预言家，属于好人阵营。"

    if role == "witch":
        potion = _witch_potion_status(context).rstrip("\n")
        return f"[身份提示] 你是女巫，属于好人阵营。{potion}"

    if role == "hunter":
        if is_day:
            return ("[身份提示] 你是猎人，属于好人阵营。"
                    "被放逐或夜间被杀时可以开枪带走一名玩家"
                    "（被女巫毒杀则无法开枪）。")
        else:
            return ("[身份提示] 你是猎人，属于好人阵营。"
                    "今晚你没有夜间行动。")

    return ""


def _phase_instructions(phase: str, context: dict) -> str:
    """Get instructions for the current phase."""
    alive = context.get("alive_players", [])
    valid_targets = context.get("valid_targets", alive)
    valid_str = "、".join(str(p) for p in sorted(valid_targets))

    instructions = {
        "NIGHT_WEREWOLF_CHAT": (
            f"现在是第{context.get('round', 1)}夜狼人讨论阶段。"
            + ("注意：这是第一夜，还没有进行过任何白天讨论，也没有玩家公开发言过。"
               "请根据狼队友身份和游戏策略讨论，不要引用尚未发生的发言或事件。\n"
               if context.get('round', 1) == 1 else
               "请根据之前的游戏进展和狼队友讨论。\n")
            + f"你的狼人同伴是：{context.get('werewolf_teammates', [])}号玩家。\n"
            "请与同伴讨论以下内容：\n"
            "1. 今晚击杀目标（可协商空刀不杀人以制造平安夜混淆好人）\n"
            "2. 白天发言分工（谁悍跳预言家、谁冲锋带节奏、谁倒钩站边好人、谁深水低调）\n"
            "3. 投票协调（绑票目标，集中投票放逐一名好人）\n"
            "speech 是你说给队友的话，禁止在 speech 中用括号插入自言自语或备注。\n"
            "请与同伴讨论今晚要击杀的目标。\n"
            "请以 JSON 格式输出（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你的分析和建议", "speech": "你的发言（纯口头语句，无括号心理活动）", "action": {"type": "discuss"}}'
        ),

        "NIGHT_WEREWOLF_KILL": (
            f"请投票选择今晚要击杀的目标。可选目标：{valid_str}\n"
            "注意：你也可选择不杀人（空刀），但需全队一致同意才会生效。\n"
            "输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你的推理过程", "action": {"type": "kill_vote", "target": <目标玩家编号>}}'
        ),

        "NIGHT_SEER": (
            f"请选择你要查验的一名玩家。可选目标：{valid_str}\n"
            "输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你想查验谁，为什么", "action": {"type": "check", "target": <目标玩家编号>}}'
        ),

        "NIGHT_WITCH_ANTIDOTE": (
            f"现在是第{context.get('round', 1)}夜。"
            + _witch_potion_status(context)
            + f"昨晚{context.get('killed_player')}号玩家被狼人杀害。\n"
            f"你要使用解药救活TA吗？"
            + ("（注意：你本人被杀害，首夜可以自救。）" if context.get("can_self_save") else "")
            + "\n输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你的考虑", "action": {"type": "use_antidote"} 或 {"type": "pass"}}'
        ),

        "NIGHT_WITCH_POISON": (
            f"现在是第{context.get('round', 1)}夜。"
            + _witch_potion_status(context)
            + (f"你已知道昨晚被杀害的玩家是{context.get('killed_player')}号。\n"
               if context.get('killed_player') is not None else "")
            + f"你要使用毒药毒杀一名玩家吗？可选目标：{valid_str}\n"
            "输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你的考虑", "action": {"type": "use_poison", "target": <目标>} 或 {"type": "pass"}}'
        ),

        "DAY_DISCUSSION": (
            f"现在是第{context.get('round', 1)}天依次发言阶段。轮到你了。\n"
            "请根据当前局势发表你的分析和怀疑。\n"
            "speech 必须是你口头说出的完整语句，绝对禁止在 speech 中用括号插入心理活动、策略备注或自言自语。\n"
            "内心分析和策略放在 thought 字段中。输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你的内心推理", "speech": "你的公开发言（纯口头语句，无括号心理活动）"}'
        ),

        "DAY_FREE_DISCUSSION": (
            f"现在是自由讨论第{context.get('free_round', 1)}轮。轮到你了。\n"
            "可以回应其他人的发言，提出新的分析，或质疑他人。\n"
            "speech 必须是你口头说出的完整语句，绝对禁止在 speech 中用括号插入心理活动、策略备注或自言自语。\n"
            "内心分析和策略放在 thought 字段中。输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你的内心推理", "speech": "你的公开发言（纯口头语句，无括号心理活动）"}'
        ),

        "VOTING": (
            f"请投票放逐一名玩家。可选目标：{valid_str}\n"
            "注意：平票时将进入PK发言环节，平票玩家各发言一轮后重新投票。\n"
            "输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你投票给谁，为什么", "action": {"type": "vote", "target": <目标玩家编号>}}'
        ),

        "PK_DISCUSSION": (
            f"你（{context.get('speaker_id')}号）在PK台上！请为自己辩护。\n"
            f"PK台玩家：{'、'.join(str(p) for p in context.get('pk_candidates', []))}号。\n"
            "请说明为什么你不应该被放逐，或指出其他PK台玩家的可疑之处。\n"
            "speech 必须是你口头说出的完整语句，绝对禁止在 speech 中用括号插入心理活动或自言自语。\n"
            "输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你的内心推理", "speech": "你的PK发言（纯口头语句，无括号心理活动）"}'
        ),

        "PK_VOTING": (
            f"PK投票阶段。请从PK台玩家中选择一人放逐。可选目标：{valid_str}\n"
            "注意：PK台上的玩家没有投票权，仅其他玩家投票。\n"
            "输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你投票给谁，为什么", "action": {"type": "vote", "target": <目标玩家编号>}}'
        ),

        "LAST_WORDS": (
            f"你（{context.get('eliminated_player')}号）被放逐了。\n"
            "请发表你的遗言。speech 是你口头说出的遗言，禁止在 speech 中用括号插入心理活动。\n"
            "输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你的想法", "speech": "你的遗言（纯口头语句，无括号心理活动）"}'
        ),

        "HUNTER_SHOOT": (
            ("你被狼人夜间杀害。" if context.get("cause") == "night_kill"
             else "你被投票放逐。")
            + f"你是猎人！你必须开枪带走一名玩家（除非被女巫毒杀则无法开枪，但你未被毒杀）。"
            + f"可选目标：{valid_str}\n"
            "你可以亮明身份（亮牌）来增加说服力，也可以直接开枪不亮牌。\n"
            "输出格式（只输出 JSON，不要有其他文字）：\n"
            '{"thought": "你要带谁走，为什么", "action": {"type": "shoot", "target": <目标玩家编号>}}'
        ),
    }

    return instructions.get(phase, f"当前阶段：{phase}\n请做出你的决策。")


def _format_event(event: dict) -> str:
    """Format a history event as a readable string."""
    etype = event.get("type", "")
    if etype == "day_announce":
        return f"天亮公告：{event.get('message', '')}"
    elif etype == "death":
        return f"{event['player_id']}号玩家死亡"
    elif etype == "elimination":
        return f"{event['player_id']}号玩家被放逐"
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
