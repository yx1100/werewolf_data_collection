"""Role definitions and team assignments."""

from enum import Enum


class Team(Enum):
    GOOD = "good"      # Villager team (好人阵营)
    WEREWOLF = "werewolf"  # Werewolf team (狼人阵营)


class Role(Enum):
    WEREWOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    HUNTER = "hunter"
    VILLAGER = "villager"


ROLE_TEAM = {
    Role.WEREWOLF: Team.WEREWOLF,
    Role.SEER: Team.GOOD,
    Role.WITCH: Team.GOOD,
    Role.HUNTER: Team.GOOD,
    Role.VILLAGER: Team.GOOD,
}

ROLE_NAMES_ZH = {
    Role.WEREWOLF: "狼人",
    Role.SEER: "预言家",
    Role.WITCH: "女巫",
    Role.HUNTER: "猎人",
    Role.VILLAGER: "平民",
}

TEAM_NAMES_ZH = {
    Team.GOOD: "好人阵营",
    Team.WEREWOLF: "狼人阵营",
}

# Role skill descriptions for prompts
ROLE_SKILLS_ZH = {
    Role.WEREWOLF: "你每晚可以与其他狼人交流并共同决定击杀一名玩家。白天你需要伪装成好人，避免被投票放逐。",
    Role.SEER: "你每晚可以查验一名玩家的身份是「好人」还是「狼人」。你是好人阵营的信息核心。",
    Role.WITCH: "你拥有一瓶解药和一瓶毒药。解药可以救活当晚被狼人杀害的玩家（仅首夜可以自救）；使用解药后，你将不再被告知当晚被狼人杀害的目标。毒药可以毒杀任意一名玩家。每瓶药全局只能使用一次。",
    Role.HUNTER: "当你被投票放逐或被狼人杀害时，可以亮明身份并开枪带走场上任意一名玩家。但如果被女巫毒杀，则无法开枪。",
    Role.VILLAGER: "你没有特殊技能，但你的投票和发言是好人阵营的重要力量。你需要通过推理找出狼人。",
}
