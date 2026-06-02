# 🐺 狼人杀 LLM 数据收集平台

全自动 AI 玩家对战 · 同步收集文本数据 · 实时 Web 观战

一个基于大语言模型的狼人杀完全自动化游戏平台。9 名 AI 玩家各自拥有独立的角色和性格设定，通过 LLM API 进行推理、发言、投票和夜间行动。所有游戏过程以结构化 JSON 格式记录，适合 LLM 多智能体行为研究、社会推理分析和对话数据收集。

## 核心特性

- **全自动对战** — 9 个 AI Agent 自动完成整局狼人杀，无需人工干预
- **实时观战** — 基于 SSE（Server-Sent Events）的 Web 界面，实时展示发言、投票、夜间行动
- **多 LLM 支持** — 同时支持 DeepSeek 和 Qwen（通义千问），可在 Web UI 中切换模型
- **深度思考模式** — 支持 Chain-of-Thought 推理，可观察 AI 的内心思考过程
- **性格系统** — 12 种预设玩家性格（谨慎老玩家、戏精影帝、数据狂人……），可为每位玩家单独设定
- **结构化数据收集** — 每局游戏保存为完整 JSON 文件，包含所有阶段记录、发言、投票、行动和结果
- **游戏控制** — 支持暂停/继续/停止游戏
- **双模式运行** — Web 界面 + CLI 命令行两种启动方式

## 快速开始

### 环境要求

- Python 3.10+
- DeepSeek 或 Qwen API Key

### 安装

```bash
git clone https://github.com/yx1100/werewolf_data_collection.git
cd werwolf_data_collection
pip install -r requirements.txt
```

### 配置 API Key

```bash
# DeepSeek
export DEEPSEEK_API_KEY=your_api_key

# Qwen（通义千问）
export QWEN_API_KEY=your_api_key
```

也可以在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_api_key
QWEN_API_KEY=your_api_key
```

### 运行

**Web 模式（推荐）：**

```bash
python run.py                          # 默认 DeepSeek
python run.py --provider qwen          # 使用 Qwen
python run.py --provider deepseek --thinking  # 开启深度思考
python run.py --port 8080              # 自定义端口
```

启动后访问 `http://127.0.0.1:8000` 进入游戏大厅。

**CLI 模式：**

```bash
python run.py --cli --provider deepseek --thinking
python run.py --cli --model qwen3.7-max                        # 指定模型名称
python run.py --cli --personalities "1=戏精影帝,5=理性分析师"   # 指定玩家 1、5 的性格，其余随机
python run.py --cli --personalities random                      # 全部随机分配性格
```

> `--model` 覆盖 `api_config.yaml` 中配置的默认模型名称，仅 CLI 模式生效。模型名必须与提供商匹配（见下方「模型名称验证」章节）。
>
> `--personalities` 通过 `"玩家号=性格名,..."` 指定任意玩家的性格（玩家号 1-9，逗号分隔）。特殊关键字 `random` 表示为所有 9 名玩家随机分配性格。未指定或名字无效的玩家也会随机分配。性格名见 `werewolf/personalities/traits.yaml`。

### 测试 API 连接

```bash
python test_api_connection.py
```

### 运行测试

```bash
python test_mock_game.py
```

## 游戏规则

采用标准 9 人局配置：

| 阵营 | 角色 | 人数 | 技能 |
|------|------|------|------|
| 🐺 狼人 | 狼人 | 3 | 每晚讨论并击杀一名玩家 |
| 🛡️ 好人 | 预言家 | 1 | 每晚查验一名玩家的身份 |
| 🛡️ 好人 | 女巫 | 1 | 拥有一瓶解药和一瓶毒药（首夜可自救） |
| 🛡️ 好人 | 猎人 | 1 | 被投票或刀杀时可开枪带走一人（被毒杀无法开枪） |
| 🛡️ 好人 | 平民 | 3 | 无特殊技能，依靠发言和投票 |

**狼人阵营获胜条件**：存活狼人数 ≥ 存活好人数
**好人阵营获胜条件**：所有狼人被消灭

## 项目结构

```
werewolf_data_collection/
├── run.py                      # 入口脚本（Web + CLI）
├── requirements.txt            # Python 依赖
├── test_api_connection.py      # API 连接测试
├── test_mock_game.py           # Mock 集成测试
├── data/                       # 游戏数据输出（按日期/游戏ID 组织）
│   └── YYYY-MM-DD/
│       └── game_XXXXXXXX/
│           └── game_data.json
└── werewolf/
    ├── agents/                 # AI 玩家 Agent
    │   ├── agent.py            # Agent 主逻辑（多轮对话、决策接口）
    │   ├── prompt_builder.py   # 系统提示词 + 阶段消息模板
    │   └── decision_parser.py  # LLM JSON 响应解析器
    ├── collector/
    │   └── collector.py        # 游戏数据序列化
    ├── config/
    │   ├── api_config.yaml     # LLM API 配置
    │   └── game_config.yaml    # 游戏参数配置
    ├── engine/                 # 游戏引擎
    │   ├── game.py             # 游戏状态机（核心编排器）
    │   ├── roles.py            # 角色/阵营定义
    │   ├── rules.py            # 胜负判定 + 行动校验
    │   └── state.py            # 游戏状态数据结构
    ├── llm/
    │   └── __init__.py         # LLM 客户端抽象（DeepSeek + Qwen）
    ├── personalities/
    │   └── traits.yaml         # 12 种 AI 性格配置
    └── web/                    # Web 界面
        ├── app.py              # FastAPI 路由 + 后台游戏运行 + SSE
        └── templates/          # Jinja2 模板
            ├── index.html      # 游戏大厅
            ├── game.html       # 实时观战页面
            └── history.html    # 历史记录
```

### 核心流程

```
Web UI 启动游戏 → 随机分配角色+性格 → 创建 9 个 Agent
    → Game 状态机循环：
       NIGHT（狼人讨论→投票杀→预言家查验→女巫用药）
       → DAY_ANNOUNCE（公布死讯）
       → DAY_DISCUSSION（顺序发言）
       → DAY_FREE_DISCUSSION（自由讨论）
       → VOTING（投票放逐）
       → DAY_RESULT（遗言+猎人开枪）
       → GAME_OVER
    → Collector 写入 game_data.json
```

每个阶段 Agent 接收当前游戏上下文，调用 LLM API 生成 JSON 格式的决策（思考过程 + 发言内容 + 行动指令），前端通过 SSE 实时推送所有事件。

## 数据格式

每局游戏保存为一个 JSON 文件，结构如下：

```json
{
  "game_id": "game_a1b2c3d4",
  "timestamp": "2026-05-28T12:34:56.789012",
  "config": {
    "api_provider": "deepseek",
    "model": "deepseek-v4-pro",
    "players": [
      {"id": 1, "role": "werewolf", "personality": "戏精影帝"},
      {"id": 2, "role": "seer", "personality": "理性分析师"}
    ]
  },
  "phases": [
    {
      "phase": "NIGHT",
      "round": 1,
      "werewolf_chat": [{"player_id": 1, "message": "...", "thought": "..."}],
      "actions": [{"player_id": 1, "role": "werewolf", "action": "kill", "target": 5}]
    },
    {
      "phase": "DAY_DISCUSSION",
      "round": 1,
      "speeches": [{"player_id": 3, "speech": "我怀疑...", "thought": "..."}]
    }
  ],
  "result": {
    "winner": "good",
    "surviving_players": [2, 4, 7],
    "roles_revealed": {"1": "werewolf", "2": "seer"}
  }
}
```

## 配置

### API 配置 (`werewolf/config/api_config.yaml`)

支持 DeepSeek 和 Qwen 两个 provider，配置各自的 base_url、model、max_tokens、temperature 等参数。API key 通过 `${ENV_VAR}` 语法从环境变量读取。

### 游戏配置 (`werewolf/config/game_config.yaml`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `free_discussion_rounds` | 自由讨论轮数 | 2 |
| `phase_delay_seconds` | 阶段间延迟（秒） | 3 |
| `deep_thinking` | 深度思考模式 | false |
| `thinking_budget` | 推理最大 token 数（Qwen） | 2048 |
| `witch_self_save_first_night` | 首夜女巫可自救 | true |

### 性格配置 (`werewolf/personalities/traits.yaml`)

预设 12 种 AI 玩家性格，每种包含 `traits`（性格特征）和 `speaking_style`（说话风格）。可在 Web UI 中为每位玩家单独选择性格，也可随机分配。

### 模型名称验证

CLI 和 Web UI 均会对用户输入的模型名称进行验证。支持的模型列表定义在 `werewolf/llm/__init__.py` 的 `VALID_MODELS` 常量中：

| 提供商 | 支持模型 |
|--------|---------|
| DeepSeek | `deepseek-v4-flash`、`deepseek-v4-pro` |
| Qwen | `qwen3.6-flash`、`qwen3.7-plus`、`qwen3.7-max` |

CLI 模式下，若 `--model` 指定了无效模型名，程序会报错并立即退出。Web UI 在服务端同样会校验，返回 400 错误。

### 上下文长度控制

系统对每个 Agent 的对话历史进行两层控制，防止超长上下文导致 API 调用失败：

1. **轮数控制**：`agent.py` 中的 `MAX_HISTORY_TURNS = 20` 限制存储的对话轮数（保留最近 20 轮用户 + 助手交换），防止历史无限增长。

2. **字符数控制**：所有发送给 LLM 的对话消息总字符数上限为 **1,000,000 字符**（约 1M 字符）。超过此限制时，系统自动丢弃最早的历史消息，保留系统提示词和最新内容，确保 API 调用不会因超长上下文而失败。

## 技术栈

- **后端**：Python / FastAPI / uvicorn
- **LLM**：OpenAI-compatible API（DeepSeek / Qwen）
- **前端**：Jinja2 模板 + 原生 JavaScript + SSE
- **数据**：结构化 JSON 文件存储
- **配置**：YAML

## License

MIT
