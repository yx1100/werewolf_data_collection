# Project: 狼人杀 LLM 数据收集平台 (Werewolf LLM Data Collection)

## Project Context
基于大语言模型的全自动狼人杀游戏平台。9 名 AI 玩家各自拥有独立角色与性格，通过 LLM API
完成推理、发言、投票和夜间行动。每局以结构化 JSON 记录全过程，用于 LLM 多智能体行为研究、
社会推理分析与对话数据收集。提供实时 Web 观战（SSE）与 CLI 两种运行方式。

## Tech Stack
- **Language**: Python 3.10+
- **Backend**: FastAPI + uvicorn
- **LLM**: OpenAI-compatible API，支持 DeepSeek 与 Qwen（通义千问），可在 Web UI 切换
- **Frontend**: Jinja2 模板 + 原生 JavaScript + SSE（Server-Sent Events）
- **Config**: YAML（API 配置、游戏参数、性格配置）
- **Data**: 结构化 JSON 文件存储（无数据库）
- **Key Dependencies**: fastapi, uvicorn[standard], openai, pyyaml, pydantic, jinja2, python-dotenv

## Project Structure
```
werwolf_data_collection/
├── run.py                      — 入口脚本（Web + CLI），含参数解析与启动 banner
├── requirements.txt            — Python 依赖
├── test_api_connection.py      — API 连接测试脚本
├── test_mock_game.py           — Mock 集成测试（不消耗真实 API）
├── data/                       — 游戏数据输出，按 YYYY-MM-DD/game_XXXX/ 组织
├── pilot/                      — 试验性脚本（pilot_stage1.py，README 未记录）
└── werewolf/
    ├── agents/                 — AI 玩家 Agent
    │   ├── agent.py            — Agent 主逻辑（多轮对话、决策接口）
    │   ├── prompt_builder.py   — 系统提示词 + 阶段消息模板
    │   └── decision_parser.py  — LLM JSON 响应解析器
    ├── collector/collector.py  — 游戏数据序列化
    ├── config/
    │   ├── api_config.yaml     — LLM API 配置（${ENV_VAR} 读取 key）
    │   └── game_config.yaml    — 游戏参数（讨论轮数、延迟、深度思考等）
    ├── engine/                 — 游戏引擎
    │   ├── game.py             — 游戏状态机（核心编排器）
    │   ├── roles.py            — 角色/阵营定义
    │   ├── rules.py            — 胜负判定 + 行动校验
    │   └── state.py            — 游戏状态数据结构
    ├── llm/__init__.py         — LLM 客户端抽象（DeepSeek + Qwen）
    ├── personalities/traits.yaml — 12 种 AI 性格配置
    └── web/
        ├── app.py              — FastAPI 路由 + 后台游戏运行 + SSE（核心 Web 逻辑）
        ├── routes.py           — 占位，逻辑全部委托给 app.py
        └── templates/          — index.html（大厅）/ game.html（观战）/ history.html
```

## Entry Points
- `run.py` — 主入口，解析命令行参数后启动 Web 服务或 CLI 单局对战
- `werewolf/web/app.py` — FastAPI 应用：游戏大厅、后台游戏运行、SSE 实时事件推送
- `werewolf/engine/game.py` — 游戏状态机循环（NIGHT → DAY_ANNOUNCE → DAY_DISCUSSION → DAY_FREE_DISCUSSION → VOTING → DAY_RESULT → GAME_OVER）

## Common Commands
- **Web（推荐）**: `python run.py`（默认 DeepSeek，访问 http://127.0.0.1:8000）
  - `python run.py --provider qwen` / `--thinking`（深度思考）/ `--port 8080`
- **CLI 单局**: `python run.py --cli --provider deepseek --thinking`
  - 指定性格: `python run.py --cli --personalities "1=戏精影帝,5=理性分析师"`（玩家号=性格名，逗号分隔，其余随机）
- **API 连接测试**: `python test_api_connection.py`
- **集成测试**: `python test_mock_game.py`

## Important Notes
- **环境要求**: Python 3.10+，需 DeepSeek 或 Qwen 的 API Key
- **API Key**: 通过环境变量 `DEEPSEEK_API_KEY` / `QWEN_API_KEY` 提供，或在根目录 `.env` 文件中配置
  （参考 `.env.example`）。`api_config.yaml` 用 `${ENV_VAR}` 语法读取
- **游戏配置**: 9 人标准局（3 狼人 / 预言家 / 女巫 / 猎人 / 3 平民）。可在
  `werewolf/config/game_config.yaml` 调整自由讨论轮数、阶段延迟、深度思考、女巫首夜自救等
- **数据输出**: 每局生成一个 `game_data.json`，含 config / phases / result 三部分；
  `data/` 已在 `.gitignore` 中忽略，不会提交
- **无数据库 / 无前端构建**: 纯文件存储 + Jinja2 服务端渲染，无需 npm/build 步骤
