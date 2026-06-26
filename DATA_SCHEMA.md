# 狼人杀对局数据格式规范 · DATA_SCHEMA v2.0

> 用途：DEDUCE 论文的 pilot + 主实验共用的数据契约。
> 版本：v2.0 — 覆盖所有 28 局已有实验数据的实际格式，并作为后续生成的规范。

---

## 0. 一条铁律（最高优先级）

> 🔴 **私密信息绝不能进入"模型可见"的数据流。**
> 玩家心理活动（thought）、狼人夜间密聊（werewolf_chat）、夜间技能动作（查验/救毒）
> 全部是 **private**，**只用于打标签和分析，永不可作为模型输入证据**。
> 一旦泄漏，整个实验结果作废。

判断标准：把一局数据里所有 `visibility=public` 的内容抽出来，应当**恰好等于**一个旁观者
在现场能看到/听到的全部信息（公开发言 + 投票 + 系统公告），不多一个字。

---

## 1. 顶层结构

```json
{
  "schema_version": "2.0",
  "game_id": "game_0671dbaf",
  "timestamp": "2026-06-04T21:08:31.781897",
  "config":  { ... },
  "phases":  [ ... ],
  "deaths":  [ ... ],
  "result":  { ... }
}
```

- `phases` 中每条记录含 `seq`（全局递增序号，从 0 开始），按时间升序。
- `deaths` 是从 phases 中提取的死亡汇总数组（由 collector 自动生成），方便下游直接使用。

---

## 2. 受控词表

| 维度 | 合法值 |
|---|---|
| `role` | `werewolf` `seer` `witch` `hunter` `villager` |
| `phase` | `NIGHT` `DAY_ANNOUNCE` `DAY_DISCUSSION` `DAY_FREE_DISCUSSION` `VOTING` `PK_VOTING` `PK_DISCUSSION` `DAY_RESULT` `GAME_OVER` |
| `visibility` | `public` `private` |
| 夜间 `action.type` | `discuss` `kill_vote` `check` `use_antidote` `use_poison` `pass` |
| 死亡 `cause` | `werewolf_kill` `witch_poison` `vote_out` `hunter_shoot` |
| `winner` | `good` `werewolves` |

---

## 3. config（对局配置 + 真值标签）

```json
"config": {
  "api_provider": "qwen",
  "model": "qwen3.7-max",
  "temperature": 1.0,
  "reasoning_effort": "high",
  "deep_thinking": false,
  "thinking_budget": 2048,
  "preserve_thinking": false,
  "phase_delay_seconds": 3,
  "free_discussion_rounds": 2,
  "witch_self_save_first_night": true,
  "players": [
    {"id": 1, "role": "werewolf", "personality": "戏精影帝"},
    {"id": 2, "role": "seer",     "personality": ""}
  ]
}
```
- `players[].role` = **ground truth**，评测的唯一标准答案。
- `id` 从 1 开始连续；全局所有地方引用玩家都用这个 `id`。
- `personality` 可为空字符串（表示未指定性格，LLM 按自身语言能力运行）。

---

## 4. phases（事件流，按 seq 升序）

每个 phase 记录至少有 `phase` + `round` + `seq`。下面分类型说明。

### 4.1 NIGHT（夜晚，全部 private）

```json
{
  "seq": 0, "phase": "NIGHT", "round": 1,
  "events": [
    {"type": "werewolf_chat", "speaker": 4, "thought": {"text": "...", "visibility": "private"}, "speech": "杀5号", "visibility": "private"},
    {"type": "werewolf_chat", "speaker": 5, "thought": {"text": "...", "visibility": "private"}, "speech": "同意", "visibility": "private"},
    {"type": "kill_vote", "actor": 4, "target": 5, "thought": {"text": "...", "visibility": "private"}, "visibility": "private"},
    {"type": "check", "actor": 2, "role": "seer", "target": 1, "result": "werewolf", "thought": {"text": "...", "visibility": "private"}, "visibility": "private"},
    {"type": "use_antidote", "actor": 6, "role": "witch", "target": 5, "thought": {"text": "...", "visibility": "private"}, "visibility": "private"},
    {"type": "pass", "actor": 6, "role": "witch", "action_type": "poison", "thought": {"text": "...", "visibility": "private"}, "visibility": "private"}
  ],
  "visibility": "private"
}
```
- `thought` 统一为 `{"text": "...", "visibility": "private"}` 结构。
- `check` 必须带 `target` + `result`（`werewolf` / `good`）。
- NIGHT 阶段全部 `visibility: "private"`。

### 4.2 DAY_ANNOUNCE（天亮公告，public）

```json
{
  "seq": 7, "phase": "DAY_ANNOUNCE", "round": 2,
  "events": [
    {"type": "death", "player_id": 4, "cause": "werewolf_kill", "visibility": "public"},
    {"type": "last_words", "player_id": 4, "thought": {"text": "...", "visibility": "private"}, "speech": "我是平民...", "visibility": "public"}
  ],
  "night_summary": {"data": [...], "visibility": "private"},
  "visibility": "public"
}
```
- `events` 中每个死亡是结构化 `{player_id, cause}`，`cause` 在受控词表内。
- 平安夜时无 `death` 事件。
- `night_summary` 为 private（仅供观战前端展示），不在 LLM 上下文中。

### 4.3 DAY_DISCUSSION / DAY_FREE_DISCUSSION / PK_DISCUSSION（白天发言）

```json
{
  "seq": 8, "phase": "DAY_DISCUSSION", "round": 1,
  "events": [
    {
      "player_id": 1, "turn": 1,
      "thought": {"text": "我是狼，得装好人...", "visibility": "private"},
      "speech": "我是1号，底牌好人……",
      "visibility": "public"
    }
  ],
  "visibility": "public"
}
```
- 每条发言一个 record：`speech`（公开）与 `thought`（私密）**分字段**。
- `turn` = 本阶段内发言顺序。
- `DAY_FREE_DISCUSSION` 额外带 `free_round` 字段在每个 event 中。
- 🔴 下游只读 `speech`；`thought` 仅用于分析。

### 4.4 VOTING / PK_VOTING（放逐投票，public）

```json
{
  "seq": 15, "phase": "VOTING", "round": 1,
  "events": [
    {"voter_id": 1, "target": 3, "thought": {"text": "...", "visibility": "private"}, "visibility": "public"},
    {"voter_id": 2, "target": 3, "thought": {"text": "...", "visibility": "private"}, "visibility": "public"},
    {"voter_id": 3, "target": 5, "thought": {"text": "...", "visibility": "private"}, "visibility": "public"}
  ],
  "result": {"eliminated": 3, "note": ""},
  "visibility": "public"
}
```
- 每个存活玩家投一票：`voter_id` → `target`。弃票 `target` 为 `null`。
- `result.eliminated` = 被放逐的玩家 ID；平票 / 无人出局时为 `null`，在 `note` 中说明。

### 4.5 DAY_RESULT（放逐结算，public）

```json
{
  "seq": 16, "phase": "DAY_RESULT", "round": 1,
  "events": [
    {"type": "elimination", "player_id": 3, "cause": "vote_out", "visibility": "public"},
    {"type": "last_words", "player_id": 3, "thought": {"text": "...", "visibility": "private"}, "speech": "遗言...", "visibility": "public"}
  ],
  "visibility": "public"
}
```
- 猎人被票出后开枪：在 `events` 中追加 `hunter_shoot` 事件（附带目标 last_words）。
- 无放逐时（平票 / 无人出局）：无 `elimination` 事件。

### 4.6 GAME_OVER（游戏结束，public）

```json
{
  "seq": 24, "phase": "GAME_OVER", "round": 3,
  "winner": "werewolves",
  "visibility": "public"
}
```

---

## 5. deaths（死亡汇总，顶层数组）

从 phases 中提取，每个死亡一条：

```json
"deaths": [
  {"player_id": 4, "round": 2, "cause": "werewolf_kill"},
  {"player_id": 5, "round": 2, "cause": "hunter_shoot"},
  {"player_id": 3, "round": 1, "cause": "vote_out"}
]
```

- `cause` 值：`werewolf_kill` | `witch_poison` | `vote_out` | `hunter_shoot`

---

## 6. result（对局结果）

```json
"result": {
  "winner": "good",
  "surviving_players": [2, 5, 6, 8],
  "roles_revealed": {"1": "werewolf", "2": "seer", "3": "villager", "4": "werewolf", "5": "hunter", "6": "witch", "7": "villager", "8": "werewolf", "9": "villager"},
  "duration_seconds": 342.5,
  "duration_display": "5分43秒"
}
```
- `winner`: `"good"` | `"werewolves"`
- `surviving_players`: 存活玩家的 id 列表

---

## 7. 校验清单（每一局都必须全过）

- [ ] 每个 `player` 有合法 `role`，9 名玩家覆盖全部 5 种角色
- [ ] 🔴 每条记录可判定 public/private；`speech` 公开、`thought` 私密、夜间动作/密聊全 private
- [ ] 🔴 把所有 public 内容抽出后，**不含任何**身份真值、心理活动、夜间查验结果
- [ ] 每个 phase 有合法 `phase` + `round` + `seq`，seq 连续递增
- [ ] 🔴 每轮 `VOTING` 有结构化 `events[{voter_id, target}]` + `result.eliminated`
- [ ] 🔴 每个死亡是结构化 `{player_id, cause}`，`cause` 在受控词表内
- [ ] 夜间 `check` 带 `target` + `result`；witch 动作为 `use_antidote` / `use_poison` / `pass`
- [ ] `result` 含 `winner` + `surviving_players`
- [ ] `game_id` 唯一；`speaker/voter_id/target/actor` 均为合法 player id
- [ ] 所有文本非空、UTF-8、无截断

---

## 8. 公开观测流 O_t 的定义（下游契约）

对任意轮次 t，"模型能看到的证据" $O_{1:t}$ = 按时间顺序拼接以下 **public** 内容：
1. `DAY_ANNOUNCE.events[].death`（系统公告，认知标签 = Fact）
2. `DAY_DISCUSSION / DAY_FREE_DISCUSSION / PK_DISCUSSION.events[].speech`（玩家发言，认知标签 = Claim）
3. `VOTING / PK_VOTING.events[{voter_id, target}]`（投票行为）

> 不包含：任何 `thought`、`werewolf_chat`、夜间 `actions`、`night_summary`。
> 这一条是 OSM 模块、似然计算、评测三处共同的输入边界。

---

## 9. 文件组织 & 划分

```
data/
  <YYYY-MM-DD>/
    game_<id>/game_data.json
splits.json          # {"train":[game_id...], "val":[...], "test":[...]}（待创建）
```
- 建议 train/val/test = 70/15/15，写进 `splits.json`。
- 🔴 **test 集只在最终评测用一次，全程禁止在其上调参/选模型。**

---

## 10. v1.0 → v2.0 变更摘要

| v1.0 | v2.0 | 说明 |
|---|---|---|
| `schema_version: "1.0"` | `"2.0"` | |
| `text` / `private_thought` | `speech` / `thought`（`{text, visibility}` 结构） | thought 统一为嵌套对象 |
| `DAY_VOTE` | `VOTING` | 阶段名修改 |
| 无 `DAY_RESULT` | `DAY_RESULT` | 新增放逐结算阶段 |
| 无 `PK_VOTING` / `PK_DISCUSSION` | 新增 | 平票加赛机制 |
| 无 `GAME_OVER` | `GAME_OVER` | 新增终局阶段 |
| `voter` / `target` | `voter_id` / `target` | 字段名修改 |
| `survivors` | `surviving_players` | 字段名修改 |
| `end_round` | 无 | 移除（用 `duration_seconds` 替代） |
| `reason` | 无 | 移除 |
| `split` 字段 | 无 | 暂未实现，待后续添加 |
| `wolf_kill` | `werewolf_kill` | cause 值修改 |
| `elimination`（cause） | `vote_out` | cause 更精确 |
| 无顶层 `deaths` | 顶层 `deaths` 数组 | 新增死亡汇总 |
| config 无 LLM 参数 | config 含 `api_provider`, `model`, `temperature`, `deep_thinking` 等 | 扩展配置字段 |
| `n_players` / `role_setup` | 无（隐含 9 人标准局） | |
