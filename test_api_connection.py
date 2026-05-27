#!/usr/bin/env python3
"""Quick API connectivity test — verifies API keys work before running a full game."""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(override=True)

from werewolf.llm import create_llm_client


async def test_provider(name: str):
    """Test a single provider's connectivity and structured output."""
    key_vars = {"qwen": "QWEN_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
    var_name = key_vars[name]
    api_key = os.environ.get(var_name)

    if not api_key:
        print(f"  ⏭️  {name}: 跳过（{var_name} 未设置）")
        return False

    print(f"  🔌 {name}: 测试连接...")

    try:
        # Import config manually (avoid web.app import)
        import yaml
        from pathlib import Path
        root = Path(__file__).parent

        with open(root / "werewolf" / "config" / "api_config.yaml") as f:
            raw = yaml.safe_load(f)
        config = {}
        for k, v in raw.get(name, {}).items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                config[k] = os.environ.get(v[2:-1], "")
            else:
                config[k] = v

        # Test without deep thinking
        client = create_llm_client(name, config, deep_thinking=False)
        messages = [
            {"role": "system", "content": "你是一个狼人杀游戏助手，请用中文回复，只输出JSON。"},
            {"role": "user", "content": """请按格式回复：{"thought": "你的推理", "speech": "你的发言", "action": {"type": "pass"}}"""},
        ]

        response = await client.chat(messages, response_format={"type": "json_object"})
        content = response["choices"][0]["message"]["content"]
        data = json.loads(content)

        print(f"     ✅ 连接成功 (model={config.get('model')}, tokens={response.get('usage', {}).get('total_tokens', '?')})")
        print(f"     回复预览: {content[:120]}...")

        # Test with deep thinking
        client2 = create_llm_client(name, config, deep_thinking=True)
        response2 = await client2.chat(messages, response_format={"type": "json_object"})
        content2 = response2["choices"][0]["message"]["content"]
        reasoning = response2["choices"][0]["message"].get("reasoning_content", "")

        print(f"     ✅ 深度思考模式正常 (reasoning={len(reasoning)} chars)")
        print(f"     回复预览: {content2[:120]}...")

        return True

    except Exception as e:
        print(f"     ❌ 连接失败: {e}")
        return False


async def main():
    print("=" * 50)
    print("API 连接测试")
    print("=" * 50)

    qwen_ok = await test_provider("qwen")
    deepseek_ok = await test_provider("deepseek")

    print()
    if qwen_ok or deepseek_ok:
        print("✅ 至少一个 provider 可用，可以运行游戏。")
        if qwen_ok:
            print("   python run.py --provider qwen")
        if deepseek_ok:
            print("   python run.py --provider deepseek --thinking")
    else:
        print("❌ 所有 provider 都不可用。请检查：")
        print("   1. API Key 是否正确设置（环境变量或 .env 文件）")
        print("   2. 网络是否能访问 API endpoint")
        print("   3. API 账户是否有余额")


if __name__ == "__main__":
    asyncio.run(main())
