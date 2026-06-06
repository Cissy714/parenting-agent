"""
育儿智能助手 - 命令行入口
用法: python main.py
"""
from graph import app
from langchain_core.messages import HumanMessage
from logger import get_logger
import uuid

logger = get_logger("main")


def main():
    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
    thread_config = {"configurable": {"thread_id": thread_id}}

    print("=" * 50)
    print("  育儿智能助手 (CLI)")
    print("  输入 'quit' 退出, 'reset' 重置对话")
    print("=" * 50)

    first_turn = True

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("再见！")
            break
        if user_input.lower() == "reset":
            thread_id = f"cli-{uuid.uuid4().hex[:8]}"
            thread_config = {"configurable": {"thread_id": thread_id}}
            first_turn = True
            print("对话已重置。")
            continue

        if first_turn:
            invoke_input = {
                "messages": [HumanMessage(content=user_input)],
                "current_baby_id": None,
                "baby_profile": None,
            }
            first_turn = False
        else:
            invoke_input = {"messages": [HumanMessage(content=user_input)]}

        try:
            result = app.invoke(invoke_input, thread_config)
        except Exception as e:
            logger.error(f"调用失败: {e}")
            print(f"错误: {e}")
            continue

        # 打印最后一条AI消息
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content and msg.type == "ai":
                print(f"\n{msg.content}")
                break


if __name__ == "__main__":
    main()
