"""
Agent 主入口（重构后）

代码已按功能模块拆分：
- config.py: 配置和环境变量
- state.py: 状态类型定义
- tools.py: 工具函数定义
- messages.py: 消息格式转换
- nodes.py: 工作流节点实现
- graph.py: 图构建和编译
- main.py: 测试入口

使用方式：
    from graph import app
    result = app.invoke({...})
"""

# 向后兼容：重新导出主要组件
from graph import app
from state import ParentingAgentState
from tools import tools, parenting_knowledge, manage_baby_profile

__all__ = ['app', 'ParentingAgentState', 'tools', 'parenting_knowledge', 'manage_baby_profile']


if __name__ == "__main__":
    import sys
    print("⚠️  agent.py 已重构，请运行 main.py 进行测试")
    print("   python main.py")
    sys.exit(1)
