# scripts/build_kb.py
import sys
sys.path.append('.')  # 确保能导入项目模块
from knowledge_base import build_vector_store

if __name__ == "__main__":
    build_vector_store()
    print("✅ 知识库构建完成")