"""
知识库构建脚本 - 处理原始书籍数据

使用方式:
    python scripts/build_knowledge_base.py

功能:
    1. 数据清洗 - 去除版权页、ISBN、页眉页脚等噪音
    2. 智能分块 - 按章节结构分割，保持语义完整
    3. 构建向量库 - 存入 Chroma 向量数据库
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from logger import get_logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from knowledge.knowledge_base import embeddings

logger = get_logger("build_kb")


@dataclass
class BookConfig:
    """书籍配置"""
    filename: str
    title: str
    author: str
    skip_patterns: List[str]  # 需要跳过的正则模式
    chapter_pattern: str      # 章节识别正则


# 三本书的配置
BOOK_CONFIGS = {
    "book0.txt": BookConfig(
        filename="book0.txt",
        title="美国儿科学会育儿百科",
        author="塔尼娅·奥尔特曼/美国儿科学会",
        skip_patterns=[
            r"^\s*图书在版编目.*?$",
            r"^\s*ISBN.*?$",
            r"^\s*著作权合同登记号.*?$",
            r"^\s*译\s*者.*?$",
            r"^\s*策划编辑.*?$",
            r"^\s*责任编辑.*?$",
            r"^\s*责任校对.*?$",
            r"^\s*责任印制.*?$",
            r"^\s*图文制作.*?$",
            r"^\s*出\s*版\s*人.*?$",
            r"^\s*出版发行.*?$",
            r"^\s*社\s*址.*?$",
            r"^\s*邮政编码.*?$",
            r"^\s*电话传真.*?$",
            r"^\s*网\s*址.*?$",
            r"^\s*印\s*刷.*?$",
            r"^\s*印\s*张.*?$",
            r"^\s*开\s*本.*?$",
            r"^\s*字\s*数.*?$",
            r"^\s*版\s*次.*?$",
            r"^\s*印\s*次.*?$",
            r"^\s*定\s*价.*?$",
            r"^\s*The publication is a translation.*?$",
            r"^\s*Simplified Chinese translation copyright.*?$",
        ],
        chapter_pattern=r"^第[一二三四五六七八九十0-9]+章|^Chapter\s+\d+"
    ),
    "book1.txt": BookConfig(
        filename="book1.txt",
        title="崔玉涛谈自然养育",
        author="崔玉涛",
        skip_patterns=[
            r"^\s*崔玉涛谈自然养育.*$",
            r"^\s*序言\s*$",
            r"^\s*在此，感谢.*?$",
            r"^\s*20\d{2}年.*?$",  # 日期
        ],
        chapter_pattern=r"^第[一二三四五六七八九十0-9]+章|^第一章|^第二章|^第三章"
    ),
    "book2.txt": BookConfig(
        filename="book2.txt",
        title="西尔斯亲密育儿百科",
        author="威廉·西尔斯",
        skip_patterns=[
            r"^\s*献给我们的.*$",
            r"^\s*詹姆斯$|^\s*罗伯特$|^\s*彼得$|^\s*海登$",  # 献词页
            r"^\s*艾琳$|^\s*马修$|^\s*史蒂芬$|^\s*劳伦$",
            r"^\s*以及$",
            r"^\s*——威廉·西尔斯$",
            r"^\s*以及我们的孙子孙女$",
            r"^\s*安德鲁$|^\s*莉$|^\s*亚历克斯$|^\s*乔纳森$",
            r"^\s*乔舒亚$|^\s*阿什顿$|^\s*摩根$|^\s*托马斯$|^\s*兰登$",
        ],
        chapter_pattern=r"^第[一二三四五六七八九十0-9]+章|^Chapter\s+\d+|^第\d+章"
    )
}


class BookProcessor:
    """书籍处理器"""

    def __init__(self, config: BookConfig):
        self.config = config
        self.chapters = []

    def clean_text(self, raw_text: str) -> str:
        """清洗文本 - 去除噪音"""
        lines = raw_text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            # 跳过匹配的模式
            should_skip = False
            for pattern in self.config.skip_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    should_skip = True
                    break

            if should_skip:
                continue

            # 去除页码（纯数字行）
            if re.match(r'^\s*\d+\s*$', line):
                continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _is_valid_chapter(self, chapter: Dict[str, str]) -> bool:
        """过滤掉目录碎片等伪章节"""
        content = chapter["content"].strip()
        if len(content) < 80:
            return False

        lines = [l.strip() for l in content.split('\n') if l.strip()]
        if not lines:
            return False

        # 90%以上的行都少于15字，认为是目录列表
        short_lines = sum(1 for line in lines if len(line) < 15)
        if short_lines / len(lines) > 0.85:
            return False

        # 超过70%的行是 LESSON/第X章/序号条目，认为是标题目录
        pattern_lines = sum(1 for line in lines if re.match(
            r'^(LESSON\s+\d+|第[一二三四五六七八九十0-9]+[章节节]|第\d+[章节节]|\d+[\.\、])',
            line
        ))
        if pattern_lines / len(lines) > 0.70:
            return False

        return True

    def extract_chapters(self, text: str) -> List[Dict[str, str]]:
        """提取章节结构"""
        chapters = []
        current_chapter = {"title": "前言/介绍", "content": []}

        lines = text.split('\n')

        for line in lines:
            # 检测章节标题
            if re.match(self.config.chapter_pattern, line.strip()):
                # 保存上一章（过滤伪章节）
                if current_chapter["content"]:
                    if self._is_valid_chapter(current_chapter):
                        chapters.append({
                            "title": current_chapter["title"],
                            "content": '\n'.join(current_chapter["content"])
                        })
                    else:
                        # 伪章节的内容合并到下一个章节
                        pass

                # 开始新章节
                current_chapter = {
                    "title": line.strip(),
                    "content": []
                }
            else:
                current_chapter["content"].append(line)

        # 保存最后一章
        if current_chapter["content"] and self._is_valid_chapter(current_chapter):
            chapters.append({
                "title": current_chapter["title"],
                "content": '\n'.join(current_chapter["content"])
            })

        return chapters

    def split_into_chunks(self, chapters: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """将章节分块"""
        chunks = []

        # 使用递归文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
            length_function=len,
        )

        for chapter in chapters:
            chapter_chunks = text_splitter.split_text(chapter["content"])

            for i, chunk in enumerate(chapter_chunks):
                # 过滤太短的块
                if len(chunk.strip()) < 50:
                    continue

                chunks.append({
                    "content": chunk.strip(),
                    "metadata": {
                        "source": self.config.title,
                        "author": self.config.author,
                        "chapter": chapter["title"],
                        "chunk_id": i,
                        "book": self.config.filename
                    }
                })

        return chunks

    def process(self, raw_text: str) -> List[Dict[str, str]]:
        """完整处理流程"""
        logger.info(f"开始处理: {self.config.title}")

        # 1. 清洗
        cleaned = self.clean_text(raw_text)
        logger.info(f"  清洗后文本长度: {len(cleaned)} 字符")

        # 2. 提取章节
        chapters = self.extract_chapters(cleaned)
        logger.info(f"  提取章节数: {len(chapters)}")

        # 3. 分块
        chunks = self.split_into_chunks(chapters)
        logger.info(f"  生成文本块: {len(chunks)} 个")

        return chunks


def build_vector_database(all_chunks: List[Dict[str, str]], collection_name: str = "parenting_books"):
    """构建向量数据库"""
    from knowledge.knowledge_base import DB_DIR

    logger.info(f"开始构建向量库: {collection_name}")
    logger.info(f"总文本块数: {len(all_chunks)}")

    # 准备数据
    texts = [chunk["content"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]

    # 创建或加载向量库
    if os.path.exists(DB_DIR):
        logger.info(f"加载现有向量库: {DB_DIR}")
    else:
        logger.info(f"创建新向量库: {DB_DIR}")

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=collection_name
    )

    # 分批嵌入，每批 200 个
    batch_size = 200
    total = len(texts)
    num_batches = (total + batch_size - 1) // batch_size
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batch_num = i // batch_size + 1
        logger.info(f"  嵌入批次 {batch_num}/{num_batches}: {i+1}-{end}/{total}")
        vectorstore.add_texts(
            texts[i:end],
            metadatas=metadatas[i:end]
        )
        logger.info(f"  批次完成，累计嵌入 {end}/{total}")

    logger.info(f"向量库构建完成")
    return vectorstore


def main():
    """主函数"""
    raw_books_dir = Path(__file__).parent.parent / "raw_books"

    logger.info("=" * 60)
    logger.info("开始构建育儿知识库")
    logger.info("=" * 60)

    all_chunks = []

    # 处理每本书
    for book_file, config in BOOK_CONFIGS.items():
        book_path = raw_books_dir / book_file

        if not book_path.exists():
            logger.warning(f"文件不存在，跳过: {book_path}")
            continue

        logger.info(f"\n处理书籍: {config.title}")

        # 读取原始文本
        with open(book_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        logger.info(f"  原始文本长度: {len(raw_text)} 字符")

        # 处理
        processor = BookProcessor(config)
        chunks = processor.process(raw_text)
        all_chunks.extend(chunks)

        logger.info(f"  完成处理，生成 {len(chunks)} 个文本块")

    # 构建向量库
    logger.info("\n" + "=" * 60)
    logger.info("构建向量数据库")
    logger.info("=" * 60)

    vectorstore = build_vector_database(all_chunks)

    # 统计信息
    logger.info("\n" + "=" * 60)
    logger.info("知识库构建完成")
    logger.info("=" * 60)
    logger.info(f"总文本块数: {len(all_chunks)}")
    logger.info(f"向量库路径: {os.path.abspath('chroma_parenting_db')}")

    # 按书籍统计
    book_stats = {}
    for chunk in all_chunks:
        book = chunk["metadata"]["book"]
        book_stats[book] = book_stats.get(book, 0) + 1

    logger.info("\n各书籍文本块统计:")
    for book, count in book_stats.items():
        logger.info(f"  {BOOK_CONFIGS[book].title}: {count} 块")

    return vectorstore


if __name__ == "__main__":
    main()
