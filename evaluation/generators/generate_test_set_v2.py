"""
基于实际书本内容生成高质量检索评估数据集 v2

特点：
1. 查询贴近真实用户口吻（带场景、数据、情绪）
2. 每个问题只标注 1-3 个真正相关的 chunk
3. 从数据库中动态匹配确认 doc_id 存在
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_chroma import Chroma
from knowledge.knowledge_base import embeddings

# ===================== 手工定义的高质量问题 =====================
# 格式: (query, [(book, chapter_substring, chunk_id), ...], keywords, is_safety, description)
HANDCRAFTED_QUESTIONS = [
    # ========== 崔玉涛谈自然养育 ==========
    (
        "我家宝宝6个月了还不会翻身，是不是发育迟缓啊？",
        [("book1.txt", "发育 追寻的点点历程", 0)],
        ["发育", "翻身", "迟缓"],
        False,
        "发育迟缓担忧"
    ),
    (
        "发育迟缓和生长缓慢是一回事吗？怎么区分？",
        [("book1.txt", "发育 追寻的点点历程", 0), ("book1.txt", "生长 最自然的成长过程", 0)],
        ["发育", "生长", "区别"],
        False,
        "发育与生长区别"
    ),
    (
        "我家宝宝出生时只有2.8kg，比隔壁家孩子轻好多，以后能追得上吗？",
        [("book1.txt", "生长 最自然的成长过程", 0)],
        ["出生体重", "生长", "追赶"],
        False,
        "低出生体重担忧"
    ),
    (
        "生长曲线到底怎么看啊？我该怎么给宝宝画？",
        [("book1.txt", "生长 最自然的成长过程", 1)],
        ["生长曲线", "监测", "测量"],
        False,
        "生长曲线使用方法"
    ),
    (
        "快生了，待产包里到底哪些东西是真正必需的？我看网上清单好长",
        [("book1.txt", "第一次 亲密接触", 0)],
        ["待产包", "必需品", "准备"],
        False,
        "待产包准备"
    ),
    (
        "医院待产包东西全吗？自己还需要额外准备什么？",
        [("book1.txt", "第一次 亲密接触", 1)],
        ["待产包", "医院", "纸尿裤"],
        False,
        "医院待产包"
    ),
    (
        "宝宝刚出生是怎么开始呼吸和吃奶的？脐带剪断了营养从哪来？",
        [("book1.txt", "食物 随生长发育而变化", 0), ("book1.txt", "食物 随生长发育而变化", 1)],
        ["呼吸", "进食", "脐带", "新生儿"],
        False,
        "新生儿生理变化"
    ),
    (
        "母乳是宝宝最好的食物吗？配方奶和母乳差在哪？",
        [("book1.txt", "食物 随生长发育而变化", 0)],
        ["母乳", "配方奶", "营养"],
        False,
        "母乳vs配方奶"
    ),
    (
        "宝宝4个月感冒了，流鼻涕鼻塞，怎么预防下次再感冒？",
        [("book1.txt", "感冒、发烧", 1)],
        ["感冒", "预防", "流鼻涕"],
        False,
        "感冒预防"
    ),
    (
        "宝宝发烧38.5度，是不是一定要吃退烧药？听说发烧是保护机制？",
        [("book1.txt", "感冒、发烧", 1)],
        ["发烧", "退烧药", "体温"],
        True,
        "发烧处理"
    ),
    (
        "高烧会烧坏孩子吗？什么情况下必须去医院？",
        [("book1.txt", "感冒、发烧", 1)],
        ["高烧", "就医", "危险信号"],
        True,
        "高烧就医指征"
    ),
    (
        "我家早产宝宝出院了才1300克，回家后该怎么护理？",
        [("book1.txt", "日常护理", 1)],
        ["早产儿", "护理", "低体重"],
        True,
        "早产儿家庭护理"
    ),
    (
        "宝宝耳朵里有很多耳垢，需要帮他清理吗？",
        [("book1.txt", "日常护理", 1)],
        ["耳垢", "清理", "耳朵"],
        False,
        "耳垢护理"
    ),
    (
        "怎么判断宝宝穿多了还是穿少了？摸手脚准不准？",
        [("book1.txt", "日常护理", 1)],
        ["穿衣", "冷暖", "颈部温度"],
        False,
        "宝宝穿衣判断"
    ),
    (
        "宝宝不小心吃了我的药，怎么办？需要催吐吗？",
        [("book1.txt", "日常护理", 0)],
        ["误服", "药物", "急救"],
        True,
        "误服药物急救"
    ),
    (
        "宝宝拉肚子脱水了怎么办？怎么判断脱水程度？",
        [("book1.txt", "日常护理", 0)],
        ["腹泻", "脱水", "补液"],
        True,
        "腹泻脱水处理"
    ),

    # ========== 美国儿科学会育儿百科 ==========
    (
        "宝宝1岁多了，老说'不'，什么都对着干，这是正常的吗？",
        [("book0.txt", "1岁", 0)],
        ["1岁", "不", "反抗"],
        False,
        "一岁宝宝说'不'"
    ),
    (
        "1岁半宝宝特别护东西，别人碰一下就哭，怎么引导？",
        [("book0.txt", "1岁", 1)],
        ["占有欲", "玩具", "分享"],
        False,
        "占有欲"
    ),
    (
        "1岁宝宝该打什么疫苗？有什么注意事项？",
        [("book0.txt", "1岁", 0)],
        ["疫苗", "接种", "1岁"],
        False,
        "1岁疫苗"
    ),
    (
        "孩子2岁了，整天说'不'，是不是进入了可怕的两岁？",
        [("book0.txt", "2岁", 0)],
        ["2岁", "可怕的两岁", "反抗期"],
        False,
        "可怕的两岁"
    ),
    (
        "宝宝2岁了，什么时候开始如厕训练比较合适？",
        [("book0.txt", "2岁", 0)],
        ["如厕训练", "2岁", "准备"],
        False,
        "如厕训练时机"
    ),
    (
        "2岁宝宝老发脾气，有什么办法应对吗？",
        [("book0.txt", "2岁", 0)],
        ["发脾气", "情绪", "2岁"],
        False,
        "2岁发脾气"
    ),
    (
        "家里有大宝了，再生一个小宝宝，怎么让大宝接受？",
        [("book0.txt", "2岁", 0)],
        ["二胎", "大宝", "接受"],
        False,
        "二胎适应"
    ),
    (
        "宝宝3岁了，该不该送幼儿园？怎么判断他准备好了没有？",
        [("book0.txt", "3岁", 0)],
        ["3岁", "幼儿园", "准备"],
        False,
        "3岁上幼儿园"
    ),
    (
        "3岁孩子开始跟我讨价还价了，'要是我做这个你就给我那个'，怎么应对？",
        [("book0.txt", "3岁", 1)],
        ["讨价还价", "3岁", "规矩"],
        False,
        "3岁讨价还价"
    ),
    (
        "4岁孩子情绪变化特别快，而且对日常规律很执着，稍微改一下就崩溃，正常吗？",
        [("book0.txt", "4～5岁", 0)],
        ["4岁", "情绪", "规律"],
        False,
        "4岁情绪与规律"
    ),
    (
        "4岁宝宝突然开始说脏话和不好听的话，故意看我们反应，该怎么办？",
        [("book0.txt", "4～5岁", 0), ("book0.txt", "4～5岁", 1)],
        ["4岁", "脏话", "语言"],
        False,
        "4岁说脏话"
    ),
    (
        "家里有哪些安全隐患是我没注意到的？怎么预防宝宝意外受伤？",
        [("book0.txt", "2岁", 0)],
        ["安全", "隐患", "预防"],
        True,
        "居家安全检查"
    ),
    (
        "宝宝坐车必须用安全座椅吗？有什么要求？",
        [("book0.txt", "2岁", 0)],
        ["汽车安全", "安全座椅", "出行"],
        True,
        "汽车安全座椅"
    ),

    # ========== 西尔斯亲密育儿百科 ==========
    (
        "母乳不够，想混合喂养，怎么选奶粉？不同奶粉差别大吗？",
        [("book2.txt", "安全而充满爱意地喂奶粉", 0)],
        ["混合喂养", "奶粉", "选择"],
        False,
        "混合喂养选奶粉"
    ),
    (
        "奶粉有粉状、浓缩液和即食的，哪种适合新生儿？",
        [("book2.txt", "安全而充满爱意地喂奶粉", 1)],
        ["奶粉", "粉状", "液态", "即食"],
        False,
        "奶粉包装类型"
    ),
    (
        "宝宝几个月可以加辅食？是看时间还是看宝宝自己的信号？",
        [("book2.txt", "引入固体食物", 0)],
        ["辅食", "时间", "信号"],
        False,
        "辅食添加时机"
    ),
    (
        "怎么培养宝宝健康的饮食习惯？不想让他以后只爱吃零食",
        [("book2.txt", "引入固体食物", 0)],
        ["饮食习惯", "零食", "健康"],
        False,
        "健康饮食习惯"
    ),
    (
        "被外婆催着加辅食，但宝宝看起来不感兴趣，可以等吗？",
        [("book2.txt", "引入固体食物", 1)],
        ["辅食", "外婆", "等待"],
        False,
        "长辈催加辅食"
    ),
    (
        "宝宝需要吃脂肪吗？吃什么脂肪对大脑发育好？",
        [("book2.txt", "10招让你成为家庭营养师", 0), ("book2.txt", "10招让你成为家庭营养师", 1)],
        ["脂肪", "大脑发育", "omega-3"],
        False,
        "婴儿脂肪摄入"
    ),
    (
        "宝宝要多吃鱼吗？三文鱼对婴儿有什么好处？",
        [("book2.txt", "10招让你成为家庭营养师", 1)],
        ["鱼", "三文鱼", "omega-3"],
        False,
        "婴儿吃鱼"
    ),
    (
        "1岁多的宝宝突然不爱吃饭了，体重也不怎么长，是不是有问题？",
        [("book2.txt", "给学步期宝宝喂食", 1)],
        ["挑食", "不吃饭", "体重"],
        False,
        "学步期挑食"
    ),
    (
        "宝宝18个月了只爱吃土豆，其他什么都不吃，怎么办？",
        [("book2.txt", "给学步期宝宝喂食", 0)],
        ["挑食", "18个月", "只吃一种"],
        False,
        "严重挑食"
    ),
    (
        "宝宝很喜欢被抱着，放下就哭，多抱会不会宠坏他？",
        [("book2.txt", "把宝宝", 0), ("book2.txt", "把宝宝", 1)],
        ["抱", "宠坏", "亲密"],
        False,
        "多抱会宠坏吗"
    ),
    (
        "用背巾背着宝宝有什么好处？多大开始可以用？",
        [("book2.txt", "把宝宝", 0)],
        ["背巾", "抱", "好处"],
        False,
        "背巾使用"
    ),
    (
        "宝宝晚上老醒，朋友推荐哭声免疫法，这个方法靠谱吗？",
        [("book2.txt", "夜间育儿", 0)],
        ["夜醒", "哭声免疫法", "睡眠"],
        False,
        "哭声免疫法评价"
    ),
    (
        "宝宝晚上频繁夜醒，是不是睡眠有问题？大人都会醒吗？",
        [("book2.txt", "夜间育儿", 1)],
        ["夜醒", "睡眠循环", "正常"],
        False,
        "婴儿夜醒正常吗"
    ),
    (
        "怎么帮助宝宝建立健康的睡眠态度？",
        [("book2.txt", "夜间育儿", 0)],
        ["睡眠态度", "健康", "夜间育儿"],
        False,
        "健康睡眠态度"
    ),
    (
        "宝宝傍晚开始连续哭几个小时，怎么哄都没用，是不是肠绞痛？",
        [("book2.txt", "肠痉挛", 0), ("book2.txt", "肠痉挛", 1)],
        ["肠绞痛", "傍晚", "哭闹"],
        True,
        "肠绞痛识别"
    ),
    (
        "高需求宝宝和难缠宝宝有什么区别？怎么照顾？",
        [("book2.txt", "肠痉挛", 0)],
        ["高需求", "难缠", "区别"],
        False,
        "高需求vs难缠"
    ),
    (
        "产假快结束了要回去上班，会不会影响跟宝宝的亲密关系？",
        [("book2.txt", "工作与育儿", 0)],
        ["上班", "亲密关系", "分离"],
        False,
        "上班族妈妈焦虑"
    ),
    (
        "上班族妈妈陪伴时间少，宝宝会不会跟我不亲了？",
        [("book2.txt", "工作与育儿", 1)],
        ["上班族", "不亲", "陪伴"],
        False,
        "陪伴时间担忧"
    ),
]


def find_doc_ids(vectorstore, book: str, chapter_sub: str, chunk_id: int):
    """根据 book 和章节子串查找匹配的 doc_id 列表"""
    data = vectorstore.get()
    metadatas = data["metadatas"]
    documents = data["documents"]
    ids = data["ids"]

    matches = []
    for doc_id, meta, doc in zip(ids, metadatas, documents):
        if meta.get("book") == book and chapter_sub in meta.get("chapter", ""):
            if meta.get("chunk_id") == chunk_id:
                matches.append((doc_id, len(doc)))

    return matches


def main():
    print("Loading vector database...")
    vectorstore = Chroma(
        persist_directory="chroma_parenting_db",
        embedding_function=embeddings,
        collection_name="parenting_books"
    )

    test_cases = []
    missing = []

    for query, refs, keywords, is_safety, desc in HANDCRAFTED_QUESTIONS:
        doc_ids = []
        for book, chapter_sub, chunk_id in refs:
            matches = find_doc_ids(vectorstore, book, chapter_sub, chunk_id)
            if matches:
                # 如果有多个匹配，取内容最长的那个（避免匹配到目录碎片）
                best = max(matches, key=lambda x: x[1])
                doc_ids.append(best[0])
            else:
                missing.append((query, book, chapter_sub, chunk_id))

        if not doc_ids:
            print(f"WARNING: No matching chunks for query: {query}")
            continue

        test_cases.append({
            "query": query,
            "expected_doc_ids": doc_ids,
            "expected_keywords": keywords,
            "is_safety_critical": is_safety,
            "description": desc
        })

    # 保存
    output_path = Path(__file__).parent.parent / "data" / "retrieval_test_set_v2.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated {len(test_cases)} test cases")
    print(f"Saved to: {output_path}")

    if missing:
        print(f"\nMissing {len(missing)} chunk references:")
        for q, b, c, cid in missing[:10]:
            print(f"  [{b}] chapter='{c}' chunk={cid} -> query='{q[:40]}...'")

    # 统计
    safety_count = sum(1 for tc in test_cases if tc["is_safety_critical"])
    print(f"\nSafety-critical: {safety_count}")
    print(f"Normal: {len(test_cases) - safety_count}")

    # 同时输出简化版
    simple = [{"query": tc["query"], "expected_doc_ids": tc["expected_doc_ids"]} for tc in test_cases]
    simple_path = Path(__file__).parent.parent / "data" / "retrieval_test_set_v2_simple.json"
    with open(simple_path, "w", encoding="utf-8") as f:
        json.dump(simple, f, ensure_ascii=False, indent=2)
    print(f"Simple version: {simple_path}")


if __name__ == "__main__":
    main()
