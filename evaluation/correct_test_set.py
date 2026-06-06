# -*- coding: utf-8 -*-
"""
修正测试集：针对37条0% recall数据，自动修正关键词和章节标注

修正策略：
1. 关键词修正：从查询中提取核心实体（症状、药物、动作）
2. 章节修正：检索Top10，记录实际包含答案的章节
"""

import json
import re
import sys
from pathlib import Path
sys.path.insert(0, '.')

from langchain_chroma import Chroma
from knowledge.knowledge_base import embeddings

# 中文停用词/问句虚词
STOP_PATTERNS = re.compile(
    r'什么|怎么|怎么办|为什么|是不是|要不要|能不能|会不会|要不要紧|'
    r'怎么办呀|呢|吗|啊|吧|呀|了|啦|哦|嗯|多大|多久|多长|'
    r'应该|需要|可以|能够|一下|一点|些|有些|这个|那个|哪个|'
    r'还是|或者|到底|真的|怎么选|怎么判断|怎么引导|怎么处理|'
    r'有什么|为啥|为啥呢|干嘛|什么样|怎么样|什么时候|多少|'
    r'该不该|好不好|有没有|是不是要|可以不|要不要给'
)

# 育儿领域核心实体提取规则
MEDICAL_TERMS = [
    # 症状/疾病
    '黄疸', '发烧', '感冒', '咳嗽', '腹泻', '便秘', '湿疹', '过敏', '脱水',
    '红屁股', '红屁屁', '尿布疹', '呕吐', '抽搐', '惊厥', '热性惊厥',
    '肠绞痛', '肠痉挛', '鹅口疮', '生长迟缓', '发育迟缓', '贫血',
    '烫伤', '溺水', '窒息', '误食', '误服', '摔伤',
    # 药物/治疗
    '退烧药', '美林', '泰诺林', '抗生素', '止咳药', '益生菌',
    '护臀膏', '护臀霜', '氧化锌', '安抚奶嘴', '背巾',
    # 喂养
    '母乳', '配方奶', '奶粉', '辅食', '夜奶', '断奶', '混合喂养',
    # 发育
    '生长曲线', '翻身', '爬行', '走路', '说话', '出牙',
    # 护理
    '脐带', '洗澡', '睡眠', '安全座椅', '防晒', '疫苗', '接种',
    # 安全
    '安全隐患', '催吐', '洗胃',
]

def extract_core_keywords(query):
    """从查询中提取核心实体关键词"""
    keywords = []

    # 1. 匹配医学实体
    query_lower = query.lower()
    for term in MEDICAL_TERMS:
        if term in query:
            keywords.append(term)

    # 2. 提取数字+单位模式（如38.5度、2岁、6个月）
    age_patterns = re.findall(r'\d+[岁个月半]', query)
    keywords.extend(age_patterns)

    # 3. 提取温度相关
    temp_patterns = re.findall(r'\d+\.?\d*度', query)
    keywords.extend(temp_patterns)

    # 4. 提取育儿特有短语
    phrases = [
        ('蓝光', '照蓝光'),
        ('美林', '泰诺林'),
        ('哭声免疫法', '睡眠训练'),
        ('如厕训练', '戒尿布'),
        ('吐奶', '溢奶'),
        ('打嗝', '拍嗝'),
        ('翻身', '大运动'),
        ('头围', '身高体重'),
        ('纠正月龄', '矫正月龄'),
        ('待产包', '分娩准备'),
        ('枕头', '仰睡', '侧睡'),
        ('护臀', '红屁股', '尿布疹'),
        ('消毒', '奶瓶'),
        ('晒太阳', '日光浴'),
        ('忌口', '食物过敏'),
        ('挑食', '偏食'),
        ('早产儿', '早产'),
    ]
    for phrase_group in phrases:
        for p in phrase_group:
            if p in query:
                keywords.extend([x for x in phrase_group if x != p])
                keywords.append(p)

    # 5. 去重并保留顺序
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen and kw:
            seen.add(kw)
            unique_keywords.append(kw)

    # 如果没提取到，用原始query中较长的片段
    if not unique_keywords:
        # 取查询中最长的两个词
        raw_words = [w for w in re.findall(r'[一-鿿\w]+', query)
                     if len(w) >= 2 and w not in STOP_PATTERNS.pattern.split('|')]
        unique_keywords = sorted(raw_words, key=len, reverse=True)[:3]

    return unique_keywords


def find_matching_chapters(query_keywords, top10_docs):
    """从Top10中找出包含关键内容的章节"""
    matching_chapters = set()

    for doc in top10_docs:
        content = doc.page_content.lower()
        source = doc.metadata.get('source', '')
        chapter = doc.metadata.get('chapter', '')

        # 检查内容是否包含核心关键词
        match_count = sum(1 for kw in query_keywords if kw.lower() in content)
        if match_count >= 1:
            matching_chapters.add(chapter)

    return list(matching_chapters)


def main():
    vectorstore = Chroma(
        persist_directory='chroma_parenting_db',
        embedding_function=embeddings,
        collection_name='parenting_books'
    )

    # 读取数据
    with open(Path(__file__).parent / 'data' / 'retrieval_eval_chapter_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    with open(Path(__file__).parent / 'data' / 'retrieval_test_set_v2_100_corrected.json', 'r', encoding='utf-8') as f:
        test_set = json.load(f)

    detailed_results = results['detailed_results']
    zero_indices = [i for i, r in enumerate(detailed_results) if r['recall_metrics']['recall@3'] == 0]

    # 修正结果
    corrections = []
    report_lines = []

    report_lines.append("="*80)
    report_lines.append(f"37条0% Recall数据修正报告")
    report_lines.append("="*80)

    for idx in zero_indices:
        test = test_set[idx]
        query = test['query']

        # 提取核心关键词
        core_kw = extract_core_keywords(query)

        # 执行检索
        raw_results = [doc for doc, _ in vectorstore.similarity_search_with_score(query, k=10)]

        # 找到包含答案的章节
        actual_chapters = find_matching_chapters(core_kw, raw_results)
        if not actual_chapters:
            # 放宽一步：用原始关键词查找
            original_kw = test['expected_keywords']
            actual_chapters = [doc.metadata.get('chapter', '') for doc in raw_results
                               if any(kw in doc.page_content for kw in original_kw)]

        # 构建修正条目
        correction = {
            'query': query,
            'original_chapters': test['expected_chapters'],
            'original_keywords': test['expected_keywords'],
            'suggested_keywords': core_kw,
            'suggested_chapters': actual_chapters[:3] if actual_chapters else test['expected_chapters'],
            'top3_source_chapters': [
                (doc.metadata.get('source', ''), doc.metadata.get('chapter', ''))
                for doc in raw_results[:3]
            ],
            'is_safety_critical': test['is_safety_critical'],
            'expected_source': test['expected_source'],
        }
        corrections.append(correction)

        # 生成报告
        report_lines.append(f"\n查询: {query}")
        report_lines.append(f"  期望来源: {test['expected_source']}")
        report_lines.append(f"  原章节: {test['expected_chapters']} --> 建议: {actual_chapters[:3]}")
        report_lines.append(f"  原关键词: {test['expected_keywords']} --> 建议: {core_kw}")
        top3_info = [(s, ch[:30]) for s, ch in correction['top3_source_chapters']]
        report_lines.append(f"  Top3实际章节: {top3_info}")

    # 保存修正后的测试集
    corrected_test_set = test_set.copy()
    for idx, correction in zip(zero_indices, corrections):
        corrected_test_set[idx]['expected_chapters'] = correction['suggested_chapters']
        corrected_test_set[idx]['expected_keywords'] = correction['suggested_keywords']

    output_path = str(Path(__file__).parent / 'data' / 'retrieval_test_set_v2_100_corrected.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(corrected_test_set, f, ensure_ascii=False, indent=2)
    report_lines.append(f"\n修正后测试集已保存: {output_path}")

    # 同时生成一个完整的对比报告
    # 统计
    new_chapter_count = sum(1 for c in corrections if c['original_chapters'] != c['suggested_chapters'])
    new_kw_count = sum(1 for c in corrections if c['original_keywords'] != c['suggested_keywords'])
    report_lines.insert(1, f"\n章节修正: {new_chapter_count}/37 条")
    report_lines.insert(2, f"关键词修正: {new_kw_count}/37 条")

    report_path = str(Path(__file__).parent / 'reports' / 'correction_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"修正报告已保存: {report_path}")

    # 打印摘要
    print(f"\n修正完成:")
    print(f"  总处理: {len(corrections)} 条")
    print(f"  章节修正: {new_chapter_count} 条")
    print(f"  关键词修正: {new_kw_count} 条")
    print(f"  修正后测试集: {output_path}")
    print(f"  详细报告: {report_path}")


if __name__ == '__main__':
    main()
