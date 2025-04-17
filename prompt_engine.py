import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import random
import logging
from collections import Counter
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PromptGenerationEngine:
    """提示词生成引擎主类"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 data_path: str = "data/prompts_dataset.json"):
        """
        初始化提示词生成引擎

        Args:
            model_name: 用于文本嵌入的模型名称
            data_path: 案例库数据路径
        """
        self.data_path = data_path
        self.case_library = CaseLibrary(data_path)
        self.prompt_analyzer = PromptAnalyzer()
        self.semantic_indexer = SemanticIndexer(model_name)
        self.adaptive_merger = AdaptiveMerger()
        self.feedback_loop = FeedbackLoop(self.case_library, self.semantic_indexer)

        # 如果数据文件存在，则加载数据
        if os.path.exists(data_path):
            self.case_library.load_cases()
            self.semantic_indexer.build_index(self.case_library.get_cases())
            logger.info(f"已加载 {len(self.case_library.cases)} 条案例数据")
        else:
            logger.warning(f"数据文件 {data_path} 不存在，将创建新的案例库")
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            self.case_library.save_cases()

    def add_case(self, content: str, paragraph_prompt: str, content_prompt: str) -> None:
        """
        添加新案例到案例库

        Args:
            content: 文章内容
            paragraph_prompt: 段落句式提示词
            content_prompt: 内容生成提示词
        """
        case_id = self.case_library.add_case(content, paragraph_prompt, content_prompt)
        self.semantic_indexer.add_case_embedding(case_id, content, paragraph_prompt, content_prompt)
        logger.info(f"已添加新案例，ID: {case_id}")

    def generate_prompts(self, content: str) -> Dict[str, str]:
        """
        为输入文本生成提示词

        Args:
            content: 输入的文章内容

        Returns:
            包含生成的提示词的字典
        """
        # 1. 从案例库中检索相似案例
        similar_cases = self.semantic_indexer.search_similar_cases(content, top_k=3)

        # 2. 分析输入文本特征
        content_features = self.prompt_analyzer.extract_content_features(content)

        # 3. 从相似案例中提取提示词模式
        paragraph_patterns = self.prompt_analyzer.extract_paragraph_patterns(
            [case["paragraph_prompt"] for case in similar_cases])
        content_patterns = self.prompt_analyzer.extract_content_patterns(
            [case["content_prompt"] for case in similar_cases])

        # 4. 融合生成最终提示词
        paragraph_prompt = self.adaptive_merger.merge_paragraph_prompt(content_features, paragraph_patterns,
                                                                       similar_cases)
        content_prompt = self.adaptive_merger.merge_content_prompt(content_features, content_patterns, similar_cases)

        # 5. 使用大模型API优化提示词
        optimized_prompts = self.optimize_prompts_with_llm(content, paragraph_prompt, content_prompt)

        return {
            "正文内容": content,
            "段落句式": optimized_prompts["段落句式"],
            "生成要求": optimized_prompts["生成要求"]
        }

    def optimize_prompts_with_llm(self, content: str, paragraph_prompt: str, content_prompt: str) -> Dict[str, str]:
        """
        使用大模型API优化生成的提示词

        Args:
            content: 输入的文章内容
            paragraph_prompt: 生成的段落句式提示词
            content_prompt: 生成的内容要求提示词

        Returns:
            优化后的提示词字典
        """
        # 获取API密钥
        api_key = os.getenv("LLM_API_KEY")
        api_url = os.getenv("LLM_API_URL")

        if not api_key or not api_url:
            logger.warning("未配置大模型API，跳过优化步骤")
            return {"段落句式": paragraph_prompt, "生成要求": content_prompt}

        try:
            # 构造提示
            system_prompt = """你是一个提示词优化专家。请根据原文内容，优化以下两类提示词，使其更准确地捕捉原文的特征：
                                1. 段落句式提示词：用于指导文章的句式结构、字数及分段特点
                                2. 内容生成提示词：用于指导文章的内容生成
                                请提供优化后的两类提示词，格式为JSON：{"段落句式":"优化后的段落句式提示词", "生成要求":"优化后的内容生成提示词"}"""

            user_prompt = f"""原文内容：{content}
                                当前段落句式提示词：{paragraph_prompt}
                                当前内容生成提示词：{content_prompt}
                                请优化这两类提示词，使其更准确地捕捉原文特征。"""

            # 调用API
            response = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "gpt-4",  # 根据实际使用的模型调整
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "response_format": {"type": "json_object"}
                }
            )

            if response.status_code == 200:
                result = response.json()
                optimized_prompts = json.loads(result["choices"][0]["message"]["content"])
                logger.info("成功使用大模型优化提示词")
                return optimized_prompts
            else:
                logger.error(f"API调用失败: {response.status_code} - {response.text}")
                return {"段落句式": paragraph_prompt, "生成要求": content_prompt}

        except Exception as e:
            logger.error(f"优化提示词时出错: {str(e)}")
            return {"段落句式": paragraph_prompt, "生成要求": content_prompt}

    def process_feedback(self, content: str, original_prompts: Dict[str, str],
                         adjusted_prompts: Dict[str, str], success_rating: int) -> None:
        """
        处理用户反馈并更新模型

        Args:
            content: 原始文章内容
            original_prompts: 原始生成的提示词
            adjusted_prompts: 用户调整后的提示词
            success_rating: 成功评分(1-5)
        """
        self.feedback_loop.process_feedback(
            content,
            original_prompts.get("段落句式", ""),
            original_prompts.get("生成要求", ""),
            adjusted_prompts.get("段落句式", ""),
            adjusted_prompts.get("生成要求", ""),
            success_rating
        )
        logger.info("已处理用户反馈并更新模型")


class CaseLibrary:
    """结构化案例库"""

    def __init__(self, data_path: str):
        """
        初始化案例库

        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.cases = {}
        self.next_id = 1

    def load_cases(self) -> None:
        """从文件加载案例数据"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.cases = data.get("cases", {})
                self.next_id = data.get("next_id", 1)
        except Exception as e:
            logger.error(f"加载案例数据失败: {str(e)}")
            self.cases = {}
            self.next_id = 1

    def save_cases(self) -> None:
        """保存案例数据到文件"""
        try:
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump({"cases": self.cases, "next_id": self.next_id}, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存 {len(self.cases)} 条案例数据")
        except Exception as e:
            logger.error(f"保存案例数据失败: {str(e)}")

    def add_case(self, content: str, paragraph_prompt: str, content_prompt: str, style_tag: str = None) -> str:
        """
        添加新案例

        Args:
            content: 文章内容
            paragraph_prompt: 段落句式提示词
            content_prompt: 内容生成提示词
            style_tag: 风格标签(可选)

        Returns:
            新案例的ID
        """
        case_id = str(self.next_id)
        self.next_id += 1

        # 自动生成风格标签(如果未提供)
        if not style_tag:
            style_tag = self.auto_generate_style_tag(content)

        self.cases[case_id] = {
            "content": content,
            "paragraph_prompt": paragraph_prompt,
            "content_prompt": content_prompt,
            "style_tag": style_tag,
            "feedback_score": 0,
            "usage_count": 0
        }

        self.save_cases()
        return case_id

    def auto_generate_style_tag(self, content: str) -> str:
        """
        自动生成文本风格标签

        Args:
            content: 文章内容

        Returns:
            风格标签字符串
        """
        # 这里用简单规则识别风格，实际应用中可以使用更复杂的分类器
        style_markers = {
            "新闻报道": ["报道", "记者", "消息", "新华社", "讯"],
            "学术论文": ["研究", "分析", "理论", "实验", "数据", "图表", "参考文献"],
            "营销文案": ["产品", "服务", "优惠", "限时", "折扣", "品牌", "体验"],
            "技术文档": ["功能", "特性", "安装", "配置", "使用方法", "步骤", "注意事项"],
            "故事叙述": ["故事", "情节", "人物", "场景", "对话"],
            "官方公告": ["公告", "通知", "决定", "规定", "实施"]
        }

        # 计算各类型标记在内容中出现的次数
        scores = {}
        for style, markers in style_markers.items():
            score = sum(content.count(marker) for marker in markers)
            scores[style] = score

        # 找出得分最高的风格
        if sum(scores.values()) > 0:
            max_style = max(scores, key=scores.get)
            return max_style
        else:
            return "通用文本"

    def get_cases(self) -> Dict[str, Dict[str, Any]]:
        """获取所有案例"""
        return self.cases

    def get_case(self, case_id: str) -> Dict[str, Any]:
        """获取指定ID的案例"""
        return self.cases.get(case_id, {})

    def update_case_feedback(self, case_id: str, feedback_score: int) -> None:
        """更新案例的反馈评分"""
        if case_id in self.cases:
            self.cases[case_id]["feedback_score"] = feedback_score
            self.cases[case_id]["usage_count"] += 1
            self.save_cases()


class PromptAnalyzer:
    """提示词解构模块"""

    def extract_content_features(self, content: str) -> Dict[str, Any]:
        """
        提取文本内容的特征

        Args:
            content: 文章内容

        Returns:
            文本特征字典
        """
        # 分段
        paragraphs = content.split('\n\n')
        paragraphs = [p for p in paragraphs if p.strip()]

        # 分句
        sentences = re.split(r'[。！？\.!?]', content)
        sentences = [s for s in sentences if s.strip()]

        # 计算基本特征
        features = {
            "total_length": len(content),
            "paragraph_count": len(paragraphs),
            "avg_paragraph_length": sum(len(p) for p in paragraphs) / max(1, len(paragraphs)),
            "sentence_count": len(sentences),
            "avg_sentence_length": sum(len(s) for s in sentences) / max(1, len(sentences)),
            "keywords": self.extract_keywords(content),
            "has_question": any('?' in p or '？' in p for p in paragraphs),
            "has_numbers": bool(re.search(r'\d+', content)),
            "has_quotes": bool(re.search(r'[""].*?[""]', content))
        }

        # 计算段落位置特征
        if len(paragraphs) >= 3:
            features["intro_length"] = len(paragraphs[0])
            features["conclusion_length"] = len(paragraphs[-1])
            features["body_avg_length"] = sum(len(p) for p in paragraphs[1:-1]) / len(paragraphs[1:-1])

        # 分析句式特征
        features["sentence_types"] = self.analyze_sentence_types(sentences)

        return features

    def extract_keywords(self, content: str, top_n: int = 10) -> List[str]:
        """
        提取文本关键词

        Args:
            content: 文章内容
            top_n: 返回前N个关键词

        Returns:
            关键词列表
        """
        # 简单的词频统计(实际应用中可以用TF-IDF或TextRank)
        # 1. 移除常见停用词
        stopwords = set(
            ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到',
             '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '这个', '那个'])

        # 2. 粗略分词(中文分词实际应用中应使用jieba等库)
        words = []
        for i in range(len(content) - 1):
            # 简单地取双字词
            if content[i:i + 2] not in stopwords and re.match(r'[\u4e00-\u9fa5]{2}', content[i:i + 2]):
                words.append(content[i:i + 2])

        # 3. 统计词频
        word_count = Counter(words)

        # 4. 返回高频词
        return [word for word, _ in word_count.most_common(top_n)]

    def analyze_sentence_types(self, sentences: List[str]) -> Dict[str, float]:
        """
        分析句式类型分布

        Args:
            sentences: 句子列表

        Returns:
            句式类型及其比例的字典
        """
        total = len(sentences)
        if total == 0:
            return {}

        types = {
            "question": sum(1 for s in sentences if '?' in s or '？' in s) / total,
            "exclamation": sum(1 for s in sentences if '!' in s or '！' in s) / total,
            "quote": sum(1 for s in sentences if '"' in s or '"' in s or '"' in s) / total,
            "enumeration": sum(1 for s in sentences if '、' in s or '，' in s and len(s) > 20) / total,
        }

        return types

    def extract_paragraph_patterns(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        从段落句式提示词中提取模式

        Args:
            prompts: 段落句式提示词列表

        Returns:
            段落模式列表
        """
        patterns = []

        for prompt in prompts:
            # 提取段落数量规则
            paragraph_count = re.search(r'(\d+)[段|自然段|部分]', prompt)

            # 提取字数规则
            word_count = re.search(r'(\d+)[字|词]', prompt)

            # 提取段落比例规则
            intro_ratio = re.search(r'开头[约|大约|占]?(\d+)[%|％]', prompt)
            conclusion_ratio = re.search(r'结尾[约|大约|占]?(\d+)[%|％]', prompt)

            # 提取句式规则
            has_question = bool(re.search(r'[设问|问句|疑问句]', prompt))
            has_exclamation = bool(re.search(r'[感叹|感叹句]', prompt))
            has_quote = bool(re.search(r'[引用|引述|引言]', prompt))

            pattern = {
                "paragraph_count": int(paragraph_count.group(1)) if paragraph_count else None,
                "word_count": int(word_count.group(1)) if word_count else None,
                "intro_ratio": float(intro_ratio.group(1)) / 100 if intro_ratio else None,
                "conclusion_ratio": float(conclusion_ratio.group(1)) / 100 if conclusion_ratio else None,
                "has_question": has_question,
                "has_exclamation": has_exclamation,
                "has_quote": has_quote,
                "original_prompt": prompt  # 存储原始提示词以备参考
            }

            patterns.append(pattern)

        return patterns

    def extract_content_patterns(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        从内容生成提示词中提取模式

        Args:
            prompts: 内容生成提示词列表

        Returns:
            内容模式列表
        """
        patterns = []

        for prompt in prompts:
            # 提取关键词数量
            keyword_count = re.search(r'包含(\d+)个关键词', prompt)

            # 提取数据引用要求
            data_refs = re.search(r'包含(\d+)个[数据|数字|统计]', prompt)

            # 提取论证方式
            argument_style = None
            if re.search(r'对比论证', prompt):
                argument_style = "comparison"
            elif re.search(r'举例论证', prompt):
                argument_style = "example"
            elif re.search(r'引用论证', prompt):
                argument_style = "quote"

            # 提取语气风格
            tone = None
            if re.search(r'[正式|专业|学术]', prompt):
                tone = "formal"
            elif re.search(r'[口语化|轻松|随意]', prompt):
                tone = "casual"
            elif re.search(r'[热情|积极|乐观]', prompt):
                tone = "enthusiastic"

            pattern = {
                "keyword_count": int(keyword_count.group(1)) if keyword_count else None,
                "data_references": int(data_refs.group(1)) if data_refs else None,
                "argument_style": argument_style,
                "tone": tone,
                "original_prompt": prompt  # 存储原始提示词以备参考
            }

            patterns.append(pattern)

        return patterns


class SemanticIndexer:
    """语义索引模块"""

    def __init__(self, model_name: str):
        """
        初始化语义索引器

        Args:
            model_name: 用于文本嵌入的模型名称
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"成功加载语义模型: {model_name}")
        except Exception as e:
            logger.error(f"加载语义模型失败: {str(e)}")
            # 如果加载失败，使用随机嵌入作为后备
            self.model = None

        self.content_embeddings = {}
        self.prompt_embeddings = {}

    def build_index(self, cases: Dict[str, Dict[str, Any]]) -> None:
        """
        为案例库构建语义索引

        Args:
            cases: 案例库字典
        """
        for case_id, case in cases.items():
            self.add_case_embedding(case_id, case["content"], case["paragraph_prompt"], case["content_prompt"])

        logger.info(f"已为 {len(cases)} 个案例构建语义索引")

    def add_case_embedding(self, case_id: str, content: str, paragraph_prompt: str, content_prompt: str) -> None:
        """
        添加单个案例的嵌入向量

        Args:
            case_id: 案例ID
            content: 文章内容
            paragraph_prompt: 段落句式提示词
            content_prompt: 内容生成提示词
        """
        if self.model:
            try:
                # 计算文本内容的嵌入向量
                content_embedding = self.model.encode(content)
                self.content_embeddings[case_id] = content_embedding

                # 计算提示词的嵌入向量(连接两种提示词)
                prompt_text = paragraph_prompt + " " + content_prompt
                prompt_embedding = self.model.encode(prompt_text)
                self.prompt_embeddings[case_id] = prompt_embedding
            except Exception as e:
                logger.error(f"计算案例 {case_id} 的嵌入向量失败: {str(e)}")
                # 使用随机向量作为后备
                self._add_random_embedding(case_id)
        else:
            # 如果模型加载失败，使用随机向量
            self._add_random_embedding(case_id)

    def _add_random_embedding(self, case_id: str) -> None:
        """使用随机向量作为嵌入(后备方案)"""
        dim = 384  # 标准维度
        self.content_embeddings[case_id] = np.random.randn(dim)
        self.prompt_embeddings[case_id] = np.random.randn(dim)

    def search_similar_cases(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        搜索与查询文本最相似的案例

        Args:
            query_text: 查询文本
            top_k: 返回前K个最相似案例

        Returns:
            相似案例列表
        """
        if not self.content_embeddings:
            logger.warning("语义索引为空，无法搜索相似案例")
            return []

        if self.model:
            try:
                # 计算查询文本的嵌入向量
                query_embedding = self.model.encode(query_text)

                # 计算与所有案例的相似度
                similarities = {}
                for case_id, embedding in self.content_embeddings.items():
                    similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                    similarities[case_id] = similarity

                # 排序并返回前K个最相似的案例
                top_cases = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

                from_case_library = self.get_cases_from_library(top_cases)
                logger.info(f"找到 {len(from_case_library)} 个相似案例")
                return from_case_library

            except Exception as e:
                logger.error(f"搜索相似案例失败: {str(e)}")
                return []
        else:
            # 如果模型加载失败，随机返回案例
            case_ids = list(self.content_embeddings.keys())
            if len(case_ids) <= top_k:
                selected_ids = case_ids
            else:
                selected_ids = random.sample(case_ids, top_k)

            return self.get_cases_from_library([(case_id, 0.5) for case_id in selected_ids])

    def get_cases_from_library(self, case_tuples: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """
        从案例库获取案例详情

        Args:
            case_tuples: (case_id, similarity)元组列表

        Returns:
            案例详情列表
        """
        # 这里假设案例库是通过外部传入的，实际实现时可能需要调整
        case_library_instance = CaseLibrary(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/prompts_dataset.json"))
        case_library_instance.load_cases()

        cases = []
        for case_id, similarity in case_tuples:
            case = case_library_instance.get_case(case_id)
            if case:
                case_with_similarity = case.copy()
                case_with_similarity["similarity"] = similarity
                case_with_similarity["id"] = case_id
                cases.append(case_with_similarity)

        return cases


class AdaptiveMerger:
    """自适应融合器"""

    def merge_paragraph_prompt(self, content_features: Dict[str, Any], paragraph_patterns: List[Dict[str, Any]],
                               similar_cases: List[Dict[str, Any]]) -> str:
        """
        融合生成段落句式提示词

        Args:
            content_features: 内容特征
            paragraph_patterns: 从相似案例中提取的段落模式
            similar_cases: 相似案例列表

        Returns:
            生成的段落句式提示词
        """
        # 如果没有相似案例或模式，生成基本提示词
        if not paragraph_patterns:
            return self._generate_basic_paragraph_prompt(content_features)

        # 选择相似度最高的模式作为基础
        best_pattern = None
        best_similarity = -1

        for i, case in enumerate(similar_cases):
            if i < len(paragraph_patterns) and case["similarity"] > best_similarity:
                best_pattern = paragraph_patterns[i]
                best_similarity = case["similarity"]

        if not best_pattern or best_similarity < 0.5:  # 相似度阈值
            return self._generate_basic_paragraph_prompt(content_features)

        # 使用最匹配的模式生成提示词
        prompt_parts = []

        # 段落数量
        if best_pattern["paragraph_count"]:
            prompt_parts.append(f"文章应包含{best_pattern['paragraph_count']}个自然段")
        else:
            # 根据原文特征确定段落数
            para_count = content_features["paragraph_count"]
            prompt_parts.append(f"文章应包含{para_count}个自然段")

        # 字数要求
        if best_pattern["word_count"]:
            prompt_parts.append(f"总字数约为{best_pattern['word_count']}字")
        else:
            # 根据原文特征确定字数
            word_count = content_features["total_length"]
            prompt_parts.append(f"总字数约为{word_count}字")

        # 段落结构比例
        if best_pattern["intro_ratio"]:
            intro_percent = int(best_pattern["intro_ratio"] * 100)
            prompt_parts.append(f"开头部分约占全文的{intro_percent}%")

        if best_pattern["conclusion_ratio"]:
            conclusion_percent = int(best_pattern["conclusion_ratio"] * 100)
            prompt_parts.append(f"结尾部分约占全文的{conclusion_percent}%")

        # 句式特点
        sentence_style_parts = []
        if best_pattern["has_question"] or content_features["sentence_types"].get("question", 0) > 0.1:
            sentence_style_parts.append("适当使用设问句")

        if best_pattern["has_exclamation"] or content_features["sentence_types"].get("exclamation", 0) > 0.1:
            sentence_style_parts.append("可使用感叹句增强表达力")

        if best_pattern["has_quote"] or content_features["sentence_types"].get("quote", 0) > 0.1:
            sentence_style_parts.append("适当引用观点或数据")

        if sentence_style_parts:
            prompt_parts.append("句式要求：" + "，".join(sentence_style_parts))

        # 段落长短变化
        if content_features.get("avg_paragraph_length", 0) > 200:
            prompt_parts.append("段落篇幅较长，每段可包含多个完整论点")
        else:
            prompt_parts.append("段落长度适中，重点突出")

        # 拼接最终提示词
        return "。".join(prompt_parts) + "。"

    def _generate_basic_paragraph_prompt(self, content_features: Dict[str, Any]) -> str:
        """
        根据内容特征生成基本段落句式提示词

        Args:
            content_features: 内容特征

        Returns:
            基本段落句式提示词
        """
        prompt_parts = []

        # 段落数量
        para_count = content_features["paragraph_count"]
        prompt_parts.append(f"文章应包含{para_count}个自然段")

        # 总字数
        word_count = content_features["total_length"]
        prompt_parts.append(f"总字数约为{word_count}字")

        # 句式特点
        sentence_types = content_features["sentence_types"]
        style_parts = []

        if sentence_types.get("question", 0) > 0.1:
            style_parts.append("适当使用设问句")

        if sentence_types.get("exclamation", 0) > 0.1:
            style_parts.append("可使用感叹句增强表达力")

        if sentence_types.get("quote", 0) > 0.1:
            style_parts.append("适当引用观点或数据")

        if style_parts:
            prompt_parts.append("句式要求：" + "，".join(style_parts))

        # 段落长度
        avg_para_len = content_features.get("avg_paragraph_length", 0)
        if avg_para_len > 200:
            prompt_parts.append("段落篇幅较长，每段包含多个完整论点")
        elif avg_para_len < 100:
            prompt_parts.append("段落简洁，一段一个要点")
        else:
            prompt_parts.append("段落长度适中，层次分明")

        return "。".join(prompt_parts) + "。"

    def merge_content_prompt(self, content_features: Dict[str, Any], content_patterns: List[Dict[str, Any]],
                          similar_cases: List[Dict[str, Any]]) -> str:
        """
        融合生成内容生成提示词

        Args:
            content_features: 内容特征
            content_patterns: 从相似案例中提取的内容模式
            similar_cases: 相似案例列表

        Returns:
            生成的内容生成提示词
        """
        # 如果没有相似案例或模式，生成基本提示词
        if not content_patterns:
            return self._generate_basic_content_prompt(content_features)

        # 选择相似度最高的模式作为基础
        best_pattern = None
        best_similarity = -1

        for i, case in enumerate(similar_cases):
            if i < len(content_patterns) and case["similarity"] > best_similarity:
                best_pattern = content_patterns[i]
                best_similarity = case["similarity"]

        if not best_pattern or best_similarity < 0.5:  # 相似度阈值
            return self._generate_basic_content_prompt(content_features)

        # 使用最匹配的模式生成提示词
        prompt_parts = []

        # 主题和关键词
        keywords = content_features["keywords"][:5]  # 取前5个关键词
        prompt_parts.append(f"文章应围绕以下关键词展开：{', '.join(keywords)}")

        # 关键词数量要求
        if best_pattern["keyword_count"]:
            prompt_parts.append(f"至少包含{best_pattern['keyword_count']}个相关关键词")

        # 数据引用要求
        if best_pattern["data_references"]:
            prompt_parts.append(f"文中应包含至少{best_pattern['data_references']}处数据引用或统计信息")
        elif content_features["has_numbers"]:
            prompt_parts.append("文中应包含适当的数据支持论点")

        # 论证方式
        if best_pattern["argument_style"]:
            if best_pattern["argument_style"] == "comparison":
                prompt_parts.append("使用对比论证方式，展示不同观点或方法的异同")
            elif best_pattern["argument_style"] == "example":
                prompt_parts.append("使用举例论证方式，通过具体案例支持观点")
            elif best_pattern["argument_style"] == "quote":
                prompt_parts.append("使用引用论证方式，引述权威观点或研究成果")

        # 语气风格
        if best_pattern["tone"]:
            if best_pattern["tone"] == "formal":
                prompt_parts.append("语言风格应正式、专业，避免口语化表达")
            elif best_pattern["tone"] == "casual":
                prompt_parts.append("语言风格可轻松自然，贴近读者")
            elif best_pattern["tone"] == "enthusiastic":
                prompt_parts.append("语言风格应积极热情，富有感染力")

        # 引用处理
        if content_features["has_quotes"]:
            prompt_parts.append("适当引用相关观点或言论，增强文章说服力")

        # 拼接最终提示词
        return "。".join(prompt_parts) + "。"

    def _generate_basic_content_prompt(self, content_features: Dict[str, Any]) -> str:
        """
        根据内容特征生成基本内容生成提示词

        Args:
            content_features: 内容特征

        Returns:
            基本内容生成提示词
        """
        prompt_parts = []

        # 关键词
        keywords = content_features["keywords"][:5]  # 取前5个关键词
        prompt_parts.append(f"文章应围绕以下关键词展开：{', '.join(keywords)}")

        # 语气风格推断
        if content_features.get("avg_sentence_length", 0) > 30:
            prompt_parts.append("使用正式、专业的语言风格")
        else:
            prompt_parts.append("使用清晰、易懂的语言风格")

        # 数据和引用
        if content_features["has_numbers"]:
            prompt_parts.append("文中应包含适当的数据支持论点")

        if content_features["has_quotes"]:
            prompt_parts.append("适当引用相关观点或言论")

        # 论证方式
        if content_features["has_question"]:
            prompt_parts.append("可采用设问后回答的论证方式")
        else:
            prompt_parts.append("采用清晰的论述方式展开内容")

        return "。".join(prompt_parts) + "。"


class FeedbackLoop:
    """在线反馈环"""

    def __init__(self, case_library: CaseLibrary, semantic_indexer: SemanticIndexer):
        """
        初始化反馈环

        Args:
            case_library: 案例库实例
            semantic_indexer: 语义索引器实例
        """
        self.case_library = case_library
        self.semantic_indexer = semantic_indexer

    def process_feedback(self, content: str, original_paragraph_prompt: str, original_content_prompt: str,
                        adjusted_paragraph_prompt: str, adjusted_content_prompt: str, success_rating: int) -> None:
        """
        处理用户反馈并更新模型

        Args:
            content: 原始文章内容
            original_paragraph_prompt: 原始段落句式提示词
            original_content_prompt: 原始内容生成提示词
            adjusted_paragraph_prompt: 用户调整后的段落句式提示词
            adjusted_content_prompt: 用户调整后的内容生成提示词
            success_rating: 成功评分(1-5)
        """
        # 如果评分较高，直接将调整后的提示词添加到案例库
        if success_rating >= 4:
            case_id = self.case_library.add_case(content, adjusted_paragraph_prompt, adjusted_content_prompt)
            self.semantic_indexer.add_case_embedding(case_id, content, adjusted_paragraph_prompt, adjusted_content_prompt)
            logger.info(f"已添加高评分案例，ID: {case_id}")
            return

        # 如果评分较低但用户进行了调整，添加调整后的版本
        if adjusted_paragraph_prompt != original_paragraph_prompt or adjusted_content_prompt != original_content_prompt:
            case_id = self.case_library.add_case(content, adjusted_paragraph_prompt, adjusted_content_prompt)
            self.semantic_indexer.add_case_embedding(case_id, content, adjusted_paragraph_prompt, adjusted_content_prompt)
            logger.info(f"已添加用户调整后的案例，ID: {case_id}")

        # 更新相似案例的评分
        similar_cases = self.semantic_indexer.search_similar_cases(content, top_k=3)
        for case in similar_cases:
            case_id = case["id"]
            # 如果评分较低，降低相似案例的权重
            if success_rating <= 2:
                self.case_library.update_case_feedback(case_id, max(0, case.get("feedback_score", 0) - 1))
                logger.info(f"已降低相似案例 {case_id} 的权重")


# 主函数示例
def main():
    # 初始化引擎
    engine = PromptGenerationEngine()

    # 示例文本
    sample_text = """
    《中国信托业ESG报告编制指南》课题研究通过对理论框架模型、国家政策和行业特殊性、优秀企业对标研究，构建科学合理的技术路线，编制一本符合国际ESG规范兼具中国ESG特色的信托行业的ESG报告编制指南。
	1.国内外ESG理论模型对标
    通过研究明晟（MSCI）ESG评级、标普道琼斯可持续发展指数（DJSI）、央企ESG先锋指数、信托行业CSR指标体系等国内外ESG理论模型及指标体系框架，明确国内外ESG理论模型及信托行业信息披露的规范和要求。
	2.国家政策和信托行业特殊性对标
    在国内外ESG理论模型对标的基础上，结合国家对企业ESG建设政策及中央企业、信托行业ESG发展趋势两个领域进行扫描，充分考虑信托行业的ESG工作的特殊性，形成信托行业ESG管理模型。
	3.优秀企业ESG建设对标
    通过政府及企业官网、企业ESG报告、社会责任报告等多个渠道获取信息，针对优秀外资、民营企业ESG建设；央企上市公司ESG建设和信托行业上市公司ESG建设三个层级开展对标分析，梳理ESG建设优秀经验，构建信托行业ESG管理指标体系。"
    """

    # 生成提示词
    result = engine.generate_prompts(sample_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 模拟用户反馈
    engine.process_feedback(
        content=sample_text,
        original_prompts={
            "段落句式": result["段落句式"],
            "生成要求": result["生成要求"]
        },
        adjusted_prompts={
            "段落句式": "1、二级标题从工作推进计划、研究方法/调研方法/策划思路/XX等等、与第三方合作的规划三个角度出发命名（但是二级标题的命名不要直接照抄这三个角度的名称），二级标题下的相应内容也应该从标题的角度出发。需要根据具体场景来确定二级标题的命名。2、工作推进计划包括时间安排、人员分工、工作进程等等；该部分标题以“1. 2. 3. ”作为序号，命名由具体场景生成。3、生成的内容不需要举例。生成的字数控制在600字即可。",
            "生成要求": "4、在讲述时间安排这个内容时，需要简要描述每个时间段及主要任务，且内容需要分段描述，如：“第一阶段：xxxx年x月至x月，主要任务包括xxx。第二阶段：xxxx年x月至x月，主要任务包括xxx。”。不同阶段需要分行进行说明。5、每个二级标题下的正文内容，均是编制该报告的执行路线的一部分，其中内容不要重复。如果前面的正文内容已经提及了有关时间上的部署方案，后面就不要再有提及，避免重复。6、生成的时间节点需要超过署名日期的时间，因为本部分需要生成的内容属于对未来的安排。"
        },
        success_rating=4
    )


if __name__ == "__main__":
    main()