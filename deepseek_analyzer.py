"""
DeepSeek API情感分析模块
使用DeepSeek API对电商评论进行情感分析
"""
import requests
import json
import logging
import os
import time
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

class DeepSeekAnalyzer:
    """DeepSeek情感分析器"""
    
    def __init__(self, api_key=None, base_url="https://api.deepseek.com/v1/chat/completions"):
        """
        初始化DeepSeek分析器
        
        Args:
            api_key: DeepSeek API密钥，如果为None则从环境变量DEEPSEEK_API_KEY读取
            base_url: API基础URL
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("需要提供DeepSeek API密钥，可通过参数或环境变量DEEPSEEK_API_KEY设置")
    
    def analyze_sentiment(self, text: str) -> int:
        """
        分析单条评论的情感
        
        Args:
            text: 评论文本
            
        Returns:
            int: 0-负面, 1-中性, 2-正面
        """
        prompt = f"""你是一个专业的电商评论情感分析专家。请仔细分析以下电商评论的情感倾向。

分析标准：
- 0（负面）：评论表达不满、失望、批评、投诉、差评、不推荐、退货等负面情绪
- 1（中性）：评论表达中性态度，如"一般"、"还可以"、"符合预期"、"没什么特别"、"中规中矩"等，无明显情感倾向
- 2（正面）：评论表达满意、好评、推荐、赞扬、惊喜等正面情绪

特别注意：
- 如果评论同时包含正面和负面内容，以整体情感倾向为准
- 如果评论主要是客观描述，无明显情感倾向，归类为中性（1）
- 如果评论表达强烈不满或强烈满意，应明确归类为负面（0）或正面（2）

评论内容：{text}

请仔细分析后，只返回一个数字（0、1或2），不要返回其他任何内容："""
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的电商评论情感分析专家，擅长准确判断评论的情感倾向。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,  # 降低temperature以提高一致性
                "max_tokens": 5  # 减少token数，只需要返回数字
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            # 提取返回的标签
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            
            # 尝试提取数字
            label = None
            for char in content:
                if char.isdigit():
                    label = int(char)
                    break
            
            if label is None or label not in [0, 1, 2]:
                logger.warning(f"无法从DeepSeek返回中提取有效标签: {content}, 默认返回1（中性）")
                return 1
            
            return label
            
        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {str(e)}")
            # 如果API调用失败，返回中性标签
            return 1
    
    def analyze_batch(self, texts: List[str], delay: float = 0.5) -> List[Tuple[str, int]]:
        """
        批量分析评论情感
        
        Args:
            texts: 评论文本列表
            delay: 每次API调用之间的延迟（秒），避免频率限制
            
        Returns:
            List[Tuple[str, int]]: [(文本, 标签), ...] 列表
        """
        results = []
        total = len(texts)
        
        logger.info(f"开始使用DeepSeek分析 {total} 条评论...")
        
        for i, text in enumerate(texts, 1):
            try:
                label = self.analyze_sentiment(text)
                results.append((text, label))
                logger.info(f"进度: {i}/{total} - 标签: {label}")
                
                # 延迟以避免API频率限制
                if i < total:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"分析第 {i} 条评论失败: {str(e)}")
                # 失败时使用中性标签
                results.append((text, 1))
        
        logger.info(f"DeepSeek分析完成，共分析 {len(results)} 条评论")
        return results
    
    def analyze_texts_to_csv(self, texts: List[str], output_path: str, delay: float = 0.5) -> str:
        """
        分析评论并保存为CSV文件
        
        Args:
            texts: 评论文本列表
            output_path: 输出CSV文件路径
            delay: API调用延迟
            
        Returns:
            str: 输出文件路径
        """
        import pandas as pd
        
        # 批量分析
        results = self.analyze_batch(texts, delay)
        
        # 创建DataFrame
        df = pd.DataFrame(results, columns=['text', 'label'])
        
        # 保存为CSV
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"分析结果已保存到: {output_path}")
        
        return output_path

    def generate_product_advice(
        self,
        product_name: str,
        metrics: List[Dict[str, Any]],
        label_distribution: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        结合三模型指标、评论情感分布与具体商品品名，调用 DeepSeek 生成通俗易懂的
        智能分析总结与经营建议（品名贯穿全文、分析细化）。
        """
        if not metrics:
            raise ValueError("缺少模型评估指标，请先完成训练")

        lines = [
            f"目标商品（请全文多次自然提及该名称，所有推断与建议须紧扣该具体商品，勿写成泛化的「本店」「该商品」而不提品名）：{product_name}",
            "",
            "三模型在测试集上的表现（准确率/精确率/召回率/F1）：",
        ]
        for row in metrics:
            model = row.get("Model", row.get("model", ""))
            acc = row.get("Accuracy", row.get("accuracy", ""))
            prec = row.get("Precision", row.get("precision", ""))
            rec = row.get("Recall", row.get("recall", ""))
            f1 = row.get("F1_Score", row.get("F1", row.get("f1", "")))
            lines.append(
                f"- {model}: 准确率={acc}, 精确率={prec}, 召回率={rec}, F1={f1}"
            )

        if label_distribution:
            neg = label_distribution.get("negative", {})
            neu = label_distribution.get("neutral", {})
            pos = label_distribution.get("positive", {})
            total = label_distribution.get("total", 0)
            lines.extend(
                [
                    "",
                    "当前训练所用数据集的情感标签分布（反映评论样本结构）：",
                    f"- 负面: {neg.get('count', 0)} 条, 约 {neg.get('percentage', 0)}%",
                    f"- 中性: {neu.get('count', 0)} 条, 约 {neu.get('percentage', 0)}%",
                    f"- 正面: {pos.get('count', 0)} 条, 约 {pos.get('percentage', 0)}%",
                    f"- 总样本数: {total}",
                ]
            )

        user_prompt = "\n".join(lines)
        user_prompt += (
            "\n\n写作要求（必须遵守）：\n"
            "1）必须同时用到三模型指标、评论情感分布、以及上面给出的「具体商品品名」：请根据品名推断品类/典型使用场景/常见买家关注点，"
            "把数据结论翻译成「和这件商品有什么关系」——例如差评多时要联想到该品类常见痛点（质量、物流、描述不符等），并点名该品名或简称。\n"
            "2）通俗易懂：少用英文缩写与晦涩术语；若必须提到模型或指标，请用短句大白话解释（例如「BERT 更擅长读懂拐弯抹角的话」这类说法即可，不必堆砌定义）。\n"
            "3）分析要细：总结里要点出该品名对应的舆情倾向（偏好评/偏中性/差评压力等），建议里每条尽量落到该商品可改的具体动作，避免空泛套话。\n"
            "\n"
            "请严格按以下两段输出，且顺序不可调换（全文最开头必须是第一段）：\n"
            "第一段：必须以单独一行「【智能分析总结】」作为标题；正文 4～8 句，先结合品名做整体判断，再串起三模型差异（用人话说明谁更稳、差异可能说明什么），"
            "最后把情感分布与「卖这件商品」的启示说清楚。\n"
            "第二段：必须以单独一行「【经营建议】」作为标题；用「1. 2. 3.」分条列出，每条都尽量带上该商品品名或明确指代，写店主/运营能直接照着做的动作；"
            "每条末尾可顺带一句「这和某某模型表现/某类评论多寡有关」但须保持口语化。\n"
            "不要使用 # 号标题或复杂 Markdown；总字数控制在 1100 字以内。"
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是熟悉淘宝/京东等平台的电商运营顾问，擅长把「模型打分」和「评论情感占比」讲成店主能听懂的大白话。"
                        "输出必须紧扣用户给出的具体商品名称做细化分析，避免泛泛而谈。"
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.5,
            "max_tokens": 2048,
        }
        response = requests.post(
            self.base_url, headers=headers, json=data, timeout=120
        )
        response.raise_for_status()
        result = response.json()
        content = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if not content:
            raise ValueError("DeepSeek 未返回有效建议内容")
        return content

    def generate_marketing_copy(
        self,
        product_name: str,
        business_advice: str,
        positive_reviews: list
    ) -> str:
        """
        根据商品名、经营建议和好评列表生成营销文案
        
        Args:
            product_name: 商品名称
            business_advice: 经营建议
            positive_reviews: 好评列表
            
        Returns:
            str: 生成的营销文案
        """
        # 处理好评列表
        if isinstance(positive_reviews, str):
            # 如果是字符串，按行分割
            reviews_list = [line.strip() for line in positive_reviews.split('\n') if line.strip()]
        else:
            reviews_list = positive_reviews
        
        # 取前5条好评
        sample_reviews = reviews_list[:5]
        
        prompt = f"""你是一个专业的电商文案策划师，擅长根据商品信息、经营建议和用户好评撰写吸引人的营销文案。

商品信息：
- 商品名称：{product_name}

经营建议：
{business_advice}

用户好评（选摘）：
"""
        for i, review in enumerate(sample_reviews, 1):
            prompt += f"{i}. {review}\n"
        
        prompt += """

请根据以上信息，撰写一篇吸引人的营销文案，要求：
1. 标题要吸引人，突出商品卖点
2. 正文内容要具体生动，结合用户好评
3. 语言风格要亲切自然，适合电商平台
4. 字数控制在300-500字
5. 要体现出商品的优势和价值

请直接返回营销文案，不要其他说明内容。"""

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的电商文案策划师，擅长撰写吸引人的营销文案。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            
            if not content:
                # 如果API没有返回内容，提供一个默认文案
                return self._generate_default_marketing_copy(product_name, sample_reviews)
            
            return content
            
        except Exception as e:
            logger.error(f"生成营销文案失败: {str(e)}")
            # 发生错误时返回默认文案
            return self._generate_default_marketing_copy(product_name, sample_reviews)
    
    def _generate_default_marketing_copy(self, product_name: str, reviews: list) -> str:
        """
        生成默认营销文案（当API调用失败时使用）
        
        Args:
            product_name: 商品名称
            reviews: 好评列表
            
        Returns:
            str: 默认营销文案
        """
        # 取前3条好评
        sample_reviews = reviews[:3]
        
        copy = f"""🎁 {product_name} - 用户好评推荐！

🌟 商品亮点：
- 品质可靠，用户好评如潮
- 性价比高，值得信赖
- 服务贴心，购物无忧

💬 用户真实评价：
"""
        for i, review in enumerate(sample_reviews, 1):
            copy += f"{i}. {review}\n"
        
        copy += f"""
🔥 现在购买，享受优质购物体验！
{product_name}，您值得拥有！

---
*注：更多详情请查看商品页面*
"""
        return copy

