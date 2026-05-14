import pandas as pd
import numpy as np
import jieba
import re
from sklearn.model_selection import train_test_split
import logging
from config import DATA_CONFIG

logger = logging.getLogger(__name__)

def preprocess_text(text):
    """文本预处理（优化版）"""
    if pd.isna(text) or text is None:
        return ""
    
    # 转换为字符串
    text = str(text)
    
    # 保留中文、英文、数字和基本标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、；：]', '', text)
    
    # 规范化空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空白
    text = text.strip()
    
    return text

def tokenize_chinese_text(text):
    """中文分词"""
    return ' '.join(jieba.cut(text))

def generate_sample_data(num_samples=3000):
    """生成模拟的电商评论数据，设计用于体现三个模型的差异性
    
    设计思路：
    1. BERT擅长理解：包含复杂语义、上下文依赖、反讽等
    2. BiLSTM擅长序列：包含长文本、序列模式、重复结构
    3. SVM依赖特征：包含关键词明显、特征清晰的文本
    """
    logger.info("生成模拟电商评论数据（优化版，体现模型差异性）...")
    
    # 正面评论 - 设计不同类型以体现模型差异
    # 简单关键词型（SVM容易识别）- 大幅扩展
    simple_positive = [
        # 质量相关
        "质量很好", "质量不错", "质量上乘", "质量优秀", "质量过硬", "品质优良", "品质保证", "正品保证",
        # 满意度相关
        "非常满意", "很满意", "超级满意", "十分满意", "特别满意", "相当满意", "极其满意",
        # 推荐相关
        "推荐购买", "强烈推荐", "值得推荐", "必须推荐", "五星推荐", "真心推荐", "强烈建议",
        # 评价相关
        "五星好评", "满分好评", "好评如潮", "赞不绝口", "赞赞赞", "好评", "点赞",
        # 价值相关
        "物超所值", "性价比高", "性价比超高", "物美价廉", "价格实惠", "价格合理", "价格公道",
        # 服务相关
        "服务好", "服务优质", "服务周到", "服务贴心", "服务热情", "服务态度好", "客服专业",
        # 物流相关
        "发货快", "物流快", "快递快", "配送快", "到货快", "发货迅速", "物流迅速", "包装精美",
        # 功能相关
        "功能强大", "功能齐全", "功能完善", "使用方便", "操作简单", "效果显著", "效果很好",
        # 外观相关
        "外观漂亮", "外观精美", "外观时尚", "设计精美", "做工精细", "款式好看", "颜值高",
        # 其他正面
        "值得购买", "会回购", "再次购买", "继续关注", "超出预期", "惊喜", "完美", "很棒"
    ]
    
    # 复杂语义型（BERT擅长理解）- 扩展
    complex_positive = [
        "虽然价格稍高，但考虑到品质和售后服务，这个价位还是很合理的",
        "不是最好的，但在这个价位上已经超出预期了，值得推荐",
        "如果满分是10分，我会给9分，扣1分是因为包装可以再改进",
        "整体来说不错，细节处理得很好，虽然有些小瑕疵但不影响使用",
        "刚开始还有点担心，但收到货后发现质量比想象中好很多，很满意",
        "价格虽然不便宜，但一分价钱一分货，这个质量对得起这个价格",
        "用了一段时间才来评价，确实不错，没有让我失望，值得这个价格",
        "商品和描述基本一致，没有夸大宣传，这点很满意，会继续关注",
        "虽然物流稍微慢了一点，但商品质量很好，所以还是给好评",
        "包装很用心，商品也没有任何损坏，卖家很负责任，好评",
        "对比了好几家才买的，这家性价比最高，质量也很好，推荐",
        "不是最便宜的，但质量确实好，长期使用的话还是选质量好的",
        "收到货后仔细检查了，没有发现任何问题，做工很精细，满意",
        "客服态度很好，有问题及时回复，商品质量也不错，整体满意",
        "虽然等待时间有点长，但收到货后觉得值得等待，质量很好"
    ]
    
    # 长文本序列型（BiLSTM擅长处理）- 扩展
    long_positive = [
        "商品收到后非常惊喜，包装很精美，打开一看质量确实不错，使用了一段时间感觉很好，物流也很快，卖家服务态度也很好，有问题及时回复，总体来说是一次很满意的购物体验，会继续关注这家店铺",
        "这个商品我已经关注很久了，终于等到活动价就下单了，收到货后仔细检查了一下，做工精细，用料扎实，使用起来很顺手，功能也很齐全，完全符合我的预期，性价比很高，值得推荐给大家",
        "第一次在这家店买东西，本来还有点担心，但收到货后完全打消了顾虑，商品质量很好，包装也很用心，物流速度也快，客服服务态度也很好，解答问题很耐心，总体来说非常满意，以后会经常来这家店",
        "商品包装很精美，打开后看到实物比图片还要好看，质量也很好，使用起来很方便，功能很强大，完全满足我的需求，价格也很合理，性价比很高，非常满意这次购物，会推荐给朋友",
        "收到商品后迫不及待地打开了，外观很漂亮，做工也很精细，使用了一段时间感觉很好，功能齐全，操作简单，效果也很明显，物流速度也快，卖家服务态度也很好，非常满意，五星好评",
        "这个商品我已经用了快一个月了，质量确实很好，没有出现任何问题，功能也很强大，完全符合我的需求，价格也很合理，性价比很高，卖家服务态度也很好，有问题及时回复，非常满意，会回购",
        "商品收到后仔细检查了一下，没有发现任何瑕疵，做工很精细，质量也很好，使用起来很方便，效果也很明显，物流速度也快，包装也很用心，卖家服务态度也很好，总体来说非常满意，值得推荐",
        "对比了好几家店，最终选择了这家，收到货后觉得选择是对的，商品质量很好，包装也很精美，使用起来很方便，功能也很齐全，价格也很合理，卖家服务态度也很好，非常满意，会继续关注"
    ]
    
    # 负面评论 - 同样设计不同类型 - 大幅扩展
    # 简单关键词型
    simple_negative = [
        # 质量相关
        "质量差", "质量不好", "质量一般", "质量堪忧", "质量有问题", "品质差", "假货", "次品",
        # 满意度相关
        "不推荐", "很失望", "非常失望", "太失望", "差评", "差评如潮", "不满意", "很不满意",
        # 物流相关
        "物流慢", "快递慢", "发货慢", "配送慢", "到货慢", "物流太慢", "快递太慢", "等了好久",
        # 服务相关
        "服务差", "服务态度差", "客服差", "态度恶劣", "服务不好", "客服不理人", "售后差",
        # 商品问题
        "破损", "有划痕", "有瑕疵", "有缺陷", "有损坏", "有异味", "有污渍", "颜色不对", "尺寸不对",
        # 描述不符
        "与描述不符", "和图片不一样", "实物差距大", "夸大宣传", "虚假宣传", "描述不实",
        # 功能问题
        "不好用", "用不了", "用着不行", "功能不全", "效果不好", "不实用", "没用",
        # 价格问题
        "价格贵", "不值这个价", "性价比低", "价格虚高", "买贵了", "不划算",
        # 其他负面
        "退货", "退款", "换货", "不会再来", "不会再买", "不推荐", "避雷", "踩雷"
    ]
    
    # 复杂语义型（反讽、双重否定等）- 扩展
    complex_negative = [
        "不是说不好，但确实没有想象中的那么好，可能是我期望太高了",
        "虽然价格便宜，但质量确实一般，一分价钱一分货吧",
        "商品本身还可以，但和描述差距有点大，感觉被夸大了",
        "不是最差的，但绝对算不上好，只能说勉强能用",
        "价格虽然不贵，但质量也确实一般，用不了多久可能就要换了",
        "商品外观还可以，但质量真的不敢恭维，用了一次就出问题了",
        "虽然客服态度还可以，但商品质量确实有问题，最后还是退货了",
        "不是说完全不能用，但和预期差距太大，感觉不值这个价格",
        "包装倒是挺精美的，但商品质量真的不行，有点失望",
        "虽然物流很快，但商品质量有问题，联系客服处理也很麻烦",
        "商品本身还可以，但售后服务太差了，有问题也不给解决",
        "虽然价格便宜，但质量确实一般，用不了多久就坏了",
        "不是说完全不好，但确实没有宣传的那么好，有点失望",
        "商品外观还可以，但使用起来效果不好，和预期差距大",
        "虽然包装很好，但商品本身质量有问题，不太满意"
    ]
    
    # 长文本序列型 - 扩展
    long_negative = [
        "收到商品后很失望，包装简陋，打开一看商品有划痕，联系客服说这是正常现象，但我觉得不应该这样，物流也很慢，等了好几天才到，总体来说这次购物体验很差，不会再买了",
        "这个商品质量真的不行，用了一次就出问题了，联系卖家处理，态度也不好，拖了很久才解决，虽然最后退款了，但浪费了很多时间和精力，非常不推荐购买",
        "商品收到后发现和描述差距很大，实物质量很差，做工也很粗糙，使用起来效果也不好，联系客服要求退货，但处理速度很慢，态度也不好，非常不满意，不会再来这家店了",
        "等了很久才收到货，打开一看商品有损坏，联系客服说可以换货，但又要等很久，而且商品质量本身也不太好，做工粗糙，功能也不全，和描述差距很大，非常失望，不推荐购买",
        "这个商品用了一段时间就出问题了，联系卖家售后，态度很差，也不给解决，拖了很久才处理，虽然最后换了货，但新货也有问题，质量真的不行，非常不满意，不会再买了",
        "商品收到后仔细检查了一下，发现有很多瑕疵，做工很粗糙，质量也不好，使用起来效果很差，和描述完全不符，联系客服也不理人，非常失望，差评，不推荐购买",
        "第一次在这家店买东西就遇到这种情况，商品质量很差，包装也很简陋，收到货就有损坏，联系客服处理速度很慢，态度也不好，虽然最后退款了，但浪费了很多时间，非常不满意",
        "商品和图片差距很大，实物质量很差，做工粗糙，使用起来效果也不好，价格也不便宜，性价比很低，联系客服也不给解决，非常失望，不会再来这家店了"
    ]
    
    # 中性评论 - 扩展
    simple_neutral = [
        "一般般", "还可以", "马马虎虎", "中规中矩", "普普通通",
        "没什么特别", "符合预期", "普通", "还行", "凑合",
        "一般", "正常", "标准", "常规", "平常", "平淡",
        "不过不失", "不温不火", "平平无奇", "中规中矩", "尚可"
    ]
    
    complex_neutral = [
        "说不上好也说不上坏，就是普通水平，对得起这个价格",
        "没有特别突出的地方，但也没有明显的缺点，就是很普通",
        "中规中矩的产品，符合基本需求，但不要有太高期望",
        "整体来说还可以，但也没有特别惊喜的地方，就是正常水平",
        "价格还算合理，质量也还可以，但也没有什么特别的地方",
        "商品本身还可以，但和预期还是有点差距，不过也能接受",
        "虽然没有什么大问题，但也没有什么亮点，就是很普通的产品",
        "质量还可以，价格也合理，但使用起来感觉一般，没有特别满意",
        "商品收到后感觉还可以，没有太大问题，但也没有特别满意的地方",
        "整体来说还可以，符合基本需求，但不要期望太高，就是普通水平",
        "价格不贵，质量也还可以，但使用起来感觉一般，没有什么特别的地方",
        "商品本身还可以，但和描述还是有点差距，不过也能用，就是一般",
        "虽然没有什么大问题，但也没有什么特别的地方，就是很普通的产品",
        "质量还可以，功能也基本满足需求，但使用起来感觉一般，就是普通水平"
    ]
    
    long_neutral = [
        "商品收到后看了一下，外观还可以，质量也还行，使用起来没什么大问题，但也没有特别突出的优点，就是很普通的一个商品，价格还算合理，总体来说就是中规中矩",
        "这个商品用了一段时间，感觉还可以，功能基本满足需求，质量也还可以，但也没有什么特别的地方，就是很普通的产品，对得起价格，但也不会特别推荐",
        "收到商品后仔细检查了一下，外观还可以，质量也还行，使用起来没什么问题，但也没有什么特别满意的地方，就是很普通的一个商品，价格还算合理，总体来说就是一般",
        "这个商品用了一段时间，功能基本满足需求，质量也还可以，但使用起来感觉一般，没有什么特别的地方，就是很普通的产品，对得起价格，但也不会特别推荐给别人",
        "商品收到后看了一下，外观还可以，质量也还行，使用起来没什么大问题，但也没有什么特别突出的优点，就是很普通的一个商品，价格还算合理，总体来说就是中规中矩，符合预期",
        "这个商品用了一段时间，感觉还可以，功能基本满足需求，质量也还可以，但也没有什么特别的地方，就是很普通的产品，对得起价格，但也不会特别推荐，就是一般水平",
        "收到商品后仔细检查了一下，外观还可以，质量也还行，使用起来没什么问题，但也没有什么特别满意的地方，就是很普通的一个商品，价格还算合理，总体来说就是一般，符合基本需求",
        "这个商品用了一段时间，功能基本满足需求，质量也还可以，但使用起来感觉一般，没有什么特别的地方，就是很普通的产品，对得起价格，但也不会特别推荐给别人，就是正常水平"
    ]
    
    texts = []
    labels = []
    
    samples_per_class = num_samples // 3
    
    # 生成正面评论 - 混合不同类型
    for i in range(samples_per_class):
        if i < samples_per_class * 0.3:  # 30% 简单关键词型（SVM优势）
            # 随机选择1-3个关键词组合，增加多样性
            num_keywords = np.random.randint(1, 4)
            selected_keywords = np.random.choice(simple_positive, num_keywords, replace=False)
            text = "，".join(selected_keywords)
            # 随机添加补充评价
            if np.random.random() > 0.4:
                supplements = ["真的很不错", "非常推荐", "五星好评", "值得购买", "会回购", "超出预期"]
                text += "，" + np.random.choice(supplements)
        elif i < samples_per_class * 0.6:  # 30% 复杂语义型（BERT优势）
            text = np.random.choice(complex_positive)
            # 偶尔添加简单关键词增强
            if np.random.random() > 0.7:
                text += "，" + np.random.choice(simple_positive)
        else:  # 40% 长文本序列型（BiLSTM优势）
            text = np.random.choice(long_positive)
        texts.append(text)
        labels.append(2)
    
    # 生成负面评论
    for i in range(samples_per_class):
        if i < samples_per_class * 0.3:  # 30% 简单关键词型
            # 随机选择1-3个关键词组合
            num_keywords = np.random.randint(1, 4)
            selected_keywords = np.random.choice(simple_negative, num_keywords, replace=False)
            text = "，".join(selected_keywords)
            # 随机添加补充评价
            if np.random.random() > 0.4:
                supplements = ["太失望了", "不会再来", "差评", "不推荐", "避雷", "踩雷"]
                text += "，" + np.random.choice(supplements)
        elif i < samples_per_class * 0.6:  # 30% 复杂语义型
            text = np.random.choice(complex_negative)
            # 偶尔添加简单关键词增强
            if np.random.random() > 0.7:
                text += "，" + np.random.choice(simple_negative)
        else:  # 40% 长文本序列型
            text = np.random.choice(long_negative)
        texts.append(text)
        labels.append(0)
    
    # 生成中性评论
    for i in range(samples_per_class):
        if i < samples_per_class * 0.3:  # 30% 简单关键词型
            # 随机选择1-2个关键词组合
            num_keywords = np.random.randint(1, 3)
            selected_keywords = np.random.choice(simple_neutral, num_keywords, replace=False)
            text = "，".join(selected_keywords)
        elif i < samples_per_class * 0.6:  # 30% 复杂语义型
            text = np.random.choice(complex_neutral)
        else:  # 40% 长文本序列型
            text = np.random.choice(long_neutral)
        texts.append(text)
        labels.append(1)
    
    # 打乱数据
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return texts, labels

def load_uploaded_data(file_path):
    """从上传的文件加载数据
    支持CSV格式，需要包含'text'和'label'列
    label: 0-负面, 1-中性, 2-正面
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 检查必需的列
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV文件必须包含'text'和'label'列")
        
        # 过滤空值
        df = df.dropna(subset=['text', 'label'])
        
        texts = df['text'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
        
        # 验证标签范围
        valid_labels = [0, 1, 2]
        filtered_data = [(t, l) for t, l in zip(texts, labels) if l in valid_labels]
        texts, labels = zip(*filtered_data) if filtered_data else ([], [])
        texts, labels = list(texts), list(labels)
        
        logger.info(f"从文件加载了 {len(texts)} 条数据")
        return texts, labels
    except Exception as e:
        logger.error(f"加载上传数据失败: {str(e)}")
        raise

def prepare_data(dataset_type='simulated', file_path=None):
    """准备训练数据
    
    Args:
        dataset_type: 'simulated'、'uploaded' 或 'deepseek'
        file_path: 如果dataset_type为'uploaded'或'deepseek'，需要提供文件路径
    """
    if dataset_type == 'simulated':
        texts, labels = generate_sample_data(DATA_CONFIG['num_samples'])
        logger.info("使用模拟数据集")
    elif dataset_type in ['uploaded', 'deepseek']:
        if file_path is None:
            raise ValueError(f"{dataset_type}数据集需要提供文件路径")
        texts, labels = load_uploaded_data(file_path)
        if dataset_type == 'deepseek':
            logger.info("使用DeepSeek分析的数据集")
        else:
            logger.info("使用上传的数据集")
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    # 划分数据集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, 
        test_size=DATA_CONFIG['test_size'], 
        random_state=DATA_CONFIG['random_state'], 
        stratify=labels
    )
    
    # 进一步划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, 
        test_size=DATA_CONFIG['val_size'], 
        random_state=DATA_CONFIG['random_state'], 
        stratify=train_labels
    )
    
    logger.info(f"训练集大小: {len(train_texts)}")
    logger.info(f"验证集大小: {len(val_texts)}")
    logger.info(f"测试集大小: {len(test_texts)}")
    
    # 计算标签分布统计
    all_labels = labels  # 使用原始标签列表
    label_counts = {
        'negative': all_labels.count(0),
        'neutral': all_labels.count(1),
        'positive': all_labels.count(2)
    }
    total = len(all_labels)
    label_distribution = {
        'negative': {
            'count': label_counts['negative'],
            'percentage': round(label_counts['negative'] / total * 100, 2) if total > 0 else 0
        },
        'neutral': {
            'count': label_counts['neutral'],
            'percentage': round(label_counts['neutral'] / total * 100, 2) if total > 0 else 0
        },
        'positive': {
            'count': label_counts['positive'],
            'percentage': round(label_counts['positive'] / total * 100, 2) if total > 0 else 0
        },
        'total': total
    }
    
    logger.info(f"标签分布 - 负面: {label_distribution['negative']['count']} ({label_distribution['negative']['percentage']}%), "
                f"中性: {label_distribution['neutral']['count']} ({label_distribution['neutral']['percentage']}%), "
                f"正面: {label_distribution['positive']['count']} ({label_distribution['positive']['percentage']}%)")
    
    return {
        'train': {'texts': train_texts, 'labels': train_labels},
        'val': {'texts': val_texts, 'labels': val_labels},
        'test': {'texts': test_texts, 'labels': test_labels},
        'label_distribution': label_distribution  # 添加标签分布信息
    }
