from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
from config import MODEL_CONFIG, PATH_CONFIG
from data_loader import preprocess_text, tokenize_chinese_text

logger = logging.getLogger(__name__)

def train_svm_model(data):
    """训练SVM模型"""
    logger.info("开始训练SVM模型...")
    
    config = MODEL_CONFIG['svm']
    train_data = data['train']
    test_data = data['test']
    
    # 预处理文本
    processed_train_texts = [preprocess_text(text) for text in train_data['texts']]
    processed_test_texts = [preprocess_text(text) for text in test_data['texts']]
    
    tokenized_train_texts = [tokenize_chinese_text(text) for text in processed_train_texts]
    tokenized_test_texts = [tokenize_chinese_text(text) for text in processed_test_texts]
    
    # TF-IDF特征提取（优化参数）
    vectorizer = TfidfVectorizer(
        max_features=config['max_features'], 
        ngram_range=config['ngram_range'],
        min_df=2,  # 最小文档频率
        max_df=0.95,  # 最大文档频率
        sublinear_tf=True  # 使用对数TF
    )
    X_train = vectorizer.fit_transform(tokenized_train_texts)
    X_test = vectorizer.transform(tokenized_test_texts)
    
    # 训练SVM（优化参数）
    svm_model = SVC(
        kernel=config.get('kernel', 'linear'),
        C=config.get('C', 1.0),
        probability=True,
        random_state=42,
        class_weight='balanced'  # 处理类别不平衡
    )
    svm_model.fit(X_train, train_data['labels'])
    
    # 评估模型
    y_pred = svm_model.predict(X_test)
    y_proba = svm_model.predict_proba(X_test)
    
    accuracy = accuracy_score(test_data['labels'], y_pred)
    precision = precision_score(test_data['labels'], y_pred, average='weighted')
    recall = recall_score(test_data['labels'], y_pred, average='weighted')
    f1 = f1_score(test_data['labels'], y_pred, average='weighted')
    
    logger.info(f"SVM模型测试集准确率: {accuracy:.4f}")
    logger.info(f"SVM模型测试集精确率: {precision:.4f}")
    logger.info(f"SVM模型测试集召回率: {recall:.4f}")
    logger.info(f"SVM模型测试集F1分数: {f1:.4f}")
    
    # 保存模型和向量器
    joblib.dump(svm_model, PATH_CONFIG['svm_model'])
    joblib.dump(vectorizer, PATH_CONFIG['vectorizer'])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred,
        'true_labels': test_data['labels'],
        'probabilities': y_proba
    }
