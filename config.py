# 配置文件
import torch

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据配置
DATA_CONFIG = {
    'num_samples': 3000,
    'test_size': 0.2,
    'val_size': 0.125,
    'random_state': 42
}

# 数据集类型配置
DATASET_TYPE = {
    'simulated': 'simulated',  # 模拟数据集
    'uploaded': 'uploaded',  # 上传的数据集
    'deepseek': 'deepseek'  # DeepSeek分析的数据集
}

# 模型配置
MODEL_CONFIG = {
    'bert': {
        'max_len': 128,
        'batch_size': 16,  # 减小batch_size以提高训练稳定性
        'epochs': 3,  # 增加epochs以提高准确率
        'learning_rate': 2e-5,
        'weight_decay': 0.01,  # 添加权重衰减防止过拟合
        'early_stopping_patience': 3  # 早停耐心值
    },
    'bilstm': {
        'embedding_dim': 128,  # 增加embedding维度
        'hidden_dim': 256,  # 增加隐藏层维度
        'n_layers': 2,
        'dropout': 0.5,  # 增加dropout防止过拟合
        'batch_size': 32,
        'epochs': 15,  # 增加训练轮数
        'learning_rate': 0.001,
        'early_stopping_patience': 5  # 早停耐心值
    },
    'svm': {
        'max_features': 10000,  # 增加特征数量
        'ngram_range': (1, 3),  # 增加n-gram范围
        'C': 1.0,  # SVM正则化参数
        'kernel': 'linear'
    }
}

# 路径配置
PATH_CONFIG = {
    'models_dir': 'models',
    'results_dir': 'results',
    'bert_model': 'models/bert_model.pth',
    'bilstm_model': 'models/bilstm_model.pth',
    'svm_model': 'models/svm_model.pkl',
    'vectorizer': 'models/tfidf_vectorizer.pkl'
}
