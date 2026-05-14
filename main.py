import logging
import os
import argparse
from data_loader import prepare_data
from bert_model import train_bert_model
from bilstm_model import train_bilstm_model
from svm_model import train_svm_model
from evaluation import (
    generate_performance_comparison, 
    generate_confusion_matrices,
    generate_classification_reports,
    save_performance_metrics,
    generate_sentiment_distribution
)
from config import PATH_CONFIG, DATASET_TYPE

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(dataset_type='simulated', file_path=None):
    """主函数：训练所有模型并比较性能
    
    Args:
        dataset_type: 'simulated' 或 'uploaded'
        file_path: 如果dataset_type为'uploaded'，需要提供文件路径
    """
    logger.info("开始电商评论情感分类系统训练和评估...")
    
    # 创建目录
    os.makedirs(PATH_CONFIG['models_dir'], exist_ok=True)
    os.makedirs(PATH_CONFIG['results_dir'], exist_ok=True)
    
    # 准备数据
    data = prepare_data(dataset_type=dataset_type, file_path=file_path)
    
    # 训练所有模型
    results = {}
    
    # 训练BERT模型
    results['BERT'] = train_bert_model(data)
    
    # 训练BiLSTM-Attention模型
    results['BiLSTM-Attention'] = train_bilstm_model(data)
    
    # 训练SVM模型
    results['SVM'] = train_svm_model(data)
    
    # 生成性能报告
    generate_performance_comparison(results)
    generate_confusion_matrices(results)
    generate_classification_reports(results)
    save_performance_metrics(results)
    
    # 生成情感分布图表
    if 'label_distribution' in data:
        generate_sentiment_distribution(data['label_distribution'])
    
    # 保存标签分布信息
    if 'label_distribution' in data:
        import json
        distribution_file = os.path.join(PATH_CONFIG['results_dir'], 'label_distribution.json')
        with open(distribution_file, 'w', encoding='utf-8') as f:
            json.dump(data['label_distribution'], f, ensure_ascii=False, indent=2)
        logger.info(f"标签分布信息已保存到: {distribution_file}")
    
    # 打印最终结果
    logger.info("\n" + "="*50)
    logger.info("最终模型性能比较:")
    logger.info("="*50)
    
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  准确率: {metrics['accuracy']:.4f}")
        logger.info(f"  精确率: {metrics['precision']:.4f}")
        logger.info(f"  召回率: {metrics['recall']:.4f}")
        logger.info(f"  F1分数: {metrics['f1_score']:.4f}")
    
    logger.info("\n所有模型训练和评估完成！")
    logger.info(f"结果已保存到 {PATH_CONFIG['results_dir']}/ 目录")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='电商评论情感分类系统')
    parser.add_argument('--dataset_type', type=str, default='simulated',
                       choices=['simulated', 'uploaded', 'deepseek'],
                       help='数据集类型: simulated(模拟)、uploaded(上传) 或 deepseek(DeepSeek分析)')
    parser.add_argument('--file_path', type=str, default=None,
                       help='数据集文件路径（当dataset_type为uploaded或deepseek时必需）')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.dataset_type in ['uploaded', 'deepseek'] and args.file_path is None:
        logger.error(f"使用{args.dataset_type}数据集时必须提供 --file_path 参数")
        exit(1)
    
    main(dataset_type=args.dataset_type, file_path=args.file_path)