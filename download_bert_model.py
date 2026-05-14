#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手动下载BERT模型的辅助脚本
如果自动下载失败，可以运行此脚本来手动下载模型
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_bert_model():
    """手动下载BERT模型和tokenizer"""
    
    # 设置镜像源
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
    
    logger.info("开始下载BERT模型和tokenizer...")
    logger.info(f"使用镜像源: {os.environ.get('HF_ENDPOINT')}")
    
    try:
        from transformers import BertTokenizer, BertModel
        
        model_name = 'bert-base-chinese'
        
        # 下载tokenizer
        logger.info("正在下载BERT tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(
            model_name,
            cache_dir=None,
            local_files_only=False,
            resume_download=True
        )
        logger.info("✓ BERT tokenizer下载成功")
        
        # 下载模型
        logger.info("正在下载BERT模型（这可能需要几分钟）...")
        model = BertModel.from_pretrained(
            model_name,
            cache_dir=None,
            local_files_only=False,
            resume_download=True
        )
        logger.info("✓ BERT模型下载成功")
        
        logger.info("\n" + "="*50)
        logger.info("✓ 所有文件下载完成！")
        logger.info("现在可以运行训练脚本了。")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"下载失败: {e}")
        logger.error("\n如果持续失败，请尝试：")
        logger.error("1. 检查网络连接")
        logger.error("2. 使用VPN或代理")
        logger.error("3. 手动设置代理：")
        logger.error("   export HTTP_PROXY=http://your-proxy:port")
        logger.error("   export HTTPS_PROXY=http://your-proxy:port")
        logger.error("4. 或者手动从 https://huggingface.co/bert-base-chinese 下载")
        return False

if __name__ == '__main__':
    success = download_bert_model()
    sys.exit(0 if success else 1)










