import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from tqdm import tqdm
import logging
import os
import time
from config import DEVICE, MODEL_CONFIG, PATH_CONFIG

logger = logging.getLogger(__name__)

# 设置Hugging Face镜像源（如果网络连接有问题）
# 优先使用镜像源，如果失败则使用原始源
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')  # 5分钟超时

# 设置镜像源 - 必须在导入transformers之前设置
# 使用hf-mirror.com作为镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 尝试禁用HF_TRANSFER以使用标准下载方式
try:
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
except:
    pass

# 设置huggingface_hub的镜像源（如果可用）
try:
    import huggingface_hub
    # 设置镜像端点
    if hasattr(huggingface_hub, 'constants'):
        huggingface_hub.constants.ENDPOINT = 'https://hf-mirror.com'
except:
    pass

class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERTClassifier(nn.Module):
    def __init__(self, n_classes=3, bert_model_name='bert-base-chinese'):
        super(BERTClassifier, self).__init__()
        logger.info(f"正在加载BERT模型: {bert_model_name}")
        logger.info(f"使用镜像源: {os.environ.get('HF_ENDPOINT', '默认')}")
        
        # 重试机制
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"尝试下载BERT模型 (第 {attempt + 1}/{max_retries} 次)...")
                # 使用镜像源下载
                self.bert = BertModel.from_pretrained(
                    bert_model_name,
                    cache_dir=None,
                    local_files_only=False,
                    resume_download=True
                )
                logger.info("BERT模型加载成功")
                break
            except Exception as e:
                logger.warning(f"第 {attempt + 1} 次尝试失败: {str(e)[:200]}")
                if attempt < max_retries - 1:
                    logger.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    # 最后一次尝试：使用原始源
                    logger.info("尝试使用原始源下载...")
                    try:
                        original_endpoint = os.environ.pop('HF_ENDPOINT', None)
                        self.bert = BertModel.from_pretrained(
                            bert_model_name,
                            cache_dir=None,
                            local_files_only=False,
                            resume_download=True
                        )
                        if original_endpoint:
                            os.environ['HF_ENDPOINT'] = original_endpoint
                        logger.info("使用原始源成功加载BERT模型")
                        break
                    except Exception as e2:
                        logger.error(f"所有下载尝试都失败了")
                        raise Exception(f"无法下载BERT模型，已尝试 {max_retries} 次。\n"
                                      f"请检查：\n"
                                      f"1. 网络连接是否正常\n"
                                      f"2. 是否可以访问 huggingface.co 或 hf-mirror.com\n"
                                      f"3. 防火墙设置是否阻止了连接\n"
                                      f"4. 尝试使用VPN或代理\n"
                                      f"最后错误: {str(e2)[:500]}")
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.fc(output)

def train_bert_model(data):
    """训练BERT模型"""
    logger.info("开始训练BERT模型...")
    
    config = MODEL_CONFIG['bert']
    train_data = data['train']
    val_data = data['val']
    test_data = data['test']
    
    # 初始化tokenizer和模型
    logger.info("正在加载BERT tokenizer...")
    
    # 重试机制
    max_retries = 3
    retry_delay = 5
    tokenizer = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"尝试下载BERT tokenizer (第 {attempt + 1}/{max_retries} 次)...")
            # 使用镜像源下载tokenizer
            tokenizer = BertTokenizer.from_pretrained(
                'bert-base-chinese',
                cache_dir=None,
                local_files_only=False,
                resume_download=True
            )
            logger.info("BERT tokenizer加载成功")
            break
        except Exception as e:
            logger.warning(f"第 {attempt + 1} 次尝试失败: {str(e)[:200]}")
            if attempt < max_retries - 1:
                logger.info(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                # 最后一次尝试：使用原始源
                logger.info("尝试使用原始源下载tokenizer...")
                try:
                    original_endpoint = os.environ.pop('HF_ENDPOINT', None)
                    tokenizer = BertTokenizer.from_pretrained(
                        'bert-base-chinese',
                        cache_dir=None,
                        local_files_only=False,
                        resume_download=True
                    )
                    if original_endpoint:
                        os.environ['HF_ENDPOINT'] = original_endpoint
                    logger.info("使用原始源成功加载BERT tokenizer")
                    break
                except Exception as e2:
                    logger.error(f"所有下载尝试都失败了")
                    raise Exception(f"无法下载BERT tokenizer，已尝试 {max_retries} 次。\n"
                                  f"请检查：\n"
                                  f"1. 网络连接是否正常\n"
                                  f"2. 是否可以访问 huggingface.co 或 hf-mirror.com\n"
                                  f"3. 防火墙设置是否阻止了连接\n"
                                  f"4. 尝试使用VPN或代理\n"
                                  f"最后错误: {str(e2)[:500]}")
    
    if tokenizer is None:
        raise Exception("无法加载BERT tokenizer")
    
    logger.info("正在初始化BERT模型...")
    model = BERTClassifier(n_classes=3).to(DEVICE)
    
    # 创建数据加载器
    train_dataset = BERTDataset(train_data['texts'], train_data['labels'], tokenizer)
    val_dataset = BERTDataset(val_data['texts'], val_data['labels'], tokenizer)
    test_dataset = BERTDataset(test_data['texts'], test_data['labels'], tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # 设置优化器（添加权重衰减）
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0.01))
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器（使用ReduceLROnPlateau，根据验证准确率调整）
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # 早停机制
    best_val_accuracy = 0.0
    patience_counter = 0
    patience = config.get('early_stopping_patience', 3)
    best_model_state = None
    
    # 训练循环
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        correct_predictions = 0
        
        for batch in tqdm(train_loader, desc=f'BERT Epoch {epoch+1}/{config["epochs"]}'):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels_batch = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels_batch)
        
        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions.double() / len(train_dataset)
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels_batch = batch['label'].to(DEVICE)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                
                val_correct += torch.sum(preds == labels_batch)
                val_total += len(labels_batch)
        
        val_accuracy = val_correct.double() / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新学习率（基于验证准确率）
        scheduler.step(val_accuracy)
        
        logger.info(f'BERT Epoch {epoch+1}: 训练损失 = {avg_loss:.4f}, 训练准确率 = {train_accuracy:.4f}, 验证准确率 = {val_accuracy:.4f}, 学习率 = {current_lr:.2e}')
        
        # 早停检查
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            logger.info(f'BERT模型改进，验证准确率: {best_val_accuracy:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'BERT早停触发，最佳验证准确率: {best_val_accuracy:.4f}')
                model.load_state_dict(best_model_state)
                break
    
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # 在测试集上评估
    return evaluate_model(model, test_loader, 'BERT')

def evaluate_model(model, test_loader, model_name):
    """评估模型"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            if model_name == 'BERT':
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                text = batch['text'].to(DEVICE)
                text_lengths = batch['text_length'].to(DEVICE)
                outputs = model(text, text_lengths)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['label'].cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    logger.info(f"{model_name}模型测试集准确率: {accuracy:.4f}")
    logger.info(f"{model_name}模型测试集精确率: {precision:.4f}")
    logger.info(f"{model_name}模型测试集召回率: {recall:.4f}")
    logger.info(f"{model_name}模型测试集F1分数: {f1:.4f}")
    
    # 保存模型
    if model_name == 'BERT':
        torch.save(model.state_dict(), PATH_CONFIG['bert_model'])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': predictions,
        'true_labels': true_labels,
        'probabilities': probabilities
    }
