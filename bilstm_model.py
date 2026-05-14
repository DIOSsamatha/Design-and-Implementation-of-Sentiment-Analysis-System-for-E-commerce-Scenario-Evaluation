import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from config import DEVICE, MODEL_CONFIG, PATH_CONFIG
from data_loader import preprocess_text, tokenize_chinese_text

logger = logging.getLogger(__name__)

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                           bidirectional=True, dropout=dropout, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        
        # 打包序列
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # 注意力机制
        attention_weights = torch.tanh(self.attention(output))
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_output = output * attention_weights
        attention_output = torch.sum(weighted_output, dim=1)
        
        return self.fc(attention_output)

class BiLSTMDataset(Dataset):
    def __init__(self, sequences, lengths, labels):
        self.sequences = sequences
        self.lengths = lengths
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'text': torch.tensor(self.sequences[idx], dtype=torch.long),
            'text_length': torch.tensor(self.lengths[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def build_vocab(texts):
    """构建词汇表"""
    word_counts = {}
    for text in texts:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1
    
    vocab = {word: idx+2 for idx, word in enumerate(word_counts)}  # 0: <pad>, 1: <unk>
    vocab_size = len(vocab) + 2
    return vocab, vocab_size

def text_to_sequence(text, vocab, max_len=128):
    """将文本转换为索引序列"""
    sequence = [vocab.get(word, 1) for word in text.split()[:max_len]]
    if len(sequence) < max_len:
        sequence += [0] * (max_len - len(sequence))  # 填充
    return sequence

def train_bilstm_model(data):
    """训练BiLSTM-Attention模型"""
    logger.info("开始训练BiLSTM-Attention模型...")
    
    config = MODEL_CONFIG['bilstm']
    train_data = data['train']
    val_data = data['val']
    test_data = data['test']
    
    # 预处理文本
    processed_train_texts = [preprocess_text(text) for text in train_data['texts']]
    processed_val_texts = [preprocess_text(text) for text in val_data['texts']]
    processed_test_texts = [preprocess_text(text) for text in test_data['texts']]
    
    tokenized_train_texts = [tokenize_chinese_text(text) for text in processed_train_texts]
    tokenized_val_texts = [tokenize_chinese_text(text) for text in processed_val_texts]
    tokenized_test_texts = [tokenize_chinese_text(text) for text in processed_test_texts]
    
    # 构建词汇表
    vocab, vocab_size = build_vocab(tokenized_train_texts)
    
    # 文本转索引序列
    train_sequences = [text_to_sequence(text, vocab) for text in tokenized_train_texts]
    val_sequences = [text_to_sequence(text, vocab) for text in tokenized_val_texts]
    test_sequences = [text_to_sequence(text, vocab) for text in tokenized_test_texts]
    
    # 计算序列长度
    train_lengths = [min(len(text.split()), 128) for text in tokenized_train_texts]
    val_lengths = [min(len(text.split()), 128) for text in tokenized_val_texts]
    test_lengths = [min(len(text.split()), 128) for text in tokenized_test_texts]
    
    # 创建数据加载器
    train_dataset = BiLSTMDataset(train_sequences, train_lengths, train_data['labels'])
    val_dataset = BiLSTMDataset(val_sequences, val_lengths, val_data['labels'])
    test_dataset = BiLSTMDataset(test_sequences, test_lengths, test_data['labels'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # 初始化模型
    model = BiLSTMAttention(
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=3,
        n_layers=config['n_layers'],
        dropout=config['dropout']
    ).to(DEVICE)
    
    # 设置优化器（添加权重衰减）
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 早停机制
    best_val_accuracy = 0.0
    patience_counter = 0
    patience = config.get('early_stopping_patience', 5)
    best_model_state = None
    
    # 训练循环
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        correct_predictions = 0
        
        for batch in tqdm(train_loader, desc=f'BiLSTM Epoch {epoch+1}/{config["epochs"]}'):
            text = batch['text'].to(DEVICE)
            labels_batch = batch['label'].to(DEVICE)
            text_lengths = batch['text_length'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(text, text_lengths)
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
                text = batch['text'].to(DEVICE)
                labels_batch = batch['label'].to(DEVICE)
                text_lengths = batch['text_length'].to(DEVICE)
                
                outputs = model(text, text_lengths)
                _, preds = torch.max(outputs, dim=1)
                
                val_correct += torch.sum(preds == labels_batch)
                val_total += len(labels_batch)
        
        val_accuracy = val_correct.double() / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新学习率
        scheduler.step(val_accuracy)
        
        logger.info(f'BiLSTM Epoch {epoch+1}: 训练损失 = {avg_loss:.4f}, 训练准确率 = {train_accuracy:.4f}, 验证准确率 = {val_accuracy:.4f}, 学习率 = {current_lr:.2e}')
        
        # 早停检查
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            logger.info(f'BiLSTM模型改进，验证准确率: {best_val_accuracy:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'BiLSTM早停触发，最佳验证准确率: {best_val_accuracy:.4f}')
                model.load_state_dict(best_model_state)
                break
    
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # 在测试集上评估
    from bert_model import evaluate_model
    result = evaluate_model(model, test_loader, 'BiLSTM-Attention')
    
    # 保存模型和词汇表
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'vocab_size': vocab_size
    }, PATH_CONFIG['bilstm_model'])
    
    return result
