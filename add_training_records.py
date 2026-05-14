import sqlite3
import json
from datetime import datetime, timedelta

print('开始添加训练记录...')

# 连接数据库
conn = sqlite3.connect('frontend/user.db')
c = conn.cursor()

try:
    # 获取admin用户ID
    c.execute('SELECT id FROM users WHERE username = ?', ('admin',))
    admin = c.fetchone()
    if not admin:
        print('未找到admin用户')
        conn.close()
        exit()
    admin_id = admin[0]
    
    # 清空现有训练记录
    c.execute('DELETE FROM model_training_records')
    print('已清空现有训练记录')
    
    # 示例模型性能数据
    models_data = [
        {'Model': 'BERT', 'Accuracy': 0.9678, 'Precision': 0.9656, 'Recall': 0.9678, 'F1_Score': 0.9662},
        {'Model': 'BiLSTM-Attention', 'Accuracy': 0.9345, 'Precision': 0.9312, 'Recall': 0.9345, 'F1_Score': 0.9324},
        {'Model': 'SVM', 'Accuracy': 0.8956, 'Precision': 0.8923, 'Recall': 0.8956, 'F1_Score': 0.8931}
    ]
    
    # 添加训练记录
    training_records = []
    
    # 为不同时间添加记录
    for i in range(5):
        for model_data in models_data:
            days_ago = 10 - (i * 2)
            record = {
                'user_id': admin_id,
                'product_id': None,
                'model_name': model_data['Model'],
                'dataset_type': 'simulated',
                'training_params': json.dumps({'batch_size': 32, 'epochs': 5 + i}),
                'metrics': json.dumps([{
                    'Model': model_data['Model'],
                    'Accuracy': model_data['Accuracy'] - i * 0.01,
                    'Precision': model_data['Precision'] - i * 0.01,
                    'Recall': model_data['Recall'] - i * 0.01,
                    'F1_Score': model_data['F1_Score'] - i * 0.01
                }]),
                'label_distribution': json.dumps({'0': 10 + i, '1': 20 + i, '2': 15 + i}),
                'status': 'completed',
                'created_at': (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S'),
                'completed_at': (datetime.now() - timedelta(days=days_ago, hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            }
            training_records.append(record)
    
    # 插入训练记录
    count = 0
    for record in training_records:
        try:
            c.execute('''
                INSERT INTO model_training_records 
                (user_id, product_id, model_name, dataset_type, training_params, metrics, label_distribution, status, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (record['user_id'], record['product_id'], record['model_name'], record['dataset_type'],
                  record['training_params'], record['metrics'], record['label_distribution'],
                  record['status'], record['created_at'], record['completed_at']))
            count += 1
        except Exception as e:
            print(f'添加训练记录失败: {e}')
    
    conn.commit()
    print(f'已添加 {count} 条训练记录')
    
    # 验证是否成功添加
    c.execute('SELECT COUNT(*) FROM model_training_records')
    total = c.fetchone()[0]
    print(f'数据库中共有 {total} 条训练记录')
    
    # 显示最近的几条记录
    c.execute('''
        SELECT id, model_name, status, created_at
        FROM model_training_records
        ORDER BY created_at DESC
        LIMIT 5
    ''')
    print('\n最近5条记录:')
    for row in c.fetchall():
        print(f'ID: {row[0]}, 模型: {row[1]}, 状态: {row[2]}, 时间: {row[3]}')
        
    print('\n训练记录添加成功！')
        
except Exception as e:
    print(f'操作失败: {e}')
    conn.rollback()
finally:
    conn.close()
    print('操作完成')
