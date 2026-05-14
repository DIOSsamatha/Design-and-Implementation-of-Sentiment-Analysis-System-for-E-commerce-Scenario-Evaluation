import sqlite3
import json
from datetime import datetime, timedelta

# 连接数据库
conn = sqlite3.connect('frontend/user.db')
c = conn.cursor()

print('开始添加分析历史记录...')

try:
    # 获取admin用户ID
    c.execute('SELECT id FROM users WHERE username = ?', ('admin',))
    admin = c.fetchone()
    if not admin:
        print('未找到admin用户')
        conn.close()
        exit()
    admin_id = admin[0]
    
    # 获取商品列表
    c.execute('SELECT id, name FROM products WHERE user_id = ?', (admin_id,))
    products = c.fetchall()
    product_dict = {name: product_id for product_id, name in products}
    print(f'找到 {len(products)} 个商品')
    
    # 过去10天的示例历史记录
    sample_histories = []
    
    # 为前8个商品添加历史
    product_names = list(product_dict.keys())[:8]
    for i, product_name in enumerate(product_names):
        product_id = product_dict[product_name]
        sample_histories.append({
            'user_id': admin_id,
            'product_id': product_id,
            'source_type': 'product',
            'comment_count': 5 + (i % 4),
            'sentiment_labels': json.dumps({'0': i % 2, '1': (i + 1) % 2, '2': 4 + (i % 3)}),
            'created_at': (datetime.now() - timedelta(days=10 - i)).strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # 添加其他类型的历史
    sample_histories.extend([
        {
            'user_id': admin_id,
            'source_type': 'input',
            'comment_count': 25,
            'sentiment_labels': json.dumps({'0': 5, '1': 10, '2': 10}),
            'created_at': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'user_id': admin_id,
            'source_type': 'document',
            'comment_count': 50,
            'sentiment_labels': json.dumps({'0': 10, '1': 20, '2': 20}),
            'created_at': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'user_id': admin_id,
            'source_type': 'input',
            'comment_count': 30,
            'sentiment_labels': json.dumps({'0': 3, '1': 12, '2': 15}),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    ])
    
    # 插入历史记录
    count = 0
    for history in sample_histories:
        try:
            if 'product_id' in history:
                c.execute('''
                    INSERT INTO analysis_history (user_id, product_id, source_type, comment_count, sentiment_labels, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (history['user_id'], history['product_id'], history['source_type'], history['comment_count'], 
                      history['sentiment_labels'], history['created_at']))
            else:
                c.execute('''
                    INSERT INTO analysis_history (user_id, source_type, comment_count, sentiment_labels, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (history['user_id'], history['source_type'], history['comment_count'], 
                      history['sentiment_labels'], history['created_at']))
            count += 1
        except Exception as e:
            print(f'添加分析历史记录失败: {e}')
    
    conn.commit()
    print(f'已添加 {count} 条分析历史记录')
    
    # 验证是否成功添加
    c.execute('SELECT COUNT(*) FROM analysis_history')
    total = c.fetchone()[0]
    print(f'数据库中共有 {total} 条分析历史记录')
    
    # 显示最近的几条记录
    c.execute('''
        SELECT ah.id, ah.source_type, ah.comment_count, ah.created_at, p.name
        FROM analysis_history ah
        LEFT JOIN products p ON ah.product_id = p.id
        ORDER BY ah.created_at DESC
        LIMIT 5
    ''')
    print('\n最近5条记录:')
    for row in c.fetchall():
        print(f'ID: {row[0]}, 类型: {row[1]}, 评论数: {row[2]}, 时间: {row[3]}, 商品: {row[4] or "无"}')
        
except Exception as e:
    print(f'操作失败: {e}')
    conn.rollback()
finally:
    conn.close()
    print('操作完成')
