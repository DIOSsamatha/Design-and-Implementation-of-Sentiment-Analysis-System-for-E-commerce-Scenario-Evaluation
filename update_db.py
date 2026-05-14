import sqlite3
import json
from datetime import datetime, timedelta

print('开始更新数据库...')

# 连接数据库
conn = sqlite3.connect('frontend/user.db')
c = conn.cursor()

try:
    # 检查并添加product_id列
    c.execute("PRAGMA table_info(analysis_history)")
    columns = [col[1] for col in c.fetchall()]
    
    if 'product_id' not in columns:
        print('添加product_id列到analysis_history表...')
        c.execute('ALTER TABLE analysis_history ADD COLUMN product_id INTEGER')
        c.execute('ALTER TABLE analysis_history ADD COLUMN FOREIGN KEY (product_id) REFERENCES products (id)')
    
    # 获取admin用户ID
    c.execute('SELECT id FROM users WHERE username = ?', ('admin',))
    admin = c.fetchone()
    if not admin:
        print('未找到admin用户')
        conn.close()
        exit()
    admin_id = admin[0]
    
    # 清空现有历史记录（可选）
    c.execute('DELETE FROM analysis_history')
    print('已清空现有历史记录')
    
    # 创建一些示例商品（如果没有的话）
    sample_products = [
        ('华为 Mate 60 Pro', '手机数码', '全新华为旗舰手机，麒麟9000S芯片，卫星通话功能'),
        ('小米 14 Ultra', '手机数码', '徕卡影像旗舰，骁龙8Gen3处理器'),
        ('Nike Air Max 270', '服装鞋包', '经典运动鞋，气垫缓震，舒适百搭'),
        ('iPhone 15 Pro Max', '手机数码', '苹果旗舰，钛金属边框，A17Pro芯片'),
        ('AirPods Pro 2', '数码配件', '主动降噪，空间音频，无线充电'),
        ('MacBook Pro 14', '电脑办公', 'M3芯片，Liquid Retina XDR屏幕'),
        ('戴森 V15 Detect', '家居生活', '激光探测，智能除尘，无绳吸尘器'),
        ('索尼 WH-1000XM5', '数码配件', '头戴式降噪耳机，音质卓越'),
        ('Switch OLED', '游戏娱乐', '任天堂游戏机，7英寸OLED屏幕'),
        ('格力空调', '家居生活', '一级能效，智能控制，静音制冷')
    ]
    
    product_ids = []
    for product in sample_products:
        try:
            c.execute('INSERT INTO products (user_id, name, category, description) VALUES (?, ?, ?, ?)', 
                     (admin_id, product[0], product[1], product[2]))
            product_ids.append(c.lastrowid)
        except:
            c.execute('SELECT id FROM products WHERE user_id = ? AND name = ?', (admin_id, product[0]))
            existing = c.fetchone()
            if existing:
                product_ids.append(existing[0])
    
    print(f'商品数量: {len(product_ids)}')
    
    # 添加分析历史记录
    sample_histories = []
    
    # 为商品添加历史
    for i, product_id in enumerate(product_ids[:8]):
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
    
    print('\n数据库更新成功！')
        
except Exception as e:
    print(f'操作失败: {e}')
    conn.rollback()
finally:
    conn.close()
    print('操作完成')
