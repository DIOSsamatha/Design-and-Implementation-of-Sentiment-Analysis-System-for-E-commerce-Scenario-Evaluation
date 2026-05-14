import sqlite3
import json
from datetime import datetime, timedelta

print('开始添加分析历史记录...')

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
    
    # 清空现有历史记录
    c.execute('DELETE FROM analysis_history')
    print('已清空现有历史记录')
    
    # 添加分析历史记录
    sample_histories = [
        {
            'user_id': admin_id,
            'source_type': 'input',
            'comment_count': 25,
            'sentiment_labels': json.dumps({'0': 5, '1': 10, '2': 10}),
            'created_at': (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'user_id': admin_id,
            'source_type': 'document',
            'comment_count': 50,
            'sentiment_labels': json.dumps({'0': 10, '1': 20, '2': 20}),
            'created_at': (datetime.now() - timedelta(days=9)).strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'user_id': admin_id,
            'source_type': 'input',
            'comment_count': 30,
            'sentiment_labels': json.dumps({'0': 3, '1': 12, '2': 15}),
            'created_at': (datetime.now() - timedelta(days=8)).strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'user_id': admin_id,
            'source_type': 'input',
            'comment_count': 15,
            'sentiment_labels': json.dumps({'0': 2, '1': 5, '2': 8}),
            'created_at': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'user_id': admin_id,
            'source_type': 'document',
            'comment_count': 35,
            'sentiment_labels': json.dumps({'0': 5, '1': 15, '2': 15}),
            'created_at': (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'user_id': admin_id,
            'source_type': 'input',
            'comment_count': 20,
            'sentiment_labels': json.dumps({'0': 4, '1': 8, '2': 8}),
            'created_at': (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'user_id': admin_id,
            'source_type': 'input',
            'comment_count': 40,
            'sentiment_labels': json.dumps({'0': 8, '1': 16, '2': 16}),
            'created_at': (datetime.now() - timedelta(days=4)).strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'user_id': admin_id,
            'source_type': 'document',
            'comment_count': 60,
            'sentiment_labels': json.dumps({'0': 12, '1': 24, '2': 24}),
            'created_at': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S')
        },
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
    ]
    
    # 插入历史记录
    count = 0
    for history in sample_histories:
        try:
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
        SELECT id, source_type, comment_count, created_at
        FROM analysis_history
        ORDER BY created_at DESC
        LIMIT 5
    ''')
    print('\n最近5条记录:')
    for row in c.fetchall():
        print(f'ID: {row[0]}, 类型: {row[1]}, 评论数: {row[2]}, 时间: {row[3]}')
        
    print('\n分析历史记录添加成功！')
        
except Exception as e:
    print(f'操作失败: {e}')
    conn.rollback()
finally:
    conn.close()
    print('操作完成')
