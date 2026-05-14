#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
永久移除 BOM 字符的脚本
"""

import os
import sys

def remove_bom_from_file(filepath):
    """移除单个文件的 BOM 字符"""
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # 检查并移除 UTF-8 BOM
        if content.startswith(b'\xef\xbb\xbf'):
            print(f'发现 BOM: {filepath}，正在移除...')
            content = content.lstrip(b'\xef\xbb\xbf')
            
            # 写回文件
            with open(filepath, 'wb') as f:
                f.write(content)
            print(f'已成功移除: {filepath}')
            return True
        return False
    except Exception as e:
        print(f'处理文件 {filepath} 时出错: {e}')
        return False

def main():
    """主函数"""
    # 要处理的文件列表
    files_to_process = [
        r'k:\毕业设计\sentiment-analysis\frontend\app.py',
        r'k:\毕业设计\sentiment-analysis\frontend\templates\index.html'
    ]
    
    removed_count = 0
    for filepath in files_to_process:
        if os.path.exists(filepath):
            if remove_bom_from_file(filepath):
                removed_count += 1
        else:
            print(f'文件不存在: {filepath}')
    
    print(f'\n处理完成！共移除了 {removed_count} 个文件的 BOM 字符')

if __name__ == '__main__':
    main()
