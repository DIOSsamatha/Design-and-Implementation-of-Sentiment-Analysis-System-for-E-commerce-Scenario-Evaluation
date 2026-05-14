#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动前端服务的便捷脚本
"""
import sys
import os

# 添加frontend目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'frontend'))

# 切换到frontend目录
os.chdir(os.path.join(os.path.dirname(__file__), 'frontend'))

# 导入并运行app
from app import app

if __name__ == '__main__':
    print("=" * 50)
    print("电商评论情感分析系统 - 前端服务")
    print("=" * 50)
    print("服务地址: http://localhost:5001")
    print("按 Ctrl+C 停止服务")
    print("=" * 50)
    app.run(debug=True, port=5001, host='0.0.0.0')













