# DeepSeek API密钥设置指南

## 方法一：在前端界面输入（推荐，最简单）

1. 启动前端服务：
```bash
python start_frontend.py
```

2. 访问 `http://localhost:5001`

3. 选择"输入评论文本（使用DeepSeek分析）"

4. 在"DeepSeek API密钥"输入框中输入您的API密钥

5. 输入评论并开始训练

**优点**：无需配置环境变量，使用方便

---

## 方法二：设置环境变量（适合长期使用）

### Windows PowerShell
```powershell
# 临时设置（当前会话有效）
$env:DEEPSEEK_API_KEY="your_api_key_here"

# 永久设置（需要管理员权限）
[System.Environment]::SetEnvironmentVariable('DEEPSEEK_API_KEY', 'your_api_key_here', 'User')
```

### Windows CMD
```cmd
# 临时设置
set DEEPSEEK_API_KEY=your_api_key_here

# 永久设置（需要管理员权限）
setx DEEPSEEK_API_KEY "your_api_key_here"
```

### Linux/Mac
```bash
# 临时设置（当前终端会话有效）
export DEEPSEEK_API_KEY="your_api_key_here"

# 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export DEEPSEEK_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

---

## 如何获取DeepSeek API密钥

1. 访问 [DeepSeek平台](https://platform.deepseek.com/)
2. 注册账号并登录
3. 进入"API管理"或"密钥管理"页面
4. 创建新的API密钥
5. 复制密钥（注意：密钥只显示一次，请妥善保存）

---

## 验证设置

设置完成后，可以通过以下方式验证：

### Python验证脚本
```python
import os
api_key = os.getenv('DEEPSEEK_API_KEY')
if api_key:
    print(f"API密钥已设置（前10个字符）：{api_key[:10]}...")
else:
    print("API密钥未设置")
```

### 在代码中验证
```python
from deepseek_analyzer import DeepSeekAnalyzer

try:
    analyzer = DeepSeekAnalyzer()
    print("API密钥设置成功！")
except ValueError as e:
    print(f"错误：{e}")
```

---

## 注意事项

1. **安全性**：
   - 不要将API密钥提交到Git仓库
   - 不要在前端代码中硬编码API密钥
   - 使用环境变量或前端输入（推荐前端输入）

2. **API配额**：
   - 检查您的API配额是否充足
   - DeepSeek可能有免费额度限制

3. **网络连接**：
   - 确保能访问 `https://api.deepseek.com`
   - 如果在中国大陆，可能需要配置代理

4. **错误处理**：
   - 如果API调用失败，系统会使用中性标签（1）作为默认值
   - 查看后端日志获取详细错误信息

---

## 故障排除

### 问题1：仍然提示需要API密钥
- 检查环境变量是否正确设置
- 重启终端/IDE
- 使用前端输入方式（方法一）

### 问题2：API调用失败
- 检查API密钥是否正确
- 检查网络连接
- 检查API配额
- 查看后端日志获取详细错误

### 问题3：分析速度慢
- 这是正常的，每条评论需要调用一次API
- 系统已设置0.5秒延迟以避免频率限制
- 建议批量分析时耐心等待

---

## 推荐使用方式

**对于临时使用**：使用方法一（前端输入），最简单方便

**对于长期使用**：使用方法二（环境变量），一次配置，长期使用













