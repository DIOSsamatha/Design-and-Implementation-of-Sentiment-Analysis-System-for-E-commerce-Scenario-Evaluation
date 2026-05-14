# DeepSeek API配置说明

## 功能说明

系统现在支持使用DeepSeek API自动分析电商评论的情感倾向，并将分析结果用于模型训练。

## 配置步骤

### 1. 获取DeepSeek API密钥

1. 访问 [DeepSeek官网](https://platform.deepseek.com/)
2. 注册账号并登录
3. 在API管理页面创建API密钥

### 2. 设置环境变量

**Windows (PowerShell):**
```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

**Windows (CMD):**
```cmd
set DEEPSEEK_API_KEY=your_api_key_here
```

**Linux/Mac:**
```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

或者在项目根目录创建 `.env` 文件（需要安装python-dotenv）：
```
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. 使用方法

1. 启动前端服务：
```bash
python start_frontend.py
```

2. 访问 `http://localhost:5001`

3. 选择"输入评论文本（使用DeepSeek分析）"

4. 在文本框中输入电商评论，每行一条，至少10条

5. 点击"开始训练模型"

6. 系统将：
   - 使用DeepSeek API分析每条评论的情感倾向
   - 自动标注为：0=负面, 1=中性, 2=正面
   - 将结果保存为CSV文件
   - 使用该CSV文件训练模型

## 注意事项

- DeepSeek API调用需要网络连接
- API调用有频率限制，系统会自动添加延迟（默认0.5秒）
- 如果API调用失败，系统会使用中性标签（1）作为默认值
- 确保API密钥有效且有足够的配额

## 示例评论格式

```
商品质量很好，物流很快，包装精美，非常满意！
质量很差，用了一次就坏了，不推荐
商品一般，没有特别突出的地方
性价比高，使用效果很棒，会再次购买
物流太慢，等了一个星期才到，失望
```

## 故障排除

如果遇到API调用失败：
1. 检查API密钥是否正确设置
2. 检查网络连接
3. 检查API配额是否充足
4. 查看后端日志获取详细错误信息













