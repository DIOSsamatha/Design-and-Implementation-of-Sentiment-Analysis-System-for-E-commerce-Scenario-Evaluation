import codecs
import sys

file_path = r'k:\毕业设计\sentiment-analysis\frontend\app.py'

# 读取文件
with open(file_path, 'rb') as f:
    content = f.read()

# 检查并去除BOM
bom = codecs.BOM_UTF8
if content.startswith(bom):
    content = content[len(bom):]
    print("BOM found and removed")
else:
    print("No BOM found")

# 重新保存为UTF-8无BOM
with open(file_path, 'wb') as f:
    f.write(content)

print("File saved without BOM")
