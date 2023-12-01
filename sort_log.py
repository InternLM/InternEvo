import re

# 读取日志信息
with open("ht.log", "r") as file:
    log_content = file.read()

# 使用正则表达式提取以 "ht debug" 开头、以 "dtype=***" 结尾的日志信息块
log_blocks = re.findall(r"ht debug.*?device=[^\n]*", log_content, re.DOTALL)

# 将日志信息块按照 "rank:" 后的整数值进行正序排序
sorted_log_blocks = sorted(log_blocks, key=lambda x: int(re.search(r"rank:(\d+)", x).group(1)))

# 将排序后的日志信息块写入新的文件
with open("sorted.log", "w") as file:
    file.write("\n\n".join(sorted_log_blocks))

print("日志信息块已按照 rank: 后的整数值进行正序排序，并保存在 sorted_log_blocks.txt 文件中。")
