#!/bin/bash

# 检查是否提供了文件名
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <python_file>"
    exit 1
fi

file="$1"

# 检查文件是否存在
if [ ! -f "$file" ]; then
    echo "Error: File '$file' not found!"
    exit 1
fi

# 备份原始文件（可选）
cp "$file" "${file}.bak"

# 使用sed替换所有"cuda"为"xpu"
sed -i 's/cuda/xpu/g' "$file"

echo "Replaced all 'cuda' with 'xpu' in $file"