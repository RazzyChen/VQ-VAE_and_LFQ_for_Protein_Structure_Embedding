#!/bin/bash

# 脚本功能：解压所有的*.pdb.gz文件，成功后删除原压缩文件

# 检查当前目录是否有pdb.gz文件
if ls *.pdb.gz 1> /dev/null 2>&1; then
    echo "发现PDB压缩文件，开始处理..."
    
    # 遍历所有的.pdb.gz文件
    for file in *.pdb.gz; do
        echo "正在解压: $file"
        
        # 解压文件
        if gunzip -k "$file"; then
            echo "成功解压: $file"
            
            # 检查解压是否成功（检查是否存在对应的.pdb文件）
            pdb_file="${file%.gz}"
            if [ -f "$pdb_file" ]; then
                echo "删除压缩文件: $file"
                rm "$file"
            else
                echo "警告：解压似乎成功但找不到解压后的文件: $pdb_file"
            fi
        else
            echo "解压失败: $file，跳过删除"
        fi
    done
    
    echo "所有文件处理完成!"
else
    echo "当前目录没有找到*.pdb.gz文件"
fi
