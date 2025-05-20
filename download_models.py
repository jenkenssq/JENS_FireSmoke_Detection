#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JENS火灾烟雾报警系统模型下载脚本
下载预训练的YOLOv5模型用于火灾烟雾检测
"""

import os
import sys
import torch
import requests
from pathlib import Path
from tqdm import tqdm

# 模型目录
MODELS_DIR = Path("JENS_models")
MODELS_DIR.mkdir(exist_ok=True)

def download_file(url, target_path, chunk_size=1024):
    """下载文件到指定路径"""
    # 获取文件大小
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # 创建目标文件
    target_path = Path(target_path)
    
    # 检查是否已存在
    if target_path.exists():
        if target_path.stat().st_size == total_size:
            print(f"文件已存在: {target_path}，跳过下载")
            return True
        else:
            print(f"文件已存在但不完整，重新下载: {target_path}")
    
    # 下载文件
    print(f"下载文件: {url} -> {target_path}")
    
    with open(target_path, 'wb') as f:
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))
        
        progress_bar.close()
    
    return True

def download_yolov5s():
    """下载YOLOv5s模型"""
    url = "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt"
    target_path = MODELS_DIR / "yolov5s.pt"
    return download_file(url, target_path)

def download_yolov5m():
    """下载YOLOv5m模型"""
    url = "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt"
    target_path = MODELS_DIR / "yolov5m.pt"
    return download_file(url, target_path)

def download_custom_model():
    """下载自定义训练的火灾烟雾检测模型（示例）"""
    # 注意：这里是一个示例URL，实际使用时应替换为真实的已训练模型URL
    url = "https://github.com/spacewalk01/yolov5-fire-detection/blob/main/model/yolov5s_best.pt"
    target_path = MODELS_DIR / "fire_smoke_yolov5s.pt"
    return download_file(url, target_path)

def main():
    """主函数"""
    print("JENS火灾烟雾报警系统 - 模型下载工具")
    print("====================================")
    
    # 创建必要的目录结构
    create_dirs()
    
    # 下载模型选项
    options = {
        "1": ("YOLOv5s (小型)", download_yolov5s),
        "2": ("YOLOv5m (中型)", download_yolov5m),
        "3": ("火灾烟雾专用模型", download_custom_model),
        "4": ("全部模型", None)
    }
    
    while True:
        print("\n请选择要下载的模型:")
        for key, (name, _) in options.items():
            print(f"{key}. {name}")
        print("0. 退出")
        
        choice = input("请输入选项 (0-4): ")
        
        if choice == "0":
            break
        elif choice in options:
            if choice == "4":
                # 下载所有模型
                download_yolov5s()
                download_yolov5m()
                download_custom_model()
                print("所有模型下载完成！")
            else:
                _, download_func = options[choice]
                if download_func():
                    print("模型下载完成！")
        else:
            print("无效选项，请重新选择。")

def create_dirs():
    """创建必要的目录结构"""
    dirs = [
        MODELS_DIR,
        Path("JENS_assets/icons"),
        Path("JENS_assets/sounds"),
        Path("JENS_assets/styles")
    ]
    
    for d in dirs:
        d.mkdir(exist_ok=True, parents=True)
        print(f"检查目录: {d}")

if __name__ == "__main__":
    main() 