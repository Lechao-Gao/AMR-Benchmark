#!/usr/bin/env python3
"""
CUDA环境变量动态设置工具
用于解决ptxas版本不兼容问题
"""

import os
import subprocess
import sys
from pathlib import Path

def find_cuda_bin_path():
    """查找CUDA的bin目录"""
    possible_paths = [
        "/usr/local/cuda/bin",
        "/opt/cuda/bin",
        "/usr/local/cuda-*/bin",
        os.path.expanduser("~/cuda/bin"),
        "/usr/bin"
    ]
    
    for path_pattern in possible_paths:
        import glob
        for path in glob.glob(path_pattern):
            if os.path.isdir(path) and "ptxas" in os.listdir(path):
                return path
    return None

def get_cuda_version_info():
    """获取CUDA版本信息"""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
    except:
        pass
    return "无法获取CUDA版本信息"

def set_cuda_environment_variables():
    """设置CUDA相关的环境变量"""
    
    # 1. 查找CUDA路径
    cuda_bin_path = find_cuda_bin_path()
    
    if cuda_bin_path:
        print(f"找到CUDA bin路径: {cuda_bin_path}")
        
        # 将CUDA bin添加到PATH前面，确保使用正确的ptxas
        current_path = os.environ.get("PATH", "")
        if cuda_bin_path not in current_path:
            new_path = f"{cuda_bin_path}:{current_path}"
            os.environ["PATH"] = new_path
            print(f"更新PATH环境变量: {cuda_bin_path} 已添加到PATH")
    
    # 2. 设置其他CUDA相关的环境变量
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 3. 设置TensorFlow特定的环境变量
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 减少TensorFlow日志输出
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # 动态分配GPU内存
    
    # 4. 可选：设置PTXAS路径（如果知道具体位置）
    ptxas_path = os.path.join(cuda_bin_path, "ptxas") if cuda_bin_path else None
    if ptxas_path and os.path.exists(ptxas_path):
        os.environ["PTXAS_PATH"] = ptxas_path
        print(f"设置PTXAS_PATH: {ptxas_path}")
    
    return cuda_bin_path is not None

def print_environment_info():
    """打印当前环境信息"""
    print("=== CUDA环境信息 ===")
    print(f"PATH: {os.environ.get('PATH', '未设置')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    print(f"CUDA_DEVICE_ORDER: {os.environ.get('CUDA_DEVICE_ORDER', '未设置')}")
    print(f"TF_CPP_MIN_LOG_LEVEL: {os.environ.get('TF_CPP_MIN_LOG_LEVEL', '未设置')}")
    print(f"TF_FORCE_GPU_ALLOW_GROWTH: {os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', '未设置')}")
    
    print("\n=== CUDA版本信息 ===")
    print(get_cuda_version_info())
    
    print("\n=== 当前ptxas路径 ===")
    try:
        result = subprocess.run(["which", "ptxas"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"ptxas路径: {result.stdout.strip()}")
            
            # 获取ptxas版本
            version_result = subprocess.run(["ptxas", "--version"], capture_output=True, text=True)
            if version_result.returncode == 0:
                print(f"ptxas版本: {version_result.stdout.strip()}")
        else:
            print("ptxas未找到")
    except Exception as e:
        print(f"查找ptxas时出错: {e}")

if __name__ == "__main__":
    print("正在设置CUDA环境变量...")
    success = set_cuda_environment_variables()
    
    if success:
        print("✅ CUDA环境变量设置成功")
    else:
        print("⚠️ 未找到CUDA路径，请检查CUDA安装")
    
    print_environment_info()