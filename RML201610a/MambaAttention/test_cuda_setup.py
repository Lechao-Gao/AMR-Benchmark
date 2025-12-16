#!/usr/bin/env python3
"""
测试CUDA环境设置是否成功解决ptxas问题
"""

import os
import sys
import subprocess
import tensorflow as tf

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from set_cuda_env import set_cuda_environment_variables, print_environment_info

def test_tensorflow_gpu():
    """测试TensorFlow GPU是否正常工作"""
    print("=== TensorFlow GPU测试 ===")
    
    # 检查GPU是否可用
    gpus = tf.config.list_physical_devices('GPU')
    print(f"可用GPU数量: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
            
        # 测试简单的GPU操作
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
                print("✅ GPU矩阵乘法测试成功")
                print(f"结果: {c}")
        except Exception as e:
            print(f"❌ GPU操作失败: {e}")
    else:
        print("❌ 未检测到GPU")

def test_ptxas_compatibility():
    """测试ptxas兼容性"""
    print("\n=== PTXAS兼容性测试 ===")
    
    try:
        # 检查ptxas版本
        result = subprocess.run(["ptxas", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version_info = result.stdout.strip()
            print(f"ptxas版本信息: {version_info}")
            
            # 检查是否包含版本号
            import re
            version_match = re.search(r'release (\d+\.\d+)', version_info)
            if version_match:
                version = float(version_match.group(1))
                print(f"ptxas版本: {version}")
                
                # 检查版本是否支持CC 8.9
                if version >= 11.8:
                    print("✅ ptxas版本支持Compute Capability 8.9")
                else:
                    print("⚠️ ptxas版本可能不支持Compute Capability 8.9")
            else:
                print("⚠️ 无法解析ptxas版本")
        else:
            print("❌ 无法获取ptxas版本信息")
            
    except Exception as e:
        print(f"❌ 检查ptxas时出错: {e}")

def test_cuda_memory_allocation():
    """测试CUDA内存分配"""
    print("\n=== CUDA内存分配测试 ===")
    
    try:
        # 检查GPU内存
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"GPU详情: {details}")
                
                # 设置内存增长
                tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU内存增长已启用")
                
        # 测试内存分配
        with tf.device('/GPU:0'):
            large_tensor = tf.random.normal([1000, 1000])
            print("✅ GPU内存分配测试成功")
            
    except Exception as e:
        print(f"❌ CUDA内存分配测试失败: {e}")

def main():
    """主测试函数"""
    print("开始测试CUDA环境设置...")
    
    # 设置环境变量
    print("正在设置CUDA环境变量...")
    success = set_cuda_environment_variables()
    
    if success:
        print("✅ CUDA环境变量设置完成")
    else:
        print("⚠️ 未找到CUDA路径")
    
    print("\n=== 环境信息 ===")
    print_environment_info()
    
    # 测试各项功能
    test_tensorflow_gpu()
    test_ptxas_compatibility()
    test_cuda_memory_allocation()
    
    print("\n=== 测试完成 ===")
    print("如果所有测试都通过，ptxas警告应该已经解决或减少了")

if __name__ == "__main__":
    main()