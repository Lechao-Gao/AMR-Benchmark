#!/usr/bin/env python3
"""
MambaAttention模型断点续训脚本
支持从MambaAttention_best.h5继续训练
"""

import os, random
import sys
import argparse

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import keras
import keras.backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.optimizers import Adam
from common import mltools, rmldataset2016
import rmlmodels.MambaAttentionModel as mamba_attention

def check_checkpoint_exists(filepath='weights/MambaAttention_best.h5'):
    """检查检查点文件是否存在"""
    if os.path.exists(filepath):
        print(f"[INFO] 找到检查点文件: {filepath}")
        print(f"[INFO] 文件大小: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
        return True
    else:
        print(f"[WARNING] 未找到检查点文件: {filepath}")
        print("[INFO] 将从头开始训练")
        return False

def load_existing_model(filepath='weights/MambaAttention_best.h5'):
    """加载已保存的模型权重"""
    try:
        print("\n=== 重建模型结构 ===")
        model = mamba_attention.MambaAttentionModel(
            input_shape=[2, 128],
            classes=11,
            d_model=256,
            num_mamba_layers=4,
            num_attention_layers=2
        )
        
        print(f"[INFO] 正在加载权重: {filepath}")
        model.load_weights(filepath)
        print("[SUCCESS] 权重加载成功")
        
        return model
    except Exception as e:
        print(f"[ERROR] 加载权重失败: {e}")
        raise e

def get_initial_epoch(log_file='training_log.csv'):
    """从训练日志中获取已训练的轮次"""
    if not os.path.exists(log_file):
        print("[INFO] 未找到训练日志，从第0轮开始")
        return 0
    
    try:
        import pandas as pd
        df = pd.read_csv(log_file)
        if len(df) > 0:
            last_epoch = df['epoch'].max() + 1  # +1因为epoch从0开始
            print(f"[INFO] 已训练到第{last_epoch-1}轮，将从第{last_epoch}轮继续")
            return last_epoch
        else:
            print("[INFO] 训练日志为空，从第0轮开始")
            return 0
    except Exception as e:
        print(f"[WARNING] 读取训练日志失败: {e}，从第0轮开始")
        return 0

def setup_callbacks(resume_epoch=0, total_epochs=100):
    """设置训练回调函数"""
    os.makedirs('weights', exist_ok=True)
    
    # 计算剩余训练轮次
    remaining_epochs = max(0, total_epochs - resume_epoch)
    
    callbacks = [
        ModelCheckpoint(
            'weights/MambaAttention_best.h5',
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            verbose=1,
            patience=10,
            min_lr=1e-7
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=50,
            verbose=1,
            mode='auto',
            restore_best_weights=True
        ),
        CSVLogger(
            'training_log.csv',
            append=True  # 追加模式，不会覆盖之前的日志
        )
    ]
    
    return callbacks, remaining_epochs

def resume_training(total_epochs=100, batch_size=128, learning_rate=0.001):
    """断点续训主函数"""
    print("=== MambaAttention模型断点续训 ===")
    
    # 检查检查点
    checkpoint_exists = check_checkpoint_exists()
    
    # 加载数据
    print("\n正在加载RML2016.10a数据集...")
    (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = \
        rmldataset2016.load_data()
    
    # 数据预处理
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    X_val = np.expand_dims(X_val, axis=3)
    
    print(f"训练数据形状: {X_train.shape}")
    print(f"验证数据形状: {X_val.shape}")
    
    # 创建或加载模型
    if checkpoint_exists:
        model = load_existing_model()
    else:
        print("\n=== 创建新模型 ===")
        model = mamba_attention.MambaAttentionModel(
            input_shape=[2, 128],
            classes=11,
            d_model=256,
            num_mamba_layers=4,
            num_attention_layers=2
        )
    
    # 编译模型
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=optimizer
    )
    
    print("\n模型结构概览:")
    model.summary()
    print(f"模型总参数量: {model.count_params():,}")
    
    # 获取初始轮次
    initial_epoch = get_initial_epoch()
    
    # 设置回调函数
    callbacks, remaining_epochs = setup_callbacks(initial_epoch, total_epochs)
    
    if remaining_epochs <= 0:
        print(f"[INFO] 已完成{total_epochs}轮训练，无需继续")
        return model
    
    print(f"\n训练计划:")
    print(f"- 起始轮次: {initial_epoch}")
    print(f"- 剩余轮次: {remaining_epochs}")
    print(f"- 总轮次: {total_epochs}")
    
    # 开始训练
    print("\n开始断点续训...")
    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=total_epochs,  # 总轮次
        initial_epoch=initial_epoch,  # 起始轮次
        verbose=2,
        validation_data=(X_val, Y_val),
        callbacks=callbacks
    )
    
    print("\n[SUCCESS] 断点续训完成")
    return model

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MambaAttention断点续训')
    parser.add_argument('--epochs', type=int, default=100,
                       help='总训练轮次（默认100）')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小（默认128）')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率（默认0.001）')
    
    args = parser.parse_args()
    
    try:
        model = resume_training(
            total_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
        
        # 可选：继续执行评估
        print("\n是否继续执行模型评估？(y/n)")
        choice = input().lower()
        if choice == 'y':
            from main import predict_and_analyze
            predict_and_analyze(model)
            
    except KeyboardInterrupt:
        print("\n[INFO] 训练被用户中断")
    except Exception as e:
        print(f"[ERROR] 训练失败: {e}")
        raise e

if __name__ == "__main__":
    main()