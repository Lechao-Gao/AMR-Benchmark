#!/usr/bin/env python3
"""
轻量级Mamba模型训练与自动评估
训练结束后自动进行预测和性能分析
"""

import os
import sys
import argparse
import csv
import pickle

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import keras
import keras.backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib_chinese_init
from common import mltools, rmldataset2016
import rmlmodels.LightweightMambaModel as lightweight_mamba

def train_and_evaluate(epochs=50, batch_size=256, learning_rate=0.001):
    """训练轻量级Mamba模型并自动评估"""
    
    print("=== 轻量级Mamba模型训练与自动评估 ===")
    print("优化配置:")
    print("- Mamba层数: 4层 → 2层")
    print("- 模型维度: 256维 → 128维")
    print("- 注意力层: 2层 → 1层")
    print("- 参数量减少: ~70%")
    print("- 预期训练时间减少: ~65-70%")
    
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
    print(f"测试数据形状: {X_test.shape}")
    
    classes = mods
    print(f"调制类型: {classes}")
    print(f"信噪比范围: {snrs}")
    
    # 创建优化模型
    print("\n正在构建轻量级Mamba模型...")
    model = lightweight_mamba.LightweightMambaModel(
        input_shape=[2, 128],
        classes=11,
        d_model=128,  # 减少维度
        num_mamba_layers=2,  # 减少到2层
        num_attention_layers=1  # 减少到1层
    )
    
    # 编译模型
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=optimizer
    )
    
    print("\n优化后模型结构:")
    model.summary()
    print(f"优化后模型总参数量: {model.count_params():,}")
    
    # 计算优化效果
    from rmlmodels.MambaAttentionModel import MambaAttentionModel
    original_model = MambaAttentionModel(
        input_shape=[2, 128],
        classes=11,
        d_model=256,
        num_mamba_layers=4,
        num_attention_layers=2
    )
    
    original_params = original_model.count_params()
    optimized_params = model.count_params()
    reduction = (original_params - optimized_params) / original_params * 100
    
    print(f"\n优化效果:")
    print(f"  原始参数量: {original_params:,}")
    print(f"  优化后参数量: {optimized_params:,}")
    print(f"  参数量减少: {reduction:.1f}%")
    
    # 设置回调函数
    os.makedirs('weights', exist_ok=True)
    os.makedirs('figure', exist_ok=True)
    os.makedirs('predictresult', exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            'weights/LightweightMamba_best.h5',
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
            patience=80,
            verbose=1,
            mode='auto',
            restore_best_weights=True
        ),
        CSVLogger(
            'training_log_lightweight.csv',
            append=False
        )
    ]
    
    print("\n开始训练...")
    print("注意: 由于模型优化，训练时间将大幅减少")
    
    # 训练模型
    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(X_val, Y_val),
        callbacks=callbacks
    )
    
    # 评估模型
    print("\n正在评估模型性能...")
    score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print(f"测试损失: {score[0]:.4f}")
    print(f"测试准确率: {score[1]:.4f}")
    
    # 训练结束后自动进行预测和评估
    print("\n=== 训练完成，开始自动评估 ===")
    predict_and_analyze(model, X_test, Y_test, classes, snrs, test_idx, lbl, batch_size)
    
    return model, history

def predict_and_analyze(model, X_test, Y_test, classes, snrs, test_idx, lbl, batch_size=256):
    """训练结束后的自动预测和评估"""
    print("\n=== 轻量级Mamba模型自动评估 ===")
    
    # 整体预测
    print("[INFO] 正在进行整体预测...")
    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    
    # 计算整体准确率
    overall_acc = np.mean(np.argmax(test_Y_hat, axis=1) == np.argmax(Y_test, axis=1))
    print(f"[RESULT] 整体测试准确率: {overall_acc:.4f}")
    
    # 生成混淆矩阵
    print("[INFO] 正在生成混淆矩阵...")
    confnorm, _, _ = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, classes)
    
    # 保存混淆矩阵图
    mltools.plot_confusion_matrix(
        confnorm,
        labels=['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK','GFSK','4-PAM','16-QAM','64-QAM','QPSK','WBFM'],
        title="轻量级Mamba模型混淆矩阵",
        save_filename='figure/light_mamba/lightweight_mamba_total_confusion'
    )
    
    # 按信噪比分析性能
    print("[INFO] 正在分析各信噪比下的性能...")
    acc = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    
    for i, snr in enumerate(snrs):
        test_SNRs = [lbl[x][1] for x in test_idx]
        
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]
        
        if len(test_X_i) == 0:
            continue
            
        # 预测
        test_Y_i_hat = model.predict(test_X_i, batch_size=batch_size)
        confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, classes)
        
        acc[snr] = 1.0 * cor / (cor + ncor)
        
        # 保存准确率结果
        with open('predictresult/lightweight_accuray_res.csv', 'a', newline='') as f0:
            write0 = csv.writer(f0)
            write0.writerow([acc[snr]])
        
        # 生成混淆矩阵图
        mltools.plot_confusion_matrix(
            confnorm_i,
            labels=['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK','GFSK','4-PAM','16-QAM','64-QAM','QPSK','WBFM'],
            title=f"轻量级Mamba模型 (SNR={snr}dB)",
            save_filename=f"figure/light_mamba/LightweightMamba_Confusion(SNR={snr})(ACC={100.0*acc[snr]:.2f}).png"
        )
        
        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i)/np.sum(confnorm_i, axis=1), 3)
    
    # 绘制性能曲线
    print("[INFO] 正在生成性能曲线...")
    plot_performance_curves(acc, acc_mod_snr, classes, snrs)
    
    # 保存结果
    save_evaluation_results(acc, acc_mod_snr, classes, snrs)
    
    return acc, acc_mod_snr

def plot_performance_curves(acc, acc_mod_snr, classes, snrs):
    """绘制性能曲线"""
    
    # 绘制各调制类型的准确率曲线
    plt.figure(figsize=(12, 8))
    
    for i, mod in enumerate(classes):
        plt.plot(snrs, acc_mod_snr[i], label=mod, marker='o')
        # 添加数值标签
        for x, y in zip(snrs, acc_mod_snr[i]):
            if not np.isnan(y):
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel("信噪比 (dB)")
    plt.ylabel("分类准确率")
    plt.title("轻量级Mamba模型 - 各调制类型分类准确率")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figure/light_mamba/lightweight_mamba_acc_with_mod.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制整体准确率曲线
    plt.figure(figsize=(10, 6))
    snr_list = list(acc.keys())
    acc_list = [acc[x] for x in snr_list]
    
    plt.plot(snr_list, acc_list, 'b-o', linewidth=2, markersize=6)
    plt.xlabel("信噪比 (dB)")
    plt.ylabel("分类准确率")
    plt.title("轻量级Mamba模型 - 整体分类准确率曲线")
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for x, y in zip(snr_list, acc_list):
        plt.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figure/light_mamba/lightweight_mamba_overall_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_evaluation_results(acc, acc_mod_snr, classes, snrs):
    """保存评估结果"""
    
    # 保存各调制类型准确率数据
    with open('predictresult/lightweight_acc_for_mod.dat', 'wb') as f:
        pickle.dump(acc_mod_snr, f)
    
    # 保存整体准确率数据
    with open('predictresult/lightweight_results.dat', 'wb') as f:
        pickle.dump(acc, f)
    
    # 保存总体准确率（ResNet格式）
    print(acc)
    with open('predictresult/mamba_attention_analyzer_d0.5.dat', 'wb') as f:
        pickle.dump(("mamba_attention_analyzer", 0.5, acc), f)
    
    # 计算性能统计
    avg_acc = np.mean(list(acc.values()))
    max_acc = max(acc.values())
    min_acc = min(acc.values())
    
    print(f"\n=== 轻量级Mamba模型性能总结 ===")
    print(f"平均准确率: {avg_acc:.4f}")
    print(f"最高准确率: {max_acc:.4f} (SNR = {max(acc, key=acc.get)} dB)")
    print(f"最低准确率: {min_acc:.4f} (SNR = {min(acc, key=acc.get)} dB)")
    
    return avg_acc, max_acc, min_acc

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='轻量级Mamba模型训练与自动评估')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='学习率')
    
    args = parser.parse_args()
    
    print("=== 轻量级Mamba模型训练与自动评估 ===")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    
    # 训练模型并自动评估
    model, history = train_and_evaluate(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("\n=== 训练与评估完成 ===")
    print("结果文件:")
    print("- 最佳模型权重: weights/LightweightMamba_best.h5")
    print("- 训练日志: training_log_lightweight.csv")
    print("- 混淆矩阵图: figure/light_mamba/lightweight_mamba_*.png")
    print("- 准确率曲线: figure/light_mamba/lightweight_mamba_*.png")
    print("- 结果数据: predictresult/lightweight_*.dat")

if __name__ == "__main__":
    main()