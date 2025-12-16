"""
独立的 t-SNE 可视化脚本（MambaAttention 系列）
无需重新训练，直接加载已有权重并输出特征分布图。
"""

import os
import sys
import argparse

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

try:
    from tensorflow import keras
except ImportError:
    import keras

from common import mltools, rmldataset2016
import rmlmodels.EnhancedMambaModel as enhanced_mamba


def load_weights_or_exit(model, weights_path):
    """加载权重; 若失败则打印错误并退出。"""
    if not os.path.exists(weights_path):
        print(f"✗ 权重文件不存在: {weights_path}")
        sys.exit(1)

    try:
        model.load_weights(weights_path)
        print(f"✓ 成功加载权重: {weights_path}")
    except Exception as e:
        print(f"✗ 加载权重失败: {e}")
        sys.exit(1)


def pick_snr_subset(snrs, desired=None, max_count=5):
    """筛选代表性 SNR 列表。"""
    desired = desired or [-20, -10,-6,-4,-2, 0, 2, 4, 6, 10, 18]
    filtered = [s for s in desired if s in snrs]
    if len(filtered) < max_count:
        extras = [s for s in snrs if s not in filtered]
        filtered.extend(extras[:max_count - len(filtered)])
    return filtered


def main():
    parser = argparse.ArgumentParser(description="MambaAttention t-SNE 可视化脚本")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/EnhancedMamba_best.h5",
        help="模型权重路径（.h5）",
    )
    parser.add_argument(
        "--feature-layer",
        type=str,
        default="dense3",
        help="提取特征的层名称（需存在于模型中）",
    )
    parser.add_argument(
        "--samples-per-snr",
        type=int,
        default=800,
        help="每个 SNR 使用的最大样本数",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="figure/tsne_mamba_attention",
        help="输出图片前缀",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MambaAttention t-SNE 可视化脚本")
    print("=" * 60)

    os.makedirs("figure", exist_ok=True)

    print("\n正在加载 RML2016.10a 数据集...")
    (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = (
        rmldataset2016.load_data()
    )
    X_test = np.expand_dims(X_test, axis=3)
    print(f"测试集形状: {X_test.shape}")
    print(f"调制方式: {mods}")
    print(f"SNR 范围: {min(snrs)} ~ {max(snrs)} dB")

    print("\n正在构建 EnhancedMamba 模型骨干（推理用）...")
    model = enhanced_mamba.EnhancedMambaModel(
        input_shape=[2, 128],
        classes=len(mods),
        d_model=256,
        num_mamba_layers=3,
        num_attention_layers=2,
    )

    load_weights_or_exit(model, args.weights)

    test_SNRs = [lbl[idx][1] for idx in test_idx]
    selected_snrs = pick_snr_subset(snrs)
    print(f"\nt-SNE 将使用以下 SNR: {selected_snrs}")

    print("\n开始生成 t-SNE 可视化图...")
    try:
        mltools.plot_tsne_visualization(
            model=model,
            X_data=X_test,
            Y_data=Y_test,
            snr_data=test_SNRs,
            classes=mods,
            snrs=snrs,
            feature_layer_name=args.feature_layer,
            save_filename=args.output_prefix,
            selected_snrs=selected_snrs,
            n_samples_per_snr=args.samples_per_snr,
            perplexity=30,
            n_iter=1000,
        )
        print("✓ t-SNE 可视化完成，图像输出至 figure/ 目录。")
    except Exception as e:
        print(f"✗ 生成 t-SNE 可视化失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

