#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成轻量版 MambaAttention 模型的结构图与 JSON 描述。

输出:
- PNG/SVG: 使用 keras.utils.plot_model
- JSON: 便于用 Netron 可视化

依赖:
- 安装 Graphviz 并将其 bin 加入 PATH
- pip install pydot graphviz
"""

import argparse
import os
import shutil
import sys

# 将项目根目录加入路径，便于导入模型
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

try:
    from tensorflow.keras.utils import plot_model
    from tensorflow import keras
except ImportError:
    from keras.utils import plot_model
    import keras

from MambaAttention.rmlmodels.MambaAttentionModel import MambaAttentionModel


def build_light_model():
    """构建轻量版 MambaAttention 模型（与 main_lightweight.py 配置一致）。"""
    return MambaAttentionModel(
        input_shape=[2, 128],
        classes=11,
        d_model=128,
        num_mamba_layers=2,
        num_attention_layers=1,
    )


def export_plot(model, output_path: str, show_shapes: bool, show_layer_names: bool):
    if shutil.which("dot") is None:
        print("[WARN] 未检测到 Graphviz 的 dot 可执行文件，跳过绘图。")
        print("       请安装 Graphviz 并将其 bin 加入 PATH，示例路径：C:\\Program Files\\Graphviz\\bin")
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        plot_model(
            model,
            to_file=output_path,
            show_shapes=show_shapes,
            show_layer_names=show_layer_names,
            expand_nested=True,
            dpi=200,
        )
        print(f"[OK] 模型结构图已保存: {output_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 绘图失败: {e}")
        print("       常见原因：未安装 Graphviz，或 pydot 版本问题。")
        print("       解决建议：")
        print("         1) 安装 Graphviz，并将其 bin 加入 PATH")
        print("         2) pip install --upgrade pydot graphviz")
        print("       已跳过绘图，后续 JSON 仍会导出。")


def export_json(model, json_path: str):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(model.to_json())
    print(f"[OK] 模型 JSON 已保存: {json_path}（可用 Netron 打开）")


def main():
    parser = argparse.ArgumentParser(
        description="生成轻量版 MambaAttention 的框架图与 JSON。"
    )
    parser.add_argument(
        "--png",
        default="figure/mamba_attention_light.png",
        help="输出的 PNG/SVG 路径（后缀决定格式）",
    )
    parser.add_argument(
        "--json",
        default="figure/mamba_attention_light.json",
        help="导出的模型 JSON 路径",
    )
    parser.add_argument(
        "--show-shapes",
        action="store_true",
        help="在图中显示张量形状",
    )
    parser.add_argument(
        "--show-layer-names",
        action="store_true",
        help="在图中显示层名称",
    )
    args = parser.parse_args()

    print("=== 构建轻量版 MambaAttention 模型 ===")
    model = build_light_model()

    print("=== 导出模型结构图 ===")
    export_plot(
        model=model,
        output_path=args.png,
        show_shapes=args.show_shapes,
        show_layer_names=args.show_layer_names,
    )

    print("=== 导出模型 JSON ===")
    export_json(model, args.json)

    print("全部完成。")


if __name__ == "__main__":
    main()

