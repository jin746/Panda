#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大熊猫个体识别测试脚本
使用训练好的ArcFace+Triplet+原型网络模型进行推理
确保与训练时的模型完全一致
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np
from PIL import Image
import json
from collections import defaultdict, Counter
import re


# 聚类和可视化相关导入
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        roc_auc_score,
        roc_curve,
        average_precision_score,
        precision_recall_curve,
    )
    import seaborn as sns

    CLUSTERING_AVAILABLE = True
except ImportError as e:
    print(f"警告: 聚类可视化功能不可用，缺少依赖: {e}")
    CLUSTERING_AVAILABLE = False

# ======================= 修改记录 =======================
# 2025-12-07:
#   1) 增加 ReID 开放集评估指标 (mAP / AUROC / OSCR / ODR / FMR)，并与年龄 MAE、性别准确率一并输出；
#   2) 为 custom 模式增加一组与 video_open_world_reid_yewai.py 中 USER_DEFAULTS 一致的默认推理参数，
#      便于直接一行命令进行评估与推理。
# 2025-12-08:
#   1) 新增 --compute-open-set-metrics 评估开关。
#   2) 默认阈值/采样点可通过 --open-set-threshold 与 --oscr-points 控制，满足 GPU 24G 4090 推理需求。
#   3) 统一清理 Tab -> 4 空格、修复 parse_option / apply_mode_settings 等处缩进与不可见字符问题，保证脚本无 TabError / 语法错误。
#   4) 在“测试集与训练集完全无交集”的设定下，重构 compute_open_set_metrics：完全移除 ODR/FMR/OSCR 与 is_new_id 依赖，
#      仅基于真值 ID + 特征向量计算 mAP / AUROC / Rank-1,5,10 / CMC / TAR@FAR / P-R-F1 / intra-ID 与 inter-ID 相似度统计，
#      并将主要指标写入 open_set_metrics.json，避免出现对当前场景无语义的开放集评分。
#   5) 修复聚类结果打印中 if/else 缩进混用导致的 SyntaxError / TabError 问题。
#   6) 在 compute_open_set_metrics 中增加成对关系的单一阈值准确率 accuracy = (TP+TN)/M，
#      使用基于 precision-recall 曲线得到的最佳 F1 阈值作为固定判定规则。
#   7) 为避免大规模数据集上构造 N×N 相似度矩阵导致的 283GiB OOM，对 compute_open_set_metrics 进行按 ID / probe / pair 子采样
#      与分块点积重写，在不改变指标含义的前提下显著降低内存与计算开销。
#   8) 根据“只需 JSON 结果、不需要逐 ID 图像可视化”的实际需求，简化 generate_output：不再为每个预测 ID 复制原图和 ROI，
#      仅输出识别结果 JSON 与必要统计信息，减少 I/O 与代码冗余。
# ======================================================

# 与 video_open_world_reid_yewai.py 保持一致的推理默认参数（custom 模式）
INFERENCE_DEFAULTS = {
    "mode": "custom",
    "similarity_threshold": 0.2,
    "base_threshold": 0.2,
    "adaptive_threshold_min": 0.15,
    "adaptive_threshold_max": 0.6,
    "confidence_threshold": 0.2,
    "quality_threshold": 0.05,
    "gender_threshold": 0.5,
    "gender_hysteresis": 0.03,
    "age_scope": "video",
    "age_display": "median",
    "sim_cosine_w": 0.7,
    "sim_euclid_w": 0.3,
    "aux_gender_penalty": 0.2,
    "aux_age_reweight": 0.1,
    "aux_min_age_sigma": 2.0,
}

# 开放集评估指标设置
OPEN_SET_METRICS = {
    "odr_threshold": 0.5,  # FMR/ODR 统计使用的相似度阈值
    "oscr_sample_points": 200,  # OSCR 曲线的采样点数量
}


# 添加项目路径
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.append(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

from config import get_config
from data.panda_dataset import PandaDataset, build_panda_transform
from models.panda_reid_model import build_panda_reid_model
from models.prototype_reid_network import PrototypeReIDNetwork
from models.open_world_metrics import compute_open_world_cluster_metrics
from logger import create_logger


class ClusteringVisualizer:
    """聚类可视化器"""

    def __init__(self, max_clusters=10, method="auto", dimensions=2):
        """
        初始化聚类可视化器

        Args:
            max_clusters: 最大聚类数量
            method: 聚类方法
            dimensions: 可视化维度
        """
        self.max_clusters = max_clusters
        self.method = method
        self.dimensions = dimensions
        self.colors = plt.cm.Set3(np.linspace(0, 1, max_clusters))

    def find_optimal_clusters(self, features, min_clusters=2):
        """
        寻找最优聚类数量

        Args:
            features: 特征矩阵 [N, D]
            min_clusters: 最小聚类数量

        Returns:
            optimal_k: 最优聚类数量
            scores: 各种评估分数
        """
        if not CLUSTERING_AVAILABLE:
            return min_clusters, {}

        print(f" 寻找最优聚类数量 (范围: {min_clusters}-{self.max_clusters})...")

        k_range = range(min_clusters, min(self.max_clusters + 1, len(features)))
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []

        for k in k_range:
            # KMeans聚类
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)

            # 计算评估指标
            sil_score = silhouette_score(features, cluster_labels)
            cal_score = calinski_harabasz_score(features, cluster_labels)
            db_score = davies_bouldin_score(features, cluster_labels)

            silhouette_scores.append(sil_score)
            calinski_scores.append(cal_score)
            davies_bouldin_scores.append(db_score)

            print(
                f"  K={k}: 轮廓系数={sil_score:.3f}, CH指数={cal_score:.1f}, DB指数={db_score:.3f}"
            )

        # 选择最优K值（轮廓系数最高）
        optimal_idx = np.argmax(silhouette_scores)
        optimal_k = k_range[optimal_idx]

        scores = {
            "k_range": list(k_range),
            "silhouette_scores": silhouette_scores,
            "calinski_scores": calinski_scores,
            "davies_bouldin_scores": davies_bouldin_scores,
            "optimal_k": optimal_k,
            "optimal_silhouette": silhouette_scores[optimal_idx],
        }

        print(
            f" 最优聚类数量: K={optimal_k} (轮廓系数={scores['optimal_silhouette']:.3f})"
        )

        return optimal_k, scores

    def perform_clustering(self, features, n_clusters=None):
        """
        执行聚类

        Args:
            features: 特征矩阵 [N, D]
            n_clusters: 聚类数量，None时自动选择

        Returns:
            cluster_labels: 聚类标签
            cluster_centers: 聚类中心（如果适用）
        """
        if not CLUSTERING_AVAILABLE:
            return np.zeros(len(features)), None

        if n_clusters is None:
            n_clusters, _ = self.find_optimal_clusters(features)

        print(f" 执行聚类: 方法={self.method}, 聚类数={n_clusters}")

        if self.method == "auto" or self.method == "kmeans":
            # KMeans聚类
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(features)
            cluster_centers = clusterer.cluster_centers_

        elif self.method == "dbscan":
            # DBSCAN聚类
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(features)
            cluster_centers = None

        elif self.method == "hierarchical":
            # 层次聚类
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(features)
            cluster_centers = None

        # 统计聚类结果
        unique_labels = np.unique(cluster_labels)
        print(f" 聚类完成: 发现{len(unique_labels)}个聚类")

        for label in unique_labels:
            count = np.sum(cluster_labels == label)
            if label == -1:
                print(f"  噪声点: {count}个")
            else:
                print(f"  聚类{label}: {count}个样本")

        return cluster_labels, cluster_centers

    def reduce_dimensions(self, features, method="pca"):
        """
        降维处理

        Args:
            features: 特征矩阵 [N, D]
            method: 降维方法 ('pca' 或 'tsne')

        Returns:
            reduced_features: 降维后的特征 [N, dimensions]
        """
        if not CLUSTERING_AVAILABLE:
            return features[:, : self.dimensions]

        print(f" 降维处理: {method.upper()} -> {self.dimensions}D")

        if method == "pca":
            reducer = PCA(n_components=self.dimensions, random_state=42)
            reduced_features = reducer.fit_transform(features)
            explained_ratio = reducer.explained_variance_ratio_
            print(f"  PCA解释方差比: {explained_ratio}")

        elif method == "tsne":
            # 如果特征维度太高，先用PCA降到50维
            if features.shape[1] > 50:
                pca = PCA(n_components=50, random_state=42)
                features = pca.fit_transform(features)
                print(f"  预处理: PCA降维到50维")

            reducer = TSNE(
                n_components=self.dimensions,
                random_state=42,
                perplexity=min(30, len(features) - 1),
            )
            reduced_features = reducer.fit_transform(features)

        return reduced_features

    def create_clustering_visualization(
        self, features, cluster_labels, image_paths, output_path, title="聚类可视化"
    ):
        """
        创建聚类可视化图

        Args:
            features: 降维后的特征 [N, dimensions]
            cluster_labels: 聚类标签
            image_paths: 图像路径列表
            output_path: 输出路径
            title: 图标题
        """
        if not CLUSTERING_AVAILABLE:
            print("聚类可视化功能不可用")
            return

        print(f" 创建聚类可视化图...")

        # 设置图形大小和样式
        plt.style.use("seaborn-v0_8")
        # 设置中文字体
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        if self.dimensions == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        else:
            fig = plt.figure(figsize=(20, 8))
            ax1 = fig.add_subplot(121, projection="3d")
            ax2 = fig.add_subplot(122)

        # 获取唯一的聚类标签
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)

        # 生成颜色
        colors = plt.cm.Set3(np.linspace(0, 1, max(n_clusters, 10)))

        # 主聚类图
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            color = colors[i % len(colors)]

            if label == -1:
                # 噪声点
                if self.dimensions == 2:
                    ax1.scatter(
                        features[mask, 0],
                        features[mask, 1],
                        c="black",
                        marker="x",
                        s=50,
                        alpha=0.6,
                        label="噪声",
                    )
                else:
                    ax1.scatter(
                        features[mask, 0],
                        features[mask, 1],
                        features[mask, 2],
                        c="black",
                        marker="x",
                        s=50,
                        alpha=0.6,
                        label="噪声",
                    )
        else:
            # 正常聚类
            if self.dimensions == 2:
                ax1.scatter(
                    features[mask, 0],
                    features[mask, 1],
                    c=[color],
                    s=60,
                    alpha=0.7,
                    label=f"聚类 {label}",
                )
            else:
                ax1.scatter(
                    features[mask, 0],
                    features[mask, 1],
                    features[mask, 2],
                    c=[color],
                    s=60,
                    alpha=0.7,
                    label=f"聚类 {label}",
                )

        ax1.set_title(f"{title} - 聚类结果", fontsize=14, fontweight="bold")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        # 聚类统计图
        cluster_counts = []
        cluster_names = []

        for label in unique_labels:
            count = np.sum(cluster_labels == label)
            if label == -1:
                cluster_names.append("噪声")
            else:
                cluster_names.append(f"聚类 {label}")
            cluster_counts.append(count)

        bars = ax2.bar(
            cluster_names,
            cluster_counts,
            color=[colors[i % len(colors)] for i in range(len(unique_labels))],
        )
        ax2.set_title("聚类分布统计", fontsize=14, fontweight="bold")
        ax2.set_ylabel("样本数量")
        ax2.tick_params(axis="x", rotation=45)

        # 在柱状图上添加数值标签
        for bar, count in zip(bars, cluster_counts):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{count}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f" 聚类可视化图已保存: {output_path}")

    def create_evaluation_plots(self, scores, output_dir):
        """
        创建聚类评估图表

        Args:
            scores: 评估分数字典
            output_dir: 输出目录
        """
        if not CLUSTERING_AVAILABLE or not scores:
            return

        print(f" 创建聚类评估图表...")

        # 设置中文字体
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        k_range = scores["k_range"]

        # 轮廓系数
        axes[0, 0].plot(
            k_range, scores["silhouette_scores"], "bo-", linewidth=2, markersize=8
        )
        axes[0, 0].axvline(
            x=scores["optimal_k"], color="red", linestyle="--", alpha=0.7
        )
        axes[0, 0].set_title("轮廓系数 (Silhouette Score)", fontweight="bold")
        axes[0, 0].set_xlabel("聚类数量 K")
        axes[0, 0].set_ylabel("轮廓系数")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].text(
            scores["optimal_k"],
            max(scores["silhouette_scores"]) * 0.9,
            f'最优K={scores["optimal_k"]}',
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

        # Calinski-Harabasz指数
        axes[0, 1].plot(
            k_range, scores["calinski_scores"], "go-", linewidth=2, markersize=8
        )
        axes[0, 1].axvline(
            x=scores["optimal_k"], color="red", linestyle="--", alpha=0.7
        )
        axes[0, 1].set_title("Calinski-Harabasz 指数", fontweight="bold")
        axes[0, 1].set_xlabel("聚类数量 K")
        axes[0, 1].set_ylabel("CH指数")
        axes[0, 1].grid(True, alpha=0.3)

        # Davies-Bouldin指数
        axes[1, 0].plot(
            k_range, scores["davies_bouldin_scores"], "ro-", linewidth=2, markersize=8
        )
        axes[1, 0].axvline(
            x=scores["optimal_k"], color="red", linestyle="--", alpha=0.7
        )
        axes[1, 0].set_title("Davies-Bouldin 指数", fontweight="bold")
        axes[1, 0].set_xlabel("聚类数量 K")
        axes[1, 0].set_ylabel("DB指数 (越小越好)")
        axes[1, 0].grid(True, alpha=0.3)

        # 综合评估
        # 归一化分数
        sil_norm = np.array(scores["silhouette_scores"])
        cal_norm = np.array(scores["calinski_scores"]) / max(scores["calinski_scores"])
        db_norm = 1 - np.array(scores["davies_bouldin_scores"]) / max(
            scores["davies_bouldin_scores"]
        )

        composite_score = (sil_norm + cal_norm + db_norm) / 3

        axes[1, 1].plot(
            k_range, composite_score, "mo-", linewidth=2, markersize=8, label="综合分数"
        )
        axes[1, 1].axvline(
            x=scores["optimal_k"], color="red", linestyle="--", alpha=0.7
        )
        axes[1, 1].set_title("综合评估分数", fontweight="bold")
        axes[1, 1].set_xlabel("聚类数量 K")
        axes[1, 1].set_ylabel("综合分数")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        eval_path = os.path.join(output_dir, "clustering_evaluation.png")
        plt.savefig(eval_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f" 聚类评估图表已保存: {eval_path}")


def parse_option():
    """解析命令行参数"""
    parser = argparse.ArgumentParser("Panda ReID Inference Script")

    # 基本路径参数
    parser.add_argument("--cfg", type=str, required=True, help="配置文件路径")
    parser.add_argument(
        "--model-path", type=str, required=True, help="训练好的模型路径"
    )
    parser.add_argument(
        "--test-image-root",
        type=str,
        required=False,
        help="测试图像根目录（与ROI配合使用）",
    )
    parser.add_argument(
        "--test-roi-root",
        type=str,
        required=False,
        help="测试ROI根目录（与原图配合使用）",
    )
    parser.add_argument(
        "--test-roiimg-root",
        type=str,
        required=False,
        help="测试ROI图像根目录 (可选，提供该项即可不需要test-image-root/test-roi-root)",
    )
    parser.add_argument("--output-root", type=str, required=True, help="输出结果根目录")
    parser.add_argument(
        "--output-roi-root",
        type=str,
        help='ROI图像输出根目录 (可选，默认为output-root + "_roi")',
    )

    # 掩码 / 热力图相关
    parser.add_argument(
        "--roi-format",
        type=str,
        default="mask",
        choices=["yolo", "mask", "auto"],
        help="ROI数据格式 (默认: mask分割掩码, yolo: YOLO格式, auto: 自动检测)",
    )
    parser.add_argument(
        "--mask-root", type=str, help="掩码文件根目录（默认使用test-roi-root）"
    )
    parser.add_argument(
        "--heatmap", action="store_true", help="是否生成热力图并保存到输出目录/heatmap"
    )
    parser.add_argument(
        "--heatmap-overlay-original",
        action="store_true",
        help="热力图是否贴回原图对应ROI位置",
    )
    parser.add_argument(
        "--heatmap-image-root",
        type=str,
        required=False,
        help="原始图像根目录（仅ROIIMG输入时用于将热力图贴回原图）",
    )
    parser.add_argument(
        "--heatmap-roi-root",
        type=str,
        required=False,
        help="热力图贴回所用的ROI坐标根目录（若与mask-root不同，可单独指定；支持mask/yolo）",
    )

    # 控制ID数量的关键参数（默认与视频脚本 USER_DEFAULTS 保持一致）
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=INFERENCE_DEFAULTS["similarity_threshold"],
        help="相似度阈值 (降低=减少ID数量, 建议范围: 0.1-0.6)",
    )
    parser.add_argument(
        "--base-threshold",
        type=float,
        default=INFERENCE_DEFAULTS["base_threshold"],
        help="基础阈值 (降低=减少ID数量, 建议范围: 0.1-0.5)",
    )
    parser.add_argument(
        "--adaptive-threshold-min",
        type=float,
        default=INFERENCE_DEFAULTS["adaptive_threshold_min"],
        help="自适应阈值最小值 (降低=减少ID数量)",
    )
    parser.add_argument(
        "--adaptive-threshold-max",
        type=float,
        default=INFERENCE_DEFAULTS["adaptive_threshold_max"],
        help="自适应阈值最大值 (降低=减少ID数量)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=INFERENCE_DEFAULTS["confidence_threshold"],
        help="置信度阈值 (降低=减少ID数量)",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=INFERENCE_DEFAULTS["quality_threshold"],
        help="质量阈值 (降低=减少ID数量)",
    )
    parser.add_argument(
        "--use-simple-logic",
        action="store_true",
        default=True,
        help="使用简化判断逻辑 (仅基于相似度)",
    )

    # 预设模式
    parser.add_argument(
        "--mode",
        type=str,
        choices=["strict", "balanced", "loose", "custom"],
        default=INFERENCE_DEFAULTS["mode"],
        help="预设模式: strict(少ID), balanced(平衡), loose(多ID), custom(自定义)",
    )

    # 相似度融合 + 性别/年龄重加权参数
    parser.add_argument(
        "--sim-cosine-w",
        type=float,
        default=INFERENCE_DEFAULTS["sim_cosine_w"],
        help="相似度融合中“余弦相似度”的权重",
    )
    parser.add_argument(
        "--sim-euclid-w",
        type=float,
        default=INFERENCE_DEFAULTS["sim_euclid_w"],
        help="相似度融合中“欧氏相似度”的权重 (自动归一化到和为1)",
    )
    parser.add_argument(
        "--aux-gender-penalty",
        type=float,
        default=INFERENCE_DEFAULTS["aux_gender_penalty"],
        help="性别不一致时对相似度的最大惩罚比例 (0~1)",
    )
    parser.add_argument(
        "--aux-age-reweight",
        type=float,
        default=INFERENCE_DEFAULTS["aux_age_reweight"],
        help="年龄高斯重加权的幅度 (0~1)",
    )
    parser.add_argument(
        "--aux-min-age-sigma",
        type=float,
        default=INFERENCE_DEFAULTS["aux_min_age_sigma"],
        help="年龄高斯最小标准差 σ (单位: 岁)，越大越不敏感",
    )

    # 聚类可视化参数
    parser.add_argument(
        "--enable-clustering-viz", action="store_true", help="启用聚类可视化"
    )
    parser.add_argument("--max-clusters", type=int, default=10, help="最大聚类数量")
    parser.add_argument(
        "--clustering-method",
        type=str,
        default="auto",
        choices=["auto", "kmeans", "dbscan", "hierarchical"],
        help="聚类方法",
    )
    parser.add_argument(
        "--viz-dimensions", type=int, default=2, choices=[2, 3], help="可视化维度"
    )
    parser.add_argument(
        "--min-samples-per-id",
        type=int,
        default=1,
        help="每个ID的最小样本数量，低于此数量的ID将被过滤",
    )

    # 性别/年龄评估参数
    parser.add_argument(
        "--compute-aux-metrics",
        action="store_true",
        help="测试结束后计算并输出性别准确率与年龄误差/准确率（需目录包含真值）",
    )
    parser.add_argument(
        "--gender-threshold",
        type=float,
        default=INFERENCE_DEFAULTS["gender_threshold"],
        help="性别判定阈值（male_prob >= 阈值 视为雄性）",
    )
    parser.add_argument(
        "--age-accuracy-tolerances",
        type=str,
        default="1,2,3",
        help='年龄准确率公差（年），如 "1,2,3" 表示±1/±2/±3 岁范围内命中率',
    )
    parser.add_argument(
        "--gender-hysteresis",
        type=float,
        default=INFERENCE_DEFAULTS["gender_hysteresis"],
        help="性别输出的迟滞边界，避免频繁翻转",
    )
    parser.add_argument(
        "--age-scope",
        type=str,
        default=INFERENCE_DEFAULTS["age_scope"],
        choices=["track", "video", "global"],
        help="年龄统计作用域: track=轨迹内；video=当前视频；global=跨视频汇总",
    )
    parser.add_argument(
        "--age-display",
        type=str,
        default=INFERENCE_DEFAULTS["age_display"],
        choices=["instant", "median", "mean"],
        help="年龄显示口径: instant=当前帧；median=中位数；mean=均值",
    )

    # 开放集评估参数（当前场景为“测试集与训练集完全无交集”，仅保留基于真值ID的特征级 ReID 指标）
    parser.add_argument(
        "--compute-open-set-metrics",
        action="store_true",
        help="计算基于真值ID的 ReID 特征评估指标 (mAP/AUROC/Rank-k/CMC/阈值化验证指标)，并同时计算辅助任务指标(等价于 --compute-aux-metrics)；需目录包含真值",
    )
    # 以下两个参数为兼容旧脚本而保留，当前实现中不再使用，可忽略
    parser.add_argument(
        "--open-set-threshold",
        type=float,
        default=OPEN_SET_METRICS["odr_threshold"],
        help="[兼容占位] 旧版 ODR/FMR 相似度阈值参数，当前版本未使用",
    )
    parser.add_argument(
        "--oscr-points",
        type=int,
        default=OPEN_SET_METRICS["oscr_sample_points"],
        help="[兼容占位] 旧版 OSCR 曲线采样点数量参数，当前版本未使用",
    )

    # 其他参数
    parser.add_argument("--batch-size", type=int, default=32, help="批处理大小")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")

    # 兼容 get_config 的 opts 参数
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs",
        default=[],
        nargs="+",
    )

    args, unparsed = parser.parse_known_args()
    if unparsed:
        unknown_flags = [x for x in unparsed if str(x).startswith("-")]
        if unknown_flags:
            print(
                f"[WARN] 未识别的参数将被忽略: {unknown_flags} "
                "(通常是拼写错误，例如 --compute-open-set-metrics)"
            )
    # 便捷用法：只要计算 open-set 指标，就同时计算辅助任务指标
    if getattr(args, "compute_open_set_metrics", False) and not getattr(
        args, "compute_aux_metrics", False
    ):
        args.compute_aux_metrics = True
    config = get_config(args)
    return args, config


class PandaReIDInference:
    """大熊猫个体识别推理器"""

    def __init__(self, config, model_path, args):
        """
        初始化推理器

        Args:
            config: 配置对象
            model_path: 训练好的模型路径
            args: 命令行参数
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args

        # 应用预设模式
        self.apply_mode_settings()

        # 设置阈值参数
        self.similarity_threshold = self.args.similarity_threshold

        # 构建模型 - 使用检查点中的类别数，避免与配置文件不一致
        inferred_num_classes = None
        try:
            ckpt_meta = torch.load(model_path, map_location=self.device)
            state_dict_meta = (
                ckpt_meta.get("model", ckpt_meta)
                if isinstance(ckpt_meta, dict)
                else ckpt_meta
            )
            if isinstance(ckpt_meta, dict) and "num_classes" in ckpt_meta:
                inferred_num_classes = int(ckpt_meta["num_classes"])
            elif isinstance(state_dict_meta, dict):
                if "neck.classifier.weight" in state_dict_meta:
                    inferred_num_classes = int(
                        state_dict_meta["neck.classifier.weight"].shape[0]
                    )
                elif "neck.classifier.bias" in state_dict_meta:
                    inferred_num_classes = int(
                        state_dict_meta["neck.classifier.bias"].shape[0]
                    )
        except Exception:
            pass
        if inferred_num_classes is None:
            inferred_num_classes = int(getattr(config.MODEL, "NUM_CLASSES", 1000))
        self.model = build_panda_reid_model(config, inferred_num_classes)
        self.model.to(self.device)

        # 构建原型网络 - 与训练时完全一致
        self.prototype_net = PrototypeReIDNetwork(
            feature_dim=self.model.feature_dim,
            temperature=0.05,  # 与训练时一致
            momentum=0.9,  # 与训练时一致
            min_samples=1,
        )
        self.prototype_net.to(self.device)

        # 加载训练好的模型
        self.load_model(model_path)

        # 清空训练原型，准备野外应用
        self.clear_training_prototypes()

        # 动态设置原型网络参数
        self.configure_prototype_network()

        # 图像预处理 - 与训练时完全一致
        self.transform = build_panda_transform(
            is_train=False, img_size=config.DATA.IMG_SIZE
        )

        # 野外个体计数器
        self.wild_id_counter = 0

        print(f"大熊猫ReID推理器初始化完成")
        print(f"   模型特征维度: {self.model.feature_dim}")
        print(f"   相似度阈值: {self.similarity_threshold}")
        print(f"   设备: {self.device}")
        print(f"   模式: {self.args.mode}")
        if self.args.verbose:
            print(f"   基础阈值: {self.args.base_threshold}")
            print(
                f"   自适应阈值范围: [{self.args.adaptive_threshold_min}, {self.args.adaptive_threshold_max}]"
            )
            print(f"   置信度阈值: {self.args.confidence_threshold}")
            print(f"   质量阈值: {self.args.quality_threshold}")
        print(f"   已清空训练原型，准备野外个体识别")

    def apply_mode_settings(self):
        """应用预设模式设置（根据 mode 调整关键阈值）"""

        # custom 模式：先用 INFERENCE_DEFAULTS 补齐缺省参数
        if getattr(self.args, "mode", None) == "custom":
            for k, v in INFERENCE_DEFAULTS.items():
                if not hasattr(self.args, k):
                    continue
                cur = getattr(self.args, k)
                if cur is None:
                    setattr(self.args, k, v)

        print_mode = self.args.mode

        if print_mode == "strict":
            # 严格模式：最少 ID 数量
            self.args.similarity_threshold = 0.35
            self.args.base_threshold = 0.25
            self.args.adaptive_threshold_min = 0.1
            self.args.adaptive_threshold_max = 0.5
            self.args.confidence_threshold = 0.15
            self.args.quality_threshold = 0.03
            print(" 应用严格模式设置 (最少ID数量)")

        elif print_mode == "balanced":
            # 平衡模式：适中 ID 数量
            self.args.similarity_threshold = 0.45
            self.args.base_threshold = 0.3
            self.args.adaptive_threshold_min = 0.15
            self.args.adaptive_threshold_max = 0.6
            self.args.confidence_threshold = 0.2
            self.args.quality_threshold = 0.05
            print("  应用平衡模式设置 (适中ID数量)")

        elif print_mode == "loose":
            # 宽松模式：较多 ID 数量
            self.args.similarity_threshold = 0.55
            self.args.base_threshold = 0.4
            self.args.adaptive_threshold_min = 0.2
            self.args.adaptive_threshold_max = 0.7
            self.args.confidence_threshold = 0.3
            self.args.quality_threshold = 0.1
            print(" 应用宽松模式设置 (较多ID数量)")

        else:
            # 自定义模式：使用命令行参数 + INFERENCE_DEFAULTS 兜底
            print(" 使用自定义参数设置 (已与 INFERENCE_DEFAULTS 对齐缺省值)")
            if getattr(self.args, "verbose", False):
                print("   当前关键参数:")
                for k in [
                    "similarity_threshold",
                    "base_threshold",
                    "adaptive_threshold_min",
                    "adaptive_threshold_max",
                    "confidence_threshold",
                    "quality_threshold",
                    "sim_cosine_w",
                    "sim_euclid_w",
                    "aux_gender_penalty",
                    "aux_age_reweight",
                    "aux_min_age_sigma",
                    "gender_threshold",
                    "gender_hysteresis",
                    "age_scope",
                    "age_display",
                ]:
                    if hasattr(self.args, k):
                        print(f"     {k}: {getattr(self.args, k)}")

    def load_model(self, model_path):
        """加载训练好的模型权重（兼容不同类别数的检查点）"""
        try:
            print(f"加载模型: {model_path}")
            # 提示：如无完全信任的权重文件，可设置weights_only=True（需PyTorch较新版本）
            checkpoint = torch.load(model_path, map_location=self.device)

            # 兼容两种保存格式
            state_dict = checkpoint.get("model", checkpoint)

            # 优先尝试严格加载
            try:
                self.model.load_state_dict(state_dict, strict=True)
                print("模型权重加载成功 (strict)")
            except RuntimeError as e:
                print("严格加载失败，可能是分类头(类别数)不匹配，尝试跳过不匹配参数…")
                model_dict = self.model.state_dict()
                filtered_state = {}
                skipped = []
                for k, v in state_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        filtered_state[k] = v
                    else:
                        skipped.append(k)
                # 加载其余权重
                self.model.load_state_dict(filtered_state, strict=False)
                if any("classifier" in k or "neck" in k for k in skipped):
                    print(
                        "已跳过分类头相关权重（例如 neck.classifier.*），使用当前模型的分类头配置进行推理"
                    )
                print(f"模型其余部分加载成功，跳过参数数: {len(skipped)}")

            # 加载原型网络权重（若存在）
            if isinstance(checkpoint, dict) and "prototype_net" in checkpoint:
                try:
                    self.prototype_net.load_state_dict(
                        checkpoint["prototype_net"], strict=False
                    )
                    print("原型网络权重加载成功")
                except Exception as pe:
                    print(f"原型网络权重加载失败，忽略: {pe}")
        except Exception as ex:
            print(f"模型加载失败: {ex}")
            raise

        # 原型网络初始化为空（开放集识别模式）
        # 注意：不加载训练集原型，野外应用时会动态构建新原型
        print("原型网络初始化为空 (开放集识别模式)")
        print("  野外个体识别时将动态构建新原型")

        # 设置为评估模式
        self.model.eval()
        self.prototype_net.eval()

        # 显示模型信息
        if isinstance(checkpoint, dict) and "epoch" in checkpoint:
            print(f"   训练轮次: {checkpoint['epoch']}")
        if isinstance(checkpoint, dict) and "best_comprehensive_score" in checkpoint:
            print(f"   最佳综合得分: {checkpoint['best_comprehensive_score']:.4f}")

    def clear_training_prototypes(self):
        """清空训练原型，准备野外应用"""
        self.prototype_net.prototypes.clear()
        print("已清空所有训练原型，开始野外个体识别模式")

    def configure_prototype_network(self):
        """动态配置原型网络参数"""
        # 动态修改原型网络的内部参数
        # 注意：这是运行时修改，不会影响原始模型文件

        # 修改compute_adaptive_threshold方法中的参数
        def patched_compute_adaptive_threshold(original_self, query_feature):
            """动态修改的自适应阈值计算"""
            with torch.no_grad():
                quality_score = original_self.quality_net(
                    query_feature.unsqueeze(0)
                ).item()

                if original_self.global_mean is not None:
                    deviation = torch.norm(
                        query_feature - original_self.global_mean
                    ).item()
                    normalized_deviation = deviation / (
                        torch.norm(original_self.global_std).item() + 1e-8
                    )

                    # 使用动态参数
                    base_threshold = self.args.base_threshold
                    quality_adjustment = (quality_score - 0.5) * 0.05
                    deviation_adjustment = min(normalized_deviation * 0.02, 0.05)

                    adaptive_threshold = (
                        base_threshold - quality_adjustment + deviation_adjustment
                    )
                    adaptive_threshold = max(
                        self.args.adaptive_threshold_min,
                        min(self.args.adaptive_threshold_max, adaptive_threshold),
                    )

                    return adaptive_threshold
                else:
                    return self.args.base_threshold

        # 动态绑定方法
        import types

        self.prototype_net.compute_adaptive_threshold = types.MethodType(
            patched_compute_adaptive_threshold, self.prototype_net
        )

        # 运行时可调：相似度融合权重与年龄/性别重加权强度
        pn = self.prototype_net
        for name in [
            "sim_cosine_w",
            "sim_euclid_w",
            "aux_gender_penalty",
            "aux_age_reweight",
        ]:
            if hasattr(self.args, name) and getattr(self.args, name) is not None:
                try:
                    setattr(pn, name, float(getattr(self.args, name)))
                except Exception:
                    pass
        # 最小年龄sigma（年） -> sigma^2
        if (
            hasattr(self.args, "aux_min_age_sigma")
            and getattr(self.args, "aux_min_age_sigma") is not None
        ):
            try:
                sig = max(0.01, float(getattr(self.args, "aux_min_age_sigma")))
                pn.aux_min_age_sigma2 = sig * sig
            except Exception:
                pass

        if self.args.verbose:
            print(" 原型网络参数已动态配置")

    def _read_roi_data(self, image_path, roi_root):
        """
        读取ROI数据（支持YOLO格式和掩码格式）

        Args:
            image_path: 图像路径
            roi_root: ROI文件根目录

        Returns:
            掩码数组或ROI坐标，如果失败返回None
        """
        try:
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]

            # 检测ROI格式（默认mask），并健壮处理mask_root为None的情况
            roi_format = getattr(self.args, "roi_format", "mask")
            mask_root_cfg = getattr(self.args, "mask_root", None)
            mask_root = (
                mask_root_cfg or roi_root
            )  # 如果未显式提供mask_root或为None，回退到roi_root

            # 生成候选路径：支持“扁平结构”和“镜像结构（按上一级目录名分组）”
            parent_dir_name = os.path.basename(os.path.dirname(image_path))
            mask_candidates = [
                os.path.join(mask_root, base_name + ".txt"),
                os.path.join(mask_root, parent_dir_name, base_name + ".txt"),
            ]
            yolo_candidates = [
                os.path.join(roi_root, base_name + ".txt"),
                os.path.join(roi_root, parent_dir_name, base_name + ".txt"),
            ]

            # 先尝试直接候选路径匹配
            if roi_format in ("mask", "auto") and mask_root:
                for coords_path in mask_candidates:
                    if os.path.exists(coords_path):
                        mask = self._read_mask_from_coords(coords_path, image_path)
                        if mask is not None:
                            return {"type": "mask", "data": mask}
                # 递归回退：按文件名在mask_root下查找
                for root_dir, _, files in os.walk(mask_root):
                    for fn in files:
                        if os.path.splitext(fn)[0] == base_name and fn.lower().endswith(
                            ".txt"
                        ):
                            coords_path = os.path.join(root_dir, fn)
                            mask = self._read_mask_from_coords(coords_path, image_path)
                            if mask is not None:
                                return {"type": "mask", "data": mask}
                            break

            if roi_format in ("yolo", "auto") and roi_root:
                for roi_path in yolo_candidates:
                    if os.path.exists(roi_path):
                        roi = self._read_yolo_roi(roi_path)
                        if roi is not None:
                            return {"type": "yolo", "data": roi}
                # 递归回退：按文件名在roi_root下查找
                for root_dir, _, files in os.walk(roi_root):
                    for fn in files:
                        if os.path.splitext(fn)[0] == base_name and fn.lower().endswith(
                            ".txt"
                        ):
                            roi_path = os.path.join(root_dir, fn)
                            roi = self._read_yolo_roi(roi_path)
                            if roi is not None:
                                return {"type": "yolo", "data": roi}
                            break

            return None

        except Exception as e:
            if hasattr(self, "args") and self.args.verbose:
                print(f"Failed to read ROI data for {image_path}: {e}")
            return None

    def _read_yolo_roi(self, roi_path):
        """
        读取YOLO格式的ROI文件

        Args:
            roi_path: ROI文件路径

        Returns:
            (cx, cy, w, h) 归一化坐标，如果失败返回None
        """
        try:
            if os.path.exists(roi_path):
                with open(roi_path, "r") as f:
                    lines = f.readlines()
                if lines:
                    # 取第一行，格式：class cx cy w h
                    parts = lines[0].strip().split()
                    if len(parts) >= 5:
                        _, cx, cy, w, h = map(float, parts[:5])
                        return cx, cy, w, h
        except Exception as e:
            if hasattr(self, "args") and self.args.verbose:
                print(f"Failed to read YOLO ROI file {roi_path}: {e}")
        return None

    def _read_mask_from_coords(self, coords_path, image_path):
        """
        从坐标文件重建掩码（与数据加载器保持一致）
        """
        try:
            # 读取原图像获取尺寸
            image = cv2.imread(image_path)
            if image is None:
                return None
            h, w = image.shape[:2]

            # 读取坐标文件
            with open(coords_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if len(lines) < 3:
                return None

            # 解析轮廓数量（第三行）
            contour_count = int(lines[2].strip())

            if contour_count == 0:
                return None

            # 创建空掩码
            mask = np.zeros((h, w), dtype=np.uint8)

            # 解析轮廓坐标
            line_idx = 3
            for _ in range(contour_count):
                if line_idx >= len(lines):
                    break

                # 读取轮廓点数
                point_count = int(lines[line_idx].strip())
                line_idx += 1

                # 读取轮廓点坐标
                contour_points = []
                for _ in range(point_count):
                    if line_idx >= len(lines):
                        break
                    x, y = map(float, lines[line_idx].strip().split(","))
                    contour_points.append([int(x), int(y)])
                    line_idx += 1

                if contour_points:
                    # 填充轮廓
                    contour = np.array(contour_points, dtype=np.int32)
                    cv2.fillPoly(mask, [contour], 255)

            return mask.astype(bool)

        except Exception as e:
            if hasattr(self, "args") and self.args.verbose:
                print(f"Failed to read mask from coords {coords_path}: {e}")
            return None

    def _apply_mask_to_image(self, image, mask):
        """应用掩码到图像"""
        try:
            # 确保掩码和图像尺寸匹配
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            # 应用掩码：保留掩码区域，其他区域置为黑色
            masked_image = image.copy()
            masked_image[~mask] = 0

            return masked_image

        except Exception as e:
            if hasattr(self, "args") and self.args.verbose:
                print(f"应用掩码失败: {e}")
            return image

    def _process_mask_region_to_rectangle(self, image, mask):
        """
        处理掩码区域，输出矩形图像（与训练时保持一致）

        步骤：
        1. 计算掩码的边界框
        2. 扩展边界框（10%边距）
        3. 裁剪图像到边界框区域
        4. 在裁剪区域内应用掩码（掩码外区域用0填充）
        """
        try:
            # 确保掩码和图像尺寸匹配
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            # 1. 计算掩码的边界框
            rows, cols = np.where(mask)
            if len(rows) == 0:
                # 掩码为空，返回默认大小的黑色图像
                return np.zeros(
                    (self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE, 3),
                    dtype=np.uint8,
                )

            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()

            # 2. 扩展边界框（10%边距）
            h, w = image.shape[:2]
            bbox_w = x2 - x1 + 1
            bbox_h = y2 - y1 + 1

            expand_w = int(bbox_w * 0.1)  # 10%扩展
            expand_h = int(bbox_h * 0.1)

            # 计算扩展后的边界框
            x1_exp = max(0, x1 - expand_w // 2)
            y1_exp = max(0, y1 - expand_h // 2)
            x2_exp = min(w, x2 + expand_w // 2)
            y2_exp = min(h, y2 + expand_h // 2)

            # 3. 裁剪图像到边界框区域
            cropped_image = image[y1_exp:y2_exp, x1_exp:x2_exp].copy()
            cropped_mask = mask[y1_exp:y2_exp, x1_exp:x2_exp]

            # 4. 在裁剪区域内应用掩码（关键：0像素填充）
            # 掩码外的区域设置为0（黑色）
            cropped_image[~cropped_mask] = 0

            return cropped_image

        except Exception as e:
            if hasattr(self, "args") and self.args.verbose:
                print(f"处理掩码区域失败: {e}")
            # 返回默认大小的黑色图像作为fallback
            return np.zeros(
                (self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE, 3),
                dtype=np.uint8,
            )

    def _get_mask_bbox(self, mask):
        """从掩码计算边界框"""
        try:
            rows, cols = np.where(mask)

            if len(rows) == 0:
                return None

            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()

            h, w = mask.shape

            # 转换为归一化的中心坐标格式
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bbox_w = (x2 - x1 + 1) / w
            bbox_h = (y2 - y1 + 1) / h

            return cx, cy, bbox_w, bbox_h

        except Exception as e:
            if hasattr(self, "args") and self.args.verbose:
                print(f"计算边界框失败: {e}")
            return None

    def _crop_roi_with_expansion(self, image, roi, roi_expand_ratio=0.1):
        """
        根据ROI裁剪图像并扩展边距（与训练时完全一致）

        Args:
            image: 输入图像 (H, W, C)
            roi: (cx, cy, w, h) 归一化坐标
            roi_expand_ratio: ROI扩展比例

        Returns:
            裁剪后的图像
        """
        h, w = image.shape[:2]
        cx, cy, roi_w, roi_h = roi

        # 转换为像素坐标
        cx_px = cx * w
        cy_px = cy * h
        w_px = roi_w * w
        h_px = roi_h * h

        # 扩展边距
        expand_w = w_px * roi_expand_ratio
        expand_h = h_px * roi_expand_ratio

        # 计算裁剪区域
        x1 = max(0, int(cx_px - (w_px + expand_w) / 2))
        y1 = max(0, int(cy_px - (h_px + expand_h) / 2))
        x2 = min(w, int(cx_px + (w_px + expand_w) / 2))
        y2 = min(h, int(cy_px + (h_px + expand_h) / 2))

        # 裁剪图像
        cropped = image[y1:y2, x1:x2]

        return cropped

    def _generate_gradcam_heatmap(
        self,
        np_image_rgb,
        tensor_image,
        save_path,
        overlay_original=False,
        original_image_path=None,
        roi_bbox=None,
        roi_mask=None,
        overlay_mask_shape=False,
    ):
        """生成热力图
        - 基于当前ROI输入（np_image_rgb, HxWx3, RGB）和进入模型的tensor（1x3xHxW）
        - 保存到save_path
        - 若overlay_original=True且提供original_image_path与roi_bbox/roi_mask，则将热力图贴回原图对应ROI位置
        """
        try:
            import torch
            import torch.nn.functional as F
            import numpy as np
            import cv2
            import os

            self.model.eval()
            with torch.no_grad():
                # 简化注意力近似（输入通道能量）
                attn = tensor_image.squeeze(0).pow(2).sum(0).sqrt().cpu().numpy()  # HxW
                attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
                attn_color = cv2.applyColorMap(
                    (attn * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                attn_color = cv2.cvtColor(attn_color, cv2.COLOR_BGR2RGB)

                # 将热力图与ROI图对齐融合
                roi_h, roi_w = np_image_rgb.shape[:2]
                if attn_color.shape[:2] != (roi_h, roi_w):
                    attn_color = cv2.resize(
                        attn_color, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR
                    )
                overlay_roi = (
                    (0.4 * attn_color + 0.6 * np_image_rgb)
                    .clip(0, 255)
                    .astype(np.uint8)
                )

                # 保存ROI坐标系热力图
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, cv2.cvtColor(overlay_roi, cv2.COLOR_RGB2BGR))

                # 需要贴回原图
                if overlay_original and original_image_path is not None:
                    orig = cv2.imread(original_image_path)
                    if orig is not None:
                        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
                        H0, W0 = orig.shape[:2]
                        # 推断ROI矩形
                        if roi_bbox is not None:
                            x1, y1, x2, y2 = roi_bbox
                        else:
                            if roi_mask is not None:
                                rows, cols = np.where(roi_mask)
                                if len(rows) > 0:
                                    y1, y2 = rows.min(), rows.max()
                                    x1, x2 = cols.min(), cols.max()
                                else:
                                    x1 = y1 = 0
                                    x2 = W0
                                    y2 = H0
                            else:
                                x1 = y1 = 0
                                x2 = W0
                                y2 = H0
                        # 将overlay_roi按ROI矩形缩放
                        roi_w0 = max(1, x2 - x1)
                        roi_h0 = max(1, y2 - y1)
                        overlay_resized = cv2.resize(
                            overlay_roi,
                            (roi_w0, roi_h0),
                            interpolation=cv2.INTER_LINEAR,
                        )

                        orig_overlay = orig.copy()
                        if overlay_mask_shape and roi_mask is not None:
                            # 使用mask形状做alpha融合：仅覆盖mask为True的区域
                            # 需要将roi_mask裁剪到bbox并resize到overlay大小
                            roi_mask_crop = roi_mask[
                                int(y1) : int(y2), int(x1) : int(x2)
                            ].astype(np.uint8)
                            if roi_mask_crop.size == 0:
                                roi_mask_crop = np.ones(
                                    (roi_h0, roi_w0), dtype=np.uint8
                                )
                            roi_mask_resized = cv2.resize(
                                roi_mask_crop,
                                (roi_w0, roi_h0),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            alpha = (roi_mask_resized > 0).astype(np.float32)[..., None]
                            patch = orig_overlay[y1:y2, x1:x2]
                            fused = (
                                alpha * overlay_resized + (1 - alpha) * patch
                            ).astype(np.uint8)
                            orig_overlay[y1:y2, x1:x2] = fused
                        else:
                            # 默认矩形覆盖
                            orig_overlay[y1:y2, x1:x2] = overlay_resized

                        # 保存到同目录下：save_path同名加 _orig
                        sp_dir, sp_name = os.path.dirname(save_path), os.path.basename(
                            save_path
                        )
                        sp_name2 = os.path.splitext(sp_name)[0] + "_orig.png"
                        save_path2 = os.path.join(sp_dir, sp_name2)
                        cv2.imwrite(
                            save_path2, cv2.cvtColor(orig_overlay, cv2.COLOR_RGB2BGR)
                        )
        except Exception as e:
            if hasattr(self, "args") and self.args.verbose:
                print(f"生成热力图失败: {e}")

    def extract_features(self, image_path, roi_root=None):
        """
        提取单张图像的特征（与训练时完全一致）

        Args:
            image_path: 图像路径
            roi_root: ROI文件根目录

        Returns:
            归一化的特征向量
        """
        try:
            # 读取图像（与训练时一致）
            image = cv2.imread(image_path)
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 读取ROI数据并处理图像（支持掩码和YOLO格式）
            original_image_for_vis = None
            if roi_root:
                roi_data = self._read_roi_data(image_path, roi_root)
                if roi_data is not None:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    if roi_data["type"] == "mask":
                        mask = roi_data["data"]
                        # 记录用于可视化的当前ROI裁剪前图像
                        original_image_for_vis = image.copy()
                        image = self._process_mask_region_to_rectangle(image, mask)
                        if hasattr(self, "args") and self.args.verbose:
                            print(f"使用掩码区域处理进行特征提取: {base_name}")
                    elif roi_data["type"] == "yolo":
                        roi = roi_data["data"]
                        original_image_for_vis = image.copy()
                        image = self._crop_roi_with_expansion(
                            image, roi, roi_expand_ratio=0.1
                        )
                        if hasattr(self, "args") and self.args.verbose:
                            print(f"使用YOLO ROI裁剪进行特征提取: {base_name}")

            # 确保图像不为空（与训练时一致的处理）
            if image.size == 0:
                print(f"警告: 裁剪后图像为空 {image_path}")
                # 使用默认尺寸的黑色图像作为fallback
                image = np.zeros(
                    (self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE, 3),
                    dtype=np.uint8,
                )

            # 转换为PIL图像（与训练时一致）
            image_pil = Image.fromarray(image)

            # 应用变换（与训练时完全一致）
            tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

            # 如果开启热力图开关，则对当前ROI输入生成热力图（支持贴回原图）
            if hasattr(self, "args") and getattr(self.args, "heatmap", False):
                try:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    heatmap_dir = os.path.join(self.args.output_root, "heatmap")
                    os.makedirs(heatmap_dir, exist_ok=True)
                    heatmap_path = os.path.join(heatmap_dir, f"{base_name}.png")

                    overlay_orig = getattr(self.args, "heatmap_overlay_original", False)
                    roi_bbox = None
                    roi_mask_local = None
                    original_for_overlay = image_path

                    # 优先使用当前roi_root读取到的ROI信息
                    if roi_root and roi_data is not None:
                        if roi_data["type"] == "yolo":
                            roi = roi_data["data"]
                            x, y, w, h = roi
                            roi_bbox = (int(x), int(y), int(x + w), int(y + h))
                        elif roi_data["type"] == "mask":
                            roi_mask_local = roi_data["data"]
                    # ROIIMG-only模式：若开启贴回原图，尝试通过文件名在指定原图根目录中寻找原图，并读取对应ROI坐标
                    elif overlay_orig and getattr(
                        self.args, "heatmap_image_root", None
                    ):
                        base_no_ext = base_name
                        hm_img_root = self.args.heatmap_image_root
                        found_orig = None
                        for r, _, fs in os.walk(hm_img_root):
                            for fn in fs:
                                if os.path.splitext(fn)[0] == base_no_ext:
                                    found_orig = os.path.join(r, fn)
                                    break
                            if found_orig:
                                break
                        if found_orig:
                            original_for_overlay = found_orig
                            overlay_roi_root = (
                                getattr(self.args, "heatmap_roi_root", None)
                                or getattr(self.args, "mask_root", None)
                                or getattr(self.args, "test_roi_root", None)
                            )
                            roi_data2 = None
                            if overlay_roi_root:
                                roi_data2 = self._read_roi_data(
                                    found_orig, overlay_roi_root
                                )
                            if roi_data2 is not None:
                                if roi_data2["type"] == "yolo":
                                    try:
                                        orig_img = cv2.imread(found_orig)
                                        if orig_img is not None:
                                            H0, W0 = orig_img.shape[:2]
                                            cx, cy, ww, hh = roi_data2["data"]
                                            x1 = int(max(0, (cx - ww / 2) * W0))
                                            y1 = int(max(0, (cy - hh / 2) * H0))
                                            x2 = int(min(W0, (cx + ww / 2) * W0))
                                            y2 = int(min(H0, (cy + hh / 2) * H0))
                                            roi_bbox = (x1, y1, x2, y2)
                                    except Exception:
                                        pass
                                elif roi_data2["type"] == "mask":
                                    roi_mask_local = roi_data2["data"]

                    self._generate_gradcam_heatmap(
                        image,
                        tensor,
                        heatmap_path,
                        overlay_original=overlay_orig,
                        original_image_path=original_for_overlay,
                        roi_bbox=roi_bbox,
                        roi_mask=roi_mask_local,
                        overlay_mask_shape=getattr(
                            self.args, "heatmap_overlay_original", False
                        ),
                    )
                except Exception as he:
                    if self.args.verbose:
                        print(f"热力图失败 {image_path}: {he}")

            # 提取特征 + 辅助预测（性别/年龄）
            with torch.no_grad():
                feat_after_bn, _feat_before_bn, gender_logits, age_pred = (
                    self.model.forward_multitask(tensor)
                )
                feat_after_bn = F.normalize(feat_after_bn, p=2, dim=1)  # L2归一化
                male_prob = F.softmax(gender_logits, dim=1)[:, 1]

            aux = {
                "gender_prob": float(male_prob[0].item()),
                "age_pred": float(age_pred[0].item()),
            }
            return feat_after_bn[0], aux  # 返回ReID特征和辅助信息

        except Exception as e:
            print(f"特征提取失败 {image_path}: {e}")
            if hasattr(self, "args") and self.args.verbose:
                import traceback

                print(f"详细错误: {traceback.format_exc()}")
            return None

    def predict_identity(self, query_feature, image_path, aux=None):
        """
        野外个体识别逻辑 - 逐张处理，动态创建ID

        Args:
            query_feature: 查询特征
            image_path: 图像路径（用于生成ID名称）
            aux: 辅助信息字典，例如{'gender_prob': float, 'age_pred': float}

        Returns:
            识别结果
        """
        # 如果数据库为空，直接创建第一个个体
        if len(self.prototype_net.prototypes) == 0:
            self.wild_id_counter += 1
            new_id = f"Wild_Panda_{self.wild_id_counter:03d}"

            # 创建新原型（带入性别/年龄元信息）
            self.prototype_net.update_prototype(
                new_id,
                query_feature,
                1.0,
                gender_prob=(aux.get("gender_prob") if aux else None),
                age_pred=(aux.get("age_pred") if aux else None),
            )

            return {
                "predicted_id": new_id,
                "similarity": 1.0,
                "confidence": 1.0,
                "is_new_id": True,
                "action": "created_first_individual",
            }

        # 与现有原型比较
        with torch.no_grad():
            result = self.prototype_net(query_feature, aux=aux)

        best_similarity = result["similarity"]
        best_match_id = result["predicted_id"]

        # 根据相似度阈值判断
        if best_similarity >= self.similarity_threshold:
            # 归类到现有个体，更新原型（带入性别/年龄信息）
            self.prototype_net.update_prototype(
                best_match_id,
                query_feature,
                0.8,
                gender_prob=(aux.get("gender_prob") if aux else None),
                age_pred=(aux.get("age_pred") if aux else None),
            )

            return {
                "predicted_id": best_match_id,
                "similarity": best_similarity,
                "confidence": result["confidence"],
                "is_new_id": False,
                "action": "assigned_to_existing",
            }
        else:
            # 创建新个体
            self.wild_id_counter += 1
            new_id = f"Wild_Panda_{self.wild_id_counter:03d}"

            # 创建新原型（带入性别/年龄信息）
            self.prototype_net.update_prototype(
                new_id,
                query_feature,
                1.0,
                gender_prob=(aux.get("gender_prob") if aux else None),
                age_pred=(aux.get("age_pred") if aux else None),
            )

            return {
                "predicted_id": new_id,
                "similarity": best_similarity,
                "confidence": result["confidence"],
                "is_new_id": True,
                "action": "created_new_individual",
                "most_similar_id": best_match_id,
                "max_similarity_with_existing": best_similarity,
            }

    def _filter_by_min_samples(
        self, features, predicted_ids, image_paths, confidences, min_samples
    ):
        """
        根据最小样本数过滤ID

        Args:
            features: 特征列表
            predicted_ids: 预测ID列表
            image_paths: 图像路径列表
            confidences: 置信度列表
            min_samples: 最小样本数

        Returns:
            过滤后的特征、ID、路径、置信度列表
        """
        from collections import Counter

        # 统计每个ID的样本数
        id_counts = Counter(predicted_ids)

        print(f" ID样本数统计 (过滤前):")
        for id_name, count in sorted(id_counts.items()):
            print(f"  {id_name}: {count}个样本")

        # 找出满足最小样本数要求的ID
        valid_ids = {
            id_name for id_name, count in id_counts.items() if count >= min_samples
        }

        print(f"\n 过滤统计:")
        print(f"  最小样本数要求: {min_samples}")
        print(f"  过滤前ID数量: {len(id_counts)}")
        print(f"  过滤后ID数量: {len(valid_ids)}")

        if len(valid_ids) == 0:
            print("  所有ID都被过滤，降低min_samples_per_id参数")
            return features, predicted_ids, image_paths, confidences

        # 过滤数据
        filtered_features = []
        filtered_ids = []
        filtered_paths = []
        filtered_confidences = []

        for i, pred_id in enumerate(predicted_ids):
            if pred_id in valid_ids:
                filtered_features.append(features[i])
                filtered_ids.append(pred_id)
                filtered_paths.append(image_paths[i])
                filtered_confidences.append(confidences[i])

        print(f"  过滤前样本数: {len(features)}")
        print(f"  过滤后样本数: {len(filtered_features)}")

        # 显示过滤后的ID统计
        filtered_id_counts = Counter(filtered_ids)
        print(f"\n 过滤后ID样本数统计:")
        for id_name, count in sorted(filtered_id_counts.items()):
            print(f"  {id_name}: {count}个样本")

        return filtered_features, filtered_ids, filtered_paths, filtered_confidences

    def create_clustering_visualization(self, results, output_root):
        """
        创建聚类可视化

        Args:
            results: 识别结果列表
            output_root: 输出根目录
        """
        if not CLUSTERING_AVAILABLE:
            print("  聚类可视化功能不可用，请安装相关依赖")
            return

        print(f"\n 开始创建聚类可视化...")

        # 提取特征和标签
        features = []
        predicted_ids = []
        image_paths = []
        confidences = []

        for result in results:
            if "feature" in result and result["feature"] is not None:
                features.append(result["feature"])
                predicted_ids.append(result["predicted_id"])
                image_paths.append(result["image_path"])
                confidences.append(result.get("confidence", 0.0))

        if len(features) < 2:
            print("  特征数量不足，无法进行聚类可视化")
            return

        # 应用ID最小样本数过滤
        if (
            hasattr(self.args, "min_samples_per_id")
            and self.args.min_samples_per_id > 1
        ):
            features, predicted_ids, image_paths, confidences = (
                self._filter_by_min_samples(
                    features,
                    predicted_ids,
                    image_paths,
                    confidences,
                    self.args.min_samples_per_id,
                )
            )

        features = np.array(features)
        print(f" 特征矩阵形状: {features.shape}")

        # 创建聚类可视化器
        visualizer = ClusteringVisualizer(
            max_clusters=self.args.max_clusters,
            method=self.args.clustering_method,
            dimensions=self.args.viz_dimensions,
        )

        # 创建可视化输出目录
        viz_output_dir = os.path.join(output_root, "clustering_visualization")
        os.makedirs(viz_output_dir, exist_ok=True)

        # 1. 基于预测ID的可视化
        print(f"\n 创建基于预测ID的可视化...")
        self._create_predicted_id_visualization(
            features,
            predicted_ids,
            image_paths,
            confidences,
            visualizer,
            viz_output_dir,
        )

        # 2. 自动聚类可视化
        print(f"\n 创建自动聚类可视化...")
        self._create_automatic_clustering_visualization(
            features,
            predicted_ids,
            image_paths,
            confidences,
            visualizer,
            viz_output_dir,
        )

        print(f" 聚类可视化完成，结果保存在: {viz_output_dir}")

    def _create_predicted_id_visualization(
        self, features, predicted_ids, image_paths, confidences, visualizer, output_dir
    ):
        """基于预测ID的可视化"""
        # 降维
        reduced_features_pca = visualizer.reduce_dimensions(features, method="pca")
        reduced_features_tsne = visualizer.reduce_dimensions(features, method="tsne")

        # 为每个预测ID分配颜色
        unique_ids = list(set(predicted_ids))
        id_to_label = {id_name: i for i, id_name in enumerate(unique_ids)}
        cluster_labels = np.array([id_to_label[pid] for pid in predicted_ids])

        # PCA可视化
        pca_output = os.path.join(output_dir, "predicted_ids_pca.png")
        visualizer.create_clustering_visualization(
            reduced_features_pca,
            cluster_labels,
            image_paths,
            pca_output,
            "基于预测ID的聚类 (PCA降维)",
        )

        # t-SNE可视化
        tsne_output = os.path.join(output_dir, "predicted_ids_tsne.png")
        visualizer.create_clustering_visualization(
            reduced_features_tsne,
            cluster_labels,
            image_paths,
            tsne_output,
            "基于预测ID的聚类 (t-SNE降维)",
        )

        # 保存ID映射信息
        id_mapping_path = os.path.join(output_dir, "id_mapping.json")
        with open(id_mapping_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "unique_ids": unique_ids,
                    "id_to_label": id_to_label,
                    "total_individuals": len(unique_ids),
                    "total_images": len(predicted_ids),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _create_automatic_clustering_visualization(
        self, features, predicted_ids, image_paths, confidences, visualizer, output_dir
    ):
        """自动聚类可视化"""
        # 降维
        reduced_features_pca = visualizer.reduce_dimensions(features, method="pca")
        reduced_features_tsne = visualizer.reduce_dimensions(features, method="tsne")

        # 寻找最优聚类数量并执行聚类
        optimal_k, scores = visualizer.find_optimal_clusters(features)
        cluster_labels, cluster_centers = visualizer.perform_clustering(
            features, optimal_k
        )

        # 创建评估图表
        visualizer.create_evaluation_plots(scores, output_dir)

        # PCA聚类可视化
        pca_cluster_output = os.path.join(output_dir, "auto_clustering_pca.png")
        visualizer.create_clustering_visualization(
            reduced_features_pca,
            cluster_labels,
            image_paths,
            pca_cluster_output,
            f"自动聚类结果 (K={optimal_k}, PCA降维)",
        )

        # t-SNE聚类可视化
        tsne_cluster_output = os.path.join(output_dir, "auto_clustering_tsne.png")
        visualizer.create_clustering_visualization(
            reduced_features_tsne,
            cluster_labels,
            image_paths,
            tsne_cluster_output,
            f"自动聚类结果 (K={optimal_k}, t-SNE降维)",
        )

        # 分析聚类与预测ID的一致性
        self._analyze_clustering_consistency(
            cluster_labels, predicted_ids, optimal_k, output_dir
        )

    def _analyze_clustering_consistency(
        self, cluster_labels, predicted_ids, optimal_k, output_dir
    ):
        """分析聚类结果与预测ID的一致性"""
        print(f" 分析聚类一致性...")

        # 构建混淆矩阵
        unique_pred_ids = list(set(predicted_ids))
        unique_clusters = list(set(cluster_labels))

        confusion_matrix = np.zeros((len(unique_clusters), len(unique_pred_ids)))

        for cluster_id, pred_id in zip(cluster_labels, predicted_ids):
            cluster_idx = unique_clusters.index(cluster_id)
            pred_idx = unique_pred_ids.index(pred_id)
            confusion_matrix[cluster_idx, pred_idx] += 1

        # 计算一致性指标
        total_samples = len(cluster_labels)
        max_consistency = 0

        for i in range(len(unique_clusters)):
            cluster_max = np.max(confusion_matrix[i, :])
            max_consistency += cluster_max

        consistency_ratio = max_consistency / total_samples

        # 保存分析结果（修复JSON序列化问题）
        analysis_results = {
            "optimal_clusters": int(optimal_k),
            "predicted_individuals": len(unique_pred_ids),
            "consistency_ratio": float(consistency_ratio),
            "confusion_matrix": [
                [float(x) for x in row] for row in confusion_matrix.tolist()
            ],
            "cluster_labels": [
                int(x) if isinstance(x, (np.integer, np.int32, np.int64)) else x
                for x in unique_clusters
            ],
            "predicted_id_labels": unique_pred_ids,
            "analysis": {
                "high_consistency": consistency_ratio > 0.8,
                "moderate_consistency": 0.6 <= consistency_ratio <= 0.8,
                "low_consistency": consistency_ratio < 0.6,
            },
        }

        analysis_path = os.path.join(output_dir, "clustering_analysis.json")
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)

        print(f"  聚类一致性: {consistency_ratio:.3f}")
        print(f"  最优聚类数: {optimal_k}")
        print(f"  预测个体数: {len(unique_pred_ids)}")

    def process_test_images(
        self,
        test_image_root,
        test_roi_root,
        output_root,
        test_roiimg_root=None,
    ):
        """处理测试图像并生成结果"""
        print(f"\n开始处理测试图像...")
        print(f"   测试图像目录: {test_image_root}")
        print(f"   ROI目录: {test_roi_root}")
        print(f"   输出目录: {output_root}")

        # 设置ROI图像相关路径
        if test_roiimg_root:
            # ROIIMG直读：以ROI裁剪图目录作为输入根，且不使用ROI txt
            test_image_root = test_roiimg_root
            test_roi_root = None
            print(f"   ROI图像目录: {test_roiimg_root}")

        # 创建输出目录
        os.makedirs(output_root, exist_ok=True)

        # 收集所有测试图像
        test_images = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        for root, dirs, files in os.walk(test_image_root):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(root, file)
                    test_images.append(image_path)

        print(f"   找到 {len(test_images)} 张测试图像")

        if len(test_images) == 0:
            print("未找到测试图像")
            return

        # 逐张处理图像 - 野外个体识别模式
        results = []
        id_counter = defaultdict(int)
        new_individuals_count = 0

        print(f"\n开始野外个体识别...")
        print(f"模式: 逐张处理，动态创建个体ID")
        start_time = time.time()

        for i, image_path in enumerate(test_images):
            if (i + 1) % 50 == 0:
                print(
                    f"   处理进度: {i + 1}/{len(test_images)} | 当前个体数: {len(self.prototype_net.prototypes)}"
                )

            # 提取特征（使用ROI裁剪，与训练时一致）
            result = self.extract_features(image_path, roi_root=test_roi_root)
            if result is None:
                continue
            query_feature, aux = result

            # 野外个体识别
            prediction = self.predict_identity(query_feature, image_path, aux=aux)

            # 记录结果（在开启开放集评估或聚类可视化时，始终保存特征向量）
            save_feature = getattr(self.args, "compute_open_set_metrics", False) or (
                hasattr(self.args, "enable_clustering_viz")
                and self.args.enable_clustering_viz
            )
            result = {
                "image_path": image_path,
                "predicted_id": prediction["predicted_id"],
                "similarity": prediction["similarity"],
                "confidence": prediction["confidence"],
                "is_new_id": prediction["is_new_id"],
                "action": prediction["action"],
                "gender_prob": (
                    float(aux["gender_prob"])
                    if (aux and "gender_prob" in aux)
                    else None
                ),
                "age_pred": (
                    float(aux["age_pred"]) if (aux and "age_pred" in aux) else None
                ),
                "feature": query_feature.cpu().numpy() if save_feature else None,
            }

            # 添加额外信息
            if "most_similar_id" in prediction:
                result["most_similar_id"] = prediction["most_similar_id"]
                result["max_similarity_with_existing"] = prediction[
                    "max_similarity_with_existing"
                ]

            results.append(result)

            # 统计
            id_counter[prediction["predicted_id"]] += 1
            if prediction["is_new_id"]:
                new_individuals_count += 1

            # 每100张图像显示一次当前状态
            if (i + 1) % 100 == 0:
                print(
                    f"     当前识别出 {len(self.prototype_net.prototypes)} 个不同个体"
                )

        # 可选：计算性别/年龄辅助评估指标
        self.last_aux_metrics = None
        if getattr(self.args, "compute_aux_metrics", False):
            try:
                self.compute_aux_metrics(results, base_root=test_image_root)
            except Exception as e:
                print(f"计算辅助任务评估失败: {e}")

        # 可选：计算开放集 ReID 指标（完全基于真值ID的特征检索/验证：mAP / AUROC / Rank-k / CMC / 验证类指标）
        self.last_open_set_metrics = None
        if getattr(self.args, "compute_open_set_metrics", False):
            try:
                # 注意：compute_open_set_metrics 当前定义为顶层函数，这里显式传入 self
                self.last_open_set_metrics = compute_open_set_metrics(
                    self, results, base_root=test_image_root
                )
            except Exception as e:
                print(f"开放集评估失败: {e}")

        end_time = time.time()
        print(f"野外个体识别完成，耗时: {end_time - start_time:.2f}秒")
        print(f"最终识别出 {len(self.prototype_net.prototypes)} 个不同的野外个体")

        # 生成输出结果
        self.generate_output(results, output_root)

        # 显示统计信息
        self.print_statistics(results, id_counter, new_individuals_count)

        return results

    def generate_output(self, results, output_root):


        # 按预测ID分组（仅用于统计，不再创建逐ID图像目录）
        id_groups = defaultdict(list)
        for result in results:
            predicted_id = result["predicted_id"]
            id_groups[predicted_id].append(result)

        # 聚类可视化（如果启用）：只画特征空间聚类图，不做逐ID图像拷贝
        if getattr(self.args, "enable_clustering_viz", False):
            try:
                self.create_clustering_visualization(results, output_root)
            except Exception as e:
                print(f"聚类可视化生成失败（已忽略，不影响主结果）: {e}")

        # 保存详细结果 JSON（不包含特征向量）
        results_json_path = os.path.join(output_root, "wild_panda_identification_results.json")

        # 原型库摘要信息
        prototype_info = {
            "total_individuals": len(self.prototype_net.prototypes),
            "individuals": list(self.prototype_net.prototypes.keys()),
            "similarity_threshold": self.similarity_threshold,
        }

        # 过滤掉 feature 字段，仅保留可读元信息，避免 JSON 中出现巨大向量
        json_results = []
        for r in results:
            r_json = dict(r)
            r_json.pop("feature", None)
            json_results.append(r_json)

        output_data = {
            "identification_results": json_results,
            "prototype_database": prototype_info,
            "summary": {
                "total_images": len(results),
                "total_individuals": len(id_groups),
                "images_per_individual": {
                    pid: len(group) for pid, group in id_groups.items()
                },
            },
        }

        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"结果已保存到: {output_root}")
        print(f"   详细结果: {results_json_path}")
        print(f"   识别出的个体数: {len(id_groups)}")
        print(
            "   每个个体的图像数(Top-10): ",
            dict(
                sorted(
                    [(k, len(v)) for k, v in id_groups.items()],
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
        )
    
    def print_statistics(self, results, id_counter, new_individuals_count):
        """打印统计信息"""
        print(f"\n野外个体识别统计:")
        print(f"   总图像数: {len(results)}")
        print(f"   识别出的个体数: {len(id_counter)}")
        print(f"   新创建的个体数: {new_individuals_count}")
        print(f"   归类到现有个体的图像数: {len(results) - new_individuals_count}")

        if id_counter:
            print(f"\n个体分布 (按图像数量排序):")
            sorted_ids = sorted(id_counter.items(), key=lambda x: x[1], reverse=True)
            for i, (id_name, count) in enumerate(sorted_ids[:15]):  # 显示前15个
                print(f"   {i+1:2d}. {id_name}: {count} 张图像")

        # 分析创建新个体的情况
        new_individual_actions = [r for r in results if r["is_new_id"]]
        if new_individual_actions:
            print(f"\n新个体创建分析:")
            print(f"   创建新个体的图像数: {len(new_individual_actions)}")

            # 分析与现有个体的最高相似度
            similarities_when_creating_new = []
            for r in new_individual_actions:
                if "max_similarity_with_existing" in r:
                    similarities_when_creating_new.append(
                        r["max_similarity_with_existing"]
                    )

            if similarities_when_creating_new:
                print(f"   创建新个体时与现有个体的最高相似度:")
                print(f"     平均: {np.mean(similarities_when_creating_new):.3f}")
                print(f"     最高: {np.max(similarities_when_creating_new):.3f}")
                print(f"     最低: {np.min(similarities_when_creating_new):.3f}")

    def _parse_gt_from_path(self, image_path, base_root):
        """从图片路径解析真值信息（ID / 性别 / 年龄）"""
        try:
            rel = os.path.relpath(image_path, base_root)
        except Exception:
            rel = image_path
        parts = os.path.normpath(rel).split(os.sep)

        gender_label = None
        birth_year = None
        capture_year = None
        id_name = None

        # 解析个体目录: <name>_<birth_year>_<sex>
        if len(parts) >= 1:
            id_dir = parts[0]
            segs = id_dir.split("_")
            if len(segs) >= 1:
                id_name = segs[0]
            if len(segs) == 3:
                by = segs[1]
                sex_raw = segs[2].upper()
                if by.isdigit():
                    try:
                        birth_year = int(by)
                    except Exception:
                        birth_year = None
                if sex_raw in ["M", "雄"]:
                    gender_label = 1
                elif sex_raw in ["F", "雌"]:
                    gender_label = 0

        # 解析拍摄年: 优先取个体目录之后的第一个4位年份目录
        if len(parts) >= 2:
            for comp in parts[1:-1]:  # 排除文件名
                if re.fullmatch(r"(?:19|20)\d{2}", comp):
                    y = int(comp)
                    if birth_year is not None and y == birth_year:
                        continue
                    capture_year = y
                    break

        age_years = None
        if birth_year is not None and capture_year is not None:
            age_years = float(max(0, capture_year - birth_year))

        return {
            "gender_label": gender_label,
            "age_years": age_years,
            "age_valid": 1 if age_years is not None else 0,
            "id_name": id_name,
        }

    def compute_aux_metrics(self, results, base_root):
        """计算性别准确率与年龄误差/准确率，并保存到输出目录。
        - 性别: 使用 args.gender_threshold 将 male_prob 二分类，计算整体准确率与混淆矩阵；
        - 年龄: 计算 MAE/RMSE，以及±K 岁范围内命中率（K 来自 args.age_accuracy_tolerances）。
        """
        # 解析容差
        tol_str = getattr(self.args, "age_accuracy_tolerances", "1,2,3")
        tolerances = []
        for t in str(tol_str).split(","):
            t = t.strip()
            if t:
                try:
                    tolerances.append(int(t))
                except Exception:
                    pass
        if not tolerances:
            tolerances = [1, 2, 3]

        gender_threshold = float(getattr(self.args, "gender_threshold", 0.5))

        # 统计容器
        g_total = g_correct = 0
        male_tp = male_fp = male_fn = male_tn = 0
        male_gt = male_correct = 0
        female_gt = female_correct = 0

        age_diffs = []
        age_N = 0

        for r in results:
            gp = r.get("gender_prob", None)
            ap = r.get("age_pred", None)
            image_path = r.get("image_path")
            gt = self._parse_gt_from_path(image_path, base_root)

            # 性别
            gt_gender = gt.get("gender_label", None)
            if gp is not None and gt_gender is not None:
                pred_gender = 1 if float(gp) >= gender_threshold else 0
                g_total += 1
                if pred_gender == gt_gender:
                    g_correct += 1
                # 混淆（以 male=1 为正类）
                if pred_gender == 1 and gt_gender == 1:
                    male_tp += 1
                elif pred_gender == 1 and gt_gender == 0:
                    male_fp += 1
                elif pred_gender == 0 and gt_gender == 1:
                    male_fn += 1
                else:
                    male_tn += 1
                # 分类别准确率计数
                if gt_gender == 1:
                    male_gt += 1
                    if pred_gender == gt_gender:
                        male_correct += 1
                else:
                    female_gt += 1
                    if pred_gender == gt_gender:
                        female_correct += 1

            # 年龄
            gt_age = gt.get("age_years", None)
            if ap is not None and gt_age is not None:
                try:
                    diff = float(ap) - float(gt_age)
                    age_diffs.append(diff)
                    age_N += 1
                except Exception:
                    pass

        # 汇总性别
        gender_metrics = None
        if g_total > 0:
            gender_metrics = {
                "N": g_total,
                "accuracy": round(g_correct / g_total, 4),
                "confusion_matrix_male_positive": {
                    "TP": male_tp,
                    "FP": male_fp,
                    "FN": male_fn,
                    "TN": male_tn,
                },
                "per_class_accuracy": {
                    "male": round(male_correct / male_gt, 4) if male_gt > 0 else None,
                    "female": (
                        round(female_correct / female_gt, 4) if female_gt > 0 else None
                    ),
                },
            }

        # 汇总年龄
        age_metrics = None
        if age_N > 0:
            diffs = np.array(age_diffs, dtype=np.float32)
            mae = float(np.mean(np.abs(diffs)))
            rmse = float(np.sqrt(np.mean(diffs**2)))
            within = {str(k): float(np.mean(np.abs(diffs) <= k)) for k in tolerances}
            age_metrics = {
                "N": age_N,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "within_tolerance": {k: round(v, 4) for k, v in within.items()},
            }

        # 打印
        print("\n===== 辅助任务评估(测试集) =====")
        if gender_metrics is not None:
            print(
                f"性别准确率: {gender_metrics['accuracy']:.4f} (N={gender_metrics['N']})  阈值={gender_threshold}"
            )
            cm = gender_metrics["confusion_matrix_male_positive"]
            print(
                f"  混淆矩阵(M为正): TP={cm['TP']} FP={cm['FP']} FN={cm['FN']} TN={cm['TN']}"
            )
            pca = gender_metrics["per_class_accuracy"]
            print(f"  分类别准确率: male={pca['male']}  female={pca['female']}")
        else:
            print("性别评估: 无可用真值或预测，已跳过")

        if age_metrics is not None:
            print(
                f"年龄: MAE={age_metrics['MAE']:.3f}  RMSE={age_metrics['RMSE']:.3f}  (N={age_metrics['N']})"
            )
            wt = age_metrics["within_tolerance"]
            ks = ", ".join([f"±{k}y={wt[k]:.3f}" for k in wt])
            print(f"  年龄准确率(容差): {ks}")
        else:
            print("年龄评估: 无可用真值或预测，已跳过")

        # 保存JSON
        try:
            metrics = {
                "gender": gender_metrics,
                "age": age_metrics,
                "gender_threshold": gender_threshold,
                "age_accuracy_tolerances": tolerances,
            }
            out_path = os.path.join(self.args.output_root, "aux_metrics.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"辅助任务评估指标已保存: {out_path}")
        except Exception as e:
            print(f"保存辅助任务评估指标失败: {e}")

        # 相似度分布统计
        similarities = [r["similarity"] for r in results]
        if similarities:
            print(f"\n相似度分布:")
            print(f"   平均相似度: {np.mean(similarities):.3f}")
            print(f"   最高相似度: {np.max(similarities):.3f}")
            print(f"   最低相似度: {np.min(similarities):.3f}")
            print(
                f"   阈值({self.similarity_threshold})以上: {len([s for s in similarities if s >= self.similarity_threshold])}"
            )
            print(
                f"   阈值({self.similarity_threshold})以下: {len([s for s in similarities if s < self.similarity_threshold])}"
            )


def compute_open_set_metrics(self, results, base_root):
    """基于真值 ID 的 ReID 特征评估（不再使用 is_new_id / ODR / FMR / OSCR）。

    设计目标：
    - 在你当前这种几十万 ROI 的大规模测试集下，不再构造 N×N 相似度矩阵，彻底避免 283GiB 级别的 OOM；
    - 保持原有指标逻辑不变：mAP / AUROC / Rank-k / CMC / intra/inter 统计 / TAR@FAR / 单阈值 accuracy；
    - 通过按 ID 抽样 + 按 probe 批次和按 pair 抽样的方式，给出数值稳定、工程可用的估计。
    """

    print("\n========== 基于真值 ID 的 ReID 特征评估 ==========")

    # 1. 收集具备真值 ID + 特征的样本
    gt_ids_all = []
    pred_ids_all = []
    feats_all = []
    for r in results:
        feat = r.get("feature")
        img_path = r.get("image_path")
        if feat is None or img_path is None:
            continue
        gt_info = self._parse_gt_from_path(img_path, base_root)
        if not gt_info or gt_info.get("id_name") is None:
            continue
        gt_ids_all.append(gt_info["id_name"])
        pred_ids_all.append(str(r.get("predicted_id", "UNKNOWN")))
        feats_all.append(np.asarray(feat, dtype=np.float32))

    if len(feats_all) < 2:
        print("开放集评估：有效样本(<2)过少，已跳过。")
        return {}

    gt_ids_all = np.asarray(gt_ids_all)
    feat_mat_all = np.stack(feats_all).astype(np.float32)
    num_samples = int(feat_mat_all.shape[0])
    unique_ids = np.unique(gt_ids_all)

    print(f"用于开放集 ReID 评估的样本数: {num_samples}，真值 ID 数: {len(unique_ids)}")

    # 额外输出：开放集聚类质量（聚类纯度 + 分配精度 + 个体数量准确率）
    cluster_eval = compute_open_world_cluster_metrics(
        [{"true_id": t, "predicted_id": p} for t, p in zip(gt_ids_all.tolist(), pred_ids_all)]
    )
    print(
        "开放集聚类指标: "
        f"purity={cluster_eval.get('cluster_purity', 0.0):.4f}, "
        f"assignment={cluster_eval.get('assignment_accuracy', 0.0):.4f}, "
        f"id_count_acc={cluster_eval.get('id_count_accuracy', 0.0):.4f}, "
        f"pred_ids={int(cluster_eval.get('predicted_id_count', 0))}, "
        f"true_ids={int(cluster_eval.get('true_id_count', 0))}"
    )

    # 2. 为避免 O(N) 内存，按 ID 下采样用于检索指标的样本
    MAX_PER_ID_FOR_RETRIEVAL = 500   # 每个 ID 参与检索指标的最大样本数
    MAX_PROBES_FOR_MAP = 5000        # mAP/CMC 使用的最大 probe 数
    SIM_BATCH_SIZE = 256             # 计算相似度时的 probe batch size
    rng = np.random.default_rng(42)

    selected_idx_list = []
    for uid in unique_ids:
        idx = np.where(gt_ids_all == uid)[0]
        if idx.size <= MAX_PER_ID_FOR_RETRIEVAL:
            selected_idx_list.append(idx)
        else:
            chosen = rng.choice(idx, size=MAX_PER_ID_FOR_RETRIEVAL, replace=False)
            selected_idx_list.append(chosen)
    selected_indices = np.concatenate(selected_idx_list)

    gt_ids = gt_ids_all[selected_indices]
    feat_mat = feat_mat_all[selected_indices]
    N = int(feat_mat.shape[0])

    if N < num_samples:
        print(
            f"为控制计算量：从 {num_samples} 个样本中按每 ID 最多 {MAX_PER_ID_FOR_RETRIEVAL} 抽样，"
            f"用于检索/成对评估的样本数: {N}"
        )

    # 3. 归一化特征
    norms = np.linalg.norm(feat_mat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    feat_norm = feat_mat / norms  # [N, D]

    # 4. mAP + CMC / Rank-k（按 probe 批次分块计算，不构造完整 N×N）
    max_rank = min(20, N)
    cmc_counts = np.zeros(max_rank, dtype=np.int64)
    num_valid_probes = 0
    ap_scores = []

    if N <= MAX_PROBES_FOR_MAP:
        probe_indices = np.arange(N, dtype=np.int64)
    else:
        probe_indices = rng.choice(N, size=MAX_PROBES_FOR_MAP, replace=False)
        print(
            f"mAP/CMC 仅在 {probe_indices.size} 个 probe 上估计（总样本 {N}），"
            "以控制 O(N) 计算开销。"
        )

    for start in range(0, probe_indices.size, SIM_BATCH_SIZE):
        end = min(start + SIM_BATCH_SIZE, probe_indices.size)
        batch_idx = probe_indices[start:end]
        q_feat = feat_norm[batch_idx]                # [B, D]
        sims_block = np.matmul(q_feat, feat_norm.T)  # [B, N]

        for bi, qi in enumerate(batch_idx):
            q_id = gt_ids[qi]

            pos_mask = gt_ids == q_id
            pos_mask[qi] = False
            if pos_mask.sum() == 0:
                continue

            gallery_mask = np.ones(N, dtype=bool)
            gallery_mask[qi] = False
            y_true = pos_mask[gallery_mask].astype(int)
            y_score = sims_block[bi, gallery_mask]

            # mAP：对该 probe 计算 AP
            if np.unique(y_true).size >= 2:
                try:
                    ap = average_precision_score(y_true, y_score)
                    ap_scores.append(ap)
                except Exception:
                    pass

            # CMC / Rank-k：标准 ReID 检索
            gallery_ids = gt_ids[gallery_mask]
            pos_mask_gallery = gallery_ids == q_id
            if not np.any(pos_mask_gallery):
                continue

            num_valid_probes += 1
            order = np.argsort(-y_score)
            matches = pos_mask_gallery[order][:max_rank]
            cmc_hits = np.cumsum(matches) > 0
            cmc_counts[cmc_hits] += 1

    mAP = float(np.mean(ap_scores)) if ap_scores else 0.0
    print(f"mAP (基于特征的 all-vs-all ReID，子采样估计): {mAP:.4f}")

    rank_k = {}
    cmc_curve = None
    if num_valid_probes > 0:
        cmc_curve = (cmc_counts / num_valid_probes).astype(float)
        for k in [1, 5, 10]:
            if k <= max_rank:
                rank_k[f"rank-{k}"] = float(cmc_curve[k - 1])
        if rank_k:
            print("Rank-k 检索准确率 (基于特征 all-vs-all 检索，子采样估计):")
            for k in sorted(rank_k.keys(), key=lambda x: int(x.split("-")[1])):
                print(f"  {k}: {rank_k[k]:.4f}")
    else:
        print("CMC / Rank-k: 有效 probe 数量为 0，已跳过")

    # 5. AUROC / intra-inter / TAR@FAR / accuracy：在 pair 抽样上直接点积
    max_pairs = 2_000_000
    total_pairs = N * (N - 1) // 2
    y_true_pairs = []
    y_score_pairs = []

    if total_pairs <= max_pairs:
        # 小规模：遍历全部 i<j，对每对直接 np.dot
        for i in range(N):
            fi = feat_norm[i]
            for j in range(i + 1, N):
                same = int(gt_ids[i] == gt_ids[j])
                y_true_pairs.append(same)
                y_score_pairs.append(float(np.dot(fi, feat_norm[j])))
    else:
        # 大规模：随机采样 max_pairs 对
        print(
            f"成对指标在采样的 {max_pairs} 对 (总对数约 {total_pairs:.3e}) 上估计，"
            "以控制计算量。"
        )
        for _ in range(max_pairs):
            i = int(rng.integers(0, N - 1))
            j = int(rng.integers(i + 1, N))
            same = int(gt_ids[i] == gt_ids[j])
            y_true_pairs.append(same)
            y_score_pairs.append(float(np.dot(feat_norm[i], feat_norm[j])))

    # AUROC
    if len(set(y_true_pairs)) >= 2:
        try:
            auroc = float(roc_auc_score(y_true_pairs, y_score_pairs))
        except Exception:
            auroc = 0.0
    else:
        auroc = 0.0
    print(f"AUROC (同ID/不同ID 成对特征): {auroc:.4f}")

    # intra-ID / inter-ID 相似度统计
    similarity_stats = None
    if y_true_pairs:
        y_arr = np.array(y_true_pairs, dtype=np.int8)
        s_arr = np.array(y_score_pairs, dtype=np.float32)
        same_scores = s_arr[y_arr == 1]
        diff_scores = s_arr[y_arr == 0]

        similarity_stats = {
            "num_positive_pairs": int(same_scores.size),
            "num_negative_pairs": int(diff_scores.size),
            "intra_mean": float(same_scores.mean()) if same_scores.size > 0 else None,
            "intra_std": float(same_scores.std()) if same_scores.size > 0 else None,
            "inter_mean": float(diff_scores.mean()) if diff_scores.size > 0 else None,
            "inter_std": float(diff_scores.std()) if diff_scores.size > 0 else None,
        }

        if same_scores.size > 0 and diff_scores.size > 0:
            print("intra-ID / inter-ID 相似度统计:")
            print(
                f"  同ID对: N={same_scores.size}, "
                f"mean={same_scores.mean():.4f}, std={same_scores.std():.4f}"
            )
            print(
                f"  异ID对: N={diff_scores.size}, "
                f"mean={diff_scores.mean():.4f}, std={diff_scores.std():.4f}"
            )
            # 兼容你关心的命名：Sintra/Sinter
            print(f"  Sintra={same_scores.mean():.4f}  Sinter={diff_scores.mean():.4f}")

    # 阈值化验证指标：TAR@FAR + 最佳 F1 + 固定阈值 accuracy
    verification = None
    if len(set(y_true_pairs)) >= 2:
        labels_np = np.array(y_true_pairs, dtype=np.int8)
        scores_np = np.array(y_score_pairs, dtype=np.float32)

        pos_mask = labels_np == 1
        neg_mask = labels_np == 0
        num_pos = int(pos_mask.sum())
        num_neg = int(neg_mask.sum())

        if num_pos > 0 and num_neg > 0:
            neg_scores = np.sort(scores_np[neg_mask])
            far_targets = [1e-2, 1e-3]

            tar_at_far = {}
            for target_far in far_targets:
                if target_far <= 0.0:
                    thr = float(np.nextafter(neg_scores.max(), np.inf)) if neg_scores.size > 0 else 1.0
                elif target_far >= 1.0:
                    thr = float(neg_scores.min()) if neg_scores.size > 0 else 0.0
                else:
                    k = int(np.floor(num_neg * (1.0 - target_far)))
                    k = min(max(k, 0), num_neg - 1)
                    thr = float(neg_scores[k])

                pred_pos = scores_np >= thr
                tp = int(np.sum(pred_pos & (labels_np == 1)))
                fp = int(np.sum(pred_pos & (labels_np == 0)))
                fn = int(np.sum(~pred_pos & (labels_np == 1)))
                tn = int(np.sum(~pred_pos & (labels_np == 0)))

                far = float(fp / max(num_neg, 1))
                tar = float(tp / max(num_pos, 1))
                prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                rec = tar
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

                key = f"{target_far:.0e}"
                tar_at_far[key] = {
                    "threshold": float(thr),
                    "FAR": round(far, 6),
                    "TAR": round(tar, 6),
                    "precision": round(prec, 6),
                    "recall": round(rec, 6),
                    "F1": round(f1, 6),
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                }

            # 基于 P-R 曲线的最佳 F1 阈值
            precisions, recalls, pr_thresholds = precision_recall_curve(labels_np, scores_np)
            f1_scores = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-12, None)
            best_idx = int(np.nanargmax(f1_scores))
            best_f1 = float(f1_scores[best_idx])
            best_prec = float(precisions[best_idx])
            best_rec = float(recalls[best_idx])
            if best_idx == 0 and len(pr_thresholds) > 0:
                best_thr = float(pr_thresholds[0])
            elif best_idx >= len(pr_thresholds):
                best_thr = float(pr_thresholds[-1]) if len(pr_thresholds) > 0 else 0.0
            else:
                best_thr = float(pr_thresholds[best_idx - 1])

            # 基于最佳 F1 阈值的成对准确率：accuracy = (TP + TN) / M
            pred_pos_best = scores_np >= best_thr
            tp_best = int(np.sum(pred_pos_best & (labels_np == 1)))
            tn_best = int(np.sum((~pred_pos_best) & (labels_np == 0)))
            total_pair_samples = int(labels_np.size)
            acc_best = float((tp_best + tn_best) / max(total_pair_samples, 1))

            verification = {
                "num_positive_pairs": num_pos,
                "num_negative_pairs": num_neg,
                "tar_at_far": tar_at_far,
                "best_f1": {
                    "threshold": best_thr,
                    "F1": round(best_f1, 6),
                    "precision": round(best_prec, 6),
                    "recall": round(best_rec, 6),
                    "accuracy": round(acc_best, 6),
                    "num_pairs": total_pair_samples,
                    "num_correct": tp_best + tn_best,
                },
            }

            print("TAR@FAR 验证指标 (基于 all-vs-all 成对特征，采样估计):")
            for k in sorted(tar_at_far.keys()):
                info = tar_at_far[k]
                print(
                    f"  TAR@FAR={k}: TAR={info['TAR']:.4f}, "
                    f"FAR={info['FAR']:.6f}, threshold={info['threshold']:.4f}"
                )
            best = verification["best_f1"]
            print(
                f"  最佳F1点: F1={best['F1']:.4f}, "
                f"precision={best['precision']:.4f}, recall={best['recall']:.4f}, "
                f"accuracy={best['accuracy']:.4f}, threshold={best['threshold']:.4f}"
            )

    # 6. 组织并保存 JSON（不再包含 ODR/FMR/OSCR）
    sintra = None
    sinter = None
    if isinstance(similarity_stats, dict):
        sintra = similarity_stats.get("intra_mean", None)
        sinter = similarity_stats.get("inter_mean", None)

    open_set_metrics = {
        "mAP": round(mAP, 4),
        "AUROC": round(auroc, 4),
        "rank_k": {k: round(v, 4) for k, v in rank_k.items()},
        "cmc_curve": (
            [round(float(x), 4) for x in cmc_curve] if cmc_curve is not None else None
        ),
        "similarity_stats": similarity_stats,
        # 兼容命名：Sintra/Sinter = intra/inter 的平均相似度（同ID对 / 异ID对）
        "Sintra": round(float(sintra), 6) if sintra is not None else None,
        "Sinter": round(float(sinter), 6) if sinter is not None else None,
        "verification": verification,
        "num_samples": int(num_samples),
        "num_gt_ids": int(len(unique_ids)),
        "clustering": {
            "cluster_purity": round(float(cluster_eval.get("cluster_purity", 0.0)), 6),
            "assignment_accuracy": round(float(cluster_eval.get("assignment_accuracy", 0.0)), 6),
            "cluster_contamination": round(float(cluster_eval.get("cluster_contamination", 1.0)), 6),
            "id_count_accuracy": round(float(cluster_eval.get("id_count_accuracy", 0.0)), 6),
            "predicted_id_count": int(cluster_eval.get("predicted_id_count", 0)),
            "true_id_count": int(cluster_eval.get("true_id_count", 0)),
            "id_count_abs_error": int(cluster_eval.get("id_count_abs_error", 0)),
            "id_count_rel_error": round(float(cluster_eval.get("id_count_rel_error", 0.0)), 6),
        },
    }

    try:
        out_path = os.path.join(self.args.output_root, "open_set_metrics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(open_set_metrics, f, ensure_ascii=False, indent=2)
        print(f"开放集 ReID 特征评估指标已保存: {out_path}")
    except Exception as e:
        print(f"保存开放集评估指标失败: {e}")

    return open_set_metrics


def main():
    """主函数"""
    args, config = parse_option()

    # 创建输出目录
    os.makedirs(args.output_root, exist_ok=True)

    # 创建logger
    logger = create_logger(
        output_dir=args.output_root, dist_rank=0, name="panda_reid_inference"
    )

    logger.info("大熊猫个体识别推理开始")
    logger.info(f"配置文件: {args.cfg}")
    logger.info(f"模型路径: {args.model_path}")
    if args.test_image_root:
        logger.info(f"测试图像: {args.test_image_root}")
    if args.test_roi_root:
        logger.info(f"ROI目录: {args.test_roi_root}")
    logger.info(f"输出目录: {args.output_root}")
    if args.test_roiimg_root:
        logger.info(f"ROI图像目录: {args.test_roiimg_root}")
    logger.info(f"相似度阈值: {args.similarity_threshold}")

    # 初始化推理器
    inference = PandaReIDInference(config=config, model_path=args.model_path, args=args)

    # 处理测试图像
    results = inference.process_test_images(
        test_image_root=(args.test_roiimg_root or args.test_image_root),
        test_roi_root=(None if args.test_roiimg_root else args.test_roi_root),
        output_root=args.output_root,
        test_roiimg_root=args.test_roiimg_root,
    )

    logger.info("推理完成！")


if __name__ == "__main__":
    main()
