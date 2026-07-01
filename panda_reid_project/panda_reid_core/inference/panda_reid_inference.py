#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
           
      ArcFace+Triplet+          
             
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


#
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
    print(f"  :                : {e}")
    CLUSTERING_AVAILABLE = False

#
# 2025-12-07:
#
#
#
# 2025-12-08:
#
#
#
#
#
#
#
#
#
#
#
#
#
# ======================================================

#
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

#
OPEN_SET_METRICS = {
    "odr_threshold": 0.5,  #
    "oscr_sample_points": 200,  #
}


#
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_ROOT_DIR)
for _path in (_THIS_DIR, _PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.append(_path)

from panda_reid_core.config import get_config
from panda_reid_core.data.panda_dataset import PandaDataset, build_panda_transform
from panda_reid_core.models.panda_reid_model import build_panda_reid_model
from panda_reid_core.models.prototype_reid_network import PrototypeReIDNetwork
from panda_reid_core.models.open_world_metrics import compute_open_world_cluster_metrics
from panda_reid_core.logger import create_logger


class ClusteringVisualizer:
    """      """

    def __init__(self, max_clusters=10, method="auto", dimensions=2):
        """
                 

        Args:
            max_clusters:       
            method:     
            dimensions:      
        """
        self.max_clusters = max_clusters
        self.method = method
        self.dimensions = dimensions
        self.colors = plt.cm.Set3(np.linspace(0, 1, max_clusters))

    def find_optimal_clusters(self, features, min_clusters=2):
        """
                

        Args:
            features:      [N, D]
            min_clusters:       

        Returns:
            optimal_k:       
            scores:       
        """
        if not CLUSTERING_AVAILABLE:
            return min_clusters, {}

        print(f"          (  : {min_clusters}-{self.max_clusters})...")

        k_range = range(min_clusters, min(self.max_clusters + 1, len(features)))
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []

        for k in k_range:
            #
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)

            #
            sil_score = silhouette_score(features, cluster_labels)
            cal_score = calinski_harabasz_score(features, cluster_labels)
            db_score = davies_bouldin_score(features, cluster_labels)

            silhouette_scores.append(sil_score)
            calinski_scores.append(cal_score)
            davies_bouldin_scores.append(db_score)

            print(
                f"  K={k}:     ={sil_score:.3f}, CH  ={cal_score:.1f}, DB  ={db_score:.3f}"
            )

        #
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
            f"       : K={optimal_k} (    ={scores['optimal_silhouette']:.3f})"
        )

        return optimal_k, scores

    def perform_clustering(self, features, n_clusters=None):
        """
            

        Args:
            features:      [N, D]
            n_clusters:      None     

        Returns:
            cluster_labels:     
            cluster_centers:           
        """
        if not CLUSTERING_AVAILABLE:
            return np.zeros(len(features)), None

        if n_clusters is None:
            n_clusters, _ = self.find_optimal_clusters(features)

        print(f"     :   ={self.method},    ={n_clusters}")

        if self.method == "auto" or self.method == "kmeans":
            #
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(features)
            cluster_centers = clusterer.cluster_centers_

        elif self.method == "dbscan":
            #
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(features)
            cluster_centers = None

        elif self.method == "hierarchical":
            #
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(features)
            cluster_centers = None

        #
        unique_labels = np.unique(cluster_labels)
        print(f"     :   {len(unique_labels)}   ")

        for label in unique_labels:
            count = np.sum(cluster_labels == label)
            if label == -1:
                print(f"     : {count} ")
            else:
                print(f"    {label}: {count}   ")

        return cluster_labels, cluster_centers

    def reduce_dimensions(self, features, method="pca"):
        """
            

        Args:
            features:      [N, D]
            method:      ('pca'   'tsne')

        Returns:
            reduced_features:        [N, dimensions]
        """
        if not CLUSTERING_AVAILABLE:
            return features[:, : self.dimensions]

        print(f"     : {method.upper()} -> {self.dimensions}D")

        if method == "pca":
            reducer = PCA(n_components=self.dimensions, random_state=42)
            reduced_features = reducer.fit_transform(features)
            explained_ratio = reducer.explained_variance_ratio_
            print(f"  PCA     : {explained_ratio}")

        elif method == "tsne":
            #
            if features.shape[1] > 50:
                pca = PCA(n_components=50, random_state=42)
                features = pca.fit_transform(features)
                print(f"     : PCA   50 ")

            reducer = TSNE(
                n_components=self.dimensions,
                random_state=42,
                perplexity=min(30, len(features) - 1),
            )
            reduced_features = reducer.fit_transform(features)

        return reduced_features

    def create_clustering_visualization(
        self, features, cluster_labels, image_paths, output_path, title="     "
    ):
        """
                

        Args:
            features:        [N, dimensions]
            cluster_labels:     
            image_paths:       
            output_path:     
            title:    
        """
        if not CLUSTERING_AVAILABLE:
            print("          ")
            return

        print(f"         ...")

        #
        plt.style.use("seaborn-v0_8")
        #
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        if self.dimensions == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        else:
            fig = plt.figure(figsize=(20, 8))
            ax1 = fig.add_subplot(121, projection="3d")
            ax2 = fig.add_subplot(122)

        #
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)

        #
        colors = plt.cm.Set3(np.linspace(0, 1, max(n_clusters, 10)))

        #
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            color = colors[i % len(colors)]

            if label == -1:
                #
                if self.dimensions == 2:
                    ax1.scatter(
                        features[mask, 0],
                        features[mask, 1],
                        c="black",
                        marker="x",
                        s=50,
                        alpha=0.6,
                        label="  ",
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
                        label="  ",
                    )
        else:
            #
            if self.dimensions == 2:
                ax1.scatter(
                    features[mask, 0],
                    features[mask, 1],
                    c=[color],
                    s=60,
                    alpha=0.7,
                    label=f"   {label}",
                )
            else:
                ax1.scatter(
                    features[mask, 0],
                    features[mask, 1],
                    features[mask, 2],
                    c=[color],
                    s=60,
                    alpha=0.7,
                    label=f"   {label}",
                )

        ax1.set_title(f"{title} -     ", fontsize=14, fontweight="bold")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        #
        cluster_counts = []
        cluster_names = []

        for label in unique_labels:
            count = np.sum(cluster_labels == label)
            if label == -1:
                cluster_names.append("  ")
            else:
                cluster_names.append(f"   {label}")
            cluster_counts.append(count)

        bars = ax2.bar(
            cluster_names,
            cluster_counts,
            color=[colors[i % len(colors)] for i in range(len(unique_labels))],
        )
        ax2.set_title("      ", fontsize=14, fontweight="bold")
        ax2.set_ylabel("    ")
        ax2.tick_params(axis="x", rotation=45)

        #
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

        print(f"          : {output_path}")

    def create_evaluation_plots(self, scores, output_dir):
        """
                

        Args:
            scores:       
            output_dir:     
        """
        if not CLUSTERING_AVAILABLE or not scores:
            return

        print(f"         ...")

        #
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        k_range = scores["k_range"]

        #
        axes[0, 0].plot(
            k_range, scores["silhouette_scores"], "bo-", linewidth=2, markersize=8
        )
        axes[0, 0].axvline(
            x=scores["optimal_k"], color="red", linestyle="--", alpha=0.7
        )
        axes[0, 0].set_title("     (Silhouette Score)", fontweight="bold")
        axes[0, 0].set_xlabel("     K")
        axes[0, 0].set_ylabel("    ")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].text(
            scores["optimal_k"],
            max(scores["silhouette_scores"]) * 0.9,
            f'  K={scores["optimal_k"]}',
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

        #
        axes[0, 1].plot(
            k_range, scores["calinski_scores"], "go-", linewidth=2, markersize=8
        )
        axes[0, 1].axvline(
            x=scores["optimal_k"], color="red", linestyle="--", alpha=0.7
        )
        axes[0, 1].set_title("Calinski-Harabasz   ", fontweight="bold")
        axes[0, 1].set_xlabel("     K")
        axes[0, 1].set_ylabel("CH  ")
        axes[0, 1].grid(True, alpha=0.3)

        #
        axes[1, 0].plot(
            k_range, scores["davies_bouldin_scores"], "ro-", linewidth=2, markersize=8
        )
        axes[1, 0].axvline(
            x=scores["optimal_k"], color="red", linestyle="--", alpha=0.7
        )
        axes[1, 0].set_title("Davies-Bouldin   ", fontweight="bold")
        axes[1, 0].set_xlabel("     K")
        axes[1, 0].set_ylabel("DB   (    )")
        axes[1, 0].grid(True, alpha=0.3)

        #
        #
        sil_norm = np.array(scores["silhouette_scores"])
        cal_norm = np.array(scores["calinski_scores"]) / max(scores["calinski_scores"])
        db_norm = 1 - np.array(scores["davies_bouldin_scores"]) / max(
            scores["davies_bouldin_scores"]
        )

        composite_score = (sil_norm + cal_norm + db_norm) / 3

        axes[1, 1].plot(
            k_range, composite_score, "mo-", linewidth=2, markersize=8, label="    "
        )
        axes[1, 1].axvline(
            x=scores["optimal_k"], color="red", linestyle="--", alpha=0.7
        )
        axes[1, 1].set_title("      ", fontweight="bold")
        axes[1, 1].set_xlabel("     K")
        axes[1, 1].set_ylabel("    ")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        eval_path = os.path.join(output_dir, "clustering_evaluation.png")
        plt.savefig(eval_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"          : {eval_path}")


def parse_option():
    """       """
    parser = argparse.ArgumentParser("Panda ReID Inference Script")

    #
    parser.add_argument("--cfg", type=str, required=True, help="      ")
    parser.add_argument(
        "--model-path", type=str, required=True, help="        "
    )
    parser.add_argument(
        "--test-image-root",
        type=str,
        required=False,
        help="         ROI     ",
    )
    parser.add_argument(
        "--test-roi-root",
        type=str,
        required=False,
        help="  ROI            ",
    )
    parser.add_argument(
        "--test-roiimg-root",
        type=str,
        required=False,
        help="  ROI      (            test-image-root/test-roi-root)",
    )
    parser.add_argument("--output-root", type=str, required=True, help="       ")
    parser.add_argument(
        "--output-roi-root",
        type=str,
        help='ROI        (      output-root + "_roi")',
    )

    #
    parser.add_argument(
        "--roi-format",
        type=str,
        default="mask",
        choices=["yolo", "mask", "auto"],
        help="ROI     (  : mask    , yolo: YOLO  , auto:     )",
    )
    parser.add_argument(
        "--mask-root", type=str, help="            test-roi-root "
    )
    parser.add_argument(
        "--heatmap", action="store_true", help="               /heatmap"
    )
    parser.add_argument(
        "--heatmap-overlay-original",
        action="store_true",
        help="           ROI  ",
    )
    parser.add_argument(
        "--heatmap-image-root",
        type=str,
        required=False,
        help="         ROIIMG              ",
    )
    parser.add_argument(
        "--heatmap-roi-root",
        type=str,
        required=False,
        help="        ROI        mask-root           mask/yolo ",
    )

    #
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=INFERENCE_DEFAULTS["similarity_threshold"],
        help="      (  =  ID  ,     : 0.1-0.6)",
    )
    parser.add_argument(
        "--base-threshold",
        type=float,
        default=INFERENCE_DEFAULTS["base_threshold"],
        help="     (  =  ID  ,     : 0.1-0.5)",
    )
    parser.add_argument(
        "--adaptive-threshold-min",
        type=float,
        default=INFERENCE_DEFAULTS["adaptive_threshold_min"],
        help="         (  =  ID  )",
    )
    parser.add_argument(
        "--adaptive-threshold-max",
        type=float,
        default=INFERENCE_DEFAULTS["adaptive_threshold_max"],
        help="         (  =  ID  )",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=INFERENCE_DEFAULTS["confidence_threshold"],
        help="      (  =  ID  )",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=INFERENCE_DEFAULTS["quality_threshold"],
        help="     (  =  ID  )",
    )
    parser.add_argument(
        "--use-simple-logic",
        action="store_true",
        default=True,
        help="         (      )",
    )

    #
    parser.add_argument(
        "--mode",
        type=str,
        choices=["strict", "balanced", "loose", "custom"],
        default=INFERENCE_DEFAULTS["mode"],
        help="    : strict( ID), balanced(  ), loose( ID), custom(   )",
    )

    #
    parser.add_argument(
        "--sim-cosine-w",
        type=float,
        default=INFERENCE_DEFAULTS["sim_cosine_w"],
        help="                ",
    )
    parser.add_argument(
        "--sim-euclid-w",
        type=float,
        default=INFERENCE_DEFAULTS["sim_euclid_w"],
        help="                 (        1)",
    )
    parser.add_argument(
        "--aux-gender-penalty",
        type=float,
        default=INFERENCE_DEFAULTS["aux_gender_penalty"],
        help="                  (0~1)",
    )
    parser.add_argument(
        "--aux-age-reweight",
        type=float,
        default=INFERENCE_DEFAULTS["aux_age_reweight"],
        help="           (0~1)",
    )
    parser.add_argument(
        "--aux-min-age-sigma",
        type=float,
        default=INFERENCE_DEFAULTS["aux_min_age_sigma"],
        help="            (  :  )       ",
    )

    #
    parser.add_argument(
        "--enable-clustering-viz", action="store_true", help="       "
    )
    parser.add_argument("--max-clusters", type=int, default=10, help="      ")
    parser.add_argument(
        "--clustering-method",
        type=str,
        default="auto",
        choices=["auto", "kmeans", "dbscan", "hierarchical"],
        help="    ",
    )
    parser.add_argument(
        "--viz-dimensions", type=int, default=2, choices=[2, 3], help="     "
    )
    parser.add_argument(
        "--min-samples-per-id",
        type=int,
        default=1,
        help="  ID              ID    ",
    )

    #
    parser.add_argument(
        "--compute-aux-metrics",
        action="store_true",
        help="                    /            ",
    )
    parser.add_argument(
        "--gender-threshold",
        type=float,
        default=INFERENCE_DEFAULTS["gender_threshold"],
        help="       male_prob >=         ",
    )
    parser.add_argument(
        "--age-accuracy-tolerances",
        type=str,
        default="1,2,3",
        help='             "1,2,3"    1/ 2/ 3        ',
    )
    parser.add_argument(
        "--gender-hysteresis",
        type=float,
        default=INFERENCE_DEFAULTS["gender_hysteresis"],
        help="                ",
    )
    parser.add_argument(
        "--age-scope",
        type=str,
        default=INFERENCE_DEFAULTS["age_scope"],
        choices=["track", "video", "global"],
        help="       : track=    video=     global=     ",
    )
    parser.add_argument(
        "--age-display",
        type=str,
        default=INFERENCE_DEFAULTS["age_display"],
        choices=["instant", "median", "mean"],
        help="      : instant=    median=    mean=  ",
    )

    #
    parser.add_argument(
        "--compute-open-set-metrics",
        action="store_true",
        help="      ID  ReID        (mAP/AUROC/Rank-k/CMC/       )            (    --compute-aux-metrics)        ",
    )
    #
    parser.add_argument(
        "--open-set-threshold",
        type=float,
        default=OPEN_SET_METRICS["odr_threshold"],
        help="[    ]    ODR/FMR                ",
    )
    parser.add_argument(
        "--oscr-points",
        type=int,
        default=OPEN_SET_METRICS["oscr_sample_points"],
        help="[    ]    OSCR                  ",
    )

    #
    parser.add_argument("--batch-size", type=int, default=32, help="     ")
    parser.add_argument("--verbose", action="store_true", help="      ")

    #
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
                f"[WARN]           : {unknown_flags} "
                "(           --compute-open-set-metrics)"
            )
    #
    if getattr(args, "compute_open_set_metrics", False) and not getattr(
        args, "compute_aux_metrics", False
    ):
        args.compute_aux_metrics = True
    config = get_config(args)
    return args, config


class PandaReIDInference:
    """          """

    def __init__(self, config, model_path, args):
        """
              

        Args:
            config:     
            model_path:         
            args:      
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args

        #
        self.apply_mode_settings()

        #
        self.similarity_threshold = self.args.similarity_threshold

        #
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

        #
        self.prototype_net = PrototypeReIDNetwork(
            feature_dim=self.model.feature_dim,
            temperature=0.05,  #
            momentum=0.9,  #
            min_samples=1,
        )
        self.prototype_net.to(self.device)

        #
        self.load_model(model_path)

        #
        self.clear_training_prototypes()

        #
        self.configure_prototype_network()

        #
        self.transform = build_panda_transform(
            is_train=False, img_size=config.DATA.IMG_SIZE
        )

        #
        self.wild_id_counter = 0

        print(f"   ReID        ")
        print(f"         : {self.model.feature_dim}")
        print(f"        : {self.similarity_threshold}")
        print(f"     : {self.device}")
        print(f"     : {self.args.mode}")
        if self.args.verbose:
            print(f"       : {self.args.base_threshold}")
            print(
                f"          : [{self.args.adaptive_threshold_min}, {self.args.adaptive_threshold_max}]"
            )
            print(f"        : {self.args.confidence_threshold}")
            print(f"       : {self.args.quality_threshold}")
        print(f"                   ")

    def apply_mode_settings(self):
        """            mode        """

        #
        if getattr(self.args, "mode", None) == "custom":
            for k, v in INFERENCE_DEFAULTS.items():
                if not hasattr(self.args, k):
                    continue
                cur = getattr(self.args, k)
                if cur is None:
                    setattr(self.args, k, v)

        print_mode = self.args.mode

        if print_mode == "strict":
            #
            self.args.similarity_threshold = 0.35
            self.args.base_threshold = 0.25
            self.args.adaptive_threshold_min = 0.1
            self.args.adaptive_threshold_max = 0.5
            self.args.confidence_threshold = 0.15
            self.args.quality_threshold = 0.03
            print("          (  ID  )")

        elif print_mode == "balanced":
            #
            self.args.similarity_threshold = 0.45
            self.args.base_threshold = 0.3
            self.args.adaptive_threshold_min = 0.15
            self.args.adaptive_threshold_max = 0.6
            self.args.confidence_threshold = 0.2
            self.args.quality_threshold = 0.05
            print("           (  ID  )")

        elif print_mode == "loose":
            #
            self.args.similarity_threshold = 0.55
            self.args.base_threshold = 0.4
            self.args.adaptive_threshold_min = 0.2
            self.args.adaptive_threshold_max = 0.7
            self.args.confidence_threshold = 0.3
            self.args.quality_threshold = 0.1
            print("          (  ID  )")

        else:
            #
            print("           (   INFERENCE_DEFAULTS      )")
            if getattr(self.args, "verbose", False):
                print("         :")
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
        """                       """
        try:
            print(f"    : {model_path}")
            #
            checkpoint = torch.load(model_path, map_location=self.device)

            #
            state_dict = checkpoint.get("model", checkpoint)

            #
            try:
                self.model.load_state_dict(state_dict, strict=True)
                print("         (strict)")
            except RuntimeError as e:
                print("             (   )              ")
                model_dict = self.model.state_dict()
                filtered_state = {}
                skipped = []
                for k, v in state_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        filtered_state[k] = v
                    else:
                        skipped.append(k)
                #
                self.model.load_state_dict(filtered_state, strict=False)
                if any("classifier" in k or "neck" in k for k in skipped):
                    print(
                        "              neck.classifier.*                  "
                    )
                print(f"                : {len(skipped)}")

            #
            if isinstance(checkpoint, dict) and "prototype_net" in checkpoint:
                try:
                    self.prototype_net.load_state_dict(
                        checkpoint["prototype_net"], strict=False
                    )
                    print("          ")
                except Exception as pe:
                    print(f"             : {pe}")
        except Exception as ex:
            print(f"      : {ex}")
            raise

        #
        #
        print("          (       )")
        print("                 ")

        #
        self.model.eval()
        self.prototype_net.eval()

        #
        if isinstance(checkpoint, dict) and "epoch" in checkpoint:
            print(f"       : {checkpoint['epoch']}")
        if isinstance(checkpoint, dict) and "best_comprehensive_score" in checkpoint:
            print(f"         : {checkpoint['best_comprehensive_score']:.4f}")

    def clear_training_prototypes(self):
        """             """
        self.prototype_net.prototypes.clear()
        print("                    ")

    def configure_prototype_network(self):
        """          """
        #
        #

        #
        def patched_compute_adaptive_threshold(original_self, query_feature):
            """            """
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

                    #
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

        #
        import types

        self.prototype_net.compute_adaptive_threshold = types.MethodType(
            patched_compute_adaptive_threshold, self.prototype_net
        )

        #
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
        #
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
            print("            ")

    def _read_roi_data(self, image_path, roi_root):
        """
          ROI     YOLO        

        Args:
            image_path:     
            roi_root: ROI     

        Returns:
                 ROI         None
        """
        try:
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]

            #
            roi_format = getattr(self.args, "roi_format", "mask")
            mask_root_cfg = getattr(self.args, "mask_root", None)
            mask_root = (
                mask_root_cfg or roi_root
            )  #

            #
            parent_dir_name = os.path.basename(os.path.dirname(image_path))
            mask_candidates = [
                os.path.join(mask_root, base_name + ".txt"),
                os.path.join(mask_root, parent_dir_name, base_name + ".txt"),
            ]
            yolo_candidates = [
                os.path.join(roi_root, base_name + ".txt"),
                os.path.join(roi_root, parent_dir_name, base_name + ".txt"),
            ]

            #
            if roi_format in ("mask", "auto") and mask_root:
                for coords_path in mask_candidates:
                    if os.path.exists(coords_path):
                        mask = self._read_mask_from_coords(coords_path, image_path)
                        if mask is not None:
                            return {"type": "mask", "data": mask}
                #
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
                #
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
          YOLO   ROI  

        Args:
            roi_path: ROI    

        Returns:
            (cx, cy, w, h)             None
        """
        try:
            if os.path.exists(roi_path):
                with open(roi_path, "r") as f:
                    lines = f.readlines()
                if lines:
                    #
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
                             
        """
        try:
            #
            image = cv2.imread(image_path)
            if image is None:
                return None
            h, w = image.shape[:2]

            #
            with open(coords_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if len(lines) < 3:
                return None

            #
            contour_count = int(lines[2].strip())

            if contour_count == 0:
                return None

            #
            mask = np.zeros((h, w), dtype=np.uint8)

            #
            line_idx = 3
            for _ in range(contour_count):
                if line_idx >= len(lines):
                    break

                #
                point_count = int(lines[line_idx].strip())
                line_idx += 1

                #
                contour_points = []
                for _ in range(point_count):
                    if line_idx >= len(lines):
                        break
                    x, y = map(float, lines[line_idx].strip().split(","))
                    contour_points.append([int(x), int(y)])
                    line_idx += 1

                if contour_points:
                    #
                    contour = np.array(contour_points, dtype=np.int32)
                    cv2.fillPoly(mask, [contour], 255)

            return mask.astype(bool)

        except Exception as e:
            if hasattr(self, "args") and self.args.verbose:
                print(f"Failed to read mask from coords {coords_path}: {e}")
            return None

    def _apply_mask_to_image(self, image, mask):
        """       """
        try:
            #
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            #
            masked_image = image.copy()
            masked_image[~mask] = 0

            return masked_image

        except Exception as e:
            if hasattr(self, "args") and self.args.verbose:
                print(f"      : {e}")
            return image

    def _process_mask_region_to_rectangle(self, image, mask):
        """
                               

           
        1.         
        2.       10%   
        3.           
        4.                  0   
        """
        try:
            #
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            #
            rows, cols = np.where(mask)
            if len(rows) == 0:
                #
                return np.zeros(
                    (self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE, 3),
                    dtype=np.uint8,
                )

            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()

            #
            h, w = image.shape[:2]
            bbox_w = x2 - x1 + 1
            bbox_h = y2 - y1 + 1

            expand_w = int(bbox_w * 0.1)  #
            expand_h = int(bbox_h * 0.1)

            #
            x1_exp = max(0, x1 - expand_w // 2)
            y1_exp = max(0, y1 - expand_h // 2)
            x2_exp = min(w, x2 + expand_w // 2)
            y2_exp = min(h, y2 + expand_h // 2)

            #
            cropped_image = image[y1_exp:y2_exp, x1_exp:x2_exp].copy()
            cropped_mask = mask[y1_exp:y2_exp, x1_exp:x2_exp]

            #
            #
            cropped_image[~cropped_mask] = 0

            return cropped_image

        except Exception as e:
            if hasattr(self, "args") and self.args.verbose:
                print(f"        : {e}")
            #
            return np.zeros(
                (self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE, 3),
                dtype=np.uint8,
            )

    def _get_mask_bbox(self, mask):
        """        """
        try:
            rows, cols = np.where(mask)

            if len(rows) == 0:
                return None

            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()

            h, w = mask.shape

            #
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bbox_w = (x2 - x1 + 1) / w
            bbox_h = (y2 - y1 + 1) / h

            return cx, cy, bbox_w, bbox_h

        except Exception as e:
            if hasattr(self, "args") and self.args.verbose:
                print(f"       : {e}")
            return None

    def _crop_roi_with_expansion(self, image, roi, roi_expand_ratio=0.1):
        """
          ROI                   

        Args:
            image:      (H, W, C)
            roi: (cx, cy, w, h)      
            roi_expand_ratio: ROI    

        Returns:
                  
        """
        h, w = image.shape[:2]
        cx, cy, roi_w, roi_h = roi

        #
        cx_px = cx * w
        cy_px = cy * h
        w_px = roi_w * w
        h_px = roi_h * h

        #
        expand_w = w_px * roi_expand_ratio
        expand_h = h_px * roi_expand_ratio

        #
        x1 = max(0, int(cx_px - (w_px + expand_w) / 2))
        y1 = max(0, int(cy_px - (h_px + expand_h) / 2))
        x2 = min(w, int(cx_px + (w_px + expand_w) / 2))
        y2 = min(h, int(cy_px + (h_px + expand_h) / 2))

        #
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
        """     
        -     ROI   np_image_rgb, HxWx3, RGB       tensor 1x3xHxW 
        -    save_path
        -  overlay_original=True   original_image_path roi_bbox/roi_mask            ROI  
        """
        try:
            import torch
            import torch.nn.functional as F
            import numpy as np
            import cv2
            import os

            self.model.eval()
            with torch.no_grad():
                #
                attn = tensor_image.squeeze(0).pow(2).sum(0).sqrt().cpu().numpy()  # HxW
                attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
                attn_color = cv2.applyColorMap(
                    (attn * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                attn_color = cv2.cvtColor(attn_color, cv2.COLOR_BGR2RGB)

                #
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

                #
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, cv2.cvtColor(overlay_roi, cv2.COLOR_RGB2BGR))

                #
                if overlay_original and original_image_path is not None:
                    orig = cv2.imread(original_image_path)
                    if orig is not None:
                        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
                        H0, W0 = orig.shape[:2]
                        #
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
                        #
                        roi_w0 = max(1, x2 - x1)
                        roi_h0 = max(1, y2 - y1)
                        overlay_resized = cv2.resize(
                            overlay_roi,
                            (roi_w0, roi_h0),
                            interpolation=cv2.INTER_LINEAR,
                        )

                        orig_overlay = orig.copy()
                        if overlay_mask_shape and roi_mask is not None:
                            #
                            #
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
                            #
                            orig_overlay[y1:y2, x1:x2] = overlay_resized

                        #
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
                print(f"       : {e}")

    def extract_features(self, image_path, roi_root=None):
        """
                           

        Args:
            image_path:     
            roi_root: ROI     

        Returns:
                    
        """
        try:
            #
            image = cv2.imread(image_path)
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            #
            original_image_for_vis = None
            if roi_root:
                roi_data = self._read_roi_data(image_path, roi_root)
                if roi_data is not None:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    if roi_data["type"] == "mask":
                        mask = roi_data["data"]
                        #
                        original_image_for_vis = image.copy()
                        image = self._process_mask_region_to_rectangle(image, mask)
                        if hasattr(self, "args") and self.args.verbose:
                            print(f"              : {base_name}")
                    elif roi_data["type"] == "yolo":
                        roi = roi_data["data"]
                        original_image_for_vis = image.copy()
                        image = self._crop_roi_with_expansion(
                            image, roi, roi_expand_ratio=0.1
                        )
                        if hasattr(self, "args") and self.args.verbose:
                            print(f"  YOLO ROI        : {base_name}")

            #
            if image.size == 0:
                print(f"  :         {image_path}")
                #
                image = np.zeros(
                    (self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE, 3),
                    dtype=np.uint8,
                )

            #
            image_pil = Image.fromarray(image)

            #
            tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

            #
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

                    #
                    if roi_root and roi_data is not None:
                        if roi_data["type"] == "yolo":
                            roi = roi_data["data"]
                            x, y, w, h = roi
                            roi_bbox = (int(x), int(y), int(x + w), int(y + h))
                        elif roi_data["type"] == "mask":
                            roi_mask_local = roi_data["data"]
                    #
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
                        print(f"      {image_path}: {he}")

            #
            with torch.no_grad():
                feat_after_bn, _feat_before_bn, gender_logits, age_pred = (
                    self.model.forward_multitask(tensor)
                )
                feat_after_bn = F.normalize(feat_after_bn, p=2, dim=1)  #
                male_prob = F.softmax(gender_logits, dim=1)[:, 1]

            aux = {
                "gender_prob": float(male_prob[0].item()),
                "age_pred": float(age_pred[0].item()),
            }
            return feat_after_bn[0], aux  #

        except Exception as e:
            print(f"       {image_path}: {e}")
            if hasattr(self, "args") and self.args.verbose:
                import traceback

                print(f"    : {traceback.format_exc()}")
            return None

    def predict_identity(self, query_feature, image_path, aux=None):
        """
                 -          ID

        Args:
            query_feature:     
            image_path:          ID   
            aux:          {'gender_prob': float, 'age_pred': float}

        Returns:
                
        """
        #
        if len(self.prototype_net.prototypes) == 0:
            self.wild_id_counter += 1
            new_id = f"Wild_Panda_{self.wild_id_counter:03d}"

            #
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

        #
        with torch.no_grad():
            result = self.prototype_net(query_feature, aux=aux)

        best_similarity = result["similarity"]
        best_match_id = result["predicted_id"]

        #
        if best_similarity >= self.similarity_threshold:
            #
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
            #
            self.wild_id_counter += 1
            new_id = f"Wild_Panda_{self.wild_id_counter:03d}"

            #
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
                 ID

        Args:
            features:     
            predicted_ids:   ID  
            image_paths:       
            confidences:      
            min_samples:      

        Returns:
                   ID         
        """
        from collections import Counter

        #
        id_counts = Counter(predicted_ids)

        print(f" ID      (   ):")
        for id_name, count in sorted(id_counts.items()):
            print(f"  {id_name}: {count}   ")

        #
        valid_ids = {
            id_name for id_name, count in id_counts.items() if count >= min_samples
        }

        print(f"\n     :")
        print(f"         : {min_samples}")
        print(f"     ID  : {len(id_counts)}")
        print(f"     ID  : {len(valid_ids)}")

        if len(valid_ids) == 0:
            print("    ID       min_samples_per_id  ")
            return features, predicted_ids, image_paths, confidences

        #
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

        print(f"        : {len(features)}")
        print(f"        : {len(filtered_features)}")

        #
        filtered_id_counts = Counter(filtered_ids)
        print(f"\n    ID     :")
        for id_name, count in sorted(filtered_id_counts.items()):
            print(f"  {id_name}: {count}   ")

        return filtered_features, filtered_ids, filtered_paths, filtered_confidences

    def create_clustering_visualization(self, results, output_root):
        """
               

        Args:
            results:       
            output_root:      
        """
        if not CLUSTERING_AVAILABLE:
            print("                    ")
            return

        print(f"\n          ...")

        #
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
            print("                  ")
            return

        #
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
        print(f"       : {features.shape}")

        #
        visualizer = ClusteringVisualizer(
            max_clusters=self.args.max_clusters,
            method=self.args.clustering_method,
            dimensions=self.args.viz_dimensions,
        )

        #
        viz_output_dir = os.path.join(output_root, "clustering_visualization")
        os.makedirs(viz_output_dir, exist_ok=True)

        #
        print(f"\n       ID    ...")
        self._create_predicted_id_visualization(
            features,
            predicted_ids,
            image_paths,
            confidences,
            visualizer,
            viz_output_dir,
        )

        #
        print(f"\n          ...")
        self._create_automatic_clustering_visualization(
            features,
            predicted_ids,
            image_paths,
            confidences,
            visualizer,
            viz_output_dir,
        )

        print(f"              : {viz_output_dir}")

    def _create_predicted_id_visualization(
        self, features, predicted_ids, image_paths, confidences, visualizer, output_dir
    ):
        """    ID    """
        #
        reduced_features_pca = visualizer.reduce_dimensions(features, method="pca")
        reduced_features_tsne = visualizer.reduce_dimensions(features, method="tsne")

        #
        unique_ids = list(set(predicted_ids))
        id_to_label = {id_name: i for i, id_name in enumerate(unique_ids)}
        cluster_labels = np.array([id_to_label[pid] for pid in predicted_ids])

        #
        pca_output = os.path.join(output_dir, "predicted_ids_pca.png")
        visualizer.create_clustering_visualization(
            reduced_features_pca,
            cluster_labels,
            image_paths,
            pca_output,
            "    ID    (PCA  )",
        )

        #
        tsne_output = os.path.join(output_dir, "predicted_ids_tsne.png")
        visualizer.create_clustering_visualization(
            reduced_features_tsne,
            cluster_labels,
            image_paths,
            tsne_output,
            "    ID    (t-SNE  )",
        )

        #
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
        """       """
        #
        reduced_features_pca = visualizer.reduce_dimensions(features, method="pca")
        reduced_features_tsne = visualizer.reduce_dimensions(features, method="tsne")

        #
        optimal_k, scores = visualizer.find_optimal_clusters(features)
        cluster_labels, cluster_centers = visualizer.perform_clustering(
            features, optimal_k
        )

        #
        visualizer.create_evaluation_plots(scores, output_dir)

        #
        pca_cluster_output = os.path.join(output_dir, "auto_clustering_pca.png")
        visualizer.create_clustering_visualization(
            reduced_features_pca,
            cluster_labels,
            image_paths,
            pca_cluster_output,
            f"       (K={optimal_k}, PCA  )",
        )

        #
        tsne_cluster_output = os.path.join(output_dir, "auto_clustering_tsne.png")
        visualizer.create_clustering_visualization(
            reduced_features_tsne,
            cluster_labels,
            image_paths,
            tsne_cluster_output,
            f"       (K={optimal_k}, t-SNE  )",
        )

        #
        self._analyze_clustering_consistency(
            cluster_labels, predicted_ids, optimal_k, output_dir
        )

    def _analyze_clustering_consistency(
        self, cluster_labels, predicted_ids, optimal_k, output_dir
    ):
        """         ID    """
        print(f"        ...")

        #
        unique_pred_ids = list(set(predicted_ids))
        unique_clusters = list(set(cluster_labels))

        confusion_matrix = np.zeros((len(unique_clusters), len(unique_pred_ids)))

        for cluster_id, pred_id in zip(cluster_labels, predicted_ids):
            cluster_idx = unique_clusters.index(cluster_id)
            pred_idx = unique_pred_ids.index(pred_id)
            confusion_matrix[cluster_idx, pred_idx] += 1

        #
        total_samples = len(cluster_labels)
        max_consistency = 0

        for i in range(len(unique_clusters)):
            cluster_max = np.max(confusion_matrix[i, :])
            max_consistency += cluster_max

        consistency_ratio = max_consistency / total_samples

        #
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

        print(f"       : {consistency_ratio:.3f}")
        print(f"       : {optimal_k}")
        print(f"       : {len(unique_pred_ids)}")

    def process_test_images(
        self,
        test_image_root,
        test_roi_root,
        output_root,
        test_roiimg_root=None,
    ):
        """           """
        print(f"\n        ...")
        print(f"         : {test_image_root}")
        print(f"   ROI  : {test_roi_root}")
        print(f"       : {output_root}")

        #
        if test_roiimg_root:
            #
            test_image_root = test_roiimg_root
            test_roi_root = None
            print(f"   ROI    : {test_roiimg_root}")

        #
        os.makedirs(output_root, exist_ok=True)

        #
        test_images = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        for root, dirs, files in os.walk(test_image_root):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(root, file)
                    test_images.append(image_path)

        print(f"      {len(test_images)}      ")

        if len(test_images) == 0:
            print("       ")
            return

        #
        results = []
        id_counter = defaultdict(int)
        new_individuals_count = 0

        print(f"\n        ...")
        print(f"  :            ID")
        start_time = time.time()

        for i, image_path in enumerate(test_images):
            if (i + 1) % 50 == 0:
                print(
                    f"       : {i + 1}/{len(test_images)} |      : {len(self.prototype_net.prototypes)}"
                )

            #
            result = self.extract_features(image_path, roi_root=test_roi_root)
            if result is None:
                continue
            query_feature, aux = result

            #
            prediction = self.predict_identity(query_feature, image_path, aux=aux)

            #
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

            #
            if "most_similar_id" in prediction:
                result["most_similar_id"] = prediction["most_similar_id"]
                result["max_similarity_with_existing"] = prediction[
                    "max_similarity_with_existing"
                ]

            results.append(result)

            #
            id_counter[prediction["predicted_id"]] += 1
            if prediction["is_new_id"]:
                new_individuals_count += 1

            #
            if (i + 1) % 100 == 0:
                print(
                    f"           {len(self.prototype_net.prototypes)}      "
                )

        #
        self.last_aux_metrics = None
        if getattr(self.args, "compute_aux_metrics", False):
            try:
                self.compute_aux_metrics(results, base_root=test_image_root)
            except Exception as e:
                print(f"          : {e}")

        #
        self.last_open_set_metrics = None
        if getattr(self.args, "compute_open_set_metrics", False):
            try:
                #
                self.last_open_set_metrics = compute_open_set_metrics(
                    self, results, base_root=test_image_root
                )
            except Exception as e:
                print(f"       : {e}")

        end_time = time.time()
        print(f"           : {end_time - start_time:.2f} ")
        print(f"      {len(self.prototype_net.prototypes)}         ")

        #
        self.generate_output(results, output_root)

        #
        self.print_statistics(results, id_counter, new_individuals_count)

        return results

    def generate_output(self, results, output_root):


        #
        id_groups = defaultdict(list)
        for result in results:
            predicted_id = result["predicted_id"]
            id_groups[predicted_id].append(result)

        #
        if getattr(self.args, "enable_clustering_viz", False):
            try:
                self.create_clustering_visualization(results, output_root)
            except Exception as e:
                print(f"                     : {e}")

        #
        results_json_path = os.path.join(output_root, "wild_panda_identification_results.json")

        #
        prototype_info = {
            "total_individuals": len(self.prototype_net.prototypes),
            "individuals": list(self.prototype_net.prototypes.keys()),
            "similarity_threshold": self.similarity_threshold,
        }

        #
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

        print(f"      : {output_root}")
        print(f"       : {results_json_path}")
        print(f"          : {len(id_groups)}")
        print(
            "           (Top-10): ",
            dict(
                sorted(
                    [(k, len(v)) for k, v in id_groups.items()],
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
        )
    
    def print_statistics(self, results, id_counter, new_individuals_count):
        """      """
        print(f"\n        :")
        print(f"       : {len(results)}")
        print(f"          : {len(id_counter)}")
        print(f"          : {new_individuals_count}")
        print(f"              : {len(results) - new_individuals_count}")

        if id_counter:
            print(f"\n     (       ):")
            sorted_ids = sorted(id_counter.items(), key=lambda x: x[1], reverse=True)
            for i, (id_name, count) in enumerate(sorted_ids[:15]):  #
                print(f"   {i+1:2d}. {id_name}: {count}    ")

        #
        new_individual_actions = [r for r in results if r["is_new_id"]]
        if new_individual_actions:
            print(f"\n       :")
            print(f"            : {len(new_individual_actions)}")

            #
            similarities_when_creating_new = []
            for r in new_individual_actions:
                if "max_similarity_with_existing" in r:
                    similarities_when_creating_new.append(
                        r["max_similarity_with_existing"]
                    )

            if similarities_when_creating_new:
                print(f"                    :")
                print(f"       : {np.mean(similarities_when_creating_new):.3f}")
                print(f"       : {np.max(similarities_when_creating_new):.3f}")
                print(f"       : {np.min(similarities_when_creating_new):.3f}")

    def _parse_gt_from_path(self, image_path, base_root):
        """            ID /    /    """
        try:
            rel = os.path.relpath(image_path, base_root)
        except Exception:
            rel = image_path
        parts = os.path.normpath(rel).split(os.sep)

        gender_label = None
        birth_year = None
        capture_year = None
        id_name = None

        #
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
                if sex_raw in ["M", " "]:
                    gender_label = 1
                elif sex_raw in ["F", " "]:
                    gender_label = 0

        #
        if len(parts) >= 2:
            for comp in parts[1:-1]:  #
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
        """            /             
        -   :    args.gender_threshold   male_prob                  
        -   :    MAE/RMSE    K         K    args.age_accuracy_tolerances  
        """
        #
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

        #
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

            #
            gt_gender = gt.get("gender_label", None)
            if gp is not None and gt_gender is not None:
                pred_gender = 1 if float(gp) >= gender_threshold else 0
                g_total += 1
                if pred_gender == gt_gender:
                    g_correct += 1
                #
                if pred_gender == 1 and gt_gender == 1:
                    male_tp += 1
                elif pred_gender == 1 and gt_gender == 0:
                    male_fp += 1
                elif pred_gender == 0 and gt_gender == 1:
                    male_fn += 1
                else:
                    male_tn += 1
                #
                if gt_gender == 1:
                    male_gt += 1
                    if pred_gender == gt_gender:
                        male_correct += 1
                else:
                    female_gt += 1
                    if pred_gender == gt_gender:
                        female_correct += 1

            #
            gt_age = gt.get("age_years", None)
            if ap is not None and gt_age is not None:
                try:
                    diff = float(ap) - float(gt_age)
                    age_diffs.append(diff)
                    age_N += 1
                except Exception:
                    pass

        #
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

        #
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

        #
        print("\n=====       (   ) =====")
        if gender_metrics is not None:
            print(
                f"     : {gender_metrics['accuracy']:.4f} (N={gender_metrics['N']})    ={gender_threshold}"
            )
            cm = gender_metrics["confusion_matrix_male_positive"]
            print(
                f"      (M  ): TP={cm['TP']} FP={cm['FP']} FN={cm['FN']} TN={cm['TN']}"
            )
            pca = gender_metrics["per_class_accuracy"]
            print(f"        : male={pca['male']}  female={pca['female']}")
        else:
            print("    :             ")

        if age_metrics is not None:
            print(
                f"  : MAE={age_metrics['MAE']:.3f}  RMSE={age_metrics['RMSE']:.3f}  (N={age_metrics['N']})"
            )
            wt = age_metrics["within_tolerance"]
            ks = ", ".join([f" {k}y={wt[k]:.3f}" for k in wt])
            print(f"       (  ): {ks}")
        else:
            print("    :             ")

        #
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
            print(f"           : {out_path}")
        except Exception as e:
            print(f"            : {e}")

        #
        similarities = [r["similarity"] for r in results]
        if similarities:
            print(f"\n     :")
            print(f"        : {np.mean(similarities):.3f}")
            print(f"        : {np.max(similarities):.3f}")
            print(f"        : {np.min(similarities):.3f}")
            print(
                f"     ({self.similarity_threshold})  : {len([s for s in similarities if s >= self.similarity_threshold])}"
            )
            print(
                f"     ({self.similarity_threshold})  : {len([s for s in similarities if s < self.similarity_threshold])}"
            )


def compute_open_set_metrics(self, results, base_root):
    """     ID   ReID           is_new_id / ODR / FMR / OSCR  

         
    -           ROI               N N            283GiB     OOM 
    -            mAP / AUROC / Rank-k / CMC / intra/inter    / TAR@FAR /     accuracy 
    -     ID    +   probe      pair                      
    """

    print("\n==========      ID   ReID      ==========")

    #
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
        print("          (<2)       ")
        return {}

    gt_ids_all = np.asarray(gt_ids_all)
    feat_mat_all = np.stack(feats_all).astype(np.float32)
    num_samples = int(feat_mat_all.shape[0])
    unique_ids = np.unique(gt_ids_all)

    print(f"      ReID       : {num_samples}    ID  : {len(unique_ids)}")

    #
    cluster_eval = compute_open_world_cluster_metrics(
        [{"true_id": t, "predicted_id": p} for t, p in zip(gt_ids_all.tolist(), pred_ids_all)]
    )
    print(
        "       : "
        f"purity={cluster_eval.get('cluster_purity', 0.0):.4f}, "
        f"assignment={cluster_eval.get('assignment_accuracy', 0.0):.4f}, "
        f"id_count_acc={cluster_eval.get('id_count_accuracy', 0.0):.4f}, "
        f"pred_ids={int(cluster_eval.get('predicted_id_count', 0))}, "
        f"true_ids={int(cluster_eval.get('true_id_count', 0))}"
    )

    #
    MAX_PER_ID_FOR_RETRIEVAL = 500   #
    MAX_PROBES_FOR_MAP = 5000        #
    SIM_BATCH_SIZE = 256             #
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
            f"         {num_samples}        ID    {MAX_PER_ID_FOR_RETRIEVAL}    "
            f"    /        : {N}"
        )

    #
    norms = np.linalg.norm(feat_mat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    feat_norm = feat_mat / norms  # [N, D]

    #
    max_rank = min(20, N)
    cmc_counts = np.zeros(max_rank, dtype=np.int64)
    num_valid_probes = 0
    ap_scores = []

    if N <= MAX_PROBES_FOR_MAP:
        probe_indices = np.arange(N, dtype=np.int64)
    else:
        probe_indices = rng.choice(N, size=MAX_PROBES_FOR_MAP, replace=False)
        print(
            f"mAP/CMC    {probe_indices.size}   probe         {N}  "
            "    O(N)      "
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

            #
            if np.unique(y_true).size >= 2:
                try:
                    ap = average_precision_score(y_true, y_score)
                    ap_scores.append(ap)
                except Exception:
                    pass

            #
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
    print(f"mAP (      all-vs-all ReID      ): {mAP:.4f}")

    rank_k = {}
    cmc_curve = None
    if num_valid_probes > 0:
        cmc_curve = (cmc_counts / num_valid_probes).astype(float)
        for k in [1, 5, 10]:
            if k <= max_rank:
                rank_k[f"rank-{k}"] = float(cmc_curve[k - 1])
        if rank_k:
            print("Rank-k       (     all-vs-all         ):")
            for k in sorted(rank_k.keys(), key=lambda x: int(x.split("-")[1])):
                print(f"  {k}: {rank_k[k]:.4f}")
    else:
        print("CMC / Rank-k:    probe     0    ")

    #
    max_pairs = 2_000_000
    total_pairs = N * (N - 1) // 2
    y_true_pairs = []
    y_score_pairs = []

    if total_pairs <= max_pairs:
        #
        for i in range(N):
            fi = feat_norm[i]
            for j in range(i + 1, N):
                same = int(gt_ids[i] == gt_ids[j])
                y_true_pairs.append(same)
                y_score_pairs.append(float(np.dot(fi, feat_norm[j])))
    else:
        #
        print(
            f"         {max_pairs}   (     {total_pairs:.3e})     "
            "       "
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
    print(f"AUROC ( ID/  ID     ): {auroc:.4f}")

    #
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
            print("intra-ID / inter-ID      :")
            print(
                f"   ID : N={same_scores.size}, "
                f"mean={same_scores.mean():.4f}, std={same_scores.std():.4f}"
            )
            print(
                f"   ID : N={diff_scores.size}, "
                f"mean={diff_scores.mean():.4f}, std={diff_scores.std():.4f}"
            )
            #
            print(f"  Sintra={same_scores.mean():.4f}  Sinter={diff_scores.mean():.4f}")

    #
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

            #
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

            #
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

            print("TAR@FAR      (   all-vs-all          ):")
            for k in sorted(tar_at_far.keys()):
                info = tar_at_far[k]
                print(
                    f"  TAR@FAR={k}: TAR={info['TAR']:.4f}, "
                    f"FAR={info['FAR']:.6f}, threshold={info['threshold']:.4f}"
                )
            best = verification["best_f1"]
            print(
                f"    F1 : F1={best['F1']:.4f}, "
                f"precision={best['precision']:.4f}, recall={best['recall']:.4f}, "
                f"accuracy={best['accuracy']:.4f}, threshold={best['threshold']:.4f}"
            )

    #
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
        #
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
        print(f"    ReID          : {out_path}")
    except Exception as e:
        print(f"           : {e}")

    return open_set_metrics


def main():
    """   """
    args, config = parse_option()

    #
    os.makedirs(args.output_root, exist_ok=True)

    #
    logger = create_logger(
        output_dir=args.output_root, dist_rank=0, name="panda_reid_inference"
    )

    logger.info("           ")
    logger.info(f"    : {args.cfg}")
    logger.info(f"    : {args.model_path}")
    if args.test_image_root:
        logger.info(f"    : {args.test_image_root}")
    if args.test_roi_root:
        logger.info(f"ROI  : {args.test_roi_root}")
    logger.info(f"    : {args.output_root}")
    if args.test_roiimg_root:
        logger.info(f"ROI    : {args.test_roiimg_root}")
    logger.info(f"     : {args.similarity_threshold}")

    #
    inference = PandaReIDInference(config=config, model_path=args.model_path, args=args)

    #
    results = inference.process_test_images(
        test_image_root=(args.test_roiimg_root or args.test_image_root),
        test_roi_root=(None if args.test_roiimg_root else args.test_roi_root),
        output_root=args.output_root,
        test_roiimg_root=args.test_roiimg_root,
    )

    logger.info("     ")


if __name__ == "__main__":
    main()
