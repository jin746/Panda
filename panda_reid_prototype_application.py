#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于原型网络的熊猫ReID应用系统
革命性的动态ID识别方案
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
import shutil
from PIL import Image
import json
from datetime import datetime
from collections import defaultdict, Counter
import glob
from pathlib import Path
import torchvision.transforms as transforms

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from models.panda_reid_model import build_panda_reid_model
from models.prototype_reid_network import PrototypeReIDNetwork
from data.panda_dataset import build_panda_transform


def find_best_prototype_model():
    """查找最佳原型模型"""
    possible_paths = [
        "output_prototype/prototype_best_model.pth",
        "prototype_best_model.pth",
        "output_deep_fix/swinv2_large_patch4_window12_192_panda_reid_enhanced/deep_fix_training/enhanced_best_model.pth"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ 找到原型模型: {path}")
            return path
    
    print("❌ 未找到原型模型文件")
    return None


def load_prototype_models(config, model_path, device):
    """加载原型模型"""
    print(f"🔄 加载原型模型: {model_path}")
    
    # 构建主模型
    model = build_panda_reid_model(config, 42)
    model.to(device)
    
    # 构建原型网络
    prototype_net = PrototypeReIDNetwork(
        feature_dim=model.feature_dim,
        temperature=0.1,
        momentum=0.9
    )
    prototype_net.to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    if 'prototype_net' in checkpoint:
        prototype_net.load_state_dict(checkpoint['prototype_net'])
    
    model.eval()
    prototype_net.eval()
    
    print(f"✅ 原型模型加载成功，特征维度: {model.feature_dim}")
    return model, prototype_net


def read_yolo_roi(roi_path):
    """读取YOLO格式ROI标签"""
    try:
        with open(roi_path, 'r') as f:
            line = f.readline().strip()
            if not line:
                return None
            parts = line.split()
            if len(parts) < 5:
                return None
            _, cx, cy, w, h = map(float, parts[:5])
            return (cx, cy, w, h)
    except:
        return None


def crop_roi_with_expansion(image, roi, expand_ratio=0.1):
    """根据ROI裁剪图像并扩展边距"""
    h, w = image.shape[:2]
    cx, cy, roi_w, roi_h = roi
    
    cx_px = cx * w
    cy_px = cy * h
    w_px = roi_w * w
    h_px = roi_h * h
    
    expand_w = w_px * expand_ratio
    expand_h = h_px * expand_ratio
    
    x1 = max(0, int(cx_px - (w_px + expand_w) / 2))
    y1 = max(0, int(cy_px - (h_px + expand_h) / 2))
    x2 = min(w, int(cx_px + (w_px + expand_w) / 2))
    y2 = min(h, int(cy_px + (h_px + expand_h) / 2))
    
    return image[y1:y2, x1:x2]


def find_image_files(input_dir):
    """查找所有图像文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)


def find_roi_file(image_path, roi_root):
    """查找对应的ROI文件"""
    if not roi_root:
        return None
    
    # 获取相对路径
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 可能的ROI文件路径
    possible_roi_paths = [
        os.path.join(roi_root, f"{image_name}.txt"),
        os.path.join(roi_root, os.path.dirname(os.path.relpath(image_path, roi_root)), f"{image_name}.txt")
    ]
    
    for roi_path in possible_roi_paths:
        if os.path.exists(roi_path):
            return roi_path
    
    return None


def extract_image_feature(model, image_path, roi_path, transform, device):
    """提取单张图像的特征"""
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ 无法读取图像: {image_path}")
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 如果有ROI文件，进行裁剪
        if roi_path:
            roi = read_yolo_roi(roi_path)
            if roi is not None:
                image = crop_roi_with_expansion(image, roi)
        
        # 转换为PIL图像并应用变换
        image_pil = Image.fromarray(image)
        tensor = transform(image_pil).unsqueeze(0).to(device)
        
        # 提取特征
        with torch.no_grad():
            features, _ = model(tensor)
            # L2归一化
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features[0]  # 返回tensor而不是numpy
    
    except Exception as e:
        print(f"⚠️ 提取特征失败 {image_path}: {e}")
        return None


class PrototypeIDManager:
    """基于原型网络的ID管理器"""
    
    def __init__(self, prototype_net, confidence_threshold=0.6, quality_threshold=0.3):
        self.prototype_net = prototype_net
        self.confidence_threshold = confidence_threshold
        self.quality_threshold = quality_threshold
        self.id_images = defaultdict(list)  # 存储每个ID的图像路径
        self.processing_stats = {
            'total_processed': 0,
            'new_ids_created': 0,
            'existing_matches': 0,
            'low_quality_skipped': 0
        }
    
    def process_image(self, image_path, feature):
        """处理单张图像"""
        self.processing_stats['total_processed'] += 1
        
        # 使用原型网络进行预测
        result = self.prototype_net(feature)
        
        predicted_id = result['predicted_id']
        similarity = result['similarity']
        confidence = result['confidence']
        is_new_id = result['is_new_id']
        quality_score = result.get('quality_score', 0.8)
        adaptive_threshold = result['adaptive_threshold']
        
        # 质量控制
        if quality_score < self.quality_threshold:
            self.processing_stats['low_quality_skipped'] += 1
            print(f"  ⚠️ 跳过低质量图像: {os.path.basename(image_path)} (质量: {quality_score:.3f})")
            return None
        
        # 更新统计
        if is_new_id:
            self.processing_stats['new_ids_created'] += 1
            # 创建新原型
            self.prototype_net.update_prototype(predicted_id, feature, quality_score)
            status_msg = f"新ID (质量: {quality_score:.3f})"
        else:
            self.processing_stats['existing_matches'] += 1
            # 更新现有原型
            self.prototype_net.update_prototype(predicted_id, feature, quality_score)
            status_msg = f"匹配 (相似度: {similarity:.3f}, 置信度: {confidence:.3f})"
        
        # 记录图像路径
        self.id_images[predicted_id].append(image_path)
        
        return {
            'id': predicted_id,
            'similarity': similarity,
            'confidence': confidence,
            'is_new_id': is_new_id,
            'quality_score': quality_score,
            'adaptive_threshold': adaptive_threshold,
            'status_msg': status_msg
        }
    
    def get_statistics(self):
        """获取处理统计信息"""
        stats = self.processing_stats.copy()
        stats['unique_ids'] = len(self.id_images)
        stats['avg_images_per_id'] = (stats['total_processed'] - stats['low_quality_skipped']) / max(1, stats['unique_ids'])
        return stats


def save_classified_results(id_manager, output_dir):
    """保存分类结果"""
    print(f"\n💾 保存分类结果到: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    copy_count = 0
    
    for id_name, image_paths in id_manager.id_images.items():
        if not image_paths:
            continue
        
        # 创建ID文件夹
        id_dir = os.path.join(output_dir, id_name)
        os.makedirs(id_dir, exist_ok=True)
        
        print(f"📁 {id_name}: {len(image_paths)} 张图像")
        
        # 复制图像文件
        for i, image_path in enumerate(image_paths):
            try:
                ext = os.path.splitext(image_path)[1]
                new_filename = f"{id_name}_{i+1:03d}{ext}"
                new_path = os.path.join(id_dir, new_filename)
                
                shutil.copy2(image_path, new_path)
                copy_count += 1
                
                if i < 3:  # 显示前3个文件的详细信息
                    file_size = os.path.getsize(new_path)
                    print(f"   ✅ {new_filename} ({file_size} bytes)")
                    
            except Exception as e:
                print(f"⚠️ 复制文件失败 {image_path}: {e}")
    
    print(f"✅ 总计保存 {copy_count} 张图像")
    return copy_count


def save_detection_report(id_manager, output_dir, processing_time):
    """保存检测报告"""
    stats = id_manager.get_statistics()
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'processing_time_seconds': processing_time,
        'statistics': stats,
        'id_details': {}
    }
    
    # 添加每个ID的详细信息
    for id_name, image_paths in id_manager.id_images.items():
        report['id_details'][id_name] = {
            'image_count': len(image_paths),
            'sample_images': [os.path.basename(p) for p in image_paths[:5]]  # 前5个样本
        }
    
    # 保存报告
    report_path = os.path.join(output_dir, 'prototype_detection_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 原型检测报告已保存: {report_path}")
    return report_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于原型网络的熊猫ReID应用系统')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='输入图像文件夹路径')
    parser.add_argument('--roi-root', type=str,
                       help='ROI文件夹路径（可选）')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='输出文件夹路径')
    parser.add_argument('--model-path', type=str,
                       help='模型路径（可选，自动查找最佳模型）')
    parser.add_argument('--config', type=str, default='configs/panda_reid_balanced.yaml',
                       help='配置文件路径')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                       help='置信度阈值（默认0.3，更宽松）')
    parser.add_argument('--quality-threshold', type=float, default=0.1,
                       help='质量阈值（默认0.1，更宽松）')

    args = parser.parse_args()

    print("="*80)
    print("🐼 基于原型网络的熊猫ReID系统")
    print("="*80)
    print(f"输入文件夹: {args.input_dir}")
    print(f"ROI文件夹: {args.roi_root if args.roi_root else '无'}")
    print(f"输出文件夹: {args.output_dir}")
    print(f"置信度阈值: {args.confidence_threshold}")
    print(f"质量阈值: {args.quality_threshold}")

    # 检查输入文件夹
    if not os.path.exists(args.input_dir):
        print(f"❌ 输入文件夹不存在: {args.input_dir}")
        return

    # 查找模型
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = find_best_prototype_model()
        if not model_path:
            return

    # 加载配置
    class ConfigArgs:
        def __init__(self):
            self.cfg = args.config
            self.opts = []
            self.zip = False
            self.cache_mode = 'part'
            self.resume = None
            self.accumulation_steps = None
            self.use_checkpoint = False
            self.disable_amp = False
            self.amp_opt_level = None
            self.output = None
            self.tag = None
            self.eval = False
            self.throughput = False

    config_args = ConfigArgs()
    config = get_config(config_args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    model, prototype_net = load_prototype_models(config, model_path, device)

    # 创建图像变换
    transform = build_panda_transform(is_train=False, img_size=config.DATA.IMG_SIZE)

    # 查找所有图像文件
    print(f"\n🔍 扫描输入文件夹...")
    image_files = find_image_files(args.input_dir)
    print(f"找到 {len(image_files)} 张图像")

    if len(image_files) == 0:
        print("❌ 未找到任何图像文件")
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建原型ID管理器
    id_manager = PrototypeIDManager(
        prototype_net=prototype_net,
        confidence_threshold=args.confidence_threshold,
        quality_threshold=args.quality_threshold
    )

    # 开始处理
    print(f"\n🚀 开始原型网络处理...")
    start_time = time.time()

    for i, image_path in enumerate(image_files):
        # 显示进度
        if i % 50 == 0:
            print(f"处理进度: {i+1}/{len(image_files)} ({(i+1)/len(image_files)*100:.1f}%)")

        # 查找ROI文件
        roi_path = find_roi_file(image_path, args.roi_root) if args.roi_root else None

        # 提取特征
        feature = extract_image_feature(model, image_path, roi_path, transform, device)

        if feature is not None:
            # 使用原型网络处理
            result = id_manager.process_image(image_path, feature)
            
            if result and (i < 20 or result['is_new_id']):
                print(f"  📸 {os.path.basename(image_path)} -> {result['id']} ({result['status_msg']})")

    processing_time = time.time() - start_time

    # 保存结果
    print(f"\n💾 保存分类结果...")
    copy_count = save_classified_results(id_manager, args.output_dir)

    # 保存报告
    report_path = save_detection_report(id_manager, args.output_dir, processing_time)

    # 获取最终统计
    stats = id_manager.get_statistics()

    # 最终总结
    print(f"\n🎉 原型网络处理完成！")
    print("="*80)
    print(f"📁 结果保存在: {args.output_dir}")
    print(f"📊 识别出 {stats['unique_ids']} 个不同的熊猫个体")
    print(f"📸 成功分类 {copy_count} 张图像")
    print(f"🆕 创建新ID: {stats['new_ids_created']} 个")
    print(f"🔄 匹配现有ID: {stats['existing_matches']} 次")
    print(f"⚠️ 跳过低质量: {stats['low_quality_skipped']} 张")
    print(f"📊 平均每ID图像数: {stats['avg_images_per_id']:.1f}")
    print(f"📄 详细报告: {report_path}")
    print(f"⏱️  总耗时: {processing_time:.2f} 秒")

    # 智能建议
    print(f"\n💡 系统建议:")
    if stats['avg_images_per_id'] < 2:
        print("  - 平均每ID图像较少，可能需要降低质量阈值")
    if stats['unique_ids'] > len(image_files) * 0.8:
        print("  - ID数量较多，可能需要提高置信度阈值")
    if stats['low_quality_skipped'] > stats['total_processed'] * 0.2:
        print("  - 低质量图像较多，建议检查图像质量和ROI标注")
    
    print(f"✨ 原型网络系统提供了更智能的动态阈值和置信度评估！")


if __name__ == '__main__':
    import time
    main()
