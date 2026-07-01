#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
         ReID    
      ID    
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

#
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from panda_reid_core.config import get_config
from panda_reid_core.models.panda_reid_model import build_panda_reid_model
from panda_reid_core.models.prototype_reid_network import PrototypeReIDNetwork
from panda_reid_core.data.panda_dataset import build_panda_transform


def find_best_prototype_model():
    """        """
    possible_paths = [
        "output_prototype/prototype_best_model.pth",
        "prototype_best_model.pth",
        "output_deep_fix/swinv2_large_patch4_window12_192_panda_reid_enhanced/deep_fix_training/enhanced_best_model.pth"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"        : {path}")
            return path
    
    print("           ")
    return None


def load_prototype_models(config, model_path, device):
    """      """
    print(f"        : {model_path}")
    
    #
    model = build_panda_reid_model(config, 42)
    model.to(device)
    
    #
    prototype_net = PrototypeReIDNetwork(
        feature_dim=model.feature_dim,
        temperature=0.1,
        momentum=0.9
    )
    prototype_net.to(device)
    
    #
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    if 'prototype_net' in checkpoint:
        prototype_net.load_state_dict(checkpoint['prototype_net'])
    
    model.eval()
    prototype_net.eval()
    
    print(f"               : {model.feature_dim}")
    return model, prototype_net


def read_yolo_roi(roi_path):
    """  YOLO  ROI  """
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
    """  ROI         """
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
    """        """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)


def find_roi_file(image_path, roi_root):
    """     ROI  """
    if not roi_root:
        return None
    
    #
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    #
    possible_roi_paths = [
        os.path.join(roi_root, f"{image_name}.txt"),
        os.path.join(roi_root, os.path.dirname(os.path.relpath(image_path, roi_root)), f"{image_name}.txt")
    ]
    
    for roi_path in possible_roi_paths:
        if os.path.exists(roi_path):
            return roi_path
    
    return None


def extract_image_feature(model, image_path, roi_path, transform, device):
    """         """
    try:
        #
        image = cv2.imread(image_path)
        if image is None:
            print(f"         : {image_path}")
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #
        if roi_path:
            roi = read_yolo_roi(roi_path)
            if roi is not None:
                image = crop_roi_with_expansion(image, roi)
        
        #
        image_pil = Image.fromarray(image)
        tensor = transform(image_pil).unsqueeze(0).to(device)
        
        #
        with torch.no_grad():
            features, _ = model(tensor)
            #
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features[0]  #
    
    except Exception as e:
        print(f"          {image_path}: {e}")
        return None


class PrototypeIDManager:
    """       ID   """
    
    def __init__(self, prototype_net, confidence_threshold=0.6, quality_threshold=0.3):
        self.prototype_net = prototype_net
        self.confidence_threshold = confidence_threshold
        self.quality_threshold = quality_threshold
        self.id_images = defaultdict(list)  #
        self.processing_stats = {
            'total_processed': 0,
            'new_ids_created': 0,
            'existing_matches': 0,
            'low_quality_skipped': 0
        }
    
    def process_image(self, image_path, feature):
        """      """
        self.processing_stats['total_processed'] += 1
        
        #
        result = self.prototype_net(feature)
        
        predicted_id = result['predicted_id']
        similarity = result['similarity']
        confidence = result['confidence']
        is_new_id = result['is_new_id']
        quality_score = result.get('quality_score', 0.8)
        adaptive_threshold = result['adaptive_threshold']
        
        #
        if quality_score < self.quality_threshold:
            self.processing_stats['low_quality_skipped'] += 1
            print(f"            : {os.path.basename(image_path)} (  : {quality_score:.3f})")
            return None
        
        #
        if is_new_id:
            self.processing_stats['new_ids_created'] += 1
            #
            self.prototype_net.update_prototype(predicted_id, feature, quality_score)
            status_msg = f" ID (  : {quality_score:.3f})"
        else:
            self.processing_stats['existing_matches'] += 1
            #
            self.prototype_net.update_prototype(predicted_id, feature, quality_score)
            status_msg = f"   (   : {similarity:.3f},    : {confidence:.3f})"
        
        #
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
        """        """
        stats = self.processing_stats.copy()
        stats['unique_ids'] = len(self.id_images)
        stats['avg_images_per_id'] = (stats['total_processed'] - stats['low_quality_skipped']) / max(1, stats['unique_ids'])
        return stats


def save_classified_results(id_manager, output_dir):
    """      """
    print(f"\n         : {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    copy_count = 0
    
    for id_name, image_paths in id_manager.id_images.items():
        if not image_paths:
            continue
        
        #
        id_dir = os.path.join(output_dir, id_name)
        os.makedirs(id_dir, exist_ok=True)
        
        print(f"  {id_name}: {len(image_paths)}    ")
        
        #
        for i, image_path in enumerate(image_paths):
            try:
                ext = os.path.splitext(image_path)[1]
                new_filename = f"{id_name}_{i+1:03d}{ext}"
                new_path = os.path.join(id_dir, new_filename)
                
                shutil.copy2(image_path, new_path)
                copy_count += 1
                
                if i < 3:  #
                    file_size = os.path.getsize(new_path)
                    print(f"     {new_filename} ({file_size} bytes)")
                    
            except Exception as e:
                print(f"          {image_path}: {e}")
    
    print(f"       {copy_count}    ")
    return copy_count


def save_detection_report(id_manager, output_dir, processing_time):
    """      """
    stats = id_manager.get_statistics()
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'processing_time_seconds': processing_time,
        'statistics': stats,
        'id_details': {}
    }
    
    #
    for id_name, image_paths in id_manager.id_images.items():
        report['id_details'][id_name] = {
            'image_count': len(image_paths),
            'sample_images': [os.path.basename(p) for p in image_paths[:5]]  #
        }
    
    #
    report_path = os.path.join(output_dir, 'prototype_detection_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"           : {report_path}")
    return report_path


def main():
    """   """
    parser = argparse.ArgumentParser(description='         ReID    ')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='         ')
    parser.add_argument('--roi-root', type=str,
                       help='ROI         ')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='       ')
    parser.add_argument('--model-path', type=str,
                       help='                 ')
    parser.add_argument('--config', type=str, default='configs/panda_reid_arcface_triplet.yaml',
                       help='      ')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                       help='        0.3     ')
    parser.add_argument('--quality-threshold', type=float, default=0.1,
                       help='       0.1     ')

    args = parser.parse_args()

    print("="*80)
    print("           ReID  ")
    print("="*80)
    print(f"     : {args.input_dir}")
    print(f"ROI   : {args.roi_root if args.roi_root else ' '}")
    print(f"     : {args.output_dir}")
    print(f"     : {args.confidence_threshold}")
    print(f"    : {args.quality_threshold}")

    #
    if not os.path.exists(args.input_dir):
        print(f"          : {args.input_dir}")
        return

    #
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = find_best_prototype_model()
        if not model_path:
            return

    #
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
    print(f"    : {device}")

    #
    model, prototype_net = load_prototype_models(config, model_path, device)

    #
    transform = build_panda_transform(is_train=False, img_size=config.DATA.IMG_SIZE)

    #
    print(f"\n         ...")
    image_files = find_image_files(args.input_dir)
    print(f"   {len(image_files)}    ")

    if len(image_files) == 0:
        print("           ")
        return

    #
    os.makedirs(args.output_dir, exist_ok=True)

    #
    id_manager = PrototypeIDManager(
        prototype_net=prototype_net,
        confidence_threshold=args.confidence_threshold,
        quality_threshold=args.quality_threshold
    )

    #
    print(f"\n          ...")
    start_time = time.time()

    for i, image_path in enumerate(image_files):
        #
        if i % 50 == 0:
            print(f"    : {i+1}/{len(image_files)} ({(i+1)/len(image_files)*100:.1f}%)")

        #
        roi_path = find_roi_file(image_path, args.roi_root) if args.roi_root else None

        #
        feature = extract_image_feature(model, image_path, roi_path, transform, device)

        if feature is not None:
            #
            result = id_manager.process_image(image_path, feature)
            
            if result and (i < 20 or result['is_new_id']):
                print(f"    {os.path.basename(image_path)} -> {result['id']} ({result['status_msg']})")

    processing_time = time.time() - start_time

    #
    print(f"\n        ...")
    copy_count = save_classified_results(id_manager, args.output_dir)

    #
    report_path = save_detection_report(id_manager, args.output_dir, processing_time)

    #
    stats = id_manager.get_statistics()

    #
    print(f"\n           ")
    print("="*80)
    print(f"       : {args.output_dir}")
    print(f"      {stats['unique_ids']}         ")
    print(f"       {copy_count}    ")
    print(f"     ID: {stats['new_ids_created']}  ")
    print(f"      ID: {stats['existing_matches']}  ")
    print(f"        : {stats['low_quality_skipped']}  ")
    print(f"     ID   : {stats['avg_images_per_id']:.1f}")
    print(f"      : {report_path}")
    print(f"       : {processing_time:.2f}  ")

    #
    print(f"\n      :")
    if stats['avg_images_per_id'] < 2:
        print("  -    ID               ")
    if stats['unique_ids'] > len(image_files) * 0.8:
        print("  - ID                ")
    if stats['low_quality_skipped'] > stats['total_processed'] * 0.2:
        print("  -                  ROI  ")
    
    print(f"                          ")


if __name__ == '__main__':
    import time
    main()
