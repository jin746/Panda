# --------------------------------------------------------
# 大熊猫个体识别数据集
# 支持YOLO格式ROI标签和目录结构解析
# --------------------------------------------------------

import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class PandaDataset(Dataset):
    """
    大熊猫个体识别数据集
    
    目录结构：
    - 图像路径：姓名_出生年_性别/拍摄年/图像.jpg
    - ROI路径：姓名_出生年_性别/拍摄年/图像.txt (YOLO格式)
    
    YOLO格式：[class_id, cx, cy, w, h] (归一化坐标)
    """
    
    def __init__(
        self,
        image_root: str,
        roi_root: str,
        img_size: int = 192,
        roi_expand_ratio: float = 0.1,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True,
        roi_format: str = 'mask',  # 'yolo', 'mask', 'auto' - 默认使用掩码格式
        mask_root: Optional[str] = None  # 掩码文件根目录
    ):
        """
        Args:
            image_root: 图像根目录路径
            roi_root: ROI标签根目录路径
            img_size: 输出图像尺寸
            roi_expand_ratio: ROI扩展比例
            transform: 图像变换
            is_train: 是否为训练模式
            roi_format: ROI格式 (默认: 'mask'分割掩码, 'yolo': YOLO格式, 'auto': 自动检测)
            mask_root: 掩码文件根目录（默认使用roi_root）

        注意:
        - 默认使用掩码格式进行处理
        - 掩码处理固定使用0像素填充方式，掩码外区域设为黑色
        """
        self.image_root = image_root
        self.roi_root = roi_root
        self.img_size = img_size
        self.roi_expand_ratio = roi_expand_ratio
        self.transform = transform
        self.is_train = is_train
        self.roi_format = roi_format
        self.mask_root = mask_root or roi_root  # 默认使用roi_root作为掩码根目录

        # 扫描数据并构建样本列表
        self.samples = self._scan_dataset()
        self.id_to_label = self._build_id_mapping()

        # 自动检测ROI格式
        if self.roi_format == 'auto':
            self.roi_format = self._detect_roi_format()
            print(f"自动检测到ROI格式: {self.roi_format}")

        print(f"Dataset loaded: {len(self.samples)} samples, {len(self.id_to_label)} IDs")
        print(f"ROI format: {self.roi_format}")
    
    def _scan_dataset(self) -> List[Dict]:
        """扫描数据集，构建样本列表"""
        samples = []
        
        # 遍历所有图像文件
        image_pattern = os.path.join(self.image_root, "*", "*", "*.jpg")
        image_files = glob.glob(image_pattern)
        
        for img_path in image_files:
            # 解析路径获取信息
            rel_path = os.path.relpath(img_path, self.image_root)
            path_parts = rel_path.split(os.sep)
            
            if len(path_parts) != 3:
                continue
                
            id_info, year, filename = path_parts
            
            # 解析ID信息：姓名_出生年_性别
            id_parts = id_info.split('_')
            if len(id_parts) != 3:
                continue
                
            name, birth_year, gender = id_parts
            
            # 构建对应的ROI文件路径
            roi_filename = os.path.splitext(filename)[0] + '.txt'
            roi_path = os.path.join(self.roi_root, id_info, year, roi_filename)
            
            # 检查ROI文件是否存在
            if not os.path.exists(roi_path):
                continue
                
            # 计算年龄
            try:
                age = int(year) - int(birth_year)
                if age < 0:
                    continue
            except ValueError:
                continue
            
            # 添加样本
            sample = {
                'image_path': img_path,
                'roi_path': roi_path,
                'id': name,
                'age': age,
                'gender': gender,
                'year': int(year),
                'birth_year': int(birth_year)
            }
            samples.append(sample)
        
        return samples
    
    def _build_id_mapping(self) -> Dict[str, int]:
        """构建ID到标签的映射"""
        unique_ids = list(set(sample['id'] for sample in self.samples))
        unique_ids.sort()  # 保证顺序一致性
        return {id_name: idx for idx, id_name in enumerate(unique_ids)}
    
    def _read_yolo_roi(self, roi_path: str) -> Optional[Tuple[float, float, float, float]]:
        """
        读取YOLO格式ROI标签
        
        Returns:
            (cx, cy, w, h) 归一化坐标，如果读取失败返回None
        """
        try:
            with open(roi_path, 'r') as f:
                line = f.readline().strip()
                if not line:
                    return None
                    
                parts = line.split()
                if len(parts) < 5:
                    return None
                    
                # YOLO格式：[class_id, cx, cy, w, h]
                _, cx, cy, w, h = map(float, parts[:5])
                return cx, cy, w, h
        except Exception as e:
            print(f"Failed to read ROI file {roi_path}: {e}")
            return None
    
    def _crop_roi_with_expansion(self, image: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
        """
        根据ROI裁剪图像并扩展边距
        
        Args:
            image: 输入图像 (H, W, C)
            roi: (cx, cy, w, h) 归一化坐标
            
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
        expand_w = w_px * self.roi_expand_ratio
        expand_h = h_px * self.roi_expand_ratio
        
        # 计算裁剪区域
        x1 = max(0, int(cx_px - (w_px + expand_w) / 2))
        y1 = max(0, int(cy_px - (h_px + expand_h) / 2))
        x2 = min(w, int(cx_px + (w_px + expand_w) / 2))
        y2 = min(h, int(cy_px + (h_px + expand_h) / 2))
        
        # 裁剪图像
        cropped = image[y1:y2, x1:x2]
        
        return cropped

    def _detect_roi_format(self) -> str:
        """
        自动检测ROI数据格式

        Returns:
            'yolo' 或 'mask'
        """
        if not self.samples:
            return 'yolo'  # 默认格式

        # 检查前几个样本的ROI文件
        for sample in self.samples[:min(10, len(self.samples))]:
            roi_path = sample['roi_path']

            # 检查是否存在对应的掩码文件
            base_name = Path(roi_path).stem
            mask_path = Path(self.mask_root) / 'masks' / f'{base_name}.npy'

            if mask_path.exists():
                return 'mask'

        return 'yolo'  # 默认使用YOLO格式

    def _read_sam_mask(self, image_path: str) -> Optional[np.ndarray]:
        """
        读取SAM生成的分割掩码（支持txt坐标文件和npy文件）

        Args:
            image_path: 图像路径

        Returns:
            掩码数组 (H, W)，如果读取失败返回None
        """
        try:
            base_name = Path(image_path).stem

            # 优先尝试读取txt坐标文件（YOLO+SAM流水线格式）
            coords_path = Path(self.mask_root) / f'{base_name}.txt'
            if coords_path.exists():
                return self._read_mask_from_coords(coords_path, image_path)

            # 备选：读取npy掩码文件
            mask_path = Path(self.mask_root) / 'masks' / f'{base_name}.npy'
            if mask_path.exists():
                mask = np.load(str(mask_path))
                # 确保掩码是二值的
                if mask.dtype != bool:
                    mask = mask > 0.5
                return mask

            return None

        except Exception as e:
            print(f"Failed to read mask file for {image_path}: {e}")
            return None

    def _read_sam_mask_from_roi_path(self, roi_path: str) -> Optional[np.ndarray]:
        """
        从ROI路径读取SAM生成的分割掩码（修复路径问题）

        Args:
            roi_path: ROI文件路径（与YOLO格式路径一致）

        Returns:
            掩码数组 (H, W)，如果读取失败返回None
        """
        try:
            # 直接使用roi_path作为掩码坐标文件路径
            coords_path = Path(roi_path)
            if coords_path.exists():
                # 需要图像路径来获取尺寸，从roi_path推断图像路径
                image_path = self._get_image_path_from_roi_path(roi_path)
                return self._read_mask_from_coords(coords_path, image_path)

            return None

        except Exception as e:
            print(f"Failed to read mask from roi path {roi_path}: {e}")
            return None

    def _get_image_path_from_roi_path(self, roi_path: str) -> str:
        """从ROI路径推断对应的图像路径"""
        try:
            # roi_path格式: roi_root/id_info/year/filename.txt
            # image_path格式: image_root/id_info/year/filename.jpg
            roi_path_obj = Path(roi_path)

            # 获取相对路径部分
            roi_root_obj = Path(self.roi_root)
            rel_path = roi_path_obj.relative_to(roi_root_obj)

            # 替换扩展名和根目录
            image_rel_path = rel_path.with_suffix('.jpg')
            image_path = Path(self.image_root) / image_rel_path

            return str(image_path)

        except Exception as e:
            print(f"Failed to infer image path from roi path {roi_path}: {e}")
            return ""

    def _read_mask_from_coords(self, coords_path: Path, image_path: str) -> Optional[np.ndarray]:
        """
        从坐标文件重建掩码（基于detect_and_crop.py的格式）

        Args:
            coords_path: 坐标文件路径
            image_path: 图像路径（用于获取图像尺寸）

        Returns:
            重建的掩码数组 (H, W)
        """
        try:
            # 读取原图像获取尺寸
            image = cv2.imread(image_path)
            if image is None:
                return None
            h, w = image.shape[:2]

            # 读取坐标文件
            with open(coords_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if len(lines) < 3:
                return None

            # 解析基本信息（第一行）
            info_parts = lines[0].strip().split(',')
            if len(info_parts) >= 5:
                # 使用文件中记录的图像尺寸
                file_w, file_h = int(info_parts[3]), int(info_parts[4])
                # 如果尺寸不匹配，使用实际图像尺寸
                if file_w != w or file_h != h:
                    print(f"Warning: Image size mismatch in {coords_path}")

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
                    x, y = map(float, lines[line_idx].strip().split(','))
                    contour_points.append([int(x), int(y)])
                    line_idx += 1

                if contour_points:
                    # 填充轮廓
                    contour = np.array(contour_points, dtype=np.int32)
                    cv2.fillPoly(mask, [contour], 255)

            return mask.astype(bool)

        except Exception as e:
            print(f"Failed to read mask from coords {coords_path}: {e}")
            return None

    def _get_mask_bbox(self, mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        从掩码计算边界框

        Args:
            mask: 二值掩码 (H, W)

        Returns:
            (cx, cy, w, h) 归一化坐标，如果掩码为空返回None
        """
        try:
            # 找到掩码中的非零像素
            rows, cols = np.where(mask)

            if len(rows) == 0:
                return None

            # 计算边界框
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
            print(f"Failed to compute bbox from mask: {e}")
            return None

    def _apply_mask_to_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        将掩码应用到图像上

        Args:
            image: 输入图像 (H, W, C)
            mask: 二值掩码 (H, W)

        Returns:
            掩码处理后的图像
        """
        try:
            # 确保掩码和图像尺寸匹配
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8),
                                (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)

            # 应用掩码：保留掩码区域，其他区域置为黑色
            masked_image = image.copy()
            masked_image[~mask] = 0

            return masked_image

        except Exception as e:
            print(f"Failed to apply mask to image: {e}")
            return image  # 返回原图像作为fallback

    def _process_mask_region_to_rectangle(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        处理掩码区域，输出矩形图像（关键方法）

        步骤：
        1. 计算掩码的边界框
        2. 扩展边界框（10%边距）
        3. 裁剪图像到边界框区域
        4. 在裁剪区域内应用掩码（掩码外区域用0填充）

        Args:
            image: 输入图像 (H, W, C)
            mask: 二值掩码 (H, W)

        Returns:
            处理后的矩形图像，掩码外区域为0像素
        """
        try:
            # 确保掩码和图像尺寸匹配
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8),
                                (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)

            # 1. 计算掩码的边界框
            rows, cols = np.where(mask)
            if len(rows) == 0:
                # 掩码为空，返回默认大小的黑色图像
                return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()

            # 2. 扩展边界框（10%边距）
            h, w = image.shape[:2]
            bbox_w = x2 - x1 + 1
            bbox_h = y2 - y1 + 1

            expand_w = int(bbox_w * self.roi_expand_ratio)
            expand_h = int(bbox_h * self.roi_expand_ratio)

            # 计算扩展后的边界框
            x1_exp = max(0, x1 - expand_w // 2)
            y1_exp = max(0, y1 - expand_h // 2)
            x2_exp = min(w, x2 + expand_w // 2)
            y2_exp = min(h, y2 + expand_h // 2)

            # 3. 裁剪图像到边界框区域
            cropped_image = image[y1_exp:y2_exp, x1_exp:x2_exp].copy()
            cropped_mask = mask[y1_exp:y2_exp, x1_exp:x2_exp]

            # 4. 应用0像素填充（最终方案）
            # 掩码外的区域设置为0（黑色），确保模型专注学习目标特征
            cropped_image[~cropped_mask] = 0

            return cropped_image

        except Exception as e:
            print(f"Failed to process mask region: {e}")
            # 返回默认大小的黑色图像作为fallback
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取数据样本
        
        Returns:
            包含图像、标签和元信息的字典
        """
        sample = self.samples[idx]
        
        # 读取图像
        try:
            image = cv2.imread(sample['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Failed to read image {sample['image_path']}: {e}")
            # 返回黑色图像作为fallback
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # 根据ROI格式处理图像
        if self.roi_format == 'mask':
            # 使用分割掩码处理 - 正确处理掩码区域为矩形
            mask = self._read_sam_mask_from_roi_path(sample['roi_path'])
            if mask is not None:
                # 关键：处理掩码区域，确保输出矩形图像（0像素填充）
                image = self._process_mask_region_to_rectangle(image, mask)
        else:
            # 使用YOLO格式ROI
            roi = self._read_yolo_roi(sample['roi_path'])
            if roi is not None:
                # 根据ROI裁剪图像
                image = self._crop_roi_with_expansion(image, roi)
        
        # 确保图像不为空
        if image.size == 0:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # 转换为PIL图像
        image = Image.fromarray(image)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        else:
            # 默认变换：resize + 归一化
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
        
        # 构建返回数据
        gender_raw = str(sample.get('gender', '')).strip()
        gender_raw_upper = gender_raw.upper()
        if gender_raw_upper in ['M', '雄', 'MALE', '男']:
            gender_label = 1
        elif gender_raw_upper in ['F', '雌', 'FEMALE', '女']:
            gender_label = 0
        else:
            # 未知时按雌性处理（保持与旧版本兼容，避免训练时报 label 越界）
            gender_label = 0
        data = {
            'image': image,
            'id_label': self.id_to_label[sample['id']],
            'id_name': sample['id'],
            'age': sample['age'],
            'gender': gender_label,  # Male=1, Female=0
            'year': sample['year'],
            'image_path': sample['image_path']
        }
        
        return data
    
    def get_num_classes(self) -> int:
        """获取ID类别数量"""
        return len(self.id_to_label)
    
    def get_id_names(self) -> List[str]:
        """获取所有ID名称列表"""
        return list(self.id_to_label.keys())


def build_panda_transform(is_train: bool, img_size: int = 192, config=None) -> transforms.Compose:
    """
    构建大熊猫数据集的图像变换
    
    Args:
        is_train: 是否为训练模式
        img_size: 图像尺寸
        config: 可选，yacs配置；若提供则使用 config.AUG 中的增强参数（AUTO_AUGMENT/REPROB 等）
        
    Returns:
        图像变换组合
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        # 训练时的数据增强（支持从 config.AUG 读取关键增强项）
        aug = getattr(config, "AUG", None) if config is not None else None

        color_jitter = 0.2
        auto_augment = None
        re_prob = 0.0
        re_mode = "pixel"
        re_count = 1
        rrc_enable = False
        rrc_scale_min = 0.6
        gray_prob = 0.0
        blur_prob = 0.0

        if aug is not None:
            try:
                color_jitter = float(getattr(aug, "COLOR_JITTER", color_jitter) or 0.0)
            except Exception:
                pass
            try:
                aa = str(getattr(aug, "AUTO_AUGMENT", "") or "").strip()
                if aa and aa.lower() not in {"none", "false", "0", "null", "~"}:
                    auto_augment = aa
            except Exception:
                auto_augment = None
            try:
                re_prob = float(getattr(aug, "REPROB", re_prob) or 0.0)
                re_mode = str(getattr(aug, "REMODE", re_mode) or re_mode)
                re_count = int(getattr(aug, "RECOUNT", re_count) or re_count)
            except Exception:
                pass
            try:
                rrc_enable = bool(int(getattr(aug, "RRC_ENABLE", int(rrc_enable))))
            except Exception:
                rrc_enable = bool(getattr(aug, "RRC_ENABLE", rrc_enable))
            try:
                rrc_scale_min = float(getattr(aug, "RRC_SCALE_MIN", rrc_scale_min) or rrc_scale_min)
            except Exception:
                pass
            try:
                gray_prob = float(getattr(aug, "GRAY_PROB", gray_prob) or 0.0)
            except Exception:
                pass
            try:
                blur_prob = float(getattr(aug, "BLUR_PROB", blur_prob) or 0.0)
            except Exception:
                pass

        rrc_scale_min = max(0.2, min(1.0, float(rrc_scale_min)))
        gray_prob = max(0.0, min(1.0, float(gray_prob)))
        blur_prob = max(0.0, min(1.0, float(blur_prob)))

        t = []
        if rrc_enable:
            t.append(
                transforms.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(rrc_scale_min, 1.0),
                    ratio=(0.75, 1.3333333333),
                )
            )
        else:
            t.append(transforms.Resize((img_size, img_size)))
        t.append(transforms.RandomHorizontalFlip(p=0.5))

        # AUTO_AUGMENT：优先使用 timm 的 auto_augment_transform（支持 rand-m* 语法）
        if auto_augment is not None:
            try:
                from timm.data.auto_augment import auto_augment_transform

                aa_params = {
                    "translate_const": int(img_size * 0.45),
                    "img_mean": tuple([min(255, max(0, int(m * 255))) for m in mean]),
                }
                t.append(auto_augment_transform(auto_augment, aa_params))
            except Exception:
                # timm 不可用则静默回退为基础增强
                auto_augment = None

        # 基础增强：当未启用 AUTO_AUGMENT 时才使用（避免叠加过强）
        if auto_augment is None:
            if color_jitter and color_jitter > 0:
                t.append(
                    transforms.ColorJitter(
                        brightness=color_jitter,
                        contrast=color_jitter,
                        saturation=color_jitter,
                        hue=min(0.1, color_jitter / 2),
                    )
                )
            t.append(transforms.RandomRotation(degrees=10))
        if gray_prob > 0:
            t.append(transforms.RandomGrayscale(p=gray_prob))
        if blur_prob > 0:
            t.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=blur_prob))

        t.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # Random Erasing：优先使用 timm 版本（支持 mode/count），回退 torchvision.RandomErasing
        if re_prob and re_prob > 0:
            try:
                from timm.data.random_erasing import RandomErasing

                t.append(
                    RandomErasing(
                        probability=re_prob,
                        mode=re_mode,
                        max_count=re_count,
                        device="cpu",
                    )
                )
            except Exception:
                t.append(transforms.RandomErasing(p=re_prob, value=0))

        transform = transforms.Compose(t)
    else:
        # 验证/测试时的变换
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    return transform


if __name__ == "__main__":
    # 测试数据集
    image_root = r"H:\trainDataset\panda\LLava\id\all\42IDtrain"
    roi_root = r"H:\trainDataset\panda\LLava\id\all\42IDYolo\roi"
    
    transform = build_panda_transform(is_train=True)
    dataset = PandaDataset(image_root, roi_root, transform=transform)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of IDs: {dataset.get_num_classes()}")
    print(f"ID list: {dataset.get_id_names()}")
    
    # 测试加载一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape: {sample['image'].shape}")
        print(f"ID label: {sample['id_label']}, ID name: {sample['id_name']}")
        print(f"Age: {sample['age']}, Gender: {sample['gender']}")
