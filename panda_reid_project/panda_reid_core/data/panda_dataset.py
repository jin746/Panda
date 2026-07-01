# --------------------------------------------------------
#
#
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
              
    
         
    -        _   _  /   /  .jpg
    - ROI     _   _  /   /  .txt (YOLO  )
    
    YOLO   [class_id, cx, cy, w, h] (     )
    """
    
    def __init__(
        self,
        image_root: str,
        roi_root: str,
        img_size: int = 192,
        roi_expand_ratio: float = 0.1,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True,
        roi_format: str = 'mask',  #
        mask_root: Optional[str] = None  #
    ):
        """
        Args:
            image_root:        
            roi_root: ROI       
            img_size:       
            roi_expand_ratio: ROI    
            transform:     
            is_train:        
            roi_format: ROI   (  : 'mask'    , 'yolo': YOLO  , 'auto':     )
            mask_root:             roi_root 

          :
        -             
        -         0                
        """
        self.image_root = image_root
        self.roi_root = roi_root
        self.img_size = img_size
        self.roi_expand_ratio = roi_expand_ratio
        self.transform = transform
        self.is_train = is_train
        self.roi_format = roi_format
        self.mask_root = mask_root or roi_root  #

        #
        self.samples = self._scan_dataset()
        self.id_to_label = self._build_id_mapping()

        #
        if self.roi_format == 'auto':
            self.roi_format = self._detect_roi_format()
            print(f"     ROI  : {self.roi_format}")

        print(f"Dataset loaded: {len(self.samples)} samples, {len(self.id_to_label)} IDs")
        print(f"ROI format: {self.roi_format}")
    
    def _scan_dataset(self) -> List[Dict]:
        """            """
        samples = []
        
        #
        image_pattern = os.path.join(self.image_root, "*", "*", "*.jpg")
        image_files = glob.glob(image_pattern)
        
        for img_path in image_files:
            #
            rel_path = os.path.relpath(img_path, self.image_root)
            path_parts = rel_path.split(os.sep)
            
            if len(path_parts) != 3:
                continue
                
            id_info, year, filename = path_parts
            
            #
            id_parts = id_info.split('_')
            if len(id_parts) != 3:
                continue
                
            name, birth_year, gender = id_parts
            
            #
            roi_filename = os.path.splitext(filename)[0] + '.txt'
            roi_path = os.path.join(self.roi_root, id_info, year, roi_filename)
            
            #
            if not os.path.exists(roi_path):
                continue
                
            #
            try:
                age = int(year) - int(birth_year)
                if age < 0:
                    continue
            except ValueError:
                continue
            
            #
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
        """  ID      """
        unique_ids = list(set(sample['id'] for sample in self.samples))
        unique_ids.sort()  #
        return {id_name: idx for idx, id_name in enumerate(unique_ids)}
    
    def _read_yolo_roi(self, roi_path: str) -> Optional[Tuple[float, float, float, float]]:
        """
          YOLO  ROI  
        
        Returns:
            (cx, cy, w, h)               None
        """
        try:
            with open(roi_path, 'r') as f:
                line = f.readline().strip()
                if not line:
                    return None
                    
                parts = line.split()
                if len(parts) < 5:
                    return None
                    
                #
                _, cx, cy, w, h = map(float, parts[:5])
                return cx, cy, w, h
        except Exception as e:
            print(f"Failed to read ROI file {roi_path}: {e}")
            return None
    
    def _crop_roi_with_expansion(self, image: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
        """
          ROI         
        
        Args:
            image:      (H, W, C)
            roi: (cx, cy, w, h)      
            
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
        expand_w = w_px * self.roi_expand_ratio
        expand_h = h_px * self.roi_expand_ratio
        
        #
        x1 = max(0, int(cx_px - (w_px + expand_w) / 2))
        y1 = max(0, int(cy_px - (h_px + expand_h) / 2))
        x2 = min(w, int(cx_px + (w_px + expand_w) / 2))
        y2 = min(h, int(cy_px + (h_px + expand_h) / 2))
        
        #
        cropped = image[y1:y2, x1:x2]
        
        return cropped

    def _detect_roi_format(self) -> str:
        """
            ROI    

        Returns:
            'yolo'   'mask'
        """
        if not self.samples:
            return 'yolo'  #

        #
        for sample in self.samples[:min(10, len(self.samples))]:
            roi_path = sample['roi_path']

            #
            base_name = Path(roi_path).stem
            mask_path = Path(self.mask_root) / 'masks' / f'{base_name}.npy'

            if mask_path.exists():
                return 'mask'

        return 'yolo'  #

    def _read_sam_mask(self, image_path: str) -> Optional[np.ndarray]:
        """
          SAM          txt     npy   

        Args:
            image_path:     

        Returns:
                 (H, W)         None
        """
        try:
            base_name = Path(image_path).stem

            #
            coords_path = Path(self.mask_root) / f'{base_name}.txt'
            if coords_path.exists():
                return self._read_mask_from_coords(coords_path, image_path)

            #
            mask_path = Path(self.mask_root) / 'masks' / f'{base_name}.npy'
            if mask_path.exists():
                mask = np.load(str(mask_path))
                #
                if mask.dtype != bool:
                    mask = mask > 0.5
                return mask

            return None

        except Exception as e:
            print(f"Failed to read mask file for {image_path}: {e}")
            return None

    def _read_sam_mask_from_roi_path(self, roi_path: str) -> Optional[np.ndarray]:
        """
         ROI    SAM               

        Args:
            roi_path: ROI      YOLO       

        Returns:
                 (H, W)         None
        """
        try:
            #
            coords_path = Path(roi_path)
            if coords_path.exists():
                #
                image_path = self._get_image_path_from_roi_path(roi_path)
                return self._read_mask_from_coords(coords_path, image_path)

            return None

        except Exception as e:
            print(f"Failed to read mask from roi path {roi_path}: {e}")
            return None

    def _get_image_path_from_roi_path(self, roi_path: str) -> str:
        """ ROI           """
        try:
            #
            #
            roi_path_obj = Path(roi_path)

            #
            roi_root_obj = Path(self.roi_root)
            rel_path = roi_path_obj.relative_to(roi_root_obj)

            #
            image_rel_path = rel_path.with_suffix('.jpg')
            image_path = Path(self.image_root) / image_rel_path

            return str(image_path)

        except Exception as e:
            print(f"Failed to infer image path from roi path {roi_path}: {e}")
            return ""

    def _read_mask_from_coords(self, coords_path: Path, image_path: str) -> Optional[np.ndarray]:
        """
                    detect_and_crop.py    

        Args:
            coords_path:       
            image_path:               

        Returns:
                    (H, W)
        """
        try:
            #
            image = cv2.imread(image_path)
            if image is None:
                return None
            h, w = image.shape[:2]

            #
            with open(coords_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if len(lines) < 3:
                return None

            #
            info_parts = lines[0].strip().split(',')
            if len(info_parts) >= 5:
                #
                file_w, file_h = int(info_parts[3]), int(info_parts[4])
                #
                if file_w != w or file_h != h:
                    print(f"Warning: Image size mismatch in {coords_path}")

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
                    x, y = map(float, lines[line_idx].strip().split(','))
                    contour_points.append([int(x), int(y)])
                    line_idx += 1

                if contour_points:
                    #
                    contour = np.array(contour_points, dtype=np.int32)
                    cv2.fillPoly(mask, [contour], 255)

            return mask.astype(bool)

        except Exception as e:
            print(f"Failed to read mask from coords {coords_path}: {e}")
            return None

    def _get_mask_bbox(self, mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
                

        Args:
            mask:      (H, W)

        Returns:
            (cx, cy, w, h)               None
        """
        try:
            #
            rows, cols = np.where(mask)

            if len(rows) == 0:
                return None

            #
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
            print(f"Failed to compute bbox from mask: {e}")
            return None

    def _apply_mask_to_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
                 

        Args:
            image:      (H, W, C)
            mask:      (H, W)

        Returns:
                    
        """
        try:
            #
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8),
                                (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)

            #
            masked_image = image.copy()
            masked_image[~mask] = 0

            return masked_image

        except Exception as e:
            print(f"Failed to apply mask to image: {e}")
            return image  #

    def _process_mask_region_to_rectangle(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
                           

           
        1.         
        2.       10%   
        3.           
        4.                  0   

        Args:
            image:      (H, W, C)
            mask:      (H, W)

        Returns:
                           0  
        """
        try:
            #
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8),
                                (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)

            #
            rows, cols = np.where(mask)
            if len(rows) == 0:
                #
                return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()

            #
            h, w = image.shape[:2]
            bbox_w = x2 - x1 + 1
            bbox_h = y2 - y1 + 1

            expand_w = int(bbox_w * self.roi_expand_ratio)
            expand_h = int(bbox_h * self.roi_expand_ratio)

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
            print(f"Failed to process mask region: {e}")
            #
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
              
        
        Returns:
                          
        """
        sample = self.samples[idx]
        
        #
        try:
            image = cv2.imread(sample['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Failed to read image {sample['image_path']}: {e}")
            #
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        #
        if self.roi_format == 'mask':
            #
            mask = self._read_sam_mask_from_roi_path(sample['roi_path'])
            if mask is not None:
                #
                image = self._process_mask_region_to_rectangle(image, mask)
        else:
            #
            roi = self._read_yolo_roi(sample['roi_path'])
            if roi is not None:
                #
                image = self._crop_roi_with_expansion(image, roi)
        
        #
        if image.size == 0:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        #
        image = Image.fromarray(image)
        
        #
        if self.transform:
            image = self.transform(image)
        else:
            #
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
        
        #
        gender_raw = str(sample.get('gender', '')).strip()
        gender_raw_upper = gender_raw.upper()
        if gender_raw_upper in ['M', ' ', 'MALE', ' ']:
            gender_label = 1
        elif gender_raw_upper in ['F', ' ', 'FEMALE', ' ']:
            gender_label = 0
        else:
            #
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
        """  ID    """
        return len(self.id_to_label)
    
    def get_id_names(self) -> List[str]:
        """    ID    """
        return list(self.id_to_label.keys())


def build_panda_transform(is_train: bool, img_size: int = 192, config=None) -> transforms.Compose:
    """
                 
    
    Args:
        is_train:        
        img_size:     
        config:    yacs          config.AUG        AUTO_AUGMENT/REPROB   
        
    Returns:
              
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        #
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

        #
        if auto_augment is not None:
            try:
                from timm.data.auto_augment import auto_augment_transform

                aa_params = {
                    "translate_const": int(img_size * 0.45),
                    "img_mean": tuple([min(255, max(0, int(m * 255))) for m in mean]),
                }
                t.append(auto_augment_transform(auto_augment, aa_params))
            except Exception:
                #
                auto_augment = None

        #
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

        #
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
        #
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    return transform


if __name__ == "__main__":
    #
    image_root = "data/example/images"
    roi_root = "data/example/roi"
    
    transform = build_panda_transform(is_train=True)
    dataset = PandaDataset(image_root, roi_root, transform=transform)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of IDs: {dataset.get_num_classes()}")
    print(f"ID list: {dataset.get_id_names()}")
    
    #
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape: {sample['image'].shape}")
        print(f"ID label: {sample['id_label']}, ID name: {sample['id_name']}")
        print(f"Age: {sample['age']}, Gender: {sample['gender']}")
