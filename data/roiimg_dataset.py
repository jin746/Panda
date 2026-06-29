import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
LEGACY_ID_PATTERN = re.compile(r"^(.+?)_(\d{4})_([A-Za-z\u4e00-\u9fff]+)$")


def build_panda_transform(is_train: bool, img_size: int = 192) -> transforms.Compose:
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _iter_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in IMAGE_EXTS:
            files.append(path)
    files.sort()
    return files


def _imread_unicode(path: str) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except Exception:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _normalize_gender_label(raw: str) -> int:
    token = str(raw or "").strip().upper()
    if token in {"M", "MALE", "雄", "男"}:
        return 1
    if token in {"F", "FEMALE", "雌", "女"}:
        return 0
    return 0


class PandaRoiImgDataset(Dataset):
    """
    Direct ROI-image dataset reader.

    Supported layouts:
    1. Legacy panda layout:
       roiimg_root/<id>_<birth_year>_<sex>/*.jpg
    2. Generic species layout:
       roiimg_root/<species>/<individual>/*.jpg

    For the generic species layout, age/gender labels are treated as unavailable and
    the ReID class name becomes "<species>__<individual>" to avoid cross-species collisions.
    """

    def __init__(
        self,
        roiimg_root: str,
        img_size: int = 192,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True,
        capture_year_regex: Optional[str] = None,
    ):
        self.roiimg_root = str(roiimg_root)
        self.img_size = int(img_size)
        self.transform = transform or build_panda_transform(is_train=is_train, img_size=img_size)
        self.is_train = bool(is_train)
        self.capture_year_regex = capture_year_regex
        self.samples = self._scan_dataset()
        self.id_to_label = self._build_id_mapping()
        print(f"ROIIMG Dataset loaded: {len(self.samples)} samples, {len(self.id_to_label)} IDs")

    def _scan_dataset(self) -> List[Dict]:
        root = Path(self.roiimg_root)
        if not root.is_dir():
            print(f"[WARN] ROIIMG root not found: {self.roiimg_root}")
            return []

        samples: List[Dict] = []
        top_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
        if not top_dirs:
            print(f"[WARN] ROIIMG root has no sub-directories: {self.roiimg_root}")
            return samples

        for top_dir in top_dirs:
            legacy_match = LEGACY_ID_PATTERN.match(top_dir.name)
            if legacy_match:
                samples.extend(self._scan_legacy_id_dir(top_dir, legacy_match))
                continue

            # Generic layout: species/individual/*.jpg
            child_dirs = [p for p in sorted(top_dir.iterdir()) if p.is_dir()]
            if not child_dirs:
                # Fallback: treat a direct folder with images as a generic identity folder.
                samples.extend(self._scan_generic_identity_dir(species="", individual_dir=top_dir))
                continue
            for individual_dir in child_dirs:
                samples.extend(self._scan_generic_identity_dir(species=top_dir.name, individual_dir=individual_dir))

        return samples

    def _scan_legacy_id_dir(self, id_dir: Path, match: re.Match) -> List[Dict]:
        uid, birth_year_str, sex_raw = match.groups()
        birth_year = int(birth_year_str)
        gender_label = _normalize_gender_label(sex_raw)
        rows: List[Dict] = []

        for img_path in _iter_images(id_dir):
            cap_year = self._infer_capture_year(str(img_path), str(id_dir), birth_year)
            age_years = None if cap_year is None else max(0, int(cap_year) - birth_year)
            rows.append({
                "image_path": str(img_path),
                "id_name": str(uid),
                "species": None,
                "individual": str(uid),
                "gender_label": gender_label,
                "birth_year": birth_year,
                "age_years": age_years,
            })
        return rows

    def _scan_generic_identity_dir(self, species: str, individual_dir: Path) -> List[Dict]:
        id_name = f"{species}__{individual_dir.name}" if species else individual_dir.name
        rows: List[Dict] = []
        for img_path in _iter_images(individual_dir):
            rows.append({
                "image_path": str(img_path),
                "id_name": id_name,
                "species": species or None,
                "individual": individual_dir.name,
                "gender_label": 0,
                "birth_year": None,
                "age_years": None,
            })
        return rows

    def _infer_capture_year(self, img_path: str, id_dir: str, birth_year: Optional[int]) -> Optional[int]:
        def extract_years(text: str) -> List[int]:
            out = []
            for item in re.finditer(r"(?:19|20)\d{2}", text):
                year = int(item.group(0))
                if 1990 <= year <= 2035:
                    out.append(year)
            return out

        try:
            rel = os.path.relpath(img_path, id_dir)
            parts = os.path.normpath(rel).split(os.sep)
            if len(parts) >= 2:
                for comp in parts[:-1]:
                    if re.fullmatch(r"(?:19|20)\d{2}", comp):
                        year = int(comp)
                        if birth_year is not None and year == birth_year:
                            continue
                        if birth_year is None or year >= birth_year:
                            return year

            fname = os.path.basename(img_path)
            if self.capture_year_regex:
                match = re.search(self.capture_year_regex, fname)
                if match:
                    year = int(match.group(1))
                    if 1990 <= year <= 2035 and (birth_year is None or year != birth_year):
                        if birth_year is None or year >= birth_year:
                            return year

            years = extract_years(fname)
            if birth_year is not None:
                years = [y for y in years if y != birth_year]
            cand = [y for y in years if birth_year is None or y >= birth_year]
            if cand:
                return cand[-1]
            if years:
                return years[-1]
        except Exception:
            return None
        return None

    def _build_id_mapping(self) -> Dict[str, int]:
        uniq_ids = sorted({str(sample["id_name"]) for sample in self.samples})
        return {name: idx for idx, name in enumerate(uniq_ids)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        img = _imread_unicode(sample["image_path"])
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(Image.fromarray(img))
        return {
            "image": img_tensor,
            "id_label": self.id_to_label[sample["id_name"]],
            "id_name": sample["id_name"],
            "gender_label": sample["gender_label"],
            "age_years": -1.0 if sample["age_years"] is None else float(sample["age_years"]),
            "age_valid": 0 if sample["age_years"] is None else 1,
            "image_path": sample["image_path"],
            "species": sample.get("species") or "",
            "individual": sample.get("individual") or "",
        }

    def get_num_classes(self) -> int:
        return len(self.id_to_label)

    def get_id_names(self) -> List[str]:
        return list(self.id_to_label.keys())
