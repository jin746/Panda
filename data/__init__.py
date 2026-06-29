"""
数据模块初始化（精简版）
仅保留与 Panda ReID 相关的导出，防止不必要依赖。
"""
from .panda_dataset import PandaDataset, build_panda_transform
