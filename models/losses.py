# --------------------------------------------------------
# 大熊猫ReID损失函数
# 包含ArcFace损失和TripletLoss
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class ArcFaceLoss(nn.Module):
    """
    ArcFace损失函数
    论文: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    """
    
    def __init__(self, in_features: int, out_features: int, scale: float = 30.0, margin: float = 0.5):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出类别数
            scale: 缩放因子
            margin: 角度边距
        """
        super(ArcFaceLoss, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        
        # 权重矩阵
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # 预计算的常数
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, input_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: 输入特征 [B, in_features]
            labels: 真实标签 [B]
            
        Returns:
            loss: ArcFace损失
        """
        # 归一化特征和权重
        cosine = F.linear(F.normalize(input_features), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # 计算 cos(theta + margin)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # 处理数值稳定性
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 创建one-hot标签
        one_hot = torch.zeros(cosine.size(), device=input_features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # 应用margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        # 计算交叉熵损失
        loss = F.cross_entropy(output, labels)
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss with hard mining
    """
    
    def __init__(self, margin: float = 0.3, hard_mining: bool = True):
        """
        Args:
            margin: triplet margin
            hard_mining: 是否使用hard mining
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征向量 [B, D]
            labels: 标签 [B]
            
        Returns:
            loss: triplet loss
        """
        n = features.size(0)
        
        # 计算距离矩阵
        dist_mat = self._compute_distance_matrix(features)
        
        if self.hard_mining:
            return self._hard_mining_triplet_loss(dist_mat, labels)
        else:
            return self._batch_all_triplet_loss(dist_mat, labels)
    
    def _compute_distance_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """计算欧氏距离矩阵"""
        n = features.size(0)
        
        # 计算 ||a||^2 + ||b||^2 - 2*a*b
        dist_mat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_mat = dist_mat + dist_mat.t()
        dist_mat.addmm_(features, features.t(), beta=1, alpha=-2)
        
        # 数值稳定性
        dist_mat = dist_mat.clamp(min=1e-12).sqrt()
        
        return dist_mat
    
    def _hard_mining_triplet_loss(self, dist_mat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Hard mining triplet loss"""
        n = dist_mat.size(0)

        # 创建mask
        is_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
        is_neg = labels.expand(n, n).ne(labels.expand(n, n).t())

        # 移除对角线（自己与自己的距离）
        is_pos = is_pos ^ torch.eye(n, dtype=torch.bool, device=labels.device)

        # 检查每个样本是否有正样本和负样本
        has_pos = is_pos.sum(dim=1) > 0
        has_neg = is_neg.sum(dim=1) > 0
        valid_samples = has_pos & has_neg

        if valid_samples.sum() == 0:
            # 如果没有有效的triplet，返回零损失
            return torch.tensor(0.0, device=labels.device, requires_grad=True)

        # 只处理有效的样本
        valid_indices = torch.where(valid_samples)[0]

        dist_ap_list = []
        dist_an_list = []

        for i in valid_indices:
            # 获取正样本距离
            pos_distances = dist_mat[i][is_pos[i]]
            if len(pos_distances) > 0:
                dist_ap_list.append(torch.max(pos_distances))

            # 获取负样本距离
            neg_distances = dist_mat[i][is_neg[i]]
            if len(neg_distances) > 0:
                dist_an_list.append(torch.min(neg_distances))

        if len(dist_ap_list) == 0 or len(dist_an_list) == 0:
            return torch.tensor(0.0, device=labels.device, requires_grad=True)

        dist_ap = torch.stack(dist_ap_list)
        dist_an = torch.stack(dist_an_list)

        # Triplet loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss
    
    def _batch_all_triplet_loss(self, dist_mat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Batch all triplet loss"""
        n = dist_mat.size(0)
        
        # 创建mask
        is_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
        is_neg = labels.expand(n, n).ne(labels.expand(n, n).t())
        
        # 获取所有有效的triplet
        dist_ap = dist_mat[is_pos].contiguous().view(n, -1)
        dist_an = dist_mat[is_neg].contiguous().view(n, -1)
        
        # 计算triplet loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss


class CenterLoss(nn.Module):
    """
    Center Loss: 约束同类样本聚合到类中心
    论文: A Discriminative Feature Learning Approach for Deep Face Recognition (ECCV 2016)

    核心作用: 稳定特征空间,减少阈值敏感性
    """

    def __init__(self, num_classes: int, feat_dim: int, lambda_c: float = 0.003):
        """
        Args:
            num_classes: 类别数
            feat_dim: 特征维度
            lambda_c: center loss权重
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c

        # 类中心参数 [num_classes, feat_dim]
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征向量 [B, feat_dim]
            labels: 标签 [B]

        Returns:
            loss: center loss
        """
        batch_size = features.size(0)

        # 获取每个样本对应的类中心
        centers_batch = self.centers[labels]  # [B, feat_dim]

        # 计算特征到类中心的距离
        loss = torch.pow(features - centers_batch, 2).sum() / (2.0 * batch_size)

        return loss * self.lambda_c


class VarianceLoss(nn.Module):
    """
    方差正则化损失: 鼓励特征维度具有相似的方差

    核心作用: 防止特征空间坍塌,提升泛化能力
    """

    def __init__(self, lambda_var: float = 0.001):
        super(VarianceLoss, self).__init__()
        self.lambda_var = lambda_var

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征向量 [B, D]

        Returns:
            loss: variance regularization loss
        """
        # 计算每个维度的方差
        var = torch.var(features, dim=0)  # [D]

        # 鼓励方差接近1(特征已归一化)
        target_var = torch.ones_like(var) * 0.5
        loss = F.mse_loss(var, target_var)

        return loss * self.lambda_var


class CombinedLoss(nn.Module):
    """
    增强组合损失函数: ArcFace + Triplet + Center + Variance

    改进点:
    1. 添加Center Loss稳定特征空间
    2. 添加Variance Loss防止特征坍塌
    3. 优化损失权重平衡
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.5,
        triplet_margin: float = 0.3,
        arcface_weight: float = 1.0,
        triplet_weight: float = 1.0,
        center_weight: float = 0.01,
        variance_weight: float = 0.001,
        use_hard_mining: bool = True
    ):
        """
        Args:
            in_features: 特征维度
            num_classes: 类别数
            arcface_scale: ArcFace缩放因子
            arcface_margin: ArcFace边距
            triplet_margin: Triplet边距
            arcface_weight: ArcFace损失权重
            triplet_weight: Triplet损失权重
            center_weight: Center损失权重
            variance_weight: Variance损失权重
            use_hard_mining: 是否使用hard mining
        """
        super(CombinedLoss, self).__init__()

        self.arcface_weight = arcface_weight
        self.triplet_weight = triplet_weight
        self.center_weight = center_weight
        self.variance_weight = variance_weight

        # ArcFace损失
        self.arcface_loss = ArcFaceLoss(
            in_features=in_features,
            out_features=num_classes,
            scale=arcface_scale,
            margin=arcface_margin
        )

        # Triplet损失
        self.triplet_loss = TripletLoss(
            margin=triplet_margin,
            hard_mining=use_hard_mining
        )

        # Center损失 - 稳定特征空间
        self.center_loss = CenterLoss(
            num_classes=num_classes,
            feat_dim=in_features,
            lambda_c=1.0  # 内部权重设为1,外部通过center_weight控制
        )

        # Variance损失 - 防止特征坍塌
        self.variance_loss = VarianceLoss(lambda_var=1.0)

    def forward(
        self,
        features: torch.Tensor,
        cls_scores: torch.Tensor,
        labels: torch.Tensor,
        arcface_features: torch.Tensor = None,
        triplet_features: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            features: 特征向量 [B, D]
            cls_scores: 分类得分 [B, num_classes] (可以为None)
            labels: 标签 [B]
            arcface_features: 可选，用于ArcFace的特征 [B, D]（例如BN后特征）
            triplet_features: 可选，用于Triplet/Center/Variance的特征 [B, D]（例如BN前特征）

        Returns:
            loss_dict: 包含各项损失的字典
        """
        loss_dict = {}
        total_loss = 0.0

        triplet_feat = triplet_features if triplet_features is not None else features
        arcface_feat = arcface_features if arcface_features is not None else features

        # ArcFace损失
        do_arcface = self.arcface_weight > 0 and (
            arcface_features is not None or cls_scores is not None
        )
        if do_arcface:
            arcface_loss = self.arcface_loss(arcface_feat, labels)
            loss_dict['arcface_loss'] = arcface_loss
            total_loss += self.arcface_weight * arcface_loss

        # Triplet损失
        if self.triplet_weight > 0:
            triplet_loss = self.triplet_loss(triplet_feat, labels)
            loss_dict['triplet_loss'] = triplet_loss
            total_loss += self.triplet_weight * triplet_loss

        # Center损失 - 稳定特征空间
        if self.center_weight > 0:
            center_loss = self.center_loss(triplet_feat, labels)
            loss_dict['center_loss'] = center_loss
            total_loss += self.center_weight * center_loss

        # Variance损失 - 防止特征坍塌
        if self.variance_weight > 0:
            variance_loss = self.variance_loss(triplet_feat)
            loss_dict['variance_loss'] = variance_loss
            total_loss += self.variance_weight * variance_loss

        loss_dict['total_loss'] = total_loss

        return loss_dict


class SimilarityLoss(nn.Module):
    """
    相似度匹配网络的损失函数
    使用BCEWithLogitsLoss以支持AMP
    """

    def __init__(self):
        """
        简化版本，直接使用BCEWithLogitsLoss
        """
        super(SimilarityLoss, self).__init__()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 未经sigmoid的原始输出 [B]
            labels: 标签，1表示同一ID，0表示不同ID [B]

        Returns:
            loss: 相似度损失
        """
        # 确保维度匹配
        if logits.dim() > 1:
            logits = logits.squeeze()

        # 转换标签为float
        targets = labels.float()

        # 计算BCEWithLogits损失
        loss = self.bce_with_logits_loss(logits, targets)

        return loss


def build_loss_function(config, in_features: int, num_classes: int):
    """
    构建损失函数
    
    Args:
        config: 配置对象
        in_features: 特征维度
        num_classes: 类别数
        
    Returns:
        loss_fn: 损失函数
    """
    loss_type = getattr(config.MODEL, 'LOSS_TYPE', 'combined')
    
    if loss_type == 'arcface':
        loss_fn = ArcFaceLoss(
            in_features=in_features,
            out_features=num_classes,
            scale=getattr(config.MODEL, 'ARCFACE_SCALE', 30.0),
            margin=getattr(config.MODEL, 'ARCFACE_MARGIN', 0.5)
        )
    elif loss_type == 'triplet':
        loss_fn = TripletLoss(
            margin=getattr(config.MODEL, 'TRIPLET_MARGIN', 0.3),
            hard_mining=getattr(config.MODEL, 'HARD_MINING', True)
        )
    elif loss_type == 'combined':
        loss_fn = CombinedLoss(
            in_features=in_features,
            num_classes=num_classes,
            arcface_scale=getattr(config.MODEL, 'ARCFACE_SCALE', 30.0),
            arcface_margin=getattr(config.MODEL, 'ARCFACE_MARGIN', 0.5),
            triplet_margin=getattr(config.MODEL, 'TRIPLET_MARGIN', 0.3),
            arcface_weight=getattr(config.MODEL, 'ARCFACE_WEIGHT', 1.0),
            triplet_weight=getattr(config.MODEL, 'TRIPLET_WEIGHT', 1.0),
            use_hard_mining=getattr(config.MODEL, 'HARD_MINING', True)
        )
    else:
        raise ValueError(f"Unsupported loss function type: {loss_type}")
    
    return loss_fn


class SupConLoss(nn.Module):
    """监督对比损失 (Supervised Contrastive Loss) - 数值稳定版本"""

    def __init__(self, temperature=0.1, base_temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """计算监督对比损失

        Args:
            features: 特征向量 [bsz, feature_dim]
            labels: 标签 [bsz]
        """
        device = features.device
        batch_size = features.shape[0]

        # 🔥 输入检查和安全处理
        if torch.isnan(features).any():
            print("⚠️ SupConLoss输入特征包含NaN")
            return torch.tensor(0.0, device=device, requires_grad=True)

        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 🔥 安全的特征归一化
        features = F.normalize(features, p=2, dim=1, eps=1e-8)

        # 检查归一化后是否有NaN
        if torch.isnan(features).any():
            print("⚠️ 特征归一化后包含NaN")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 创建标签掩码
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 检查是否有有效的正样本对（除了对角线）
        mask_no_diag = mask - torch.eye(batch_size, device=device)
        if mask_no_diag.sum() == 0:
            # 没有正样本对，返回0损失
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 🔥 计算相似度矩阵（更稳定的方式）
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 🔥 数值稳定性处理
        similarity_matrix = torch.clamp(similarity_matrix, min=-50, max=50)

        # 减去最大值以提高数值稳定性
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # 创建负样本掩码（排除对角线）
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)

        # 🔥 安全的exp和log计算
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # 计算正样本的平均log概率
        mask_pos = mask_no_diag
        mask_pos_sum = mask_pos.sum(1)

        # 只考虑有正样本的anchor
        valid_anchors = mask_pos_sum > 0
        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 计算损失
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (mask_pos_sum + 1e-8)
        loss = -mean_log_prob_pos[valid_anchors].mean()

        # 🔥 最终NaN检查
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ SupConLoss输出NaN或Inf")
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss


class ContrastiveReIDLoss(nn.Module):
    """优化的对比学习ReID损失函数 - 针对高精度识别优化"""

    def __init__(self, feature_dim: int, num_classes: int):
        super(ContrastiveReIDLoss, self).__init__()

        # 主要损失：Triplet Loss（增强版）
        self.triplet_loss = nn.TripletMarginLoss(
            margin=0.6,  # 增大margin提高区分度
            p=2,
            reduction='mean'
        )

        # 监督对比损失（核心损失，优化参数）
        self.supcon_loss = SupConLoss(temperature=0.07)  # 降低温度增强区分度

        # 分类损失（优化版）
        self.combined_loss = CombinedLoss(
            in_features=feature_dim,
            num_classes=num_classes,
            arcface_scale=48.0,  # 增大scale提高区分度
            arcface_margin=0.4,  # 适中的margin
            triplet_margin=0.6,  # 与主triplet保持一致
            arcface_weight=0.8,  # 平衡的分类权重
            triplet_weight=1.5,  # 增加triplet权重
            use_hard_mining=True
        )

        # 优化的损失权重
        self.supcon_weight = 2.0      # 大幅增加对比学习权重
        self.classification_weight = 0.8  # 适中的分类权重

    def forward(self, features: torch.Tensor, cls_scores: torch.Tensor, labels: torch.Tensor) -> dict:
        device = features.device

        # 🔥 输入检查
        if torch.isnan(features).any():
            print("⚠️ ContrastiveReIDLoss输入特征包含NaN")
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                'total_loss': zero_loss,
                'supcon_loss': zero_loss,
                'classification_loss': zero_loss,
                'arcface_loss': zero_loss,
                'triplet_loss': zero_loss
            }

        # 主要损失：监督对比学习
        try:
            supcon_loss = self.supcon_loss(features, labels)
            if torch.isnan(supcon_loss):
                print("⚠️ SupCon损失为NaN，使用0替代")
                supcon_loss = torch.tensor(0.0, device=device, requires_grad=True)
        except Exception as e:
            print(f"⚠️ SupCon损失计算出错: {e}")
            supcon_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 辅助损失：分类损失（帮助稳定训练）
        try:
            classification_loss_dict = self.combined_loss(features, cls_scores, labels)
            classification_loss = classification_loss_dict['total_loss']
            if torch.isnan(classification_loss):
                print("⚠️ 分类损失为NaN，使用0替代")
                classification_loss = torch.tensor(0.0, device=device, requires_grad=True)
                classification_loss_dict = {
                    'total_loss': classification_loss,
                    'arcface_loss': torch.tensor(0.0, device=device),
                    'triplet_loss': torch.tensor(0.0, device=device)
                }
        except Exception as e:
            print(f"⚠️ 分类损失计算出错: {e}")
            classification_loss = torch.tensor(0.0, device=device, requires_grad=True)
            classification_loss_dict = {
                'total_loss': classification_loss,
                'arcface_loss': torch.tensor(0.0, device=device),
                'triplet_loss': torch.tensor(0.0, device=device)
            }

        # 🔥 总损失计算（安全版本）
        total_loss = (self.supcon_weight * supcon_loss +
                     self.classification_weight * classification_loss)

        # 最终NaN检查
        if torch.isnan(total_loss):
            print("⚠️ 总损失为NaN，使用0替代")
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return {
            'total_loss': total_loss,
            'supcon_loss': supcon_loss,
            'classification_loss': classification_loss,
            'arcface_loss': classification_loss_dict.get('arcface_loss', torch.tensor(0.0, device=device)),
            'triplet_loss': classification_loss_dict.get('triplet_loss', torch.tensor(0.0, device=device))
        }


if __name__ == "__main__":
    # 测试损失函数
    batch_size = 8
    feature_dim = 512
    num_classes = 42
    
    # 测试数据
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 测试ArcFace损失
    arcface_loss = ArcFaceLoss(feature_dim, num_classes)
    loss = arcface_loss(features, labels)
    print(f"ArcFace loss: {loss.item():.4f}")

    # Test Triplet loss
    triplet_loss = TripletLoss()
    loss = triplet_loss(features, labels)
    print(f"Triplet loss: {loss.item():.4f}")

    # Test combined loss
    combined_loss = CombinedLoss(feature_dim, num_classes)
    cls_scores = torch.randn(batch_size, num_classes)
    loss_dict = combined_loss(features, cls_scores, labels)
    print(f"Combined loss: {loss_dict}")
