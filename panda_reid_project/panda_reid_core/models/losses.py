# --------------------------------------------------------
#
#
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class ArcFaceLoss(nn.Module):
    """
    ArcFace    
      : ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    """
    
    def __init__(self, in_features: int, out_features: int, scale: float = 30.0, margin: float = 0.5):
        """
        Args:
            in_features:       
            out_features:      
            scale:     
            margin:     
        """
        super(ArcFaceLoss, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        
        #
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        #
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, input_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features:      [B, in_features]
            labels:      [B]
            
        Returns:
            loss: ArcFace  
        """
        #
        cosine = F.linear(F.normalize(input_features), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        #
        phi = cosine * self.cos_m - sine * self.sin_m
        
        #
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        #
        one_hot = torch.zeros(cosine.size(), device=input_features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        #
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        #
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
            hard_mining:     hard mining
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features:      [B, D]
            labels:    [B]
            
        Returns:
            loss: triplet loss
        """
        n = features.size(0)
        
        #
        dist_mat = self._compute_distance_matrix(features)
        
        if self.hard_mining:
            return self._hard_mining_triplet_loss(dist_mat, labels)
        else:
            return self._batch_all_triplet_loss(dist_mat, labels)
    
    def _compute_distance_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """        """
        n = features.size(0)
        
        #
        dist_mat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_mat = dist_mat + dist_mat.t()
        dist_mat.addmm_(features, features.t(), beta=1, alpha=-2)
        
        #
        dist_mat = dist_mat.clamp(min=1e-12).sqrt()
        
        return dist_mat
    
    def _hard_mining_triplet_loss(self, dist_mat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Hard mining triplet loss"""
        n = dist_mat.size(0)

        #
        is_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
        is_neg = labels.expand(n, n).ne(labels.expand(n, n).t())

        #
        is_pos = is_pos ^ torch.eye(n, dtype=torch.bool, device=labels.device)

        #
        has_pos = is_pos.sum(dim=1) > 0
        has_neg = is_neg.sum(dim=1) > 0
        valid_samples = has_pos & has_neg

        if valid_samples.sum() == 0:
            #
            return torch.tensor(0.0, device=labels.device, requires_grad=True)

        #
        valid_indices = torch.where(valid_samples)[0]

        dist_ap_list = []
        dist_an_list = []

        for i in valid_indices:
            #
            pos_distances = dist_mat[i][is_pos[i]]
            if len(pos_distances) > 0:
                dist_ap_list.append(torch.max(pos_distances))

            #
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
        
        #
        is_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
        is_neg = labels.expand(n, n).ne(labels.expand(n, n).t())
        
        #
        dist_ap = dist_mat[is_pos].contiguous().view(n, -1)
        dist_an = dist_mat[is_neg].contiguous().view(n, -1)
        
        #
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss


class CenterLoss(nn.Module):
    """
    Center Loss:             
      : A Discriminative Feature Learning Approach for Deep Face Recognition (ECCV 2016)

        :       ,       
    """

    def __init__(self, num_classes: int, feat_dim: int, lambda_c: float = 0.003):
        """
        Args:
            num_classes:    
            feat_dim:     
            lambda_c: center loss  
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c

        #
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features:      [B, feat_dim]
            labels:    [B]

        Returns:
            loss: center loss
        """
        batch_size = features.size(0)

        #
        centers_batch = self.centers[labels]  # [B, feat_dim]

        #
        loss = torch.pow(features - centers_batch, 2).sum() / (2.0 * batch_size)

        return loss * self.lambda_c


class VarianceLoss(nn.Module):
    """
           :              

        :         ,      
    """

    def __init__(self, lambda_var: float = 0.001):
        super(VarianceLoss, self).__init__()
        self.lambda_var = lambda_var

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features:      [B, D]

        Returns:
            loss: variance regularization loss
        """
        #
        var = torch.var(features, dim=0)  # [D]

        #
        target_var = torch.ones_like(var) * 0.5
        loss = F.mse_loss(var, target_var)

        return loss * self.lambda_var


class CombinedLoss(nn.Module):
    """
            : ArcFace + Triplet + Center + Variance

       :
    1.   Center Loss      
    2.   Variance Loss      
    3.         
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
            in_features:     
            num_classes:    
            arcface_scale: ArcFace    
            arcface_margin: ArcFace  
            triplet_margin: Triplet  
            arcface_weight: ArcFace    
            triplet_weight: Triplet    
            center_weight: Center    
            variance_weight: Variance    
            use_hard_mining:     hard mining
        """
        super(CombinedLoss, self).__init__()

        self.arcface_weight = arcface_weight
        self.triplet_weight = triplet_weight
        self.center_weight = center_weight
        self.variance_weight = variance_weight

        #
        self.arcface_loss = ArcFaceLoss(
            in_features=in_features,
            out_features=num_classes,
            scale=arcface_scale,
            margin=arcface_margin
        )

        #
        self.triplet_loss = TripletLoss(
            margin=triplet_margin,
            hard_mining=use_hard_mining
        )

        #
        self.center_loss = CenterLoss(
            num_classes=num_classes,
            feat_dim=in_features,
            lambda_c=1.0  #
        )

        #
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
            features:      [B, D]
            cls_scores:      [B, num_classes] (   None)
            labels:    [B]
            arcface_features:      ArcFace    [B, D]   BN    
            triplet_features:      Triplet/Center/Variance    [B, D]   BN    

        Returns:
            loss_dict:          
        """
        loss_dict = {}
        total_loss = 0.0

        triplet_feat = triplet_features if triplet_features is not None else features
        arcface_feat = arcface_features if arcface_features is not None else features

        #
        do_arcface = self.arcface_weight > 0 and (
            arcface_features is not None or cls_scores is not None
        )
        if do_arcface:
            arcface_loss = self.arcface_loss(arcface_feat, labels)
            loss_dict['arcface_loss'] = arcface_loss
            total_loss += self.arcface_weight * arcface_loss

        #
        if self.triplet_weight > 0:
            triplet_loss = self.triplet_loss(triplet_feat, labels)
            loss_dict['triplet_loss'] = triplet_loss
            total_loss += self.triplet_weight * triplet_loss

        #
        if self.center_weight > 0:
            center_loss = self.center_loss(triplet_feat, labels)
            loss_dict['center_loss'] = center_loss
            total_loss += self.center_weight * center_loss

        #
        if self.variance_weight > 0:
            variance_loss = self.variance_loss(triplet_feat)
            loss_dict['variance_loss'] = variance_loss
            total_loss += self.variance_weight * variance_loss

        loss_dict['total_loss'] = total_loss

        return loss_dict


class SimilarityLoss(nn.Module):
    """
                
      BCEWithLogitsLoss   AMP
    """

    def __init__(self):
        """
                 BCEWithLogitsLoss
        """
        super(SimilarityLoss, self).__init__()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:   sigmoid      [B]
            labels:    1    ID 0    ID [B]

        Returns:
            loss:      
        """
        #
        if logits.dim() > 1:
            logits = logits.squeeze()

        #
        targets = labels.float()

        #
        loss = self.bce_with_logits_loss(logits, targets)

        return loss


def build_loss_function(config, in_features: int, num_classes: int):
    """
          
    
    Args:
        config:     
        in_features:     
        num_classes:    
        
    Returns:
        loss_fn:     
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
    """       (Supervised Contrastive Loss) -       """

    def __init__(self, temperature=0.1, base_temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """        

        Args:
            features:      [bsz, feature_dim]
            labels:    [bsz]
        """
        device = features.device
        batch_size = features.shape[0]

        #
        if torch.isnan(features).any():
            print("   SupConLoss      NaN")
            return torch.tensor(0.0, device=device, requires_grad=True)

        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        #
        features = F.normalize(features, p=2, dim=1, eps=1e-8)

        #
        if torch.isnan(features).any():
            print("           NaN")
            return torch.tensor(0.0, device=device, requires_grad=True)

        #
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        #
        mask_no_diag = mask - torch.eye(batch_size, device=device)
        if mask_no_diag.sum() == 0:
            #
            return torch.tensor(0.0, device=device, requires_grad=True)

        #
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        #
        similarity_matrix = torch.clamp(similarity_matrix, min=-50, max=50)

        #
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        #
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)

        #
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        #
        mask_pos = mask_no_diag
        mask_pos_sum = mask_pos.sum(1)

        #
        valid_anchors = mask_pos_sum > 0
        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        #
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (mask_pos_sum + 1e-8)
        loss = -mean_log_prob_pos[valid_anchors].mean()

        #
        if torch.isnan(loss) or torch.isinf(loss):
            print("   SupConLoss  NaN Inf")
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss


class ContrastiveReIDLoss(nn.Module):
    """       ReID     -          """

    def __init__(self, feature_dim: int, num_classes: int):
        super(ContrastiveReIDLoss, self).__init__()

        #
        self.triplet_loss = nn.TripletMarginLoss(
            margin=0.6,  #
            p=2,
            reduction='mean'
        )

        #
        self.supcon_loss = SupConLoss(temperature=0.07)  #

        #
        self.combined_loss = CombinedLoss(
            in_features=feature_dim,
            num_classes=num_classes,
            arcface_scale=48.0,  #
            arcface_margin=0.4,  #
            triplet_margin=0.6,  #
            arcface_weight=0.8,  #
            triplet_weight=1.5,  #
            use_hard_mining=True
        )

        #
        self.supcon_weight = 2.0      #
        self.classification_weight = 0.8  #

    def forward(self, features: torch.Tensor, cls_scores: torch.Tensor, labels: torch.Tensor) -> dict:
        device = features.device

        #
        if torch.isnan(features).any():
            print("   ContrastiveReIDLoss      NaN")
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                'total_loss': zero_loss,
                'supcon_loss': zero_loss,
                'classification_loss': zero_loss,
                'arcface_loss': zero_loss,
                'triplet_loss': zero_loss
            }

        #
        try:
            supcon_loss = self.supcon_loss(features, labels)
            if torch.isnan(supcon_loss):
                print("   SupCon   NaN   0  ")
                supcon_loss = torch.tensor(0.0, device=device, requires_grad=True)
        except Exception as e:
            print(f"   SupCon      : {e}")
            supcon_loss = torch.tensor(0.0, device=device, requires_grad=True)

        #
        try:
            classification_loss_dict = self.combined_loss(features, cls_scores, labels)
            classification_loss = classification_loss_dict['total_loss']
            if torch.isnan(classification_loss):
                print("        NaN   0  ")
                classification_loss = torch.tensor(0.0, device=device, requires_grad=True)
                classification_loss_dict = {
                    'total_loss': classification_loss,
                    'arcface_loss': torch.tensor(0.0, device=device),
                    'triplet_loss': torch.tensor(0.0, device=device)
                }
        except Exception as e:
            print(f"           : {e}")
            classification_loss = torch.tensor(0.0, device=device, requires_grad=True)
            classification_loss_dict = {
                'total_loss': classification_loss,
                'arcface_loss': torch.tensor(0.0, device=device),
                'triplet_loss': torch.tensor(0.0, device=device)
            }

        #
        total_loss = (self.supcon_weight * supcon_loss +
                     self.classification_weight * classification_loss)

        #
        if torch.isnan(total_loss):
            print("       NaN   0  ")
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return {
            'total_loss': total_loss,
            'supcon_loss': supcon_loss,
            'classification_loss': classification_loss,
            'arcface_loss': classification_loss_dict.get('arcface_loss', torch.tensor(0.0, device=device)),
            'triplet_loss': classification_loss_dict.get('triplet_loss', torch.tensor(0.0, device=device))
        }


if __name__ == "__main__":
    #
    batch_size = 8
    feature_dim = 512
    num_classes = 42
    
    #
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    #
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
