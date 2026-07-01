# --------------------------------------------------------
#
#
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_transformer_v2 import SwinTransformerV2
import os

try:
    import timm
except Exception:
    timm = None


def apply_gradient_ratio(x: torch.Tensor, grad_ratio: float) -> torch.Tensor:
    """Keep forward value unchanged while scaling backward gradient."""
    ratio = float(max(0.0, min(1.0, grad_ratio)))
    if ratio >= 0.9999:
        return x
    if ratio <= 1e-6:
        return x.detach()
    return x.detach() + ratio * (x - x.detach())


class BNNeck(nn.Module):
    """
    BatchNorm Neck -           
     ReID    BNNeck             
    """
    
    def __init__(self, in_features: int, num_classes: int, neck_feat: str = 'after'):
        """
        Args:
            in_features:       
            num_classes:      
            neck_feat: 'before'   'after'     BN   BN    
        """
        super(BNNeck, self).__init__()
        
        self.neck_feat = neck_feat
        self.in_features = in_features
        self.num_classes = num_classes
        
        #
        self.bottleneck = nn.BatchNorm1d(in_features)
        self.bottleneck.bias.requires_grad_(False)  #
        
        #
        self.classifier = nn.Linear(in_features, num_classes, bias=False)
        
        #
        self.bottleneck.apply(self._init_weights)
        self.classifier.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """   BN  /         """
        feat_before_bn = features
        feat_after_bn = self.bottleneck(features)
        cls_score = self.classifier(feat_after_bn)
        return feat_before_bn, feat_after_bn, cls_score


class PandaReIDModel(nn.Module):
    """
             
      Swin Transformer V2      + BNNeck + ID  
    """
    
    def __init__(
        self,
        num_classes: int,
        img_size: int = 192,
        embed_dim: int = 192,
        depths: list = [2, 2, 18, 2],
        num_heads: list = [6, 12, 24, 48],
        window_size: int = 12,
        pretrained_window_sizes: list = [12, 12, 12, 12],
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        neck_feat: str = 'after',
        proj_dim: int = 0,
        proj_drop: float = 0.0,
        aux_detach: bool = False,
        aux_grad_ratio: float = 1.0,
        pretrained_path: str = None
    ):
        """
        Args:
            num_classes: ID    
            img_size:       
            embed_dim:     
            depths:     
            num_heads:        
            window_size:     
            pretrained_window_sizes:        
            drop_path_rate: DropPath 
            neck_feat: BNNeck    
            pretrained_path:        
        """
        super(PandaReIDModel, self).__init__()
        
        self.num_classes = num_classes
        self.neck_feat = neck_feat
        
        self.aux_detach = bool(aux_detach)
        self.aux_grad_ratio = float(aux_grad_ratio)

        # Swin Transformer V2 backbone
        self.backbone = SwinTransformerV2(
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            num_classes=0,  #
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=float(drop_rate),
            attn_drop_rate=float(attn_drop_rate),
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            pretrained_window_sizes=pretrained_window_sizes
        )
        
        #
        self.backbone_dim = int(self.backbone.num_features)

        #
        self.proj = None
        proj_dim = int(proj_dim) if proj_dim is not None else 0
        proj_drop = float(proj_drop) if proj_drop is not None else 0.0
        if proj_dim > 0 and proj_dim != self.backbone_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.backbone_dim, proj_dim),
                nn.BatchNorm1d(proj_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=proj_drop),
            )
            self.feature_dim = proj_dim
        else:
            self.feature_dim = self.backbone_dim

        #
        self.neck = BNNeck(self.feature_dim, num_classes, neck_feat)

        #
        #
        self.gender_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

        self.age_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        #
        self._init_auxiliary_heads()

        #
        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def _init_auxiliary_heads(self):
        """         """
        for module in [self.gender_head, self.age_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def load_pretrained(self, pretrained_path: str):
        """       """
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            #
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            #
            backbone_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('head.'):
                    continue  #
                backbone_state_dict[k] = v

            #
            missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)

            print(f"Pretrained model loaded: {pretrained_path}")
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")

        except Exception as e:
            print(f"Failed to load pretrained model: {e}")

    def forward(self, x, return_features=False):
        """
            

        Args:
            x:      [B, 3, H, W]
            return_features:         

        Returns:
              return_features=False:
                global_feat:      [B, feature_dim]
                cls_score:      [B, num_classes]
              return_features=True:
                    backbone  
        """
        #
        backbone_feat = self.backbone.forward_features(x)  # [B, backbone_dim]

        #
        embed_feat = self.proj(backbone_feat) if self.proj is not None else backbone_feat

        #
        feat_before_bn, feat_after_bn, cls_score = self.neck(embed_feat)

        #
        global_feat = feat_after_bn if self.neck_feat == 'after' else feat_before_bn

        if return_features:
            return global_feat, cls_score, backbone_feat
        else:
            return global_feat, cls_score

    def forward_multitask(self, x):
        """
                     
        -    ReID     (   BN  ) + ReID     (BN  ) +   /     
        -    AUX_DETACH            ReID   /        

        Returns:
            feat_after_bn:  [B, D] ReID     BN  
            feat_before_bn: [B, D] ReID     BN      Triplet 
            gender_logits:  [B, 2]
            age_pred:       [B]
        """
        backbone_feat = self.backbone.forward_features(x)
        embed_feat = self.proj(backbone_feat) if self.proj is not None else backbone_feat
        feat_before_bn, feat_after_bn, _cls_score = self.neck(embed_feat)

        if self.aux_detach:
            aux_feat = feat_after_bn.detach()
        else:
            aux_feat = apply_gradient_ratio(feat_after_bn, self.aux_grad_ratio)
        gender_logits = self.gender_head(aux_feat)
        age_pred = self.age_head(aux_feat).squeeze(1)

        return feat_after_bn, feat_before_bn, gender_logits, age_pred
    
    def extract_features(self, x):
        """
                  
        
        Args:
            x:      [B, 3, H, W]
            
        Returns:
            features:          [B, feature_dim]
        """
        with torch.no_grad():
            global_feat, _ = self.forward(x)
            #
            features = F.normalize(global_feat, p=2, dim=1)
            return features


class GeMPool2d(nn.Module):
    """Generalized mean pooling for more discriminative global descriptors."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super(GeMPool2d, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(self.p, min=1.0, max=8.0)
        x = torch.clamp(x, min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / p).flatten(1)


class PartAwarePoolingHead(nn.Module):
    """Fuse global pooling with horizontal part pooling for open-world ReID."""

    def __init__(self, in_channels: int, num_parts: int = 3, pool_type: str = 'avgmax'):
        super(PartAwarePoolingHead, self).__init__()
        self.in_channels = int(in_channels)
        self.num_parts = max(1, int(num_parts))
        self.pool_type = str(pool_type).lower()
        self.gem_pool = GeMPool2d(p=3.0)
        self.part_norm = nn.LayerNorm(self.in_channels)
        self.part_gate = nn.Sequential(
            nn.LayerNorm(self.in_channels),
            nn.Linear(self.in_channels, 1),
        )

    @property
    def output_dim(self) -> int:
        if self.num_parts <= 1:
            return self.in_channels
        return self.in_channels * 3

    def _pool_global(self, feat_map: torch.Tensor) -> torch.Tensor:
        if self.pool_type == 'gem':
            return self.gem_pool(feat_map)
        avg_feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        if self.pool_type == 'avg':
            return avg_feat
        max_feat = F.adaptive_max_pool2d(feat_map, 1).flatten(1)
        return 0.5 * (avg_feat + max_feat)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        feat_map = F.gelu(feat_map)
        global_feat = self._pool_global(feat_map)
        if self.num_parts <= 1:
            return global_feat

        part_tokens = F.adaptive_avg_pool2d(feat_map, (self.num_parts, 1)).squeeze(-1).transpose(1, 2).contiguous()
        part_tokens = self.part_norm(part_tokens)
        part_weights = torch.softmax(self.part_gate(part_tokens).squeeze(-1), dim=1).unsqueeze(-1)
        part_mean = torch.sum(part_tokens * part_weights, dim=1)
        part_max = torch.max(part_tokens, dim=1).values
        return torch.cat([global_feat, part_mean, part_max], dim=1)


class TimmPandaReIDModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = 'convnextv2_base.fcmae_ft_in22k_in1k',
        backbone_pretrained: bool = True,
        backbone_weights: str = '',
        neck_feat: str = 'after',
        proj_dim: int = 0,
        proj_drop: float = 0.0,
        aux_detach: bool = False,
        aux_grad_ratio: float = 1.0,
        pool_type: str = 'avgmax',
        part_pool_enable: bool = False,
        part_num_parts: int = 3,
    ):
        super(TimmPandaReIDModel, self).__init__()
        if timm is None:
            raise RuntimeError(
                'timm is not installed, cannot use TimmPandaReIDModel. '
                'Please install timm or switch MODEL.TYPE back to swinv2.'
            )

        self.num_classes = num_classes
        self.neck_feat = neck_feat
        self.aux_detach = bool(aux_detach)
        self.aux_grad_ratio = float(aux_grad_ratio)
        self.backbone_name = str(backbone_name)
        self.backbone_weights = str(backbone_weights or '').strip()
        self.pool_type = str(pool_type).lower()
        self.part_pool_enable = bool(part_pool_enable)
        self.part_num_parts = max(1, int(part_num_parts))

        use_local_weights = bool(self.backbone_weights) and os.path.exists(self.backbone_weights)
        if use_local_weights:
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=False,
                num_classes=0,
                global_pool='',
            )
            self._load_local_backbone_weights(self.backbone_weights)
        else:
            try:
                self.backbone = timm.create_model(
                    self.backbone_name,
                    pretrained=bool(backbone_pretrained),
                    num_classes=0,
                    global_pool='',
                )
            except Exception as e:
                if bool(backbone_pretrained):
                    print(
                        f"[WARN] Failed to load pretrained timm backbone '{self.backbone_name}': {e}\n"
                        '       Fallback to pretrained=False.'
                    )
                    self.backbone = timm.create_model(
                        self.backbone_name,
                        pretrained=False,
                        num_classes=0,
                        global_pool='',
                    )
                else:
                    raise

        self.backbone_dim = int(getattr(self.backbone, 'num_features', 0))
        if self.backbone_dim <= 0:
            raise RuntimeError(f'Failed to infer num_features for backbone: {self.backbone_name}')

        self.pooling_head = PartAwarePoolingHead(
            in_channels=self.backbone_dim,
            num_parts=self.part_num_parts if self.part_pool_enable else 1,
            pool_type=self.pool_type,
        )
        pooled_dim = int(self.pooling_head.output_dim)

        self.proj = None
        proj_dim = int(proj_dim) if proj_dim is not None else 0
        proj_drop = float(proj_drop) if proj_drop is not None else 0.0
        if proj_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(pooled_dim, proj_dim),
                nn.BatchNorm1d(proj_dim),
                nn.GELU(),
                nn.Dropout(p=proj_drop),
            )
            self.feature_dim = proj_dim
        elif pooled_dim != self.backbone_dim:
            self.proj = nn.Sequential(
                nn.Linear(pooled_dim, self.backbone_dim),
                nn.BatchNorm1d(self.backbone_dim),
                nn.GELU(),
                nn.Dropout(p=proj_drop),
            )
            self.feature_dim = self.backbone_dim
        else:
            self.feature_dim = self.backbone_dim

        self.neck = BNNeck(self.feature_dim, num_classes, neck_feat)

        self.gender_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

        self.age_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

        self._init_auxiliary_heads()

    def _load_local_backbone_weights(self, weight_path: str):
        state = torch.load(weight_path, map_location='cpu')
        if isinstance(state, dict):
            if 'model' in state and isinstance(state['model'], dict):
                state = state['model']
            elif 'state_dict' in state and isinstance(state['state_dict'], dict):
                state = state['state_dict']
        if not isinstance(state, dict):
            raise RuntimeError(f'Unsupported local backbone weight format: {weight_path}')

        model_state = self.backbone.state_dict()
        filtered = {}
        skipped = []
        for k, v in state.items():
            nk = str(k)
            if nk.startswith('module.'):
                nk = nk[len('module.'):]
            if nk in model_state and tuple(model_state[nk].shape) == tuple(v.shape):
                filtered[nk] = v
            else:
                skipped.append(nk)

        missing, unexpected = self.backbone.load_state_dict(filtered, strict=False)
        print(
            f'[INFO] Loaded local timm backbone weights: {weight_path}\n'
            f'       loaded={len(filtered)}, skipped={len(skipped)}, '
            f'missing={len(missing)}, unexpected={len(unexpected)}'
        )

    def _init_auxiliary_heads(self):
        for module in [self.gender_head, self.age_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _forward_backbone_map(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x) if hasattr(self.backbone, 'forward_features') else self.backbone(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]
        if feat.dim() == 4 and feat.shape[1] != self.backbone_dim and feat.shape[-1] == self.backbone_dim:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        elif feat.dim() == 3:
            n, l, c = feat.shape
            side = int(round(l ** 0.5))
            if side * side == l:
                feat = feat.transpose(1, 2).reshape(n, c, side, side).contiguous()
            else:
                feat = feat.transpose(1, 2).unsqueeze(-1).contiguous()
        elif feat.dim() == 2:
            feat = feat.unsqueeze(-1).unsqueeze(-1)
        return feat

    def _forward_backbone(self, x):
        feat_map = self._forward_backbone_map(x)
        fused_feat = self.pooling_head(feat_map)
        if self.proj is not None:
            fused_feat = self.proj(fused_feat)
        return fused_feat, feat_map

    def forward(self, x, return_features=False):
        backbone_feat, feat_map = self._forward_backbone(x)
        feat_before_bn, feat_after_bn, cls_score = self.neck(backbone_feat)
        global_feat = feat_after_bn if self.neck_feat == 'after' else feat_before_bn

        if return_features:
            return global_feat, cls_score, feat_map
        return global_feat, cls_score

    def forward_multitask(self, x):
        backbone_feat, _feat_map = self._forward_backbone(x)
        feat_before_bn, feat_after_bn, _cls_score = self.neck(backbone_feat)

        if self.aux_detach:
            aux_feat = feat_after_bn.detach()
        else:
            aux_feat = apply_gradient_ratio(feat_after_bn, self.aux_grad_ratio)
        gender_logits = self.gender_head(aux_feat)
        age_pred = self.age_head(aux_feat).squeeze(1)
        return feat_after_bn, feat_before_bn, gender_logits, age_pred

    def extract_features(self, x):
        with torch.no_grad():
            global_feat, _ = self.forward(x)
            return F.normalize(global_feat, p=2, dim=1)


class SimilarityMatchingNetwork(nn.Module):
    """
               
                            
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 512):
        """
        Args:
            feature_dim:       
            hidden_dim:      
        """
        super(SimilarityMatchingNetwork, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        #
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        #
        #
        #
        self.similarity_net = nn.Sequential(
            nn.Linear(feature_dim * 4, hidden_dim),  #
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
            #
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, feat1, feat2):
        """
                    
        
        Args:
            feat1:   1 [B, feature_dim]
            feat2:   2 [B, feature_dim]
            
        Returns:
            similarity:       [B, 1]
        """
        #
        feat1_trans = self.feature_transform(feat1)
        feat2_trans = self.feature_transform(feat2)
        
        #
        feat1_norm = F.normalize(feat1_trans, p=2, dim=1)
        feat2_norm = F.normalize(feat2_trans, p=2, dim=1)
        
        #
        concat_feat = torch.cat([feat1_norm, feat2_norm], dim=1)  #
        product_feat = feat1_norm * feat2_norm  #
        diff_feat = torch.abs(feat1_norm - feat2_norm)  #
        
        #
        combined_feat = torch.cat([concat_feat, product_feat, diff_feat], dim=1)
        
        #
        similarity = self.similarity_net(combined_feat)
        
        return similarity


def build_panda_reid_model(config, num_classes: int):
    """
         ReID  
    
    Args:
        config:     
        num_classes: ID    
        
    Returns:
        model: PandaReIDModel  
    """
    model_type = str(getattr(config.MODEL, "TYPE", "swinv2")).lower()

    if model_type in {"timm", "convnext", "resnet", "efficientnet"}:
        backbone_name = str(
            getattr(config.MODEL, "BACKBONE_NAME", "convnextv2_base.fcmae_ft_in22k_in1k")
        )
        backbone_pretrained = bool(getattr(config.MODEL, "BACKBONE_PRETRAINED", True))
        backbone_weights = str(getattr(config.MODEL, "BACKBONE_WEIGHTS", "") or "")
        model = TimmPandaReIDModel(
            num_classes=num_classes,
            backbone_name=backbone_name,
            backbone_pretrained=backbone_pretrained,
            backbone_weights=backbone_weights,
            neck_feat=getattr(config.MODEL, "NECK_FEAT", "after"),
            proj_dim=int(getattr(config.MODEL, "PROJ_DIM", 0) or 0),
            proj_drop=float(getattr(config.MODEL, "PROJ_DROP", getattr(config.MODEL, "DROP_RATE", 0.0))),
            aux_detach=bool(getattr(config.MODEL, "AUX_DETACH", False)),
            aux_grad_ratio=float(getattr(config.MODEL, "AUX_GRAD_RATIO", 1.0)),
            pool_type=str(getattr(config.MODEL, "POOL_TYPE", "avgmax")),
            part_pool_enable=bool(getattr(config.MODEL, "PART_POOL_ENABLE", False)),
            part_num_parts=int(getattr(config.MODEL, "PART_NUM_PARTS", 3) or 3),
        )
        return model

    if model_type in {"resnet50", "resnet_official", "osnet", "transreid", "mege", "metagraph"}:
        from .external_reid_models import build_external_reid_model

        return build_external_reid_model(config, num_classes)

    model = PandaReIDModel(
        num_classes=num_classes,
        img_size=config.DATA.IMG_SIZE,
        embed_dim=config.MODEL.SWINV2.EMBED_DIM,
        depths=config.MODEL.SWINV2.DEPTHS,
        num_heads=config.MODEL.SWINV2.NUM_HEADS,
        window_size=config.MODEL.SWINV2.WINDOW_SIZE,
        pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES,
        drop_rate=float(getattr(config.MODEL, "DROP_RATE", 0.0)),
        attn_drop_rate=float(getattr(config.MODEL, "ATTN_DROP_RATE", getattr(config.MODEL, "DROP_RATE", 0.0))),
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        neck_feat=getattr(config.MODEL, 'NECK_FEAT', 'after'),
        proj_dim=int(getattr(config.MODEL, "PROJ_DIM", 0) or 0),
        proj_drop=float(getattr(config.MODEL, "PROJ_DROP", getattr(config.MODEL, "DROP_RATE", 0.0))),
        aux_detach=bool(getattr(config.MODEL, "AUX_DETACH", False)),
        aux_grad_ratio=float(getattr(config.MODEL, "AUX_GRAD_RATIO", 1.0)),
        pretrained_path=config.MODEL.PRETRAINED if hasattr(config.MODEL, 'PRETRAINED') else None
    )
    return model


if __name__ == "__main__":
    #
    model = PandaReIDModel(num_classes=42, img_size=192)
    
    #
    x = torch.randn(2, 3, 192, 192)
    
    #
    global_feat, cls_score = model(x)
    print(f"Global feature shape: {global_feat.shape}")
    print(f"Classification score shape: {cls_score.shape}")

    # Test feature extraction
    features = model.extract_features(x)
    print(f"Extracted feature shape: {features.shape}")

    # Test similarity matching network
    sim_net = SimilarityMatchingNetwork(feature_dim=1536)
    feat1 = torch.randn(2, 1536)
    feat2 = torch.randn(2, 1536)
    similarity = sim_net(feat1, feat2)
    print(f"Similarity score shape: {similarity.shape}")
