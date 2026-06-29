import os 
import sys 
import cv2 
import argparse 
import atexit 
import time 
from collections import defaultdict ,deque 
import numpy as np 
import torch 
import torch .nn .functional as F 
from PIL import Image 
from torchvision import transforms 
import shutil 
from typing import Optional 

_THIS_DIR =os .path .dirname (os .path .abspath (__file__ ))
sys .path .append (_THIS_DIR )

from inference_modules .yolo26_backend import (
bootstrap_local_yolo26_runtime ,
build_yolo_detector as _build_yolo26_detector_impl ,
resolve_detector_runtime_paths as _resolve_yolo26_runtime_paths_impl ,
resolve_existing_path as _resolve_existing_path_impl ,
)

_YOLO26_RUNTIME =bootstrap_local_yolo26_runtime (_THIS_DIR )
_YOLO26_IMPORT_ROOT =_YOLO26_RUNTIME .vendor_root 
_YOLO26_LOCAL_VENDOR =_YOLO26_RUNTIME .vendor_root 
_YOLO26_LOCAL_WEIGHT =_YOLO26_RUNTIME .default_weight 
_YOLO26_LOCAL_TRACKER =_YOLO26_RUNTIME .default_tracker 
_ULTRALYTICS_VERSION =_YOLO26_RUNTIME .ultralytics_version 
_ULTRALYTICS_FILE =_YOLO26_RUNTIME .ultralytics_file 

from inference_modules .panda_reid_inference import PandaReIDInference 
from models .panda_reid_model import build_panda_reid_model 
from train_age_gender_specialist import AgeGenderSpecialist 
# SAM (Segment Anything) 
try :
    from segment_anything import sam_model_registry ,SamPredictor  
    _HAS_SAM =True 
except Exception :
    _HAS_SAM =False 

from analyze_population import build_infer_args_for_inference 
from config import get_config 
from inference_modules.wild_folder_metrics import save_folder_level_metrics 
from inference_modules.prototype_strategies import (
IdentityPrototypeBank ,
cluster_embeddings_multiproto ,
)

# ===== User-tunable defaults =====
USER_DEFAULTS ={
# Input / Output
'input':r"H:\自制数据集\video",
'output':r"E:\pythonProjrct\panda\Swin-Transformer-main-reid20260104SAMVideo\output\infer_openworld_best",

# Best available deployment in current workspace (final default after 2026-03-12 comparison):
# - Video ReID: output_transfer_oldstrong_trainonly_10ep_aligned/prototype_best_model.pth
# - Image ReID (macro-best across testsets 1..5): output_transfer_oldstrong_trainonly_10ep_aligned/prototype_best_model.pth
# - Image fallback candidate (better on some count-sensitive sets): prototype_epoch_004.pth
# - Aux(age/gender): output_Model_20260102/prototype_best_model.pth
'cfg':r"E:\pythonProjrct\panda\Swin-Transformer-main-reid20260104SAMVideo\configs\panda_reid_arcface_triplet.yaml",
'model_path':r"E:\pythonProjrct\panda\Swin-Transformer-main-reid20260104SAMVideo\output\model\output_transfer_oldstrong_trainonly_10ep_aligned\swinv2_large_patch4_window12_192_panda_reid_arcface_triplet\arcface_triplet_enhanced_training\prototype_best_model.pth",
'video_model_path':r"E:\pythonProjrct\panda\Swin-Transformer-main-reid20260104SAMVideo\output\model\output_transfer_oldstrong_trainonly_10ep_aligned\swinv2_large_patch4_window12_192_panda_reid_arcface_triplet\arcface_triplet_enhanced_training\prototype_best_model.pth",
'image_model_path':r"E:\pythonProjrct\panda\Swin-Transformer-main-reid20260104SAMVideo\output\model\output_transfer_oldstrong_trainonly_10ep_aligned\swinv2_large_patch4_window12_192_panda_reid_arcface_triplet\arcface_triplet_enhanced_training\prototype_best_model.pth",
'aux_model_path':r"E:\pythonProjrct\panda\Swin-Transformer-main-reid20260104SAMVideo\output\model\output_Model_20260102\swinv2_large_patch4_window12_192_panda_reid_arcface_triplet\arcface_triplet_enhanced_training\prototype_best_model.pth",

# Matching
'mode':'custom',
'similarity_threshold':0.50 ,
'base_threshold':0.50 ,
'adaptive_threshold_min':0.42 ,
'adaptive_threshold_max':0.72 ,
'confidence_threshold':0.55 ,
'quality_threshold':0.30 ,

# Age/Gender
'gender_threshold':0.5 ,
'gender_hysteresis':0.03 ,
'age_scope':'global',
'age_display':'median',
'sim_cosine_w':0.7 ,
'sim_euclid_w':0.3 ,
'aux_gender_penalty':0.15 ,
'aux_age_reweight':0.10 ,
'aux_min_age_sigma':2.0 ,
'aux_feature_fuse_weight':0.2 ,
'prototype_strategy':'single' ,
'prototype_max_slots':4 ,
'prototype_spawn_similarity':0.68 ,
'prototype_update_similarity':0.58 ,
'prototype_topk':2 ,
'prototype_aggregate_weight':0.75 ,

# Detection / tracking
'yolo_repo_root':_YOLO26_LOCAL_VENDOR ,
'det_model':_YOLO26_LOCAL_WEIGHT ,
'det_conf':0.55 ,
'det_iou':0.45 ,
'use_sam':True ,
'tracker':_YOLO26_LOCAL_TRACKER ,
'roi_vis_use_detector':True ,
'roi_vis_det_conf':0.35 ,
'roi_vis_det_iou':0.45 ,
'sec_interval':1.0 ,
'vid_stride':1 ,
'max_keep_missing_sec':5.0 ,

# Runtime
'verbose':False ,
'filter_low_quality':True ,
'min_frame_brightness':20.0 ,
'min_bbox_area_ratio':0.01 ,
'max_bbox_area_ratio':0.75 ,
'bbox_border_ratio':0.0 ,
'min_mask_fill_ratio':0.20 ,
'min_blur_var':10 ,
'min_brightness':0.0 ,
'track_bad_ratio_threshold':0.8 ,
'min_track_obs':5 ,
'min_track_good':2 ,
'update_existing_min_id_conf':0.55 ,
'shuffle_images':True ,
'shuffle_seed':42 ,
'image_cluster_mode':'twopass',
'cluster_method':'agglomerative',
'cluster_agglomerative_max_n':5000 ,
'cluster_momentum':0.90 ,
'max_rois_per_image':8 ,
'twopass_threshold':0.22 ,
'twopass_threshold_auto':True ,
'twopass_fuse_full_image':True ,
'twopass_full_image_weight':0.35 ,
'twopass_cluster_aux_reid_weight':0.30 ,
'twopass_cluster_aux_gender_weight':0.10 ,
'twopass_cluster_aux_age_weight':0.05 ,
'twopass_auto_merge':False ,
'twopass_target_cluster_size':26.0 ,
'twopass_merge_min_sim':0.50 ,
'twopass_refine_split':False ,
'twopass_refine_min_size':18 ,
'twopass_refine_delta':0.08 ,
'twopass_refine_min_subcluster':4 ,
'twopass_centroid_merge':True ,
'twopass_centroid_image_only':True ,
'twopass_centroid_min_fragments':128 ,
'twopass_centroid_method':'agglomerative',
'twopass_centroid_threshold_auto':True ,
'twopass_centroid_threshold':0.18 ,
'twopass_centroid_fragment_group_size':16.0 ,
'twopass_centroid_auto_merge':False ,
'twopass_centroid_merge_min_sim':0.40 ,

# Conservative post-merge to reduce over-splitting without collapsing different IDs.
'post_merge_enable':True ,
'post_merge_enable_video':True ,
'post_merge_enable_image':False ,
'post_merge_video_min_sim':0.41 ,
'post_merge_image_min_sim':0.56 ,
'post_merge_mixed_min_sim':0.52 ,
'post_merge_gender_conf_guard':0.80 ,
'post_merge_age_gap_soft':3.0 ,
'post_merge_age_gap_hard':6.0 ,
'post_merge_fragment_relax':0.04 ,
'post_merge_tiny_image_max_size':3 ,
'post_merge_tiny_image_min_sim':0.44 ,

# Output layout
'save_vis':True ,
'group_by_id':True ,
'image_as_roi':0 ,
}
# ===============================================

# Visualization style
_VIS_LABEL_BG_BGR =(191 ,222 ,240 )# #F0DEBF
_VIS_TEXT_BGR =(42 ,42 ,178 )# #B22A2A
_VIS_BOX_BGR =_VIS_TEXT_BGR
_VIS_TEXT_SCALE_GAIN =1.22

# Path/folder priors are fully disabled in this script.
_ID_META ={}


def _is_canonical_id_name (name :str )->bool :
    """Check if name is canonical ID format like ID1/ID2."""
    if not name or not isinstance (name ,str ):
        return False 
    if not name .startswith ('ID'):
        return False 
    return name [2 :].isdigit ()


def _first_folder_of_rel_path (rel_path :Optional [str ])->Optional [str ]:
    """Return first-level folder name from a relative path."""
    if not rel_path :
        return None 
    s =str (rel_path ).replace ('/',os .sep ).replace ('\\',os .sep )
    parts =[p for p in s .split (os .sep )if p ]
    return parts [0 ]if parts else None 


def _candidate_ids_for_rel_path (rel_path :Optional [str ]):
    """Disabled: do not derive candidate IDs from input path."""
    return None 


def _ensure_display_id (global_to_display :dict ,gid :str ,next_display_id :int )->int :
    """Ensure gid has a display ID; return updated next_display_id."""
    if gid in global_to_display :
        return next_display_id 
    if _is_canonical_id_name (gid ):
        global_to_display [gid ]=gid 
        
        try :
            next_display_id =max (int (next_display_id ),int (gid [2 :])+1 )
        except Exception :
            pass 
        return next_display_id 
    global_to_display [gid ]=f"ID{next_display_id}"
    return next_display_id +1 


def _candidate_score (gid :str ,male_p :float ,age_p :float ,sim :Optional [float ]=None )->float :
    """Compute candidate score from metadata and optional similarity."""
    meta =_ID_META .get (gid )or {}
    gender =meta .get ('gender')
    age_gt =meta .get ('age')
    if gender =='M':
        gender_score =float (male_p )
    elif gender =='F':
        gender_score =float (1.0 -male_p )
    else :
        gender_score =0.5 

    age_score =0.0 
    if age_gt is not None :
        try :
            age_score =1.0 /(1.0 +abs (float (age_p )-float (age_gt )))
        except Exception :
            age_score =0.0 

    if sim is None :
    # 
        return 1.0 *gender_score +0.5 *age_score 

        
    try :
        sim01 =float ((float (sim )+1.0 )*0.5 )
    except Exception :
        sim01 =0.5 
        
    return 100.0 *sim01 +1.0 *gender_score +0.5 *age_score 


def _clamp01 (v ,default =0.0 )->float :
    try :
        x =float (v )
    except Exception :
        x =float (default )
    if x <0.0 :
        return 0.0 
    if x >1.0 :
        return 1.0 
    return x 


def _sim_to_conf01 (sim )->float :
    try :
        s =float (sim )
    except Exception :
        return 0.0 
    if 0.0 <=s <=1.0 :
        return s 
    return _clamp01 ((s +1.0 )*0.5 ,default =0.0 )


def _sigmoid01 (x ,temperature =1.0 )->float :
    """Numerically-stable sigmoid mapped to [0,1]."""
    t =max (float (temperature ),1e-6 )
    z =float (x )/t 
    if z >=0 :
        ez =np .exp (-z )
        return float (1.0 /(1.0 +ez ))
    ez =np .exp (z )
    return float (ez /(1.0 +ez ))


def _identity_confidence_01 (sim ,adaptive_th =None ,is_new =False )->float :
    """
    Unified ID confidence in [0, 1]:
    - closer to 1: more confident
    - closer to 0: less confident
    """
    sim01 =_sim_to_conf01 (sim )
    if adaptive_th is None :
        return sim01 
    th01 =_sim_to_conf01 (adaptive_th )
    margin =sim01 -th01 
    # Display-calibrated confidence:
    # margin==0 -> 0.5, larger positive/negative margins quickly approach 1.0.
    temp =0.08 
    if is_new :
        return _clamp01 (_sigmoid01 (-margin ,temperature =temp ),default =0.0 )
    return _clamp01 (_sigmoid01 (margin ,temperature =temp ),default =0.0 )


def _gender_confidence_01 (male_prob ,stable_gender =None ,threshold =0.5 )->float :
    """
    Unified gender confidence in [0, 1]:
    - closer to 1: more confident
    - closer to 0: less confident
    """
    mp =_clamp01 (male_prob ,default =0.5 )
    if stable_gender =='M':
        pred_prob =mp 
    elif stable_gender =='F':
        pred_prob =1.0 -mp 
    else :
        pred ='M'if mp >=float (threshold )else 'F'
        pred_prob =mp if pred =='M'else 1.0 -mp 
    # Display-calibrated confidence in [0,1] with 0.5 as decision boundary.
    return _clamp01 (_sigmoid01 (pred_prob -0.5 ,temperature =0.10 ),default =0.5 )


def _display_id_sort_key (display_id :str ):
    try :
        s =str (display_id )
        if _is_canonical_id_name (s ):
            return (0 ,int (s [2 :]))
    except Exception :
        pass 
    return (1 ,str (display_id ))


def _normalize_np_feature (feat )->Optional [np .ndarray ]:
    if feat is None :
        return None 
    try :
        arr =np .asarray (feat ,dtype =np .float32 ).reshape (-1 )
    except Exception :
        return None 
    if arr .size ==0 :
        return None 
    norm =float (np .linalg .norm (arr ))
    if norm <=1e-12 :
        return None 
    return arr /norm 


def _fused_similarity_np (feat_a ,feat_b ,w_cos =0.7 ,w_euc =0.3 )->float :
    a =_normalize_np_feature (feat_a )
    b =_normalize_np_feature (feat_b )
    if a is None or b is None :
        return -1.0 
    cosine_sim =float (np .dot (a ,b ))
    euclidean_dist =float (np .linalg .norm (a -b ))
    euclidean_sim =1.0 /(1.0 +euclidean_dist )
    denom =max (1e-6 ,float (w_cos )+float (w_euc ))
    w_cos =float (w_cos )/denom 
    w_euc =float (w_euc )/denom 
    return float (w_cos *cosine_sim +w_euc *euclidean_sim )


def _merge_grouped_output_folders (id_root :Optional [str ],src_disp_id :str ,dst_disp_id :str ):
    if not id_root :
        return 
    src_dir =os .path .join (id_root ,str (src_disp_id ))
    dst_dir =os .path .join (id_root ,str (dst_disp_id ))
    if not os .path .isdir (src_dir )or src_dir ==dst_dir :
        return 
    os .makedirs (dst_dir ,exist_ok =True )
    for item in os .listdir (src_dir ):
        src_path =os .path .join (src_dir ,item )
        dst_path =os .path .join (dst_dir ,item )
        if os .path .exists (dst_path ):
            base ,ext =os .path .splitext (item )
            dst_path =os .path .join (dst_dir ,f"{base}__merged_from_{src_disp_id}{ext}")
        shutil .move (src_path ,dst_path )
    try :
        if len (os .listdir (src_dir ))==0 :
            os .rmdir (src_dir )
    except Exception :
        pass 


def _identity_conflict_pairs_store (inference ):
    store =getattr (inference ,'_identity_conflict_pairs',None )
    if store is None :
        store =set ()
        setattr (inference ,'_identity_conflict_pairs',store )
    return store 


def _identity_gid_sources_store (inference ):
    store =getattr (inference ,'_identity_gid_sources',None )
    if store is None :
        store =defaultdict (set )
        setattr (inference ,'_identity_gid_sources',store )
    return store 


def _identity_gid_kinds_store (inference ):
    store =getattr (inference ,'_identity_gid_kinds',None )
    if store is None :
        store =defaultdict (set )
        setattr (inference ,'_identity_gid_kinds',store )
    return store 


def _identity_gid_features_store (inference ):
    store =getattr (inference ,'_identity_gid_features',None )
    if store is None :
        store ={}
        setattr (inference ,'_identity_gid_features',store )
    return store 


def _ensure_identity_bank (inference ,args ):
    strategy =str (getattr (args ,'prototype_strategy','single')or 'single').lower ()
    if strategy =='single':
        return None 
    bank =getattr (inference ,'identity_bank',None )
    needs_new =(
    bank is None 
    or str (getattr (bank ,'strategy','single')or 'single').lower ()!=strategy 
    )
    if needs_new :
        bank =IdentityPrototypeBank (
        strategy =strategy ,
        momentum =float (getattr (args ,'cluster_momentum',0.9 )or 0.9 ),
        max_slots =int (getattr (args ,'prototype_max_slots',4 )or 4 ),
        spawn_similarity =float (getattr (args ,'prototype_spawn_similarity',0.68 )or 0.68 ),
        update_similarity =float (getattr (args ,'prototype_update_similarity',0.58 )or 0.58 ),
        topk =int (getattr (args ,'prototype_topk',2 )or 2 ),
        aggregate_weight =float (getattr (args ,'prototype_aggregate_weight',0.75 )or 0.75 ),
        sim_cosine_w =float (getattr (args ,'sim_cosine_w',0.7 )or 0.7 ),
        sim_euclid_w =float (getattr (args ,'sim_euclid_w',0.3 )or 0.3 ),
        aux_gender_penalty =float (getattr (args ,'aux_gender_penalty',0.15 )or 0.15 ),
        aux_age_reweight =float (getattr (args ,'aux_age_reweight',0.10 )or 0.10 ),
        aux_min_age_sigma =float (getattr (args ,'aux_min_age_sigma',2.0 )or 2.0 ),
        )
        setattr (inference ,'identity_bank',bank )
    return bank 


def _prototype_quality_score (inference ,feature )->float :
    pn =getattr (inference ,'prototype_net',None )
    if pn is None :
        return 1.0 
    try :
        with torch .no_grad ():
            q =F .normalize (feature .float ().flatten (),p =2 ,dim =0 )
            return float (pn .quality_net (q .unsqueeze (0 )).item ())
    except Exception :
        return 1.0 


def _prototype_pair_confidence (inference ,query_feature ,proto_feature )->float :
    pn =getattr (inference ,'prototype_net',None )
    if pn is None or proto_feature is None :
        return 0.0 
    try :
        with torch .no_grad ():
            q =F .normalize (query_feature .float ().flatten (),p =2 ,dim =0 )
            p =F .normalize (proto_feature .to (q .device ).float ().flatten (),p =2 ,dim =0 )
            logits =pn .confidence_net (torch .cat ([q ,p ],dim =0 ).unsqueeze (0 )).squeeze (1 )
            return float (torch .sigmoid (logits ).item ())
    except Exception :
        return 0.0 


def _match_identity_with_strategy (inference ,feature ,*,aux =None ,reserved_ids =None ,candidate_ids =None ,allow_new =True ):
    args =getattr (inference ,'args',None )
    strategy =str (getattr (args ,'prototype_strategy','single')or 'single').lower ()if args is not None else 'single'
    q =F .normalize (feature .float ().flatten (),p =2 ,dim =0 )
    if strategy =='single':
        pn =getattr (inference ,'prototype_net',None )
        if pn is None :
            return {
            'predicted_id':None ,
            'similarity':0.0 ,
            'confidence':0.0 ,
            'is_new_id':True ,
            'adaptive_threshold':float (getattr (inference ,'similarity_threshold',0.5 )or 0.5 ),
            'all_similarities':{} ,
            'quality_score':1.0 ,
            }
        if candidate_ids :
            return pn (q ,known_ids =[str (x )for x in candidate_ids if x ],aux =aux )
        return pn (q ,aux =aux )

    bank =_ensure_identity_bank (inference ,args )
    pn =getattr (inference ,'prototype_net',None )
    if pn is not None :
        try :
            pn .update_global_stats (q .unsqueeze (0 ))
        except Exception :
            pass 
    adaptive_th =float (getattr (inference ,'similarity_threshold',0.5 )or 0.5 )
    if pn is not None :
        try :
            adaptive_th =float (pn .compute_adaptive_threshold (q ))
        except Exception :
            pass 
    quality_score =_prototype_quality_score (inference ,q )
    match =None if bank is None else bank .match (
    q ,
    candidate_ids =candidate_ids ,
    reserved_ids =reserved_ids ,
    aux =aux ,
    )
    if match is None :
        if not allow_new :
            return {
            'predicted_id':None ,
            'similarity':0.0 ,
            'confidence':0.0 ,
            'is_new_id':True ,
            'adaptive_threshold':adaptive_th ,
            'all_similarities':{} ,
            'quality_score':quality_score ,
            }
        inference .wild_id_counter +=1 
        return {
        'predicted_id':f"Wild_Panda_{inference .wild_id_counter:03d}" ,
        'similarity':1.0 ,
        'confidence':quality_score ,
        'is_new_id':True ,
        'adaptive_threshold':adaptive_th ,
        'all_similarities':{} ,
        'quality_score':quality_score ,
        }

    best_id =str (match .get ('gid'))
    best_similarity =float (match .get ('score',0.0 )or 0.0 )
    second_best =float (match .get ('second_identity_score',-1.0 )or -1.0 )
    best_proto_feat =match .get ('best_feature',None )
    best_conf =_prototype_pair_confidence (inference ,q ,best_proto_feat )
    sim_margin =best_similarity -second_best if second_best >=-0.5 else 1.0 
    conf_th =float (getattr (args ,'confidence_threshold',0.0 )or 0.0 )if args is not None else 0.0 
    quality_th =float (getattr (args ,'quality_threshold',0.0 )or 0.0 )if args is not None else 0.0 
    ambiguous_margin =float (getattr (pn ,'ambiguous_margin',0.03 )or 0.03 )if pn is not None else 0.03 
    ambiguous_offset =float (getattr (pn ,'ambiguous_offset',0.02 )or 0.02 )if pn is not None else 0.02 
    similarity_gate =best_similarity <adaptive_th 
    confidence_gate =best_conf <conf_th 
    quality_gate =quality_score <quality_th 
    ambiguous_gate =(sim_margin <ambiguous_margin and best_similarity <adaptive_th +ambiguous_offset )
    is_new_id =bool (similarity_gate and (confidence_gate or quality_gate or ambiguous_gate ))
    if is_new_id and allow_new :
        inference .wild_id_counter +=1 
        predicted_id =f"Wild_Panda_{inference .wild_id_counter:03d}"
    elif is_new_id :
        predicted_id =None 
    else :
        predicted_id =best_id 
    return {
    'predicted_id':predicted_id ,
    'similarity':best_similarity ,
    'confidence':best_conf ,
    'is_new_id':bool (is_new_id ),
    'adaptive_threshold':adaptive_th ,
    'all_similarities':match .get ('all_similarities',{})or {} ,
    'quality_score':quality_score ,
    'debug_info':{
    'best_similarity':best_similarity ,
    'adaptive_threshold':adaptive_th ,
    'best_confidence':best_conf ,
    'confidence_threshold':conf_th ,
    'quality_score':quality_score ,
    'quality_threshold':quality_th ,
    'sim_margin':sim_margin ,
    'num_prototypes':0 if bank is None else len (bank .ids ()),
    },
    }


def _update_identity_memory (inference ,gid ,feature ,quality_score ,*,gender_prob =None ,age_pred =None ,id_conf =None ):
    if gid is None :
        return 
    q =F .normalize (feature .float ().flatten (),p =2 ,dim =0 )
    args =getattr (inference ,'args',None )
    bank =None if args is None else _ensure_identity_bank (inference ,args )
    if bank is not None :
        try :
            bank .update (
            str (gid ),
            q ,
            quality_score =quality_score ,
            gender_prob =gender_prob ,
            age_pred =age_pred ,
            id_conf =id_conf ,
            )
        except Exception :
            pass 
    pn =getattr (inference ,'prototype_net',None )
    if pn is not None :
        try :
            pn .update_prototype (
            str (gid ),
            q ,
            quality_score ,
            gender_prob =gender_prob ,
            age_pred =age_pred ,
            )
        except Exception :
            pass 
    feat_store =_identity_gid_features_store (inference )
    if bank is not None and bank .has_id (str (gid )):
        bank_feat =bank .get_feature (str (gid ))
        if bank_feat is not None :
            feat_store [str (gid )]=bank_feat .detach ().cpu ().numpy ()
            return 
    feat_store [str (gid )]=q .detach ().cpu ().numpy ()


def _register_gid_source (inference ,gid :str ,source_name :Optional [str ],source_kind :Optional [str ]):
    if gid is None :
        return 
    if source_name :
        _identity_gid_sources_store (inference )[str (gid )].add (str (source_name ))
    if source_kind :
        _identity_gid_kinds_store (inference )[str (gid )].add (str (source_kind ))


def _register_conflict_pairs_from_gids (inference ,gid_list ):
    gids =sorted ({str (gid )for gid in gid_list if gid })
    if len (gids )<2 :
        return 
    store =_identity_conflict_pairs_store (inference )
    for i in range (len (gids )):
        for j in range (i +1 ,len (gids )):
            a ,b =gids [i ],gids [j ]
            store .add ((a ,b )if a <b else (b ,a ))


def _merge_identity_stats (
root_gid :str ,
child_gid :str ,
*,
global_to_display :dict ,
global_gender_stats :dict ,
global_age_stats :dict ,
inference ,
):
    if root_gid ==child_gid :
        return 
    pn =getattr (inference ,'prototype_net',None )
    bank =getattr (inference ,'identity_bank',None )
    root_disp =global_to_display .get (root_gid ,root_gid )
    global_to_display [child_gid ]=root_disp 

    root_g =global_gender_stats .get (root_gid )
    child_g =global_gender_stats .pop (child_gid ,None )
    if child_g is not None :
        if root_g is None :
            global_gender_stats [root_gid ]={
            'alpha_male':float (child_g .get ('alpha_male',1.0 )or 1.0 ),
            'alpha_female':float (child_g .get ('alpha_female',1.0 )or 1.0 ),
            'stable_gender':child_g .get ('stable_gender'),
            }
        else :
            root_g ['alpha_male']=float (root_g .get ('alpha_male',1.0 )or 1.0 )+float (child_g .get ('alpha_male',1.0 )or 1.0 )
            root_g ['alpha_female']=float (root_g .get ('alpha_female',1.0 )or 1.0 )+float (child_g .get ('alpha_female',1.0 )or 1.0 )
            male_mean =float (root_g ['alpha_male'])/max (1e-6 ,float (root_g ['alpha_male'])+float (root_g ['alpha_female']))
            root_g ['stable_gender']='M'if male_mean >=0.5 else 'F'

    if child_gid in global_age_stats :
        root_age =global_age_stats .setdefault (root_gid ,[])
        root_age .extend (list (global_age_stats .pop (child_gid ,[])or []))

    feat_store =_identity_gid_features_store (inference )
    root_feat =feat_store .get (root_gid )
    child_feat =feat_store .get (child_gid )
    if child_feat is not None :
        if root_feat is None :
            feat_store [root_gid ]=child_feat 
        else :
            rc =0 
            cc =0 
            if pn is not None :
                rc =int (getattr (pn ,'prototype_counts',{}).get (root_gid ,0 )or 0 )
                cc =int (getattr (pn ,'prototype_counts',{}).get (child_gid ,0 )or 0 )
            rc =max (1 ,rc )
            cc =max (1 ,cc )
            merged =(root_feat *rc +child_feat *cc )/float (rc +cc )
            feat_store [root_gid ]=_normalize_np_feature (merged )
        feat_store .pop (child_gid ,None )

    src_store =_identity_gid_sources_store (inference )
    kind_store =_identity_gid_kinds_store (inference )
    if child_gid in src_store :
        src_store [root_gid ].update (src_store .pop (child_gid ,set ()))
    if child_gid in kind_store :
        kind_store [root_gid ].update (kind_store .pop (child_gid ,set ()))

    conflict_store =_identity_conflict_pairs_store (inference )
    if conflict_store :
        updated =set ()
        for a ,b in conflict_store :
            x =root_gid if a ==child_gid else a 
            y =root_gid if b ==child_gid else b 
            if x ==y :
                continue 
            updated .add ((x ,y )if x <y else (y ,x ))
        conflict_store .clear ()
        conflict_store .update (updated )

    if pn is not None :
        protos =getattr (pn ,'prototypes',{})
        counts =getattr (pn ,'prototype_counts',{})
        meta =getattr (pn ,'prototype_meta',{})
        if root_gid in protos and child_gid in protos :
            root_t =protos [root_gid ]
            child_t =protos [child_gid ]
            rc =max (1 ,int (counts .get (root_gid ,1 )or 1 ))
            cc =max (1 ,int (counts .get (child_gid ,1 )or 1 ))
            merged =(root_t .to (child_t .device )*rc +child_t *cc )/float (rc +cc )
            protos [root_gid ]=F .normalize (merged ,p =2 ,dim =0 )
            counts [root_gid ]=rc +cc 
            del protos [child_gid ]
            if child_gid in counts :
                del counts [child_gid ]
        elif child_gid in protos and root_gid not in protos :
            protos [root_gid ]=protos .pop (child_gid )
            counts [root_gid ]=int (counts .pop (child_gid ,1 )or 1 )
    if child_gid in meta :
        child_meta =meta .pop (child_gid )
        root_meta =meta .setdefault (root_gid ,{})
        if root_meta .get ('gender_prob')is None and child_meta .get ('gender_prob')is not None :
            root_meta ['gender_prob']=child_meta .get ('gender_prob')
        if root_meta .get ('age_mean')is None and child_meta .get ('age_mean')is not None :
            root_meta ['age_mean']=child_meta .get ('age_mean')
    if bank is not None :
        try :
            bank .merge (root_gid ,child_gid )
            bank_feat =bank .get_feature (root_gid )
            if bank_feat is not None :
                _identity_gid_features_store (inference )[root_gid ]=bank_feat .detach ().cpu ().numpy ()
        except Exception :
            pass 


def _consolidate_runtime_identities (
*,
inference ,
candidate_gids ,
global_to_display :dict ,
global_gender_stats :dict ,
global_age_stats :dict ,
args ,
gid_feature_map :Optional [dict ]=None ,
id_root :Optional [str ]=None ,
log_prefix :str ="post-merge",
):
    if not bool (getattr (args ,'post_merge_enable',True )):
        return {}
    gid_list =sorted ({str (gid )for gid in candidate_gids if gid is not None },key =str )
    if len (gid_list )<2 :
        return {}

    pn =getattr (inference ,'prototype_net',None )
    stored_features =_identity_gid_features_store (inference )
    for gid ,feat in (gid_feature_map or {}).items ():
        nf =_normalize_np_feature (feat )
        if nf is not None :
            stored_features [str (gid )]=nf 

    def _make_info (gid :str ):
        feat =None 
        if gid_feature_map is not None and gid in gid_feature_map :
            feat =gid_feature_map .get (gid )
        if feat is None and gid in stored_features :
            feat =stored_features .get (gid )
        if feat is None and pn is not None and gid in getattr (pn ,'prototypes',{}):
            try :
                feat =getattr (pn ,'prototypes',{})[gid ].detach ().cpu ().numpy ()
            except Exception :
                feat =None 
        feat =_normalize_np_feature (feat )

        gstat =global_gender_stats .get (gid ,{})
        a_male =float (gstat .get ('alpha_male',1.0 )or 1.0 )
        a_female =float (gstat .get ('alpha_female',1.0 )or 1.0 )
        total =a_male +a_female 
        male_mean =a_male /max (1e-6 ,total )
        gender ='M'if male_mean >=float (getattr (args ,'gender_threshold',0.5 )or 0.5 )else 'F'
        gender_conf =_gender_confidence_01 (male_mean ,stable_gender =gender ,threshold =getattr (args ,'gender_threshold',0.5 ))

        ages =[]
        for age in global_age_stats .get (gid ,[])or []:
            try :
                ages .append (float (age ))
            except Exception :
                pass 
        age_mean =float (np .median (ages ))if len (ages )>0 else None 

        count =len (ages )
        if pn is not None :
            count =max (count ,int (getattr (pn ,'prototype_counts',{}).get (gid ,0 )or 0 ))
        count =max (1 ,count )
        kinds =set (_identity_gid_kinds_store (inference ).get (gid ,set ()))
        if not kinds :
            kinds ={'image'}
        sources =set (_identity_gid_sources_store (inference ).get (gid ,set ()))
        return {
        'gid':gid ,
        'display_id':global_to_display .get (gid ,gid ),
        'feature':feat ,
        'count':count ,
        'alpha_male':a_male ,
        'alpha_female':a_female ,
        'gender':gender ,
        'gender_conf':gender_conf ,
        'age_mean':age_mean ,
        'ages':ages ,
        'kinds':kinds ,
        'sources':sources ,
        'members':{gid },
        }

    active ={gid :_make_info (gid )for gid in gid_list }
    conflicts =_identity_conflict_pairs_store (inference )
    near_miss_log =[]

    def _has_conflict (info_a ,info_b ):
        for ga in info_a ['members']:
            for gb in info_b ['members']:
                if ga ==gb :
                    continue 
                pair =(ga ,gb )if ga <gb else (gb ,ga )
                if pair in conflicts :
                    return True 
        return False 

    merge_log =[]
    while True :
        keys =sorted (active .keys (),key =lambda gid :_display_id_sort_key (active [gid ]['display_id']))
        best =None 
        for i in range (len (keys )):
            for j in range (i +1 ,len (keys )):
                gid_a =keys [i ]
                gid_b =keys [j ]
                info_a =active [gid_a ]
                info_b =active [gid_b ]
                if info_a ['feature']is None or info_b ['feature']is None :
                    continue 

                sim =_fused_similarity_np (
                info_a ['feature'],
                info_b ['feature'],
                w_cos =getattr (args ,'sim_cosine_w',0.7 ),
                w_euc =getattr (args ,'sim_euclid_w',0.3 ),
                )
                if sim <=0.0 :
                    continue 

                blocked_by_conflict =_has_conflict (info_a ,info_b )

                kinds =set (info_a ['kinds'])|set (info_b ['kinds'])
                if kinds =={'video'}:
                    th =float (getattr (args ,'post_merge_video_min_sim',0.41 )or 0.41 )
                elif kinds =={'image'}:
                    th =float (getattr (args ,'post_merge_image_min_sim',0.56 )or 0.56 )
                else :
                    th =float (getattr (args ,'post_merge_mixed_min_sim',0.52 )or 0.52 )

                small =min (int (info_a ['count']),int (info_b ['count']))
                big =max (int (info_a ['count']),int (info_b ['count']))
                relax =float (getattr (args ,'post_merge_fragment_relax',0.04 )or 0.04 )
                if big >=3 and small <=max (3 ,int (0.35 *big )):
                    th -=relax 
                if kinds =={'image'}:
                    tiny_max =int (getattr (args ,'post_merge_tiny_image_max_size',3 )or 3 )
                    tiny_min_sim =float (getattr (args ,'post_merge_tiny_image_min_sim',0.44 )or 0.44 )
                    if small <=tiny_max and big >=max (8 ,tiny_max *3 ):
                        th =min (th ,tiny_min_sim )

                if info_a ['gender']!=info_b ['gender']:
                    guard =float (getattr (args ,'post_merge_gender_conf_guard',0.80 )or 0.80 )
                    if info_a ['gender_conf']>=guard and info_b ['gender_conf']>=guard :
                        continue 
                    th +=0.03 

                age_a =info_a ['age_mean']
                age_b =info_b ['age_mean']
                if age_a is not None and age_b is not None :
                    age_gap =abs (float (age_a )-float (age_b ))
                    hard_gap =float (getattr (args ,'post_merge_age_gap_hard',6.0 )or 6.0 )
                    soft_gap =float (getattr (args ,'post_merge_age_gap_soft',3.0 )or 3.0 )
                    if age_gap >hard_gap :
                        continue 
                    if age_gap >soft_gap :
                        th +=0.03 

                near_miss_log .append ({
                'gid_a':gid_a ,
                'gid_b':gid_b ,
                'disp_a':info_a ['display_id'],
                'disp_b':info_b ['display_id'],
                'sim':float (sim ),
                'th':float (th ),
                'blocked_by_conflict':bool (blocked_by_conflict ),
                'count_a':int (info_a ['count']),
                'count_b':int (info_b ['count']),
                })
                if blocked_by_conflict :
                    continue 
                if sim >=th :
                    cand =(sim ,-th ,gid_a ,gid_b )
                    if best is None or cand >best :
                        best =cand 

        if best is None :
            break 

        sim ,neg_th ,gid_a ,gid_b =best 
        info_a =active [gid_a ]
        info_b =active [gid_b ]
        root_gid =gid_a if _display_id_sort_key (info_a ['display_id'])<=_display_id_sort_key (info_b ['display_id'])else gid_b 
        child_gid =gid_b if root_gid ==gid_a else gid_a 
        root =active [root_gid ]
        child =active [child_gid ]

        root_count =max (1 ,int (root ['count']))
        child_count =max (1 ,int (child ['count']))
        merged_feat =None 
        if root ['feature']is not None and child ['feature']is not None :
            merged_feat =(root ['feature']*root_count +child ['feature']*child_count )/float (root_count +child_count )
            merged_feat =_normalize_np_feature (merged_feat )

        root ['feature']=merged_feat if merged_feat is not None else (root ['feature']or child ['feature'])
        root ['count']=root_count +child_count 
        root ['alpha_male']=float (root ['alpha_male'])+float (child ['alpha_male'])
        root ['alpha_female']=float (root ['alpha_female'])+float (child ['alpha_female'])
        total =root ['alpha_male']+root ['alpha_female']
        male_mean =root ['alpha_male']/max (1e-6 ,total )
        root ['gender']='M'if male_mean >=float (getattr (args ,'gender_threshold',0.5 )or 0.5 )else 'F'
        root ['gender_conf']=_gender_confidence_01 (male_mean ,stable_gender =root ['gender'],threshold =getattr (args ,'gender_threshold',0.5 ))
        root ['ages'].extend (child ['ages'])
        root ['age_mean']=float (np .median (root ['ages']))if len (root ['ages'])>0 else None 
        root ['kinds'].update (child ['kinds'])
        root ['sources'].update (child ['sources'])
        root ['members'].update (child ['members'])
        active .pop (child_gid ,None )

        merge_log .append ({
        'root_gid':root_gid ,
        'child_gid':child_gid ,
        'root_display_id':global_to_display .get (root_gid ,root_gid ),
        'child_display_id':global_to_display .get (child_gid ,child_gid ),
        'similarity':float (sim ),
        'threshold':float (-neg_th ),
        })

    if not merge_log :
        if near_miss_log and bool (getattr (args ,'verbose',False )):
            print (f"[INFO] {log_prefix}: no merge accepted, top candidate pairs:")
            for item in sorted (near_miss_log ,key =lambda x :x ['sim'],reverse =True )[:5 ]:
                print (
                f"  {item['disp_a']}({item['count_a']}) vs {item['disp_b']}({item['count_b']}): "
                f"sim={item['sim']:.3f}, th={item['th']:.3f}, conflict={item['blocked_by_conflict']}"
                )
        return {}

    alias_map ={}
    for item in merge_log :
        root_gid =item ['root_gid']
        child_gid =item ['child_gid']
        _merge_identity_stats (
        root_gid ,
        child_gid ,
        global_to_display =global_to_display ,
        global_gender_stats =global_gender_stats ,
        global_age_stats =global_age_stats ,
        inference =inference ,
        )
        alias_map [child_gid ]=root_gid 
        _merge_grouped_output_folders (
        id_root ,
        item ['child_display_id'],
        item ['root_display_id'],
        )

    print (f"[INFO] {log_prefix}: merged {len(merge_log)} identity pairs")
    for item in merge_log :
        print (
        f"  {item['child_display_id']}({item['child_gid']}) -> "
        f"{item['root_display_id']}({item['root_gid']}), "
        f"sim={item['similarity']:.3f}, th={item['threshold']:.3f}"
        )
    return alias_map 


def _remap_gid_to_root (gid :Optional [str ],alias_map :dict )->Optional [str ]:
    if gid is None :
        return None 
    cur =str (gid )
    visited =set ()
    while cur in alias_map and cur not in visited :
        visited .add (cur )
        cur =str (alias_map [cur ])
    return cur 


def _remap_detection_records (detection_records ,global_to_display :dict ,alias_map :dict ):
    if not detection_records :
        return detection_records 
    remapped =[]
    for rel_path ,id_records in detection_records :
        grouped ={}
        for disp_id ,gender ,age in id_records :
            gid =None 
            for k ,v in global_to_display .items ():
                if str (v )==str (disp_id ):
                    gid =k 
                    break 
            if gid is not None :
                gid =_remap_gid_to_root (gid ,alias_map )
                new_disp =global_to_display .get (gid ,disp_id )
            else :
                new_disp =disp_id 
            item =grouped .setdefault (str (new_disp ),{'genders':[],'ages':[]})
            g =str (gender ).strip ()if gender is not None else ''
            if g in {'M','F'}:
                item ['genders'].append (g )
            if age is not None :
                try :
                    item ['ages'].append (float (age ))
                except Exception :
                    pass 
        merged_records =[]
        for key in sorted (grouped .keys (),key =_display_id_sort_key ):
            item =grouped [key ]
            if item ['genders']:
                m =sum (1 for x in item ['genders']if x =='M')
                f =sum (1 for x in item ['genders']if x =='F')
                gender ='M'if m >f else 'F'if f >m else item ['genders'][-1 ]
            else :
                gender ='-'
            age_val =float (np .median (item ['ages']))if item ['ages']else None 
            merged_records .append ((key ,gender ,age_val ))
        remapped .append ((rel_path ,merged_records ))
    return remapped 




def parse_args ():
    p =argparse .ArgumentParser ("Open-world video/image ReID inference")

    # ReID model options
    p .add_argument ('--cfg',type =str ,default =None )
    p .add_argument ('--model-path',type =str ,default =None )
    p .add_argument ('--aux-model-path',type =str ,default =None ,help ='age/gender specialist model path (2-model deployment)')
    p .add_argument ('--mode',type =str ,default =None ,choices =['custom','strict','balanced','loose'])
    p .add_argument ('--similarity-threshold',type =float ,default =None )
    p .add_argument ('--verbose',action ='store_true')

    # Adaptive threshold options
    p .add_argument ('--base-threshold',type =float ,default =None ,help ='base threshold for adaptive policy')
    p .add_argument ('--adaptive-threshold-min',type =float ,default =None ,help ='minimum adaptive threshold')
    p .add_argument ('--adaptive-threshold-max',type =float ,default =None ,help ='maximum adaptive threshold')
    p .add_argument ('--confidence-threshold',type =float ,default =None ,help ='minimum confidence')
    p .add_argument ('--quality-threshold',type =float ,default =None ,help ='minimum quality score')

    # Age/Gender specialist options
    p .add_argument ('--gender-threshold',type =float ,default =None )
    p .add_argument ('--gender-hysteresis',type =float ,default =None ,help ='hysteresis for gender stability')
    p .add_argument ('--age-scope',type =str ,default =None ,choices =['track','video','global'],help ='age aggregation scope')
    p .add_argument ('--age-display',type =str ,default =None ,choices =['instant','median','mean'])
    p .add_argument ('--sim-cosine-w',type =float ,default =None ,help ='cosine similarity weight')
    p .add_argument ('--sim-euclid-w',type =float ,default =None ,help ='euclidean similarity weight')
    p .add_argument ('--aux-gender-penalty',type =float ,default =None ,help ='gender mismatch penalty (0-1)')
    p .add_argument ('--aux-age-reweight',type =float ,default =None ,help ='age reweight coefficient (0-1)')
    p .add_argument ('--aux-min-age-sigma',type =float ,default =None ,help ='minimum sigma for age reweighting')
    p .add_argument ('--aux-feature-fuse-weight',type =float ,default =None ,help ='fuse aux ReID embedding into main embedding [0,1]')

    # Input/Output
    p .add_argument ('--input',type =str ,default =None )
    p .add_argument ('--output',type =str ,default =None ,help ='output directory')
    p .add_argument ('--file-list',type =str ,default =None ,help ='UTF-8 text file with absolute or input-root relative media paths to process')

    # Detector/tracker
    p .add_argument ('--yolo-repo-root',type =str ,default =None ,help ='preferred local ultralytics repo root for YOLO26')
    p .add_argument ('--det-model',type =str ,default =None )
    p .add_argument ('--det-conf',type =float ,default =None )
    p .add_argument ('--det-iou',type =float ,default =None )
    p .add_argument ('--use-sam',type =int ,default =None ,choices =[0 ,1 ],help ='enable SAM refinement for YOLO boxes')
    p .add_argument ('--tracker',type =str ,default =None )

    # Output behavior
    p .add_argument ('--save-vis',type =int ,default =None ,choices =[0 ,1 ],help ='save visualization images')
    p .add_argument ('--group-by-id',action ='store_true',help ='group output ROIs by predicted ID')

    # Tracking behavior
    p .add_argument ('--sec-interval',type =float ,default =None ,help ='status update interval (seconds)')
    p .add_argument ('--vid-stride',type =int ,default =None ,help ='process every Nth video frame; 1 keeps full-frame tracking')
    p .add_argument ('--max-keep-missing-sec',type =float ,default =None ,help ='keep lost track state for N seconds')

    # Quality filtering
    p .add_argument ('--filter-low-quality',type =int ,default =None ,choices =[0 ,1 ],help ='enable low-quality filtering')
    p .add_argument ('--min-frame-brightness',type =float ,default =None ,help ='minimum frame brightness')
    p .add_argument ('--min-bbox-area-ratio',type =float ,default =None ,help ='minimum bbox area ratio')
    p .add_argument ('--max-bbox-area-ratio',type =float ,default =None ,help ='maximum bbox area ratio')
    p .add_argument ('--bbox-border-ratio',type =float ,default =None ,help ='minimum bbox border ratio')
    p .add_argument ('--min-mask-fill-ratio',type =float ,default =None ,help ='minimum mask fill ratio')
    p .add_argument ('--min-blur-var',type =float ,default =None ,help ='minimum blur variance')
    p .add_argument ('--min-brightness',type =float ,default =None ,help ='minimum ROI brightness')

    # Track-level suppression
    p .add_argument ('--track-bad-ratio-threshold',type =float ,default =None ,help ='suppress track if bad/(bad+good) exceeds threshold')
    p .add_argument ('--min-track-obs',type =int ,default =None ,help ='minimum observations before ID update')
    p .add_argument ('--min-track-good',type =int ,default =None ,help ='minimum good observations before ID update')
    p .add_argument ('--update-existing-min-id-conf',type =float ,default =None ,help ='minimum id confidence to update existing prototype')
    p .add_argument ('--shuffle-images',type =int ,default =None ,choices =[0 ,1 ],help ='shuffle image-file inference order to reduce order bias')
    p .add_argument ('--shuffle-seed',type =int ,default =None ,help ='seed for image order shuffling')
    p .add_argument ('--prototype-strategy',type =str ,default =None ,choices =['single','multi','spherical_multi'],help ='online identity memory strategy')
    p .add_argument ('--prototype-max-slots',type =int ,default =None ,help ='max sub-prototypes per identity')
    p .add_argument ('--prototype-spawn-similarity',type =float ,default =None ,help ='spawn new sub-prototype below this similarity')
    p .add_argument ('--prototype-update-similarity',type =float ,default =None ,help ='low-similarity update soft gate')
    p .add_argument ('--prototype-topk',type =int ,default =None ,help ='top-k slots used in multi-prototype scoring')
    p .add_argument ('--prototype-aggregate-weight',type =float ,default =None ,help ='weight of best slot vs top-k mean')
    p .add_argument ('--image-cluster-mode',type =str ,default =None ,choices =['online','twopass'],help ='image-directory ID assignment mode')
    p .add_argument ('--cluster-method',type =str ,default =None ,choices =['auto','agglomerative','sequential','multi_proto','spherical_multi'],help ='global clustering method for twopass mode')
    p .add_argument ('--twopass-threshold',type =float ,default =None ,help ='similarity threshold used in twopass image clustering')
    p .add_argument ('--twopass-threshold-auto',type =int ,default =None ,choices =[0 ,1 ],help ='auto-select twopass clustering threshold from feature statistics')
    p .add_argument ('--cluster-agglomerative-max-n',type =int ,default =None ,help ='max ROI count for agglomerative clustering')
    p .add_argument ('--cluster-momentum',type =float ,default =None ,help ='prototype momentum for sequential fallback clustering')
    p .add_argument ('--max-rois-per-image',type =int ,default =None ,help ='max kept ROIs per image in twopass mode')
    p .add_argument ('--twopass-fuse-full-image',type =int ,default =None ,choices =[0 ,1 ],help ='fuse full-image feature with ROI feature in twopass image mode')
    p .add_argument ('--twopass-full-image-weight',type =float ,default =None ,help ='fusion weight of full-image feature [0,1]')
    p .add_argument ('--twopass-cluster-aux-reid-weight',type =float ,default =None ,help ='concat aux ReID embedding into twopass clustering feature [0,1]')
    p .add_argument ('--twopass-cluster-aux-gender-weight',type =float ,default =None ,help ='concat gender cue into twopass clustering feature [0,1]')
    p .add_argument ('--twopass-cluster-aux-age-weight',type =float ,default =None ,help ='concat age cue into twopass clustering feature [0,1]')
    p .add_argument ('--twopass-auto-merge',type =int ,default =None ,choices =[0 ,1 ],help ='auto-merge close clusters after twopass clustering')
    p .add_argument ('--twopass-target-cluster-size',type =float ,default =None ,help ='target samples per identity for auto merge')
    p .add_argument ('--twopass-merge-min-sim',type =float ,default =None ,help ='minimum centroid similarity to allow cluster merge')
    p .add_argument ('--twopass-refine-split',type =int ,default =None ,choices =[0 ,1 ],help ='refine low-compactness clusters by secondary split')
    p .add_argument ('--twopass-refine-min-size',type =int ,default =None ,help ='minimum cluster size to trigger split refinement')
    p .add_argument ('--twopass-refine-delta',type =float ,default =None ,help ='threshold delta used in split refinement')
    p .add_argument ('--twopass-refine-min-subcluster',type =int ,default =None ,help ='minimum subcluster size to accept split refinement')
    p .add_argument ('--post-merge-enable',type =int ,default =None ,choices =[0 ,1 ],help ='enable conservative post-merge of fragmented IDs')
    p .add_argument ('--post-merge-enable-video',type =int ,default =None ,choices =[0 ,1 ],help ='enable post-merge for video / cross-video identities')
    p .add_argument ('--post-merge-enable-image',type =int ,default =None ,choices =[0 ,1 ],help ='enable post-merge for image-directory identities')
    p .add_argument ('--post-merge-video-min-sim',type =float ,default =None ,help ='minimum similarity for video-level identity post-merge')
    p .add_argument ('--post-merge-image-min-sim',type =float ,default =None ,help ='minimum similarity for image-level identity post-merge')
    p .add_argument ('--post-merge-mixed-min-sim',type =float ,default =None ,help ='minimum similarity for mixed-source identity post-merge')
    p .add_argument ('--post-merge-gender-conf-guard',type =float ,default =None ,help ='block opposite-gender post-merge above this confidence')
    p .add_argument ('--post-merge-age-gap-soft',type =float ,default =None ,help ='soft age-gap penalty for post-merge')
    p .add_argument ('--post-merge-age-gap-hard',type =float ,default =None ,help ='hard age-gap cutoff for post-merge')
    p .add_argument ('--post-merge-fragment-relax',type =float ,default =None ,help ='similarity relaxation for tiny fragments in post-merge')
    p .add_argument ('--post-merge-tiny-image-max-size',type =int ,default =None ,help ='max image-only cluster size treated as tiny fragment')
    p .add_argument ('--post-merge-tiny-image-min-sim',type =float ,default =None ,help ='minimum similarity for tiny image-fragment merge')

    # ROI-only mode: do not run detector, treat each input image as one ROI
    p .add_argument ('--image-as-roi',type =int ,default =None ,choices =[0 ,1 ],help ='1: each image is ROI; 0: run detector first')
    p .add_argument ('--roi-vis-use-detector',type =int ,default =None ,choices =[0 ,1 ],help ='when image_as_roi=1, use detector bbox for visualization')
    p .add_argument ('--roi-vis-det-conf',type =float ,default =None ,help ='detector conf for roi visualization box')
    p .add_argument ('--roi-vis-det-iou',type =float ,default =None ,help ='detector iou for roi visualization box')
    return p .parse_args ()

def _ensure_dir (p ):
    """Load image via cv2.imdecode for Windows-compatible Unicode paths."""
    if p is None or p =="":
        return "."
    os .makedirs (p ,exist_ok =True )
    return p 


def _imread_unicode (path :str ,flags =cv2 .IMREAD_COLOR ):
    """Read image bytes then decode with OpenCV to avoid path issues."""
    try :
        img =cv2 .imread (path ,flags )
        if img is not None :
            return img 
    except Exception :
        img =None 
    try :
        data =np .fromfile (path ,dtype =np .uint8 )
        if data is None or data .size ==0 :
            return None 
        return cv2 .imdecode (data ,flags )
    except Exception :
        return None 


def _imwrite_unicode (path :str ,image :np .ndarray ,params =None )->bool :
    """Encode and save image via OpenCV in a Windows-safe way."""
    try :
        out_dir =os .path .dirname (path )
        if out_dir :
            os .makedirs (out_dir ,exist_ok =True )
    except Exception :
        pass 

    params =params or []
    try :
        ok =cv2 .imwrite (path ,image ,params )
        if bool (ok ):
            return True 
    except Exception :
        pass 

    ext =os .path .splitext (path )[1 ]
    if not ext :
        ext =".jpg"
        path =path +ext 
    try :
        ok ,buf =cv2 .imencode (ext ,image ,params )
        if not ok :
            return False 
        buf .tofile (path )
        return True 
    except Exception :
        return False 


def _get_adaptive_text_style (
canvas_bgr :np .ndarray ,
base_scale :float =0.7 ,
min_scale :float =0.55 ,
max_scale :float =2.2 ,
):
    """Return (font, scale, thickness, pad) scaled by image size."""
    font =cv2 .FONT_HERSHEY_SIMPLEX 
    if canvas_bgr is None or getattr (canvas_bgr ,'size',0 )==0 :
        scale =float (base_scale )*_VIS_TEXT_SCALE_GAIN 
        thickness =max (2 ,int (round (scale *2.35 )))
        pad =max (6 ,int (round (scale *7.5 )))
        return font ,scale ,thickness ,pad 

    h ,w =canvas_bgr .shape [:2 ]
    short_side =max (1.0 ,float (min (h ,w )))
    scale_factor =float (np .sqrt (short_side /720.0 ))
    scale =float (np .clip (
    base_scale *scale_factor *_VIS_TEXT_SCALE_GAIN ,
    min_scale *_VIS_TEXT_SCALE_GAIN ,
    max_scale *_VIS_TEXT_SCALE_GAIN ,
    ))
    thickness =max (2 ,int (round (scale *2.35 )))
    pad =max (6 ,int (round (scale *7.5 )))
    return font ,scale ,thickness ,pad 


def _draw_text_block (
canvas_bgr :np .ndarray ,
text :str ,
x_txt :int ,
y_txt :int ,
font ,
font_scale :float ,
thickness :int ,
pad_txt :int ,
*,
bg_color =_VIS_LABEL_BG_BGR ,
text_color =_VIS_TEXT_BGR ,
border_color =_VIS_TEXT_BGR ,
):
    """Draw a high-contrast text block and return its background rectangle."""
    if canvas_bgr is None or getattr (canvas_bgr ,'size',0 )==0 or not text :
        return None 
    (tw ,thh ),baseline =cv2 .getTextSize (text ,font ,font_scale ,thickness )
    x_bg =max (0 ,int (x_txt -pad_txt //2 ))
    y_bg =max (0 ,int (y_txt -thh -baseline -pad_txt //2 ))
    max_x =max (0 ,int (canvas_bgr .shape [1 ]-1 ))
    max_y =max (0 ,int (canvas_bgr .shape [0 ]-1 ))
    x_bg2 =min (max_x ,int (x_txt +tw +pad_txt ))
    y_bg2 =min (max_y ,int (y_txt +baseline +pad_txt //2 ))
    cv2 .rectangle (canvas_bgr ,(x_bg ,y_bg ),(x_bg2 ,y_bg2 ),bg_color ,-1 )
    cv2 .rectangle (
    canvas_bgr ,
    (x_bg ,y_bg ),
    (x_bg2 ,y_bg2 ),
    border_color ,
    max (1 ,int (round (thickness *0.6 ))),
    )
    cv2 .putText (canvas_bgr ,text ,(int (x_txt ),int (y_txt )),font ,font_scale ,text_color ,thickness ,cv2 .LINE_AA )
    return (x_bg ,y_bg ,x_bg2 ,y_bg2 )


def _draw_status_label (canvas_bgr :np .ndarray ,text :str ,color =(0 ,0 ,255 )):
    """Resolve ROI input list from directory tree or single image file."""
    try :
        font ,font_scale ,thickness ,pad =_get_adaptive_text_style (
        canvas_bgr ,base_scale =0.95 ,min_scale =0.72 ,max_scale =2.6 
        )
        (_tw ,thh ),_baseline =cv2 .getTextSize (text ,font ,font_scale ,thickness )
        x_txt =pad *2 
        y_txt =pad *2 +thh 
        _draw_text_block (
        canvas_bgr ,
        text ,
        x_txt ,
        y_txt ,
        font ,
        font_scale ,
        thickness ,
        pad ,
        border_color =_VIS_TEXT_BGR if color is None else tuple (color ),
        )
    except Exception :
        pass 


def _resolve_output_path (output_arg ,input_path ,is_image :bool ,is_video :bool =False ):# noqa: ARG001
    """--output 
    - 
      * 
      * .mp4mp4v
    - 
    -  _out 
    """
    image_exts ={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
    video_exts ={'.mp4','.avi','.mov','.mkv','.wmv','.flv','.webm','.ts','.m4v'}

    in_dir ,in_name =os .path .split (input_path )
    stem ,in_ext =os .path .splitext (in_name )

    # 1 ->  + *_out
    if output_arg is None or output_arg =='':
        out_dir =_ensure_dir (in_dir )
        out_name =f"{stem}_out{in_ext or '.jpg'}"if is_image else f"{stem}_out.mp4"
        return os .path .join (out_dir ,out_name )

        
    if os .path .isdir (output_arg ):
        out_dir =_ensure_dir (output_arg )
        out_name =in_name if is_image else (stem +'.mp4')
        return os .path .join (out_dir ,out_name )

        
    ext =os .path .splitext (output_arg )[1 ].lower ()
    looks_like_file =(ext in image_exts )or (ext in video_exts )
    if not looks_like_file :
        out_dir =_ensure_dir (output_arg )
        out_name =in_name if is_image else (stem +'.mp4')
        return os .path .join (out_dir ,out_name )

        # 4
    out_dir =os .path .dirname (output_arg )
    if out_dir :
        _ensure_dir (out_dir )
    return output_arg 


def _resolve_existing_path (path_value :Optional [str ],bases =None )->Optional [str ]:
    return _resolve_existing_path_impl (path_value ,bases =bases )


def _resolve_detector_runtime_paths (args ):
    return _resolve_yolo26_runtime_paths_impl (args ,_YOLO26_RUNTIME )


def _build_yolo_detector (args ):
    return _build_yolo26_detector_impl (args ,_YOLO26_RUNTIME ,verbose =True )


def _crop_roi_with_expansion_numpy (image_rgb ,bbox_xyxy ,expand_ratio =0.1 ):
    """PandaReIDInference._crop_roi_with_expansion .
    bbox: (x1,y1,x2,y2) 
    """
    H ,W =image_rgb .shape [:2 ]
    x1 ,y1 ,x2 ,y2 =bbox_xyxy 
    w =max (1 ,x2 -x1 )
    h =max (1 ,y2 -y1 )
    ex =int (w *expand_ratio /2 )
    ey =int (h *expand_ratio /2 )
    x1e =max (0 ,x1 -ex )
    y1e =max (0 ,y1 -ey )
    x2e =min (W ,x2 +ex )
    y2e =min (H ,y2 +ey )
    return image_rgb [y1e :y2e ,x1e :x2e ]


def _is_dark_frame (frame_bgr :np .ndarray ,args =None ):
    """ReIDD"""
    if args is None :
        return False ,None 
    th =float (getattr (args ,'min_frame_brightness',0.0 )or 0.0 )
    if th <=0 :
        return False ,None 
    if frame_bgr is None or getattr (frame_bgr ,'size',0 )==0 :
        return True ,0.0 
    try :
        gray =cv2 .cvtColor (frame_bgr ,cv2 .COLOR_BGR2GRAY )
        mean_val =float (gray .mean ())
    except Exception :
        mean_val =float (np .mean (frame_bgr ))
    return (mean_val <th ),mean_val 


def _is_low_quality_roi (
frame_rgb :np .ndarray ,
bbox_xyxy ,
crop_rgb :np .ndarray ,
mask :np .ndarray =None ,
args =None ,
):
    """Choose top-quality ROI result and keep deterministic selection."""
    if args is None :
        return False ,[]
    if not bool (getattr (args ,'filter_low_quality',False )):
        return False ,[]

    H ,W =frame_rgb .shape [:2 ]
    x1 ,y1 ,x2 ,y2 =[int (v )for v in bbox_xyxy ]
    x1 =max (0 ,min (W -1 ,x1 ))
    y1 =max (0 ,min (H -1 ,y1 ))
    x2 =max (0 ,min (W ,x2 ))
    y2 =max (0 ,min (H ,y2 ))
    bw =max (1 ,x2 -x1 )
    bh =max (1 ,y2 -y1 )

    reasons =[]

    # 1) bbox 
    area_ratio =float (bw *bh )/max (1.0 ,float (H *W ))
    min_area =float (getattr (args ,'min_bbox_area_ratio',0.0 )or 0.0 )
    max_area =float (getattr (args ,'max_bbox_area_ratio',0.0 )or 0.0 )
    if min_area >0 and area_ratio <min_area :
        reasons .append (f"small({area_ratio:.3f}<{min_area})")
    if max_area >0 and area_ratio >max_area :
        reasons .append (f"large({area_ratio:.3f}>{max_area})")

        # 2) 
    border_ratio =float (getattr (args ,'bbox_border_ratio',0.0 )or 0.0 )
    if border_ratio >0 :
        margin =int (round (min (H ,W )*border_ratio ))
        if margin >0 and (x1 <=margin or y1 <=margin or x2 >=(W -margin )or y2 >=(H -margin )):
            reasons .append ("touch_border")

            
    min_fill =float (getattr (args ,'min_mask_fill_ratio',0.0 )or 0.0 )
    if min_fill >0 and mask is not None :
        try :
            if mask .shape [:2 ]==frame_rgb .shape [:2 ]:
                m =mask [y1 :y2 ,x1 :x2 ]
            else :
                m =mask 
            fill =float (np .sum (m ))/max (1.0 ,float (bw *bh ))
            if fill <min_fill :
                reasons .append (f"mask_fill({fill:.3f}<{min_fill})")
        except Exception :
            pass 

            # 4) 
    min_blur =float (getattr (args ,'min_blur_var',0.0 )or 0.0 )
    min_bright =float (getattr (args ,'min_brightness',0.0 )or 0.0 )
    if (min_blur >0 or min_bright >0 )and crop_rgb is not None and crop_rgb .size >0 :
        try :
            gray =cv2 .cvtColor (crop_rgb ,cv2 .COLOR_RGB2GRAY )
            if min_blur >0 :
                blur_var =float (cv2 .Laplacian (gray ,cv2 .CV_64F ).var ())
                if blur_var <min_blur :
                    reasons .append (f"blur({blur_var:.1f}<{min_blur})")
            if min_bright >0 :
                bright =float (gray .mean ())
                if bright <min_bright :
                    reasons .append (f"dark({bright:.1f}<{min_bright})")
        except Exception :
            pass 

    return (len (reasons )>0 ),reasons 


def _is_low_quality_roi_image (roi_rgb :np .ndarray ,args =None ):
    """ROIROI

    OIbox/
    - aplacian var
    - ean gray
    """
    if args is None :
        return False ,[]
    if not bool (getattr (args ,'filter_low_quality',False )):
        return False ,[]
    if roi_rgb is None or getattr (roi_rgb ,'size',0 )==0 :
        return True ,['empty_roi']

    reasons =[]
    min_blur =float (getattr (args ,'min_blur_var',0.0 )or 0.0 )
    min_bright =float (getattr (args ,'min_brightness',0.0 )or 0.0 )
    if min_blur <=0 and min_bright <=0 :
        return False ,[]
    try :
        gray =cv2 .cvtColor (roi_rgb ,cv2 .COLOR_RGB2GRAY )
        if min_blur >0 :
            blur_var =float (cv2 .Laplacian (gray ,cv2 .CV_64F ).var ())
            if blur_var <min_blur :
                reasons .append (f"blur({blur_var:.1f}<{min_blur})")
        if min_bright >0 :
            bright =float (gray .mean ())
            if bright <min_bright :
                reasons .append (f"dark({bright:.1f}<{min_bright})")
    except Exception :
        pass 
    return (len (reasons )>0 ),reasons 


def _crop_with_mask_patch (frame_rgb :np .ndarray ,bbox_xyxy ,mask_patch :np .ndarray ,expand_ratio :float =0.1 ):
    """bbox + mask_patch  ROI _process_mask_region_to_rectangle ask

    bbox_xyxy: (x1,y1,x2,y2) 2/y2
    mask_patch: bbox bool/0-1 (y2-y1, x2-x1)
    """
    if frame_rgb is None or getattr (frame_rgb ,'size',0 )==0 :
        return None 
    H ,W =frame_rgb .shape [:2 ]
    x1 ,y1 ,x2 ,y2 =[int (v )for v in bbox_xyxy ]
    x1 =max (0 ,min (W -1 ,x1 ))
    y1 =max (0 ,min (H -1 ,y1 ))
    x2 =max (0 ,min (W ,x2 ))
    y2 =max (0 ,min (H ,y2 ))
    bw =max (1 ,x2 -x1 )
    bh =max (1 ,y2 -y1 )

    expand_w =int (bw *float (expand_ratio ))
    expand_h =int (bh *float (expand_ratio ))
    x1_exp =max (0 ,x1 -expand_w //2 )
    y1_exp =max (0 ,y1 -expand_h //2 )
    x2_exp =min (W ,x2 +expand_w //2 )
    y2_exp =min (H ,y2 +expand_h //2 )

    cropped_image =frame_rgb [y1_exp :y2_exp ,x1_exp :x2_exp ].copy ()
    if mask_patch is None :
        return cropped_image 

    try :
        mp =mask_patch .astype (bool )
        
        if mp .shape [0 ]!=(y2 -y1 )or mp .shape [1 ]!=(x2 -x1 ):
            mp =cv2 .resize (mp .astype (np .uint8 ),(x2 -x1 ,y2 -y1 ),interpolation =cv2 .INTER_NEAREST ).astype (bool )

        cropped_mask =np .zeros ((y2_exp -y1_exp ,x2_exp -x1_exp ),dtype =bool )
        ix1 ,iy1 =max (x1 ,x1_exp ),max (y1 ,y1_exp )
        ix2 ,iy2 =min (x2 ,x2_exp ),min (y2 ,y2_exp )
        if ix2 >ix1 and iy2 >iy1 :
            cropped_mask [iy1 -y1_exp :iy2 -y1_exp ,ix1 -x1_exp :ix2 -x1_exp ]=mp [
            iy1 -y1 :iy2 -y1 ,ix1 -x1 :ix2 -x1 
            ]
        cropped_image [~cropped_mask ]=0 
    except Exception :
        pass 
    return cropped_image 


def _split_instances_from_sam_multimask (masks ,scores ,bbox_xyxy ,args =None ,
min_comp_area_ratio :float =0.05 ,
max_top12_ratio :float =10.0 ,
max_instances :int =4 ):
    """YOLO  SAM maskmask

    : list[{'bbox':(x1,y1,x2,y2), 'mask':mask_patch(bool), 'area':int}] None
    """
    if masks is None :
        return None 
    try :
        m =np .asarray (masks )
        if m .ndim ==2 :
            m =m [None ,...]
    except Exception :
        return None 

    x1 ,y1 ,x2 ,y2 =[int (v )for v in bbox_xyxy ]
    H ,W =m .shape [1 ],m .shape [2 ]
    x1 =max (0 ,min (W -1 ,x1 ))
    y1 =max (0 ,min (H -1 ,y1 ))
    x2 =max (0 ,min (W ,x2 ))
    y2 =max (0 ,min (H ,y2 ))
    bw =max (1 ,x2 -x1 )
    bh =max (1 ,y2 -y1 )
    box_area =float (bw *bh )
    min_area =max (int (box_area *float (min_comp_area_ratio )),2000 )

    candidates =[]
    for mi in range (m .shape [0 ]):
        try :
            mask_u8 =(m [mi ]>0 ).astype (np .uint8 )
            roi =mask_u8 [y1 :y2 ,x1 :x2 ]
            if roi .size ==0 :
                continue 
            num_labels ,labels =cv2 .connectedComponents (roi )
            if num_labels <=2 :
                continue 
            areas =[]
            for lab in range (1 ,num_labels ):
                a =int ((labels ==lab ).sum ())
                if a >=min_area :
                    areas .append (a )
            if len (areas )<2 :
                continue 
            areas .sort (reverse =True )
            ratio =float (areas [0 ])/float (max (1 ,areas [1 ]))
            if ratio >max_top12_ratio :
                continue 
            score =float (scores [mi ])if scores is not None and len (scores )>mi else 0.0 
            candidates .append ({'mi':mi ,'count':len (areas ),'ratio':ratio ,'score':score })
        except Exception :
            continue 

    if not candidates :
        return None 

        
    candidates .sort (key =lambda c :(-c ['count'],c ['ratio'],-c ['score']))
    best_mi =int (candidates [0 ]['mi'])

    try :
        mask_u8 =(m [best_mi ]>0 ).astype (np .uint8 )
        roi =mask_u8 [y1 :y2 ,x1 :x2 ]
        num_labels ,labels =cv2 .connectedComponents (roi )
        comps =[]
        for lab in range (1 ,num_labels ):
            a =int ((labels ==lab ).sum ())
            if a <min_area :
                continue 
            ys ,xs =np .where (labels ==lab )
            if len (ys )==0 :
                continue 
            y1c ,y2c =int (ys .min ()),int (ys .max ())+1 
            x1c ,x2c =int (xs .min ()),int (xs .max ())+1 
            bbox =(x1 +x1c ,y1 +y1c ,x1 +x2c ,y1 +y2c )
            mask_patch =(labels [y1c :y2c ,x1c :x2c ]==lab )
            comps .append ({'bbox':bbox ,'mask':mask_patch ,'area':a })
        comps .sort (key =lambda d :d ['area'],reverse =True )
        comps =comps [:max_instances ]
        if len (comps )<2 :
            return None 
        if getattr (args ,'verbose',False ):
            print (f"   SAM det_bbox={tuple(map(int,bbox_xyxy))} -> {len(comps)} instances (mask#{best_mi})")
        return comps 
    except Exception :
        return None 


def _track_lowq_stats (state ):
    """Update per-track state when new ROI observations arrive."""
    if not isinstance (state ,dict ):
        return 0 ,0 ,0 ,0.0 
    bad =int (state .get ('lq_bad',0 )or 0 )
    good =int (state .get ('lq_good',0 )or 0 )
    obs =bad +good 
    bad_ratio =(float (bad )/float (obs ))if obs >0 else 0.0 
    return bad ,good ,obs ,bad_ratio 


def _should_skip_track_by_lowq (state ,args ):
    """(track)ID /

    
    - bad_ratio >= track_bad_ratio_threshold obs>=min_track_obs
    - good < min_track_good 1 ID
    """
    if args is None or not bool (getattr (args ,'filter_low_quality',False )):
        return False ,{}
    bad ,good ,obs ,bad_ratio =_track_lowq_stats (state )
    th =float (getattr (args ,'track_bad_ratio_threshold',0.8 )or 0.8 )
    min_obs =int (getattr (args ,'min_track_obs',0 )or 0 )
    min_good =int (getattr (args ,'min_track_good',0 )or 0 )

    info ={
    'lq_bad':bad ,
    'lq_good':good ,
    'lq_obs':obs ,
    'lq_bad_ratio':bad_ratio ,
    'track_bad_ratio_threshold':th ,
    'min_track_obs':min_obs ,
    'min_track_good':min_good ,
    }

    if min_good >0 and good <min_good :
        info ['skip_reason']='min_track_good'
        return True ,info 

    if obs >0 and min_obs >0 and obs >=min_obs and bad_ratio >=th :
        info ['skip_reason']='bad_ratio'
        return True ,info 

    return False ,info 


def _aggregate_track_features (feature_list ,method ='mean'):
    """.

    Args:
        feature_list: list of torch.Tensor,  [D]
        method: 'mean', 'weighted_mean'

    Returns:
        aggregated: torch.Tensor [D], L2
    """
    if not feature_list :
        return None 
    if len (feature_list )==1 :
        return F .normalize (feature_list [0 ],p =2 ,dim =0 )

    stacked =torch .stack (feature_list ,dim =0 )# [N, D]
    if method =='weighted_mean':
    
        N =stacked .shape [0 ]
        weights =torch .linspace (0.5 ,1.5 ,N ,device =stacked .device ).unsqueeze (1 )# [N, 1]
        aggregated =(stacked *weights ).sum (dim =0 )/weights .sum ()
    else :
        aggregated =stacked .mean (dim =0 )

    return F .normalize (aggregated ,p =2 ,dim =0 )


_STDOUT_TEE_ENABLED =False 


class _TeeIO :
    def __init__ (self ,*streams ):
        self ._streams =streams 

    def write (self ,data ):
        for s in self ._streams :
            try :
                s .write (data )
            except Exception :
                pass 
        return len (data )

    def flush (self ):
        for s in self ._streams :
            try :
                s .flush ()
            except Exception :
                pass 

    def isatty (self ):
        for s in self ._streams :
            try :
                if s .isatty ():
                    return True 
            except Exception :
                pass 
        return False 

    @property 
    def encoding (self ):
        for s in self ._streams :
            enc =getattr (s ,'encoding',None )
            if enc :
                return enc 
        return 'utf-8'


def _enable_stdout_tee (log_path :str ):
    """Print progress/status logs when verbose mode is enabled."""
    global _STDOUT_TEE_ENABLED 
    if _STDOUT_TEE_ENABLED :
        return 
    if not log_path :
        return 
    log_dir =os .path .dirname (os .path .abspath (log_path ))
    if log_dir :
        _ensure_dir (log_dir )
    f =open (log_path ,'a',encoding ='utf-8',errors ='replace')
    orig_out ,orig_err =sys .stdout ,sys .stderr 
    sys .stdout =_TeeIO (orig_out ,f )
    sys .stderr =_TeeIO (orig_err ,f )

    def _cleanup ():
        try :
            sys .stdout =orig_out 
            sys .stderr =orig_err 
        except Exception :
            pass 
        try :
            f .flush ()
            f .close ()
        except Exception :
            pass 

    atexit .register (_cleanup )
    _STDOUT_TEE_ENABLED =True 


def _infer_num_classes_from_ckpt (checkpoint )->int :
    state =checkpoint .get ("model",checkpoint )if isinstance (checkpoint ,dict )else checkpoint 
    if isinstance (checkpoint ,dict )and "num_classes"in checkpoint :
        try :
            return int (checkpoint ["num_classes"])
        except Exception :
            pass 
    if isinstance (state ,dict ):
        for key in ("neck.classifier.weight","module.neck.classifier.weight"):
            if key in state and hasattr (state [key ],"shape"):
                return int (state [key ].shape [0 ])
    return 1000 


class _AuxPredictorWrapper :
    def __init__ (self ,model ,kind :str ):
        self .model =model 
        self .kind =kind 

    def __call__ (self ,x ,return_feat :bool =False ):
        if self .kind =="specialist":
            _feat ,gender_logits ,_age_logits ,age_pred =self .model (x )
            if return_feat :
                return None ,gender_logits ,age_pred 
            return gender_logits ,age_pred 
        _fa ,_fb ,gender_logits ,age_pred =self .model .forward_multitask (x )
        if return_feat :
            return _fa ,gender_logits ,age_pred 
        return gender_logits ,age_pred 


def _build_aux_eval_transform (img_size :int ):
    mean =[0.485 ,0.456 ,0.406 ]
    std =[0.229 ,0.224 ,0.225 ]
    return transforms .Compose (
    [
    transforms .Resize ((int (img_size ),int (img_size ))),
    transforms .ToTensor (),
    transforms .Normalize (mean =mean ,std =std ),
    ]
    )


def _load_aux_predictor (aux_model_path :str ,device :torch .device ,fallback_cfg_path :Optional [str ]=None ):
    if not aux_model_path :
        return None ,None 
    if not os .path .isfile (aux_model_path ):
        print (f"[WARN] aux model not found: {aux_model_path}; fallback to ReID model heads.")
        return None ,None 

    ckpt =torch .load (aux_model_path ,map_location ="cpu")

    # Dedicated specialist checkpoint from train_age_gender_specialist.py
    if isinstance (ckpt ,dict )and ("backbone_name"in ckpt or "max_age_bin"in ckpt ):
        backbone_name =ckpt .get ("backbone_name","convnextv2_base.fcmae_ft_in22k_in1k")
        max_age_bin =int (ckpt .get ("max_age_bin",40 ))
        aux_img_size =int (ckpt .get ("img_size",224 ))
        age_expected_mix =float (ckpt .get ("age_expected_mix",0.6 ))
        model =AgeGenderSpecialist (
        backbone_name =backbone_name ,
        num_age_bins =max_age_bin +1 ,
        dropout =0.2 ,
        age_expected_mix =age_expected_mix ,
        )
        state =ckpt .get ("model",ckpt )
        model .load_state_dict (state ,strict =False )
        model .to (device )
        model .eval ()
        print (
        f"[INFO] Aux specialist loaded: {backbone_name}, "
        f"max_age_bin={max_age_bin}, img_size={aux_img_size}, age_mix={age_expected_mix:.2f}"
        )
        return _AuxPredictorWrapper (model ,kind ="specialist"),aux_img_size 

        # ReID-style checkpoint with multitask heads, reused as aux model.
    aux_cfg =ckpt .get ("config",None )if isinstance (ckpt ,dict )else None 
    if aux_cfg is None :
        class _Tmp :
            pass 
        tmp =_Tmp ()
        tmp .cfg =fallback_cfg_path or USER_DEFAULTS .get ("cfg")
        tmp .opts =[]
        aux_cfg =get_config (tmp )

    state =ckpt .get ("model",ckpt )if isinstance (ckpt ,dict )else ckpt 
    num_classes =_infer_num_classes_from_ckpt (ckpt )
    model =build_panda_reid_model (aux_cfg ,num_classes =num_classes )
    model_sd =model .state_dict ()
    filtered ={}
    for k ,v in state .items ():
        if k in model_sd and getattr (v ,"shape",None )==getattr (model_sd [k ],"shape",None ):
            filtered [k ]=v 
    model .load_state_dict (filtered ,strict =False )
    model .to (device )
    model .eval ()
    aux_img_size =int (getattr (aux_cfg .DATA ,"IMG_SIZE",192 ))
    print (
    f"[INFO] Aux ReID checkpoint loaded: num_classes={num_classes}, "
    f"img_size={aux_img_size}, loaded={len(filtered)}"
    )
    return _AuxPredictorWrapper (model ,kind ="reid_aux"),aux_img_size 


def _attach_aux_predictor (inference ,args ):
    aux_predictor ,aux_img_size =_load_aux_predictor (
    aux_model_path =getattr (args ,"aux_model_path",None ),
    device =inference .device ,
    fallback_cfg_path =getattr (args ,"cfg",None ),
    )
    setattr (inference ,"aux_predictor",aux_predictor )
    if aux_predictor is not None and aux_img_size is not None :
        setattr (inference ,"aux_transform",_build_aux_eval_transform (int (aux_img_size )))
    else :
        setattr (inference ,"aux_transform",None )
    return aux_predictor is not None 


def _forward_reid_and_aux (inference ,reid_input ,crops_rgb =None ,return_aux_feat :bool =False ,fuse_aux_into_main :bool =True ):
    feat_after_bn ,_feat_before_bn ,gender_logits ,age_pred =inference .model .forward_multitask (reid_input )
    feat_after_bn =F .normalize (feat_after_bn ,p =2 ,dim =1 )
    aux_feat_out =None

    aux_predictor =getattr (inference ,"aux_predictor",None )
    aux_transform =getattr (inference ,"aux_transform",None )
    can_use_aux =(
    aux_predictor is not None 
    and aux_transform is not None 
    and isinstance (crops_rgb ,(list ,tuple ))
    and len (crops_rgb )==int (reid_input .shape [0 ])
    )
    if can_use_aux :
        aux_tensors =[]
        for crop in crops_rgb :
            if crop is None or getattr (crop ,"size",0 )==0 :
                can_use_aux =False 
                break 
            aux_tensors .append (aux_transform (Image .fromarray (crop )))
        if can_use_aux and len (aux_tensors )>0 :
            x_aux =torch .stack (aux_tensors ,dim =0 ).to (reid_input .device ,non_blocking =True )
            args =getattr (inference ,'args',None )
            aux_fuse_w =0.0 if args is None else float (getattr (args ,'aux_feature_fuse_weight',0.0 )or 0.0 )
            aux_fuse_w =float (min (1.0 ,max (0.0 ,aux_fuse_w )))
            if str (getattr (aux_predictor ,'kind',''))=='reid_aux':
                aux_feat ,gender_logits ,age_pred =aux_predictor (x_aux ,return_feat =True )
                if aux_feat is not None :
                    aux_feat =F .normalize (aux_feat ,p =2 ,dim =1 )
                    aux_feat_out =aux_feat 
                    if aux_fuse_w >0.0 and fuse_aux_into_main and tuple (aux_feat .shape )==tuple (feat_after_bn .shape ):
                        feat_after_bn =F .normalize (
                        (1.0 -aux_fuse_w )*feat_after_bn +aux_fuse_w *aux_feat ,
                        p =2 ,
                        dim =1 ,
                        )
            else :
                gender_logits ,age_pred =aux_predictor (x_aux )

    if getattr (age_pred ,"ndim",1 )>1 :
        age_pred =age_pred .view (age_pred .shape [0 ],-1 )[:,0 ]
    male_probs =F .softmax (gender_logits ,dim =1 )[:,1 ]
    if return_aux_feat :
        return feat_after_bn ,male_probs ,age_pred ,aux_feat_out 
    return feat_after_bn ,male_probs ,age_pred 


def _patch_prototype_threshold_no_quality (inference ,verbose =False ):
    """Keep PandaReIDInference runtime threshold patch as-is (no override here)."""
    _ =inference 
    if verbose :
        print ("[INFO] Using PandaReIDInference adaptive-threshold policy (no extra override).")


class _GidRangeReservation :
    """gid

    
    -  ByteTrack ID switch tid
    -  tid finalize
    -  set()  gid  gid gid
    """

    def __init__ (self ):
        self ._ranges_by_gid =defaultdict (list )# gid -> list[(start,end)]
        self ._cur_range =None # (start,end)

    def set_current_range (self ,start_frame ,end_frame ):
        try :
            if start_frame is None or end_frame is None :
                self ._cur_range =None 
            else :
                self ._cur_range =(int (start_frame ),int (end_frame ))
        except Exception :
            self ._cur_range =None 

    def __contains__ (self ,gid ):
        ranges =self ._ranges_by_gid .get (gid )
        if not ranges :
            return False 
            
        if self ._cur_range is None :
            return True 
        s ,e =self ._cur_range 
        for s2 ,e2 in ranges :
            if s2 is None or e2 is None :
                return True 
                # overlap if not disjoint
            if not (e <s2 or e2 <s ):
                return True 
        return False 

    def add (self ,gid ):
        if self ._cur_range is None :
            self ._ranges_by_gid [gid ].append ((None ,None ))
        else :
            self ._ranges_by_gid [gid ].append (self ._cur_range )


def _pick_global_id_from_prototype_result (res ,inference ,reserved_ids ,allow_new =True ):
    """ PrototypeReIDNetwork  open-set 

    
    -  best_sim + similarity_threshold
    -  res['predicted_id'] / res['is_new_id'] / res['adaptive_threshold']
    - reserved_ids  finalize
    """
    sims =res .get ('all_similarities',{})or {}
    adaptive_th =float (res .get ('adaptive_threshold',0.0 )or 0.0 )
    if adaptive_th <=0 :
        adaptive_th =float (getattr (inference ,'similarity_threshold',0.5 )or 0.5 )

    args =getattr (inference ,'args',None )
    conf_th =float (getattr (args ,'confidence_threshold',0.0 )or 0.0 )if args is not None else 0.0 
    quality_th =float (getattr (args ,'quality_threshold',0.0 )or 0.0 )if args is not None else 0.0 
    fallback_margin =0.02 

    model_pred =res .get ('predicted_id',None )
    model_is_new =bool (res .get ('is_new_id',False ))
    model_best_sim =float (res .get ('similarity',0.0 )or 0.0 )
    model_conf =float (res .get ('confidence',0.0 )or 0.0 )
    model_quality =float (res .get ('quality_score',1.0 )or 1.0 )

    best_sim_any =float (max (sims .values ()))if sims else model_best_sim 

    # Respect prototype-net open-world decision if it is a confident existing-ID match.
    if (not model_is_new )and model_pred is not None and model_pred not in reserved_ids :
        if model_best_sim >=adaptive_th and model_conf >=conf_th and model_quality >=quality_th :
            return model_pred ,model_best_sim ,False ,adaptive_th 

    # Fallback: pick best non-reserved prototype only when similarity is clearly above threshold.
    best_id =None 
    best_sim =0.0 
    if sims :
        for pid ,sim in sorted (sims .items (),key =lambda x :x [1 ],reverse =True ):
            if pid in reserved_ids :
                continue 
            best_id =pid 
            best_sim =float (sim )
            break 

    if best_id is not None and best_sim >=(adaptive_th +fallback_margin ):
        return best_id ,best_sim ,False ,adaptive_th 

    if not allow_new :
        return None ,best_sim_any ,True ,adaptive_th 

    inference .wild_id_counter +=1 
    gid =f"Wild_Panda_{inference.wild_id_counter:03d}"
    return gid ,best_sim_any ,True ,adaptive_th 


def _finalize_track_id (state ,inference ,global_to_display ,global_gender_stats ,
video_age_stats ,global_age_stats ,finalized_tracks ,# noqa: ARG001
reserved_ids ,args ,verbose =False ,tid =None ,event ='',known_ids =None ):
    """D

    Args:
        state: 
        inference: PandaReIDInference 
        global_to_display: ID
        global_gender_stats: 
        video_age_stats: 
        global_age_stats: 
        finalized_tracks: 
        reserved_ids: ID
        args: 
        verbose: 
        tid: ID

    Returns:
        (gid, display_id, similarity, next_display_id_increment)
    """
    feature_buffer =state .get ('feature_buffer',[])
    if not feature_buffer :
        return None ,None ,None ,0 
        
    if hasattr (reserved_ids ,'set_current_range'):
        reserved_ids .set_current_range (state .get ('first_seen_frame'),state .get ('last_seen_frame'))

        # 
    aggregated_feat =_aggregate_track_features (feature_buffer ,method ='weighted_mean')
    if aggregated_feat is None :
        return None ,None ,None ,0 

        # /
    male_hist =list (state .get ('male_hist',[]))
    age_hist =list (state .get ('age_hist',[]))
    agg_male_p =float (np .mean (male_hist ))if male_hist else 0.5 
    agg_age_p =float (np .mean (age_hist ))if age_hist else 5.0 
    aux ={'gender_prob':agg_male_p ,'age_pred':agg_age_p }

    
    _created_new =False 
    if known_ids :
        candidates_all =[str (x )for x in known_ids if x ]
        # ID
        candidates =[cid for cid in candidates_all if cid not in reserved_ids ]or list (candidates_all )

        res =_match_identity_with_strategy (
        inference ,
        aggregated_feat ,
        aux =aux ,
        reserved_ids =reserved_ids ,
        candidate_ids =candidates_all ,
        allow_new =False ,
        )
        sims =res .get ('all_similarities',{})or {}
        gid =None 
        best_score =None 
        for cid in candidates :
            sc =_candidate_score (cid ,agg_male_p ,agg_age_p ,sim =sims .get (cid ))
            if best_score is None or sc >best_score :
                gid =cid 
                best_score =sc 
        if gid is None :
            return None ,None ,None ,0 
        sim_for_log =float (sims .get (gid ,0.0 )or 0.0 )
        adaptive_th =float (res .get ('adaptive_threshold',0.0 )or 0.0 )
        _created_new =False 
    else :
        res =_match_identity_with_strategy (
        inference ,
        aggregated_feat ,
        aux =aux ,
        reserved_ids =reserved_ids ,
        candidate_ids =None ,
        allow_new =True ,
        )
        gid ,sim_for_log ,_created_new ,adaptive_th =_pick_global_id_from_prototype_result (
        res ,inference ,reserved_ids ,allow_new =True 
        )
    if gid is None :
        return None ,None ,None ,0 

    id_conf_01 =_identity_confidence_01 (sim_for_log ,adaptive_th =adaptive_th ,is_new =_created_new )
    state ['last_id_confidence']=id_conf_01 

    next_display_id_inc =0 

    reserved_ids .add (gid )
    if known_ids :
        try :
            with torch .no_grad ():
                quality_score =float (inference .prototype_net .quality_net (aggregated_feat .unsqueeze (0 )).item ())
        except Exception :
            quality_score =1.0 
    else :
        quality_score =float (res .get ('quality_score',1.0 )or 1.0 )
    min_upd_conf =float (getattr (args ,'update_existing_min_id_conf',0.55 )or 0.55 )
    should_update =(bool (_created_new )or (id_conf_01 >=min_upd_conf ))
    if should_update :
        _update_identity_memory (
        inference ,
        gid ,
        aggregated_feat ,
        quality_score ,
        gender_prob =agg_male_p ,
        age_pred =agg_age_p ,
        id_conf =id_conf_01 ,
        )

    
    if gid not in global_to_display :
    
        next_display_id_inc =1 

        # 
    gstat =global_gender_stats .setdefault (gid ,{
    'alpha_male':1.0 ,'alpha_female':1.0 ,'stable_gender':None ,
    })
    for mp in male_hist :
        weight =1.0 +2.0 *abs (mp -0.5 )
        gstat ['alpha_male']+=weight *mp 
        gstat ['alpha_female']+=weight *(1.0 -mp )
    total =gstat ['alpha_male']+gstat ['alpha_female']
    male_mean =gstat ['alpha_male']/max (1e-6 ,total )
    margin =getattr (args ,'gender_hysteresis',0.03 )
    th =getattr (args ,'gender_threshold',0.5 )
    if gstat ['stable_gender']is None :
        gstat ['stable_gender']='M'if male_mean >=th else 'F'
    else :
        if gstat ['stable_gender']=='M'and male_mean <(th -margin ):
            gstat ['stable_gender']='F'
        elif gstat ['stable_gender']=='F'and male_mean >(th +margin ):
            gstat ['stable_gender']='M'

            # DID
    forced_meta =_ID_META .get (gid )
    if forced_meta is not None and forced_meta .get ('gender')in ('M','F'):
        gstat ['stable_gender']=forced_meta ['gender']

        # 
    for ap in age_hist :
        video_age_stats [gid ].append (ap )
        global_age_stats .setdefault (gid ,[]).append (ap )

        
    if tid is not None :
        if verbose :
            pred =res .get ('predicted_id',None )
            is_new =bool (res .get ('is_new_id',False ))
            qs =res .get ('quality_score',None )
            conf_s =f"{id_conf_01 :.2f}"
            qs_s =f"{float(qs):.2f}"if qs is not None else "-"
            print (
            f"  trk{tid} {event}D: {gid}, {len(feature_buffer)}, "
            f"sim={sim_for_log:.3f}, ad_th={adaptive_th:.3f}, pred={pred}, is_new={is_new}, id_conf={conf_s}, q={qs_s}"
            )
        else :
            print (f"  trk{tid} {event} ID: {gid}, feats={len(feature_buffer)}, sim={sim_for_log:.3f}, id_conf={id_conf_01 :.2f}")

    return gid ,sim_for_log ,next_display_id_inc 


def _reorganize_folders_by_id (id_root ,finalized_track_map ,global_to_display =None ):# noqa: ARG001
    """trk{id}  ID{n} 

    IDD

    Args:
        id_root: 
        finalized_track_map: { tid: { 'global_id': str, 'display_id': str } }
        global_to_display: ID
    """
    if not id_root or not os .path .isdir (id_root ):
        return 

        
    trk_to_display ={}
    for tid ,info in finalized_track_map .items ():
        trk_name =f"trk{tid}"
        display_id =info .get ('display_id')
        if display_id :
            trk_to_display [trk_name ]=display_id 

            
    existing_folders =[d for d in os .listdir (id_root )if os .path .isdir (os .path .join (id_root ,d ))]

    for folder_name in existing_folders :
        folder_path =os .path .join (id_root ,folder_name )

        
        if folder_name .startswith ('trk')and folder_name in trk_to_display :
            target_id =trk_to_display [folder_name ]
            target_path =os .path .join (id_root ,target_id )

            if folder_path ==target_path :
                continue # 

            try :
                if os .path .exists (target_path ):
                
                    for item in os .listdir (folder_path ):
                        src =os .path .join (folder_path ,item )
                        dst =os .path .join (target_path ,item )
                        if os .path .exists (dst ):
                        # 
                            base ,ext =os .path .splitext (item )
                            dst =os .path .join (target_path ,f"{base}_{folder_name}{ext}")
                        shutil .move (src ,dst )
                        # 
                    os .rmdir (folder_path )
                    print (f"    {folder_name} -> {target_id}")
                else :
                
                    os .rename (folder_path ,target_path )
                    print (f"   {folder_name} -> {target_id}")
            except Exception as e :
                print (f"    {folder_name}: {e}")


def _flatten_rel_path (rel_path :str )->str :
    """Build output paths when grouping results by predicted identity."""
    if rel_path is None :
        return ""
    s =str (rel_path ).replace ("/","__").replace ("\\","__")
    
    for ch in ['<','>',':','"','/','\\','|','?','*']:
        s =s .replace (ch ,"_")
        # 
    return s .lstrip ("_")


def _save_image_visualization (
args ,
inference ,
frame_bgr :np .ndarray ,
*,
input_rel_path :Optional [str ]=None ,
disp_ids =None ,
fallback_folder :str ="UNASSIGNED",
)->bool :
    """Save visualization image for both grouped and non-grouped output modes."""
    if not bool (getattr (args ,'save_vis',True )):
        return False 
    if frame_bgr is None or getattr (frame_bgr ,'size',0 )==0 :
        return False 

    if bool (getattr (inference ,'group_by_id',False )):
        id_root =getattr (inference ,'id_group_root',None )or ""
        if not id_root :
            return False 
        rel_name =input_rel_path if input_rel_path else os .path .basename (getattr (args ,'input',"output.jpg"))
        save_name =_flatten_rel_path (rel_name )or os .path .basename (getattr (args ,'input',"output.jpg"))
        targets =[]
        if disp_ids is not None :
            targets =[str (x )for x in disp_ids if x is not None and str (x )!=""]
        if not targets :
            targets =[str (fallback_folder )]
        ok_any =False 
        for disp_id in sorted (set (targets )):
            id_dir =os .path .join (id_root ,disp_id )
            os .makedirs (id_dir ,exist_ok =True )
            save_path =os .path .join (id_dir ,save_name )
            ok =_imwrite_unicode (save_path ,frame_bgr )
            ok_any =ok_any or bool (ok )
            if (not ok )and bool (getattr (args ,'verbose',False )):
                print (f"[WARN] Failed to save visualization: {save_path}")
        return ok_any 

    out_path =getattr (args ,'output',None )
    if not out_path :
        return False 
    ok =_imwrite_unicode (out_path ,frame_bgr )
    if (not ok )and bool (getattr (args ,'verbose',False )):
        print (f"[WARN] Failed to save visualization: {out_path}")
    return bool (ok )


def _pick_roi_visual_box_from_detector (frame_bgr :np .ndarray ,yolo ,args =None ):
    """Pick the best detection bbox for visualization in ROI-only image mode."""
    if frame_bgr is None or getattr (frame_bgr ,'size',0 )==0 or yolo is None :
        return None 
    H ,W =frame_bgr .shape [:2 ]
    if H <=1 or W <=1 :
        return None 
    try :
        conf =float (getattr (args ,'roi_vis_det_conf',None ))if args is not None and getattr (args ,'roi_vis_det_conf',None )is not None else float (getattr (args ,'det_conf',0.35 )or 0.35 )
    except Exception :
        conf =0.35 
    try :
        iou =float (getattr (args ,'roi_vis_det_iou',None ))if args is not None and getattr (args ,'roi_vis_det_iou',None )is not None else float (getattr (args ,'det_iou',0.45 )or 0.45 )
    except Exception :
        iou =0.45 
    conf =max (0.01 ,min (0.95 ,conf ))
    iou =max (0.05 ,min (0.95 ,iou ))

    try :
        results =yolo .predict (source =frame_bgr ,conf =conf ,iou =iou ,verbose =False )
    except Exception :
        return None 
    if not results :
        return None 
    boxes =results [0 ].boxes 
    if boxes is None or len (boxes )==0 :
        return None 
    try :
        xyxy =boxes .xyxy .cpu ().numpy ().astype (int )
    except Exception :
        xyxy =np .asarray (boxes .xyxy ,dtype =np .float32 ).astype (int )
    try :
        confs =boxes .conf .cpu ().numpy ().astype (float )
    except Exception :
        confs =np .ones ((len (xyxy ),),dtype =np .float32 )*0.5 

    best =None 
    best_score =-1.0 
    for i ,(x1 ,y1 ,x2 ,y2 )in enumerate (xyxy ):
        x1 =int (max (0 ,min (W -1 ,x1 )))
        y1 =int (max (0 ,min (H -1 ,y1 )))
        x2 =int (max (x1 +1 ,min (W ,x2 )))
        y2 =int (max (y1 +1 ,min (H ,y2 )))
        bw =max (1 ,x2 -x1 )
        bh =max (1 ,y2 -y1 )
        area_ratio =float (bw *bh )/max (1.0 ,float (H *W ))
        c =float (confs [i ])if i <len (confs )else 0.5 
        score =0.70 *c +0.30 *area_ratio 
        if score >best_score :
            best_score =score 
            best =(x1 ,y1 ,x2 ,y2 )
    return best 


def _sequential_cluster_assign_np (
features :np .ndarray ,
threshold :float ,
momentum :float =0.9 ,
):
    """Sequential open-world clustering used as fallback when global clustering is unavailable."""
    n ,d =features .shape 
    if n ==0 :
        return np .zeros ((0 ,),dtype =np .int32 ),0 
    capacity =128 
    protos =np .zeros ((capacity ,d ),dtype =np .float32 )
    k =0 
    pred =np .zeros ((n ,),dtype =np .int32 )
    for i in range (n ):
        f =features [i ]
        if k ==0 :
            protos [0 ]=f 
            pred [i ]=0 
            k =1 
            continue 
        sims =protos [:k ]@f 
        best_idx =int (np .argmax (sims ))
        best_sim =float (sims [best_idx ])
        if best_sim >=float (threshold ):
            pred [i ]=best_idx 
            updated =float (momentum )*protos [best_idx ]+(1.0 -float (momentum ))*f 
            protos [best_idx ]=updated /(float (np .linalg .norm (updated ))+1e-12 )
        else :
            if k >=capacity :
                new_capacity =capacity *2 
                new_protos =np .zeros ((new_capacity ,d ),dtype =np .float32 )
                new_protos [:capacity ]=protos 
                protos =new_protos 
                capacity =new_capacity 
            protos [k ]=f 
            pred [i ]=k 
            k +=1 
    return pred ,k 


def _cluster_embeddings_global (
features :np .ndarray ,
*,
threshold :float ,
method :str ='auto',
agglomerative_max_n :int =5000 ,
momentum :float =0.9 ,
sim_cosine_w :float =0.7 ,
sim_euclid_w :float =0.3 ,
mp_max_slots :int =4 ,
mp_spawn_similarity :float =0.68 ,
mp_topk :int =2 ,
mp_aggregate_weight :float =0.75 ,
seed :int =42 ,
):
    """
    Order-independent clustering for directory-image mode.
    - Prefer agglomerative clustering on cosine distance for moderate N.
    - Fallback to shuffled sequential clustering for large N or missing sklearn.
    """
    n =int (features .shape [0 ])if features is not None else 0 
    if n <=0 :
        return np .zeros ((0 ,),dtype =np .int32 ),'none'

    method =str (method or 'auto').lower ()
    threshold =float (threshold )
    agglomerative_max_n =max (1 ,int (agglomerative_max_n ))
    use_agglomerative =(method in {'auto','agglomerative'})and (n <=agglomerative_max_n )
    use_multi_proto =method in {'multi_proto','spherical_multi'}

    if use_agglomerative :
        try :
            from sklearn .cluster import AgglomerativeClustering 

            dist_th =1.0 -threshold 
            dist_th =max (1e-6 ,min (1.999 ,float (dist_th )))
            cl =AgglomerativeClustering (
            n_clusters =None ,
            metric ='cosine',
            linkage ='average',
            distance_threshold =dist_th ,
            )
            labels =cl .fit_predict (features )
            return labels .astype (np .int32 ,copy =False ),'agglomerative'
        except Exception as e :
            if method =='agglomerative':
                raise RuntimeError (f"Agglomerative clustering failed: {e}")from e 

    if use_multi_proto :
        labels =cluster_embeddings_multiproto (
        features ,
        threshold =threshold ,
        momentum =float (momentum ),
        max_slots =int (mp_max_slots ),
        spawn_similarity =float (mp_spawn_similarity ),
        topk =int (mp_topk ),
        aggregate_weight =float (mp_aggregate_weight ),
        sim_cosine_w =float (sim_cosine_w ),
        sim_euclid_w =float (sim_euclid_w ),
        spherical =bool (method =='spherical_multi'),
        seed =int (seed ),
        )
        return labels .astype (np .int32 ,copy =False ),str (method )

    # Sequential fallback with shuffle to reduce order bias.
    order =np .arange (n ,dtype =np .int32 )
    if n >1 :
        rng =np .random .default_rng (int (seed ))
        rng .shuffle (order )
    pred_perm ,_ =_sequential_cluster_assign_np (
    features [order ],
    threshold =threshold ,
    momentum =float (momentum ),
    )
    labels =np .zeros ((n ,),dtype =np .int32 )
    labels [order ]=pred_perm 
    return labels ,'sequential'


def _estimate_cluster_count_eigengap (features :np .ndarray ,max_k :int =24 )->int :
    """
    Estimate plausible cluster count from eigengap on cosine-affinity graph.
    This is label-free and used only to regularize automatic threshold search.
    """
    n =int (features .shape [0 ])if features is not None else 0 
    if n <4 :
        return 1 
    kmax =int (max (2 ,min (max_k ,n -2 )))
    try :
        S =features @features .T 
        A =np .clip ((S +1.0 )*0.5 ,0.0 ,1.0 )
        np .fill_diagonal (A ,0.0 )
        d =A .sum (axis =1 )
        d =np .maximum (d ,1e-8 )
        inv_sqrt =1.0 /np .sqrt (d )
        L =np .eye (n ,dtype =np .float64 )-(inv_sqrt [:,None ]*A *inv_sqrt [None ,:])
        evals =np .linalg .eigvalsh (L )
        evals =np .sort (np .real (evals ))
        gaps =evals [1 :kmax +1 ]-evals [:kmax ]
        if gaps .size <=0 :
            return min (9 ,kmax )
        k_est =int (np .argmax (gaps )+1 )
        k_est =int (max (2 ,min (k_est ,kmax )))
        return k_est 
    except Exception :
        return min (9 ,kmax )


def _auto_select_twopass_threshold (
features :np .ndarray ,
*,
method :str ='auto',
agglomerative_max_n :int =5000 ,
momentum :float =0.9 ,
sim_cosine_w :float =0.7 ,
sim_euclid_w :float =0.3 ,
mp_max_slots :int =4 ,
mp_spawn_similarity :float =0.68 ,
mp_topk :int =2 ,
mp_aggregate_weight :float =0.75 ,
target_cluster_size :float =26.0 ,
seed :int =42 ,
verbose :bool =False ,
):
    """
    Auto-select clustering threshold using unsupervised criteria.
    Search is fixed and shared across datasets (no per-dataset manual tuning).
    """
    n =int (features .shape [0 ])if features is not None else 0 
    if n <=1 :
        return 0.17 ,np .zeros ((n ,),dtype =np .int32 ),'none',[]

    # Fixed search grid for reproducibility.
    candidates =[round (x ,3 )for x in np .arange (0.18 ,0.58 +1e-9 ,0.02 ).tolist ()]
    k_est =_estimate_cluster_count_eigengap (features ,max_k =24 )
    target_cluster_size =float (max (4.0 ,float (target_cluster_size )))
    k_size_prior =int (round (float (n )/target_cluster_size ))
    k_size_prior =int (max (2 ,min (k_size_prior ,max (2 ,n -1 ))))
    k_low =int (max (2 ,np .floor (0.60 *k_size_prior )))
    k_high =int (max (k_low +1 ,np .ceil (1.90 *k_size_prior )))
    stats =[]
    best =None 
    labels_best =None 
    method_best ='none'
    for th in candidates :
        labels ,m_used =_cluster_embeddings_global (
        features ,
        threshold =float (th ),
        method =method ,
        agglomerative_max_n =agglomerative_max_n ,
        momentum =momentum ,
        sim_cosine_w =sim_cosine_w ,
        sim_euclid_w =sim_euclid_w ,
        mp_max_slots =mp_max_slots ,
        mp_spawn_similarity =mp_spawn_similarity ,
        mp_topk =mp_topk ,
        mp_aggregate_weight =mp_aggregate_weight ,
        seed =seed ,
        )
        uniq ,counts =np .unique (labels ,return_counts =True )
        k =int (uniq .size )
        if k <=0 :
            continue 
        max_share =float (counts .max ()/max (1 ,n ))
        tiny_ratio =float (np .mean (counts <=2 ))if counts .size >0 else 1.0 

        sil =0.0 
        if 1 <k <n :
            try :
                from sklearn .metrics import silhouette_score 
                sil =float (silhouette_score (features ,labels ,metric ='cosine'))
            except Exception :
                sil =0.0 

        # Robust unsupervised scoring:
        # - Keep high separation (silhouette)
        # - Strongly avoid collapse (too few clusters / giant cluster)
        # - Keep moderate preference around dataset-size prior (n / target_cluster_size)
        k_pen_size =float (abs (np .log ((k +1.0 )/(k_size_prior +1.0 ))))
        k_pen_eig =float (abs (np .log ((k +1.0 )/(k_est +1.0 ))))
        low_k_pen =float (max (0.0 ,(k_low -k )/max (1.0 ,float (k_low ))))
        high_k_pen =float (max (0.0 ,(k -k_high )/max (1.0 ,float (k_high ))))
        collapse_pen =float (max (0.0 ,(max_share -0.36 )/0.20 ))
        score =(
        1.10 *sil 
        +0.45 *(1.0 -max_share )
        +0.25 *(1.0 -tiny_ratio )
        -0.22 *k_pen_size 
        -0.10 *k_pen_eig 
        -1.20 *low_k_pen 
        -0.30 *high_k_pen 
        -0.90 *collapse_pen 
        )
        row ={
        'th':float (th ),
        'score':float (score ),
        'k':k ,
        'sil':float (sil ),
        'max_share':float (max_share ),
        'tiny_ratio':float (tiny_ratio ),
        'k_est':int (k_est ),
        'k_prior':int (k_size_prior ),
        'k_low':int (k_low ),
        'k_high':int (k_high ),
        'k_pen_size':float (k_pen_size ),
        'k_pen_eig':float (k_pen_eig ),
        'low_k_pen':float (low_k_pen ),
        'high_k_pen':float (high_k_pen ),
        'collapse_pen':float (collapse_pen ),
        'method':str (m_used ),
        }
        stats .append (row )
        if best is None or row ['score']>best ['score']:
            best =row 
            labels_best =labels 
            method_best =str (m_used )

    if best is None :
        labels ,m_used =_cluster_embeddings_global (
        features ,
        threshold =0.17 ,
        method =method ,
        agglomerative_max_n =agglomerative_max_n ,
        momentum =momentum ,
        sim_cosine_w =sim_cosine_w ,
        sim_euclid_w =sim_euclid_w ,
        mp_max_slots =mp_max_slots ,
        mp_spawn_similarity =mp_spawn_similarity ,
        mp_topk =mp_topk ,
        mp_aggregate_weight =mp_aggregate_weight ,
        seed =seed ,
        )
        return 0.17 ,labels ,str (m_used ),stats 

    if verbose :
        top =sorted (stats ,key =lambda x :x ['score'],reverse =True )[:5 ]
        print (
        f"[INFO] Auto-threshold search: k_est={int(k_est)}, k_prior={int(k_size_prior)}, "
        f"accept_k=[{int(k_low)},{int(k_high)}], candidates={len(stats)}"
        )
        for i ,r in enumerate (top ,start =1 ):
            print (
            f"  [{i}] th={r['th']:.3f}, score={r['score']:.4f}, k={r['k']}, "
            f"sil={r['sil']:.4f}, max_share={r['max_share']:.3f}, tiny={r['tiny_ratio']:.3f}, "
            f"lowk={r['low_k_pen']:.3f}, collapse={r['collapse_pen']:.3f}"
            )
    return float (best ['th']),labels_best ,method_best ,stats 


def _refine_labels_by_secondary_split (
labels :np .ndarray ,
features :np .ndarray ,
*,
base_threshold :float ,
method :str ='auto',
agglomerative_max_n :int =5000 ,
momentum :float =0.9 ,
sim_cosine_w :float =0.7 ,
sim_euclid_w :float =0.3 ,
mp_max_slots :int =4 ,
mp_spawn_similarity :float =0.68 ,
mp_topk :int =2 ,
mp_aggregate_weight :float =0.75 ,
seed :int =42 ,
min_cluster_size :int =18 ,
min_subcluster_size :int =4 ,
split_delta :float =0.08 ,
verbose :bool =False ,
):
    """
    Split only low-compactness large clusters to reduce contamination.
    This refinement is deterministic and uses one fixed rule across datasets.
    """
    if labels is None or features is None :
        return labels ,[]
    labels =labels .astype (np .int32 ,copy =True )
    n =int (features .shape [0 ])
    if n <=1 :
        return labels ,[]

    min_cluster_size =int (max (4 ,int (min_cluster_size )))
    min_subcluster_size =int (max (2 ,int (min_subcluster_size )))
    split_delta =float (max (0.0 ,float (split_delta )))
    base_threshold =float (max (-1.0 ,min (1.0 ,float (base_threshold ))))

    split_log =[]
    next_label =int (labels .max ()) +1 if labels .size >0 else 0
    uniq =np .unique (labels )
    for lb in uniq :
        idxs =np .where (labels ==lb )[0 ]
        m =int (idxs .size )
        if m <min_cluster_size :
            continue 

        sub_feat =features [idxs ]
        cen =np .mean (sub_feat ,axis =0 )
        cen_norm =float (np .linalg .norm (cen ))
        if cen_norm <=1e-12 :
            continue 
        cen =cen /cen_norm 
        sims =sub_feat @cen 
        q25 =float (np .percentile (sims ,25 ))
        sstd =float (np .std (sims ))
        # Keep compact clusters untouched; only split clearly mixed clusters.
        if q25 >=base_threshold +0.015 and sstd <=0.09 :
            continue 

        th_split =float (min (0.90 ,max (base_threshold +split_delta ,base_threshold )))
        sub_method ='agglomerative'if str (method )=='auto'else str (method )
        try :
            sub_labels ,_ =_cluster_embeddings_global (
            sub_feat ,
            threshold =th_split ,
            method =sub_method ,
            agglomerative_max_n =int (max (200 ,min (agglomerative_max_n ,m *4 ))),
            momentum =momentum ,
            sim_cosine_w =sim_cosine_w ,
            sim_euclid_w =sim_euclid_w ,
            mp_max_slots =mp_max_slots ,
            mp_spawn_similarity =mp_spawn_similarity ,
            mp_topk =mp_topk ,
            mp_aggregate_weight =mp_aggregate_weight ,
            seed =seed ,
            )
        except Exception :
            continue 

        su ,sc =np .unique (sub_labels ,return_counts =True )
        k_sub =int (su .size )
        if k_sub <=1 :
            continue 
        if int (sc .min ())<min_subcluster_size :
            continue 
        if k_sub >max (2 ,m //min_subcluster_size ):
            continue 

        first =True 
        new_ids =[]
        for su_i in su :
            gidx =idxs [sub_labels ==su_i ]
            if first :
                labels [gidx ]=int (lb )
                new_ids .append (int (lb ))
                first =False 
            else :
                labels [gidx ]=int (next_label )
                new_ids .append (int (next_label ))
                next_label +=1 

        split_log .append ({
        'src':int (lb ),
        'k_sub':int (k_sub ),
        'size':int (m ),
        'q25':float (q25 ),
        'std':float (sstd ),
        'th_split':float (th_split ),
        'new_ids':new_ids ,
        })

    if labels .size >0 :
        uniq2 =np .unique (labels )
        remap ={int (u ):i for i ,u in enumerate (uniq2 .tolist ())}
        out =np .zeros_like (labels )
        for old ,new in remap .items ():
            out [labels ==old ]=int (new )
        labels =out 

    if verbose and len (split_log )>0 :
        print (f"[INFO] Split-refine applied: splits={len(split_log)}")
    return labels ,split_log 


def _merge_labels_toward_target (
labels :np .ndarray ,
features :np .ndarray ,
male_probs :Optional [np .ndarray ]=None ,
*,
target_k :int ,
min_sim :float =0.45 ,
):
    """
    Greedy centroid-merge to reduce over-splitting.
    Merge only when centroid similarity is high enough.
    """
    if labels is None or features is None :
        return labels ,[]
    n =int (features .shape [0 ])
    if n <=0 :
        return labels ,[]
    labels =labels .astype (np .int32 ,copy =True )
    min_sim =float (min_sim )
    merge_log =[]

    def _reindex (lb :np .ndarray )->np .ndarray :
        uniq =np .unique (lb )
        mp ={int (u ):i for i ,u in enumerate (uniq .tolist ())}
        out =np .zeros_like (lb )
        for old ,new in mp .items ():
            out [lb ==old ]=int (new )
        return out 

    labels =_reindex (labels )
    while True :
        uniq =np .unique (labels )
        k =int (uniq .size )
        if k <=max (1 ,int (target_k )):
            break 
        centroids =[]
        male_mean =[]
        counts =[]
        for u in uniq :
            idxs =np .where (labels ==u )[0 ]
            counts .append (int (idxs .size ))
            cen =np .mean (features [idxs ],axis =0 )
            cen =cen /(float (np .linalg .norm (cen ))+1e-12 )
            centroids .append (cen )
            if male_probs is not None and male_probs .size ==n :
                male_mean .append (float (np .mean (male_probs [idxs ])))
            else :
                male_mean .append (0.5 )
        C =np .stack (centroids ,axis =0 )
        S =C @C .T 
        np .fill_diagonal (S ,-10.0 )

        best =None 
        best_score =-10.0 
        for i in range (k ):
            for j in range (i +1 ,k ):
                sim =float (S [i ,j ])
                if sim <min_sim :
                    continue 
                # Prevent obvious opposite-gender merges when both clusters are confident.
                mi =float (male_mean [i ])
                mj =float (male_mean [j ])
                ci =abs (mi -0.5 )
                cj =abs (mj -0.5 )
                gi ='M'if mi >=0.5 else 'F'
                gj ='M'if mj >=0.5 else 'F'
                if gi !=gj and ci >=0.20 and cj >=0.20 :
                    continue 
                # Prefer merging small fragments into nearest large cluster.
                sz_bonus =0.05 *float (min (counts [i ],counts [j ])/max (1 ,max (counts [i ],counts [j ])))
                score =sim +sz_bonus 
                if score >best_score :
                    best_score =score 
                    best =(i ,j ,sim )
        if best is None :
            break 

        i ,j ,sim =best 
        li =int (uniq [i ])
        lj =int (uniq [j ])
        labels [labels ==lj ]=li 
        labels =_reindex (labels )
        merge_log .append ({
        'src':int (lj ),
        'dst':int (li ),
        'sim':float (sim ),
        'k_after':int (np .unique (labels ).size ),
        })
    return labels ,merge_log 


def _merge_labels_by_centroid_topology (
labels :np .ndarray ,
features :np .ndarray ,
male_probs :Optional [np .ndarray ]=None ,
*,
method :str ='agglomerative',
auto_threshold :bool =True ,
threshold :float =0.18 ,
fragment_group_size :float =16.0 ,
min_fragments :int =128 ,
agglomerative_max_n :int =5000 ,
momentum :float =0.9 ,
sim_cosine_w :float =0.7 ,
sim_euclid_w :float =0.3 ,
mp_max_slots :int =4 ,
mp_spawn_similarity :float =0.68 ,
mp_topk :int =2 ,
mp_aggregate_weight :float =0.75 ,
auto_merge :bool =False ,
merge_min_sim :float =0.40 ,
seed :int =42 ,
verbose :bool =False ,
):
    """
    Stage-2 merge on stage-1 fragment centroids.
    This targets the common failure mode on large ROI-image sets:
    strong features + very pure micro-clusters + severe over-splitting.
    """
    if labels is None or features is None :
        return labels ,{'applied':False ,'reason':'empty'}
    labels =np .asarray (labels ,dtype =np .int32 )
    features =np .asarray (features ,dtype =np .float32 )
    if labels .size <=1 or features .shape [0 ]!=labels .size :
        return labels ,{'applied':False ,'reason':'invalid-shape'}

    uniq =np .unique (labels )
    k_frag =int (uniq .size )
    if k_frag <int (max (2 ,min_fragments )):
        return labels ,{'applied':False ,'reason':'few-fragments','stage1_fragments':k_frag}

    centroids =[]
    frag_male =[]
    for lb in uniq :
        idxs =np .where (labels ==lb )[0 ]
        cen =np .mean (features [idxs ],axis =0 )
        cen =cen /(float (np .linalg .norm (cen ))+1e-12 )
        centroids .append (cen .astype (np .float32 ,copy =False ))
        if male_probs is not None and male_probs .size ==labels .size :
            frag_male .append (float (np .mean (male_probs [idxs ])))
        else :
            frag_male .append (0.5 )
    centroids =np .stack (centroids ,axis =0 ).astype (np .float32 ,copy =False )
    frag_male =np .asarray (frag_male ,dtype =np .float32 )

    if bool (auto_threshold ):
        th2 ,frag_labels ,used_method ,_ =_auto_select_twopass_threshold (
        centroids ,
        method =str (method or 'agglomerative'),
        agglomerative_max_n =int (agglomerative_max_n ),
        momentum =float (momentum ),
        sim_cosine_w =float (sim_cosine_w ),
        sim_euclid_w =float (sim_euclid_w ),
        mp_max_slots =int (mp_max_slots ),
        mp_spawn_similarity =float (mp_spawn_similarity ),
        mp_topk =int (mp_topk ),
        mp_aggregate_weight =float (mp_aggregate_weight ),
        target_cluster_size =float (max (1.0 ,fragment_group_size )),
        seed =int (seed ),
        verbose =False ,
        )
    else :
        th2 =float (threshold )
        frag_labels ,used_method =_cluster_embeddings_global (
        centroids ,
        threshold =float (th2 ),
        method =str (method or 'agglomerative'),
        agglomerative_max_n =int (agglomerative_max_n ),
        momentum =float (momentum ),
        sim_cosine_w =float (sim_cosine_w ),
        sim_euclid_w =float (sim_euclid_w ),
        mp_max_slots =int (mp_max_slots ),
        mp_spawn_similarity =float (mp_spawn_similarity ),
        mp_topk =int (mp_topk ),
        mp_aggregate_weight =float (mp_aggregate_weight ),
        seed =int (seed ),
        )

    if bool (auto_merge ):
        k_before =int (np .unique (frag_labels ).size )
        target_k =int (round (float (k_frag )/max (1.0 ,float (fragment_group_size ))))
        target_k =int (max (2 ,min (k_before ,target_k )))
        if target_k <k_before :
            frag_labels ,_ =_merge_labels_toward_target (
            frag_labels ,
            centroids ,
            male_probs =frag_male ,
            target_k =target_k ,
            min_sim =float (merge_min_sim ),
            )

    lb_map ={int (lb ):int (frag_labels [i ])for i ,lb in enumerate (uniq .tolist ())}
    merged =np .zeros_like (labels ,dtype =np .int32 )
    for old_lb ,new_lb in lb_map .items ():
        merged [labels ==old_lb ]=int (new_lb )
    uniq2 =np .unique (merged )
    remap ={int (u ):i for i ,u in enumerate (uniq2 .tolist ())}
    out =np .zeros_like (merged ,dtype =np .int32 )
    for old_lb ,new_lb in remap .items ():
        out [merged ==old_lb ]=int (new_lb )

    info ={
    'applied':True ,
    'stage1_fragments':int (k_frag ),
    'stage2_clusters':int (np .unique (out ).size ),
    'threshold':float (th2 ),
    'method':str (used_method ),
    }
    if verbose :
        print (
        f"[INFO] Centroid-topology merge: k {int(k_frag)}->{int(info['stage2_clusters'])}, "
        f"method={str(used_method)}, threshold={float(th2):.3f}"
        )
    return out ,info 


def _build_aux_aware_cluster_features (features :np .ndarray ,samples ,args ):
    """Lightly inject age/gender cues into clustering without replacing ReID features."""
    if features is None or len (features )==0 :
        return features 
    try :
        reid_w =float (getattr (args ,'twopass_cluster_aux_reid_weight',0.0 )or 0.0 )
    except Exception :
        reid_w =0.0 
    try :
        gender_w =float (getattr (args ,'twopass_cluster_aux_gender_weight',0.0 )or 0.0 )
    except Exception :
        gender_w =0.0 
    try :
        age_w =float (getattr (args ,'twopass_cluster_aux_age_weight',0.0 )or 0.0 )
    except Exception :
        age_w =0.0 
    if reid_w <=0.0 and gender_w <=0.0 and age_w <=0.0 :
        return features 

    base =np .asarray (features ,dtype =np .float32 )
    n =int (base .shape [0 ])
    aux_parts =[]
    if reid_w >0.0 :
        try :
            aux_feat_list =[]
            valid =True 
            for s in samples :
                aux_feat =s .get ('aux_feat',None )if isinstance (s ,dict )else None 
                if aux_feat is None :
                    valid =False 
                    break 
                aux_feat_arr =np .asarray (aux_feat ,dtype =np .float32 ).reshape (-1 )
                if aux_feat_arr .size <=0 :
                    valid =False 
                    break 
                aux_feat_list .append (aux_feat_arr )
            if valid and len (aux_feat_list )==n :
                aux_feat_mat =np .stack (aux_feat_list ,axis =0 )
                aux_norms =np .linalg .norm (aux_feat_mat ,axis =1 ,keepdims =True )
                aux_feat_mat =aux_feat_mat /np .clip (aux_norms ,1e-12 ,None )
                aux_parts .append (aux_feat_mat *float (reid_w ))
        except Exception :
            pass 
    if gender_w >0.0 :
        try :
            male_vals =np .asarray ([float (s .get ('male_p',0.5 )or 0.5 )for s in samples ],dtype =np .float32 )
            male_centered =((male_vals -0.5 )*2.0 )[:,None ]
            aux_parts .append (male_centered *float (gender_w ))
        except Exception :
            pass 
    if age_w >0.0 :
        try :
            age_vals =np .asarray ([float (s .get ('age_p',0.0 )or 0.0 )for s in samples ],dtype =np .float32 )
            med =float (np .median (age_vals ))
            mad =float (np .median (np .abs (age_vals -med )))
            scale =max (1.5 ,1.4826 *mad )
            age_norm =((age_vals -med )/scale )[:,None ]
            age_norm =np .clip (age_norm ,-3.0 ,3.0 )
            aux_parts .append (age_norm *float (age_w ))
        except Exception :
            pass 
    if not aux_parts :
        return base 
    aug =np .concatenate ([base ]+aux_parts ,axis =1 )
    norms =np .linalg .norm (aug ,axis =1 ,keepdims =True )
    aug =aug /np .clip (norms ,1e-12 ,None )
    return aug .astype (np .float32 ,copy =False )


def _extract_rois_for_twopass_image (
args ,
inference ,
yolo ,
sam_predictor ,
device ,
image_path :str ,
):
    """Extract valid ROI tensors for one image without assigning IDs."""
    frame_bgr =_imread_unicode (image_path )
    if frame_bgr is None :
        return {'status':'READ_FAIL','frame_bgr':None ,'rois':[]}

    is_dark ,mean_bright =_is_dark_frame (frame_bgr ,args =args )
    if is_dark :
        return {
        'status':f"IGNORED:DARK({mean_bright:.1f})",
        'frame_bgr':frame_bgr ,
        'rois':[],
        }
    frame_rgb =cv2 .cvtColor (frame_bgr ,cv2 .COLOR_BGR2RGB )
    H ,W =frame_rgb .shape [:2 ]

    image_as_roi =bool (getattr (args ,'image_as_roi',False ))
    if isinstance (getattr (args ,'image_as_roi',None ),(int ,np .integer )):
        image_as_roi =bool (int (getattr (args ,'image_as_roi')))

    rois =[]
    if image_as_roi :
        is_bad ,reasons =_is_low_quality_roi_image (frame_rgb ,args =args )
        if is_bad :
            msg =','.join (reasons [:2 ])or 'LOW_QUALITY'
            return {'status':f"IGNORED:{msg}",'frame_bgr':frame_bgr ,'rois':[]}
        vis_box =None 
        if bool (getattr (args ,'roi_vis_use_detector',True )):
            vis_box =_pick_roi_visual_box_from_detector (frame_bgr ,yolo ,args =args )
        if vis_box is None :
            mx =int (round (W *0.08 ))
            my =int (round (H *0.08 ))
            vis_box =(max (0 ,mx ),max (0 ,my ),max (1 ,W -mx ),max (1 ,H -my ))
        crop_rgb =frame_rgb 
        crop_pil =Image .fromarray (crop_rgb )
        tensor =inference .transform (crop_pil ).unsqueeze (0 ).to (device )
        rois .append ({
        'bbox':tuple (int (x )for x in vis_box ),
        'crop_rgb':crop_rgb ,
        'tensor':tensor ,
        'det_conf':1.0 ,
        'roi_source':'full_image' ,
        })
        return {'status':'OK','frame_bgr':frame_bgr ,'rois':rois }

    # Detection + optional SAM refinement
    if sam_predictor is not None :
        try :
            sam_predictor .set_image (frame_rgb )
        except Exception :
            sam_predictor =None 

    try :
        results =yolo .predict (
        source =frame_bgr ,
        conf =float (args .det_conf ),
        iou =float (args .det_iou ),
        verbose =False ,
        )
    except Exception :
        return {'status':'NO_DET','frame_bgr':frame_bgr ,'rois':[]}
    if not results :
        return {'status':'NO_DET','frame_bgr':frame_bgr ,'rois':[]}
    boxes =results [0 ].boxes 
    if boxes is None or len (boxes )==0 :
        return {'status':'NO_DET','frame_bgr':frame_bgr ,'rois':[]}

    try :
        xyxy =boxes .xyxy .cpu ().numpy ().astype (int )
    except Exception :
        xyxy =np .asarray (boxes .xyxy ,dtype =np .float32 ).astype (int )
    try :
        confs =boxes .conf .cpu ().numpy ().astype (float )
    except Exception :
        confs =np .ones ((len (xyxy ),),dtype =np .float32 )*0.5 

    for det_i ,(x1 ,y1 ,x2 ,y2 )in enumerate (xyxy ):
        x1 =int (max (0 ,min (W -1 ,x1 )))
        y1 =int (max (0 ,min (H -1 ,y1 )))
        x2 =int (max (x1 +1 ,min (W ,x2 )))
        y2 =int (max (y1 +1 ,min (H ,y2 )))
        crop_rgb =None 
        mask =None 
        if sam_predictor is not None :
            try :
                box =np .array ([x1 ,y1 ,x2 ,y2 ],dtype =np .float32 )[None ,:]
                masks ,scores ,_ =sam_predictor .predict (box =box ,multimask_output =True )
                if masks is not None and len (masks )>0 :
                    best =int (np .argmax (scores ))
                    mask =masks [best ].astype (bool )
                    crop_rgb =inference ._process_mask_region_to_rectangle (frame_rgb ,mask )
            except Exception :
                crop_rgb =None 
                mask =None 
        if crop_rgb is None or crop_rgb .size ==0 :
            crop_rgb =_crop_roi_with_expansion_numpy (frame_rgb ,(x1 ,y1 ,x2 ,y2 ),expand_ratio =0.1 )
            mask =None 
        if crop_rgb is None or crop_rgb .size ==0 :
            continue 

        is_bad ,_ =_is_low_quality_roi (
        frame_rgb ,(x1 ,y1 ,x2 ,y2 ),crop_rgb ,mask =mask ,args =args 
        )
        if is_bad :
            continue 

        crop_pil =Image .fromarray (crop_rgb )
        tensor =inference .transform (crop_pil ).unsqueeze (0 ).to (device )
        det_conf =float (confs [det_i ])if det_i <len (confs )else 0.5 
        rois .append ({
        'bbox':(x1 ,y1 ,x2 ,y2 ),
        'crop_rgb':crop_rgb ,
        'tensor':tensor ,
        'det_conf':det_conf ,
        'roi_source':('sam_mask' if mask is not None else 'bbox_fallback') ,
        })

    if len (rois )==0 :
        return {'status':'NO_VALID_ROI','frame_bgr':frame_bgr ,'rois':[]}

    max_rois =int (getattr (args ,'max_rois_per_image',1 )or 1 )
    if max_rois >0 and len (rois )>max_rois :
        scored =[]
        for r in rois :
            x1 ,y1 ,x2 ,y2 =r ['bbox']
            area_ratio =float (max (1 ,x2 -x1 )*max (1 ,y2 -y1 ))/max (1.0 ,float (H *W ))
            score =0.75 *float (r .get ('det_conf',0.5 ))+0.25 *area_ratio 
            scored .append ((score ,r ))
        scored .sort (key =lambda x :x [0 ],reverse =True )
        rois =[r for _,r in scored [:max_rois ]]

    return {'status':'OK','frame_bgr':frame_bgr ,'rois':rois }


def run_image_dirmode_twopass (
args ,
inference ,
yolo ,
sam_predictor ,
device ,
*,
input_dir :str ,
output_dir :str ,
all_image_paths ,
):
    """
    Two-pass directory image inference:
    1) Extract all ROI features (YOLO+SAM kept).
    2) Cluster globally (order-independent when agglomerative is used).
    """
    if not all_image_paths :
        return 0 ,0 ,[]

    print ("[INFO] Image directory mode: two-pass global clustering enabled.")
    img_count =0 
    img_kept =0 
    detection_records =[]
    detection_details =[]

    samples =[]
    fallback_saved =0 
    old_input =getattr (args ,'input',None )
    old_output =getattr (args ,'output',None )

    for idx ,in_path in enumerate (all_image_paths ):
        img_count +=1 
        rel_path =os .path .relpath (in_path ,input_dir )
        extracted =_extract_rois_for_twopass_image (
        args ,inference ,yolo ,sam_predictor ,device ,in_path 
        )
        status =str (extracted .get ('status','OK'))
        frame_bgr =extracted .get ('frame_bgr',None )
        rois =extracted .get ('rois',[])or []

        if status !='OK'or len (rois )==0 :
            if frame_bgr is None :
                frame_bgr =np .zeros ((320 ,320 ,3 ),dtype =np .uint8 )
            vis =frame_bgr .copy ()
            _draw_status_label (vis ,status ,color =(0 ,0 ,255 ))
            args .input =in_path 
            if not bool (getattr (inference ,'group_by_id',False )):
                out_path =os .path .join (output_dir ,rel_path )
                if bool (getattr (args ,'save_vis',True )):
                    out_dir =os .path .dirname (out_path )
                    if out_dir :
                        _ensure_dir (out_dir )
                args .output =out_path 
            _save_image_visualization (
            args ,
            inference ,
            vis ,
            input_rel_path =rel_path ,
            fallback_folder ="IGNORED",
            )
            fallback_saved +=1 
            continue 

        blend_full =bool (getattr (args ,'twopass_fuse_full_image',False ))
        blend_w =float (getattr (args ,'twopass_full_image_weight',0.0 )or 0.0 )
        blend_w =max (0.0 ,min (1.0 ,blend_w ))
        use_blend =(blend_full and (not bool (getattr (args ,'image_as_roi',False )))and frame_bgr is not None and blend_w >0.0 )

        with torch .no_grad ():
            inp_list =[r ['tensor']for r in rois ]
            crops_for_aux =[r ['crop_rgb']for r in rois ]
            if use_blend :
                try :
                    full_rgb =cv2 .cvtColor (frame_bgr ,cv2 .COLOR_BGR2RGB )
                    full_pil =Image .fromarray (full_rgb )
                    full_tensor =inference .transform (full_pil ).unsqueeze (0 ).to (device )
                    inp_list .append (full_tensor )
                    crops_for_aux .append (full_rgb )
                except Exception :
                    use_blend =False 
            inp =torch .cat (inp_list ,dim =0 )
            feat_after_bn ,male_probs ,age_pred ,aux_feat_after_bn =_forward_reid_and_aux (
            inference ,
            inp ,
            crops_rgb =crops_for_aux ,
            return_aux_feat =True ,
            fuse_aux_into_main =False ,
            )
            feat_after_bn =F .normalize (feat_after_bn ,p =2 ,dim =1 )
            if aux_feat_after_bn is not None :
                aux_feat_after_bn =F .normalize (aux_feat_after_bn ,p =2 ,dim =1 )

        if use_blend :
            roi_feat =feat_after_bn [:-1 ]
            roi_male =male_probs [:-1 ]
            roi_age =age_pred [:-1 ]
            full_feat =feat_after_bn [-1 ]
            full_male =male_probs [-1 ]
            full_age =age_pred [-1 ]
            if aux_feat_after_bn is not None and int (aux_feat_after_bn .shape [0 ])==int (feat_after_bn .shape [0 ]):
                roi_aux_feat =aux_feat_after_bn [:-1 ]
                full_aux_feat =aux_feat_after_bn [-1 ]
            else :
                roi_aux_feat =None 
                full_aux_feat =None 
        else :
            roi_feat =feat_after_bn 
            roi_male =male_probs 
            roi_age =age_pred 
            full_feat =None 
            full_male =None 
            full_age =None 
            roi_aux_feat =aux_feat_after_bn 
            full_aux_feat =None 

        for j ,r in enumerate (rois ):
            f_j =roi_feat [j ]
            male_j =roi_male [j ]
            age_j =roi_age [j ]
            aux_f_j =None if roi_aux_feat is None else roi_aux_feat [j ]
            if use_blend and full_feat is not None :
                f_j =F .normalize ((1.0 -blend_w )*f_j +blend_w *full_feat ,p =2 ,dim =0 )
                male_j =(1.0 -blend_w )*male_j +blend_w *full_male 
                age_j =(1.0 -blend_w )*age_j +blend_w *full_age 
                if aux_f_j is not None and full_aux_feat is not None :
                    aux_f_j =F .normalize ((1.0 -blend_w )*aux_f_j +blend_w *full_aux_feat ,p =2 ,dim =0 )

            feat =f_j .detach ().cpu ().numpy ().astype (np .float32 ,copy =False )
            aux_feat_np =None 
            if aux_f_j is not None :
                aux_feat_np =aux_f_j .detach ().cpu ().numpy ().astype (np .float32 ,copy =False )
            samples .append ({
            'image_path':in_path ,
            'rel_path':rel_path ,
            'bbox':tuple (int (x )for x in r ['bbox']),
            'feat':feat ,
            'aux_feat':aux_feat_np ,
            'male_p':float (male_j .item ()),
            'age_p':float (age_j .item ()),
            'roi_source':str (r .get ('roi_source','unknown')or 'unknown'),
            })

        if idx %100 ==0 or idx ==len (all_image_paths )-1 :
            print (f"[INFO] Pass1 extraction progress: {idx + 1}/{len(all_image_paths)} images, rois={len(samples)}")

    if len (samples )==0 :
        args .input =old_input 
        args .output =old_output 
        print ("[WARN] No valid ROI extracted for two-pass clustering.")
        return img_count ,0 ,detection_records 

    features =np .stack ([s ['feat']for s in samples ],axis =0 ).astype (np .float32 ,copy =False )
    cluster_features =_build_aux_aware_cluster_features (features ,samples ,args )
    th_manual =float (
    getattr (args ,'twopass_threshold',None )
    if getattr (args ,'twopass_threshold',None )is not None
    else (
    getattr (args ,'base_threshold',None )
    if getattr (args ,'base_threshold',None )is not None
    else getattr (args ,'similarity_threshold',0.5 )or 0.5
    )
    )
    use_auto_th =bool (getattr (args ,'twopass_threshold_auto',True ))
    if use_auto_th :
        th ,labels ,cluster_method ,_th_stats =_auto_select_twopass_threshold (
        cluster_features ,
        method =str (getattr (args ,'cluster_method','auto')or 'auto'),
        agglomerative_max_n =int (getattr (args ,'cluster_agglomerative_max_n',5000 )or 5000 ),
        momentum =float (getattr (args ,'cluster_momentum',0.9 )or 0.9 ),
        sim_cosine_w =float (getattr (args ,'sim_cosine_w',0.7 )or 0.7 ),
        sim_euclid_w =float (getattr (args ,'sim_euclid_w',0.3 )or 0.3 ),
        mp_max_slots =int (getattr (args ,'prototype_max_slots',4 )or 4 ),
        mp_spawn_similarity =float (getattr (args ,'prototype_spawn_similarity',0.68 )or 0.68 ),
        mp_topk =int (getattr (args ,'prototype_topk',2 )or 2 ),
        mp_aggregate_weight =float (getattr (args ,'prototype_aggregate_weight',0.75 )or 0.75 ),
        target_cluster_size =float (getattr (args ,'twopass_target_cluster_size',26.0 )or 26.0 ),
        seed =int (getattr (args ,'shuffle_seed',42 )or 42 ),
        verbose =bool (getattr (args ,'verbose',False )),
        )
    else :
        th =th_manual 
        labels ,cluster_method =_cluster_embeddings_global (
        cluster_features ,
        threshold =th ,
        method =str (getattr (args ,'cluster_method','auto')or 'auto'),
        agglomerative_max_n =int (getattr (args ,'cluster_agglomerative_max_n',5000 )or 5000 ),
        momentum =float (getattr (args ,'cluster_momentum',0.9 )or 0.9 ),
        sim_cosine_w =float (getattr (args ,'sim_cosine_w',0.7 )or 0.7 ),
        sim_euclid_w =float (getattr (args ,'sim_euclid_w',0.3 )or 0.3 ),
        mp_max_slots =int (getattr (args ,'prototype_max_slots',4 )or 4 ),
        mp_spawn_similarity =float (getattr (args ,'prototype_spawn_similarity',0.68 )or 0.68 ),
        mp_topk =int (getattr (args ,'prototype_topk',2 )or 2 ),
        mp_aggregate_weight =float (getattr (args ,'prototype_aggregate_weight',0.75 )or 0.75 ),
        seed =int (getattr (args ,'shuffle_seed',42 )or 42 ),
        )

    if bool (getattr (args ,'twopass_refine_split',False )):
        labels_refined ,split_log =_refine_labels_by_secondary_split (
        labels ,
        cluster_features ,
        base_threshold =float (th ),
        method =str (getattr (args ,'cluster_method','auto')or 'auto'),
        agglomerative_max_n =int (getattr (args ,'cluster_agglomerative_max_n',5000 )or 5000 ),
        momentum =float (getattr (args ,'cluster_momentum',0.9 )or 0.9 ),
        sim_cosine_w =float (getattr (args ,'sim_cosine_w',0.7 )or 0.7 ),
        sim_euclid_w =float (getattr (args ,'sim_euclid_w',0.3 )or 0.3 ),
        mp_max_slots =int (getattr (args ,'prototype_max_slots',4 )or 4 ),
        mp_spawn_similarity =float (getattr (args ,'prototype_spawn_similarity',0.68 )or 0.68 ),
        mp_topk =int (getattr (args ,'prototype_topk',2 )or 2 ),
        mp_aggregate_weight =float (getattr (args ,'prototype_aggregate_weight',0.75 )or 0.75 ),
        seed =int (getattr (args ,'shuffle_seed',42 )or 42 ),
        min_cluster_size =int (getattr (args ,'twopass_refine_min_size',18 )or 18 ),
        min_subcluster_size =int (getattr (args ,'twopass_refine_min_subcluster',4 )or 4 ),
        split_delta =float (getattr (args ,'twopass_refine_delta',0.08 )or 0.08 ),
        verbose =bool (getattr (args ,'verbose',False )),
        )
        if labels_refined is not None :
            labels =labels_refined 
        if bool (getattr (args ,'verbose',False ))and len (split_log )>0 :
            print (f"[INFO] Split-refine clusters: +{sum(max(0, int(r.get('k_sub',1))-1) for r in split_log)} ids")

    if bool (getattr (args ,'twopass_auto_merge',False )):
        uniq_before =np .unique (labels )
        k_before =int (uniq_before .size )
        if k_before >1 :
            target_sz =float (getattr (args ,'twopass_target_cluster_size',26.0 )or 26.0 )
            target_k =int (round (float (features .shape [0 ])/max (4.0 ,target_sz )))
            target_k =int (max (2 ,min (k_before ,target_k )))
            if target_k <k_before :
                male_arr =np .array ([float (s .get ('male_p',0.5 ))for s in samples ],dtype =np .float32 )
                labels_merged ,merge_log =_merge_labels_toward_target (
                labels ,
                cluster_features ,
                male_probs =male_arr ,
                target_k =target_k ,
                min_sim =float (getattr (args ,'twopass_merge_min_sim',0.45 )or 0.45 ),
                )
                if labels_merged is not None :
                    labels =labels_merged 
                if bool (getattr (args ,'verbose',False )):
                    print (
                    f"[INFO] Auto-merge clusters: k {k_before}->{int(np.unique(labels).size)} "
                    f"(target={target_k}, merges={len(merge_log)})"
                    )

    centroid_merge_info ={'applied':False }
    if bool (getattr (args ,'twopass_centroid_merge',False )):
        allow_centroid_stage =True 
        if bool (getattr (args ,'twopass_centroid_image_only',True )):
            allow_centroid_stage =bool (getattr (args ,'image_as_roi',False ))
        if allow_centroid_stage :
            male_arr =np .array ([float (s .get ('male_p',0.5 ))for s in samples ],dtype =np .float32 )
            labels_merged2 ,centroid_merge_info =_merge_labels_by_centroid_topology (
            labels ,
            cluster_features ,
            male_probs =male_arr ,
            method =str (getattr (args ,'twopass_centroid_method','agglomerative')or 'agglomerative'),
            auto_threshold =bool (getattr (args ,'twopass_centroid_threshold_auto',True )),
            threshold =float (getattr (args ,'twopass_centroid_threshold',0.18 )or 0.18 ),
            fragment_group_size =float (getattr (args ,'twopass_centroid_fragment_group_size',16.0 )or 16.0 ),
            min_fragments =int (getattr (args ,'twopass_centroid_min_fragments',128 )or 128 ),
            agglomerative_max_n =int (getattr (args ,'cluster_agglomerative_max_n',5000 )or 5000 ),
            momentum =float (getattr (args ,'cluster_momentum',0.9 )or 0.9 ),
            sim_cosine_w =float (getattr (args ,'sim_cosine_w',0.7 )or 0.7 ),
            sim_euclid_w =float (getattr (args ,'sim_euclid_w',0.3 )or 0.3 ),
            mp_max_slots =int (getattr (args ,'prototype_max_slots',4 )or 4 ),
            mp_spawn_similarity =float (getattr (args ,'prototype_spawn_similarity',0.68 )or 0.68 ),
            mp_topk =int (getattr (args ,'prototype_topk',2 )or 2 ),
            mp_aggregate_weight =float (getattr (args ,'prototype_aggregate_weight',0.75 )or 0.75 ),
            auto_merge =bool (getattr (args ,'twopass_centroid_auto_merge',False )),
            merge_min_sim =float (getattr (args ,'twopass_centroid_merge_min_sim',0.40 )or 0.40 ),
            seed =int (getattr (args ,'shuffle_seed',42 )or 42 ),
            verbose =bool (getattr (args ,'verbose',False )),
            )
            if labels_merged2 is not None and bool (centroid_merge_info .get ('applied',False )):
                labels =labels_merged2 
                cluster_method =f"{cluster_method}+centroid-{str(centroid_merge_info.get('method','agglomerative'))}"

    uniq =np .unique (labels )
    cluster_info =[]
    for lb in uniq :
        idxs =np .where (labels ==lb )[0 ]
        cluster_info .append ((int (lb ),int (idxs .size ),int (idxs .min ())))
    cluster_info .sort (key =lambda t :( -t [1 ],t [2 ]))

    global_to_display =getattr (inference ,'global_to_display',None )
    if global_to_display is None :
        global_to_display ={}
        setattr (inference ,'global_to_display',global_to_display )
    next_display_id =int (getattr (inference ,'next_display_id',1 )or 1 )
    global_gender_stats =getattr (inference ,'global_gender_stats',None )
    if global_gender_stats is None :
        global_gender_stats ={}
        setattr (inference ,'global_gender_stats',global_gender_stats )
    global_age_stats =getattr (inference ,'global_age_stats',None )
    if global_age_stats is None :
        global_age_stats ={}
        setattr (inference ,'global_age_stats',global_age_stats )

    label_to_gid ={}
    label_to_centroid ={}
    for rank ,(lb ,_sz ,_first_idx )in enumerate (cluster_info ,start =1 ):
        gid =f"cluster_{next_display_id + rank - 1:05d}"
        label_to_gid [int (lb )]=gid 
        global_to_display [gid ]=f"ID{next_display_id + rank - 1}"
        idxs =np .where (labels ==lb )[0 ]
        cen =np .mean (features [idxs ],axis =0 )
        cen =cen /(float (np .linalg .norm (cen ))+1e-12 )
        label_to_centroid [int (lb )]=cen .astype (np .float32 ,copy =False )

        male_vals =[float (samples [ii ]['male_p'])for ii in idxs ]
        age_vals =[float (samples [ii ]['age_p'])for ii in idxs ]
        a_male =1.0 
        a_female =1.0 
        for mp in male_vals :
            w =1.0 +2.0 *abs (mp -0.5 )
            a_male +=w *mp 
            a_female +=w *(1.0 -mp )
        male_mean =a_male /max (1e-6 ,a_male +a_female )
        stable_gender ='M'if male_mean >=float (args .gender_threshold )else 'F'
        global_gender_stats [gid ]={
        'alpha_male':float (a_male ),
        'alpha_female':float (a_female ),
        'stable_gender':stable_gender ,
        }
        global_age_stats [gid ]=age_vals 

    next_display_id +=len (cluster_info )
    setattr (inference ,'next_display_id',next_display_id )
    setattr (inference ,'global_to_display',global_to_display )
    setattr (inference ,'global_gender_stats',global_gender_stats )
    setattr (inference ,'global_age_stats',global_age_stats )

    gid_feature_map ={label_to_gid [lb ]:label_to_centroid [lb ]for lb in label_to_gid }
    # Persist image cluster centroids so detection_results.json can expose
    # identity_feature_summary for downstream raw-output topology merging.
    try :
        _feat_store =_identity_gid_features_store (inference )
        for _gid ,_feat in gid_feature_map .items ():
            _feat_store [str (_gid )]=_normalize_np_feature (_feat )
    except Exception :
        pass 
    gid_to_image_paths =defaultdict (set )
    by_image =defaultdict (list )
    for i ,s in enumerate (samples ):
        by_image [s ['image_path']].append (i )
        gid_to_image_paths [label_to_gid [int (labels [i ])]].add (str (s ['image_path']))
    for gid ,paths in gid_to_image_paths .items ():
        for path in paths :
            _register_gid_source (inference ,gid ,path ,'image')
    for idxs in by_image .values ():
        gids_in_image ={label_to_gid [int (labels [ii ])]for ii in idxs}
        _register_conflict_pairs_from_gids (inference ,gids_in_image )

    id_root =getattr (inference ,'id_group_root',None )if bool (getattr (inference ,'group_by_id',False ))else None 
    merge_alias_map ={}
    if bool (getattr (args ,'post_merge_enable_image',getattr (args ,'post_merge_enable',True ))):
        merge_alias_map =_consolidate_runtime_identities (
        inference =inference ,
        candidate_gids =list (gid_feature_map .keys ()),
        global_to_display =global_to_display ,
        global_gender_stats =global_gender_stats ,
        global_age_stats =global_age_stats ,
        args =args ,
        gid_feature_map =gid_feature_map ,
        id_root =id_root ,
        log_prefix ="image-post-merge",
        )

    if len (uniq )>1 :
        centroids_mat =np .stack ([label_to_centroid [int (lb )]for lb in uniq ],axis =0 )
    else :
        centroids_mat =None 
    uniq_list =[int (x )for x in uniq .tolist ()]
    lb_to_pos ={lb :i for i ,lb in enumerate (uniq_list )}

    for i in range (len (samples )):
        lb =int (labels [i ])
        gid =_remap_gid_to_root (label_to_gid [lb ],merge_alias_map )or label_to_gid [lb ]
        gid_feat =_identity_gid_features_store (inference ).get (gid )
        if gid_feat is None :
            gid_feat =label_to_centroid [lb ]
        sim_same =float (np .dot (features [i ],gid_feat ))
        if centroids_mat is not None and len (uniq_list )>1 :
            sims =centroids_mat @features [i ]
            sims [lb_to_pos [lb ]]=-10.0 
            sim_other =float (np .max (sims ))
        else :
            sim_other =-1.0 
        margin =sim_same -max (th ,sim_other +0.02 )
        id_conf =_clamp01 (_sigmoid01 (margin ,temperature =0.08 ),default =0.0 )
        samples [i ]['gid']=gid 
        samples [i ]['disp_id']=global_to_display .get (gid ,gid )
        samples [i ]['sim']=sim_same 
        samples [i ]['id_conf']=id_conf 

    for in_path in all_image_paths :
        rel_path =os .path .relpath (in_path ,input_dir )
        idxs =by_image .get (in_path ,[])
        if len (idxs )==0 :
            continue 
        frame_bgr =_imread_unicode (in_path )
        if frame_bgr is None :
            continue 

        id_info_map ={}
        present_disp_ids =set ()
        for si in idxs :
            s =samples [si ]
            x1 ,y1 ,x2 ,y2 =s ['bbox']
            x1 =int (max (0 ,x1 ))
            y1 =int (max (0 ,y1 ))
            x2 =int (max (x1 +1 ,x2 ))
            y2 =int (max (y1 +1 ,y2 ))
            gid =s ['gid']
            disp_id =s ['disp_id']
            present_disp_ids .add (disp_id )

            gstat =global_gender_stats .get (gid ,{'alpha_male':1.0 ,'alpha_female':1.0 ,'stable_gender':None })
            total =float (gstat ['alpha_male'])+float (gstat ['alpha_female'])
            male_mean =float (gstat ['alpha_male'])/max (1e-6 ,total )
            gender =gstat .get ('stable_gender')or ('M'if male_mean >=float (args .gender_threshold )else 'F')
            gender_conf =_gender_confidence_01 (male_mean ,stable_gender =gender ,threshold =args .gender_threshold )

            ages =global_age_stats .get (gid ,[])
            if str (getattr (args ,'age_display','median')).lower ()=='mean'and len (ages )>0 :
                age_disp =float (np .mean (ages ))
            elif str (getattr (args ,'age_display','median')).lower ()=='instant':
                age_disp =float (s ['age_p'])
            else :
                age_disp =float (np .median (ages ))if len (ages )>0 else float (s ['age_p'])

            txt =' | '.join ([
            disp_id ,
            f"{gender}_conf:{gender_conf:.2f}",
            f"Age:{age_disp:.0f}",
            f"id_conf:{float(s['id_conf']):.2f}",
            ])
            font ,font_scale ,thickness ,pad_txt =_get_adaptive_text_style (
            frame_bgr ,base_scale =0.82 ,min_scale =0.62 ,max_scale =2.7 
            )
            color =_VIS_BOX_BGR 
            box_th =max (2 ,int (round (thickness *0.9 )))
            cv2 .rectangle (frame_bgr ,(x1 ,y1 ),(x2 ,y2 ),color ,box_th )
            (_tw ,thh ),baseline =cv2 .getTextSize (txt ,font ,font_scale ,thickness )
            x_txt =x1 
            y_txt =max (thh +baseline +pad_txt ,y1 -pad_txt )
            _draw_text_block (
            frame_bgr ,
            txt ,
            x_txt ,
            y_txt ,
            font ,
            font_scale ,
            thickness ,
            pad_txt ,
            bg_color =_VIS_LABEL_BG_BGR ,
            text_color =_VIS_TEXT_BGR ,
            border_color =color ,
            )

            id_info_map [disp_id ]={'gender':gender ,'age':age_disp }
            detection_details .append ({
            'rel_path':rel_path ,
            'true_id':_first_folder_of_rel_path (rel_path ),
            'display_id':disp_id ,
            'global_id':gid ,
            'bbox':[int (x1 ),int (y1 ),int (x2 ),int (y2 )],
            'gender':gender ,
            'gender_conf':float (gender_conf ),
            'age':float (age_disp )if age_disp is not None else None ,
            'id_conf':float (s ['id_conf']),
            'similarity':float (s ['sim']),
            'roi_source':str (s .get ('roi_source','unknown')or 'unknown'),
            })

        args .input =in_path 
        if not bool (getattr (inference ,'group_by_id',False )):
            out_path =os .path .join (output_dir ,rel_path )
            if bool (getattr (args ,'save_vis',True )):
                out_dir =os .path .dirname (out_path )
                if out_dir :
                    _ensure_dir (out_dir )
            args .output =out_path 
        _save_image_visualization (
        args ,
        inference ,
        frame_bgr ,
        input_rel_path =rel_path ,
        disp_ids =sorted (present_disp_ids )if present_disp_ids else None ,
        fallback_folder ="UNASSIGNED",
        )

        records =[(disp_id ,info ['gender'],info ['age'])for disp_id ,info in sorted (id_info_map .items ())] 
        if records :
            detection_records .append ((rel_path ,records ))
            img_kept +=1 

    args .input =old_input 
    args .output =old_output 

    run_summary ={
    'cluster_method':str (cluster_method ),
    'threshold':float (th ),
    'threshold_mode':('auto' if bool (getattr (args ,'twopass_threshold_auto',True ))else 'manual'),
    'num_rois':int (len (samples )),
    'num_ids':int (len (cluster_info )),
    'centroid_merge_applied':bool (centroid_merge_info .get ('applied',False )),
    'centroid_merge_stage1_fragments':int (centroid_merge_info .get ('stage1_fragments',0 )or 0 ),
    'centroid_merge_stage2_clusters':int (centroid_merge_info .get ('stage2_clusters',0 )or 0 ),
    'centroid_merge_threshold':float (centroid_merge_info .get ('threshold',0.0 )or 0.0 ),
    'centroid_merge_method':str (centroid_merge_info .get ('method','')),
    'fallback_saved':int (fallback_saved ),
    }
    setattr (inference ,'_last_detection_details',detection_details )
    setattr (inference ,'_last_directory_run_summary',run_summary )

    print (
    f"[INFO] Two-pass clustering finished: method={cluster_method}, "
    f"threshold={th:.3f} ({'auto' if bool(getattr(args,'twopass_threshold_auto',True)) else 'manual'}), "
    f"rois={len(samples)}, ids={len(cluster_info)}, "
    f"fallback_saved={fallback_saved}"
    )
    return img_count ,img_kept ,detection_records 


def run_image (args ,inference ,yolo ,sam_predictor ,device ,*,input_rel_path :Optional [str ]=None ):
    """Main inference loop: detect, embed, assign IDs, and save outputs."""
    assert args .input ,"input"
    if not bool (getattr (inference ,'group_by_id',False )):
        assert args .output ,"output is required when group_by_id is enabled"
    frame_bgr =_imread_unicode (args .input )
    assert frame_bgr is not None ,f": {args.input}"

    
    is_dark ,mean_bright =_is_dark_frame (frame_bgr ,args =args )
    if is_dark :
        if getattr (args ,'verbose',False ):
            th =float (getattr (args ,'min_frame_brightness',0.0 )or 0.0 )
            print (f"   [filtered] mean_gray={mean_bright:.1f} < {th}")
        vis =frame_bgr .copy ()
        _draw_status_label (vis ,f"IGNORED:DARK({mean_bright:.1f})",color =(0 ,0 ,255 ))
        _save_image_visualization (
        args ,inference ,vis ,input_rel_path =input_rel_path ,fallback_folder ="IGNORED"
        )
        return []
    frame_rgb =cv2 .cvtColor (frame_bgr ,cv2 .COLOR_BGR2RGB )

    # =ROIOIOLO/SAM
    image_as_roi =bool (getattr (args ,'image_as_roi',False ))
    if isinstance (getattr (args ,'image_as_roi',None ),(int ,np .integer )):
        image_as_roi =bool (int (getattr (args ,'image_as_roi')))

    batch_tensors ,batch_crops ,kept_boxes =[],[],[]
    ignored_rois =[]# [{'bbox': (x1,y1,x2,y2), 'reasons': [...]}, ...]

    if image_as_roi :
        is_bad ,reasons =_is_low_quality_roi_image (frame_rgb ,args =args )
        if is_bad :
            if getattr (args ,'verbose',False ):
                print (f"   OI(): {reasons}")
            vis =frame_bgr .copy ()
            _draw_status_label (vis ,f"IGNORED:{','.join(reasons[:2]) or 'LOW_QUALITY'}",color =(0 ,0 ,255 ))
            _save_image_visualization (
            args ,inference ,vis ,input_rel_path =input_rel_path ,fallback_folder ="IGNORED"
            )
            return []
        H ,W =frame_rgb .shape [:2 ]
        crop_rgb =frame_rgb 
        crop_pil =Image .fromarray (crop_rgb )
        tensor =inference .transform (crop_pil ).unsqueeze (0 ).to (device )
        batch_tensors .append (tensor )
        batch_crops .append (crop_rgb )
        vis_box =None 
        if bool (getattr (args ,'roi_vis_use_detector',True )):
            vis_box =_pick_roi_visual_box_from_detector (frame_bgr ,yolo ,args =args )
        if vis_box is None :
            # Fallback inset box to avoid full-frame edge box in ROI-only mode.
            mx =int (round (W *0.08 ))
            my =int (round (H *0.08 ))
            x1 =max (0 ,mx )
            y1 =max (0 ,my )
            x2 =max (x1 +1 ,W -mx )
            y2 =max (y1 +1 ,H -my )
            vis_box =(x1 ,y1 ,x2 ,y2 )
        kept_boxes .append (vis_box )
    else :
    # SAM 
        if sam_predictor is not None :
            try :
                sam_predictor .set_image (frame_rgb )
            except Exception :
                sam_predictor =None 

                
        results =yolo .predict (source =frame_bgr ,conf =args .det_conf ,iou =args .det_iou ,verbose =False )
        if not results :
            vis =frame_bgr .copy ()
            _draw_status_label (vis ,"NO_DET",color =(0 ,0 ,255 ))
            _save_image_visualization (
            args ,inference ,vis ,input_rel_path =input_rel_path ,fallback_folder ="NO_DET"
            )
            return []
        res0 =results [0 ]
        boxes =res0 .boxes 
        if boxes is None or len (boxes )==0 :
            # 
            vis =frame_bgr .copy ()
            _draw_status_label (vis ,"NO_DET",color =(0 ,0 ,255 ))
            _save_image_visualization (
            args ,inference ,vis ,input_rel_path =input_rel_path ,fallback_folder ="NO_DET"
            )
            return []

        try :
            xyxy =boxes .xyxy .cpu ().numpy ().astype (int )
        except Exception :
            xyxy =np .asarray (boxes .xyxy ,dtype =np .float32 ).astype (int )

            # OI
        for det_i ,(x1 ,y1 ,x2 ,y2 )in enumerate (xyxy ):
            crop_rgb =None 
            mask =None 
            if sam_predictor is not None :
                try :
                    box =np .array ([x1 ,y1 ,x2 ,y2 ],dtype =np .float32 )[None ,:]
                    masks ,scores ,_ =sam_predictor .predict (box =box ,multimask_output =True )
                    if masks is not None and len (masks )>0 :
                    # mask
                        best =int (np .argmax (scores ))
                        mask =masks [best ].astype (bool )
                        crop_rgb =inference ._process_mask_region_to_rectangle (frame_rgb ,mask )
                except Exception :
                    crop_rgb =None 
                    mask =None 
            if crop_rgb is None or crop_rgb .size ==0 :
                crop_rgb =_crop_roi_with_expansion_numpy (frame_rgb ,(x1 ,y1 ,x2 ,y2 ),expand_ratio =0.1 )
                mask =None 
            if crop_rgb .size ==0 :
                continue 

            is_bad ,reasons =_is_low_quality_roi (
            frame_rgb ,(x1 ,y1 ,x2 ,y2 ),crop_rgb ,mask =mask ,args =args 
            )
            if is_bad :
                ignored_rois .append ({
                'bbox':(int (x1 ),int (y1 ),int (x2 ),int (y2 )),
                'reasons':reasons ,
                })
                if getattr (args ,'verbose',False ):
                    print (f"   [filtered] det={det_i}, reasons={reasons}")
                continue 

            crop_pil =Image .fromarray (crop_rgb )
            tensor =inference .transform (crop_pil ).unsqueeze (0 ).to (device )
            batch_tensors .append (tensor )
            batch_crops .append (crop_rgb )
            kept_boxes .append ((int (x1 ),int (y1 ),int (x2 ),int (y2 )))

    if not batch_tensors :
    # OI
        vis =frame_bgr .copy ()
        _draw_status_label (vis ,"NO_VALID_ROI",color =(0 ,0 ,255 ))
        _save_image_visualization (
        args ,inference ,vis ,input_rel_path =input_rel_path ,fallback_folder ="IGNORED"
        )
        return []

        # 
    with torch .no_grad ():
        inp =torch .cat (batch_tensors ,dim =0 )
        feat_after_bn ,male_probs ,age_pred =_forward_reid_and_aux (
        inference ,inp ,crops_rgb =batch_crops 
        )

        # ID/
    global_to_display =getattr (inference ,'global_to_display',None )
    if global_to_display is None :
        global_to_display ={}
        setattr (inference ,'global_to_display',global_to_display )
    next_display_id =getattr (inference ,'next_display_id',1 )
    global_gender_stats =getattr (inference ,'global_gender_stats',None )
    if global_gender_stats is None :
        global_gender_stats ={}
        setattr (inference ,'global_gender_stats',global_gender_stats )
    global_age_stats =getattr (inference ,'global_age_stats',None )
    if global_age_stats is None :
        global_age_stats ={}
        setattr (inference ,'global_age_stats',global_age_stats )

        # D input_rel_path 
    candidate_ids =_candidate_ids_for_rel_path (input_rel_path )

    # D
    assignments ={}
    used_ids_in_image =set ()
    
    for i in range (len (batch_tensors )):
        f =feat_after_bn [i ]
        male_p =float (male_probs [i ].item ())
        age_p =float (age_pred [i ].item ())
        aux ={'gender_prob':male_p ,'age_pred':age_p }
        _created_new =False 
        _adaptive_th =float (getattr (inference ,'similarity_threshold',getattr (args ,'base_threshold',0.22 ))or 0.22 )

        if candidate_ids :
        
            candidates =[cid for cid in candidate_ids if cid not in used_ids_in_image ]
            if not candidates :
                candidates =list (candidate_ids )

            res =_match_identity_with_strategy (
            inference ,
            f ,
            aux =aux ,
            reserved_ids =used_ids_in_image ,
            candidate_ids =candidates ,
            allow_new =False ,
            )
            sims =res .get ('all_similarities',{})or {}

            gid =None 
            best_score =None 
            for cid in candidates :
                sc =_candidate_score (cid ,male_p ,age_p ,sim =sims .get (cid ))
                if best_score is None or sc >best_score :
                    gid =cid 
                    best_score =sc 

            if gid is None :
                continue 
            sim_for_log =float (sims .get (gid ,0.0 )or 0.0 )
            _adaptive_th =float (res .get ('adaptive_threshold',_adaptive_th )or _adaptive_th )
        else :
            res =_match_identity_with_strategy (
            inference ,
            f ,
            aux =aux ,
            reserved_ids =used_ids_in_image ,
            candidate_ids =None ,
            allow_new =True ,
            )
            gid ,sim_for_log ,_created_new ,_adaptive_th =_pick_global_id_from_prototype_result (
            res ,inference ,used_ids_in_image ,allow_new =True 
            )
            if gid is None :
                continue 

        used_ids_in_image .add (gid )
        id_conf =_identity_confidence_01 (sim_for_log ,adaptive_th =_adaptive_th ,is_new =_created_new )

        
        if candidate_ids :
            try :
                with torch .no_grad ():
                    quality_score =float (inference .prototype_net .quality_net (f .unsqueeze (0 )).item ())
            except Exception :
                quality_score =1.0 
        else :
            quality_score =float (res .get ('quality_score',1.0 )or 1.0 )

        min_upd_conf =float (getattr (args ,'update_existing_min_id_conf',0.55 )or 0.55 )
        should_update =(bool (_created_new )or (id_conf >=min_upd_conf ))
        if should_update :
            _update_identity_memory (
            inference ,
            gid ,
            f ,
            quality_score ,
            gender_prob =male_p ,
            age_pred =age_p ,
            id_conf =id_conf ,
            )
        assignments [i ]={'gid':gid ,'male_p':male_p ,'age_p':age_p ,'sim':sim_for_log ,'id_conf':id_conf }
        # 
        global_age_stats .setdefault (gid ,[]).append (age_p )
        setattr (inference ,'global_age_stats',global_age_stats )
        
        next_display_id =_ensure_display_id (global_to_display ,gid ,next_display_id )
        setattr (inference ,'next_display_id',next_display_id )


        
    frame_bgr_orig =frame_bgr .copy ()
    overlay_items =[]# D
    for i ,(x1 ,y1 ,x2 ,y2 )in enumerate (kept_boxes ):
        gid =assignments [i ]['gid']
        male_p =assignments [i ]['male_p']
        age_p =assignments [i ]['age_p']
        disp_id =global_to_display [gid ]

        
        forced_meta =_ID_META .get (gid )
        if forced_meta is not None :
            gender =forced_meta .get ('gender')or 'F'
            male_mean =1.0 if gender =='M'else 0.0 
            age_disp =float (forced_meta .get ('age',0.0 )or 0.0 )
            
            gstat =global_gender_stats .setdefault (
            gid ,{'alpha_male':1.0 ,'alpha_female':1.0 ,'stable_gender':gender }
            )
            gstat ['stable_gender']=gender 
        else :
        # +
            gstat =global_gender_stats .setdefault (gid ,{'alpha_male':1.0 ,'alpha_female':1.0 ,'stable_gender':None })
            weight =1.0 +2.0 *abs (male_p -0.5 )
            gstat ['alpha_male']+=weight *male_p 
            gstat ['alpha_female']+=weight *(1.0 -male_p )
            total =gstat ['alpha_male']+gstat ['alpha_female']
            male_mean =gstat ['alpha_male']/max (1e-6 ,total )
            margin ,th =args .gender_hysteresis ,args .gender_threshold 
            if gstat ['stable_gender']is None :
                gstat ['stable_gender']='M'if male_mean >=th else 'F'
            elif gstat ['stable_gender']=='M'and male_mean <(th -margin ):
                gstat ['stable_gender']='F'
            elif gstat ['stable_gender']=='F'and male_mean >(th +margin ):
                gstat ['stable_gender']='M'
            gender =gstat ['stable_gender']
            age_disp =float (age_p )

        gender_conf =_gender_confidence_01 (male_mean ,stable_gender =gender ,threshold =args .gender_threshold )
        age_str =f"Age:{age_disp:.0f}"

        txt =' | '.join ([
        disp_id ,
        f"{gender}_conf:{gender_conf :.2f}",
        age_str ,
        f"id_conf:{assignments[i]['id_conf']:.2f}",
        ])
        font ,font_scale ,thickness ,pad_txt =_get_adaptive_text_style (
        frame_bgr ,base_scale =0.82 ,min_scale =0.62 ,max_scale =2.7 
        )
        (tw ,thh ),baseline =cv2 .getTextSize (txt ,font ,font_scale ,thickness )
        x_txt =x1 
        y_txt =max (thh +baseline +pad_txt ,y1 -pad_txt )
        x_bg ,y_bg =x_txt -pad_txt //2 ,y_txt -thh -baseline -pad_txt //2 

        
        color =_VIS_BOX_BGR 
        box_th =max (2 ,int (round (thickness *0.9 )))
        cv2 .rectangle (frame_bgr ,(x1 ,y1 ),(x2 ,y2 ),color ,box_th )
        text_box =_draw_text_block (
        frame_bgr ,
        txt ,
        x_txt ,
        y_txt ,
        font ,
        font_scale ,
        thickness ,
        pad_txt ,
        bg_color =_VIS_LABEL_BG_BGR ,
        text_color =_VIS_TEXT_BGR ,
        border_color =color ,
        )

        
        overlay_items .append ({
        'disp_id':disp_id ,
        'gender':gender ,
        'gender_conf':gender_conf ,
        'age':age_disp ,
        'id_conf':assignments [i ]['id_conf'],
        'rect':(x1 ,y1 ,x2 ,y2 ),
        'txt':txt ,
        'text_box':text_box if text_box is not None else (x_bg ,y_bg ,x_txt +tw +6 ,y_txt +baseline +2 ),
        'text_pos':(x_txt ,y_txt ),
        'font':font ,
        'font_scale':font_scale ,
        'thickness':thickness ,
        'color':color ,
        'text_color':_VIS_TEXT_BGR 
        })

        
    if ignored_rois and not getattr (inference ,'group_by_id',False ):
        for it in ignored_rois :
            x1 ,y1 ,x2 ,y2 =it ['bbox']
            reasons =it .get ('reasons')or []
            show_reason =','.join ([str (r )for r in reasons [:2 ]])if getattr (args ,'verbose',False )else ''
            txt =f"IGNORED{(':' + show_reason) if show_reason else ''}"

            color =_VIS_BOX_BGR 
            font ,font_scale ,thickness ,pad_txt =_get_adaptive_text_style (
            frame_bgr ,base_scale =0.72 ,min_scale =0.56 ,max_scale =2.3 
            )
            box_th =max (2 ,int (round (thickness *0.9 )))
            cv2 .rectangle (frame_bgr ,(x1 ,y1 ),(x2 ,y2 ),color ,box_th )
            (_tw ,thh ),baseline =cv2 .getTextSize (txt ,font ,font_scale ,thickness )
            x_txt =x1 
            y_txt =max (thh +baseline +pad_txt ,y1 -pad_txt )
            _draw_text_block (
            frame_bgr ,
            txt ,
            x_txt ,
            y_txt ,
            font ,
            font_scale ,
            thickness ,
            pad_txt ,
            bg_color =_VIS_LABEL_BG_BGR ,
            text_color =_VIS_TEXT_BGR ,
            border_color =color ,
            )

            
    id_info_map ={}
    for item in overlay_items :
        disp_id =item ['disp_id']
        if disp_id not in id_info_map :
            id_info_map [disp_id ]={'gender':item ['gender'],'age':item ['age']}
    detected_records =[(disp_id ,info ['gender'],info ['age'])for disp_id ,info in sorted (id_info_map .items ())]

    
    if bool (getattr (args ,'save_vis',True )):
        present_disp_ids ={item ['disp_id']for item in overlay_items }
        _save_image_visualization (
        args ,
        inference ,
        frame_bgr ,
        input_rel_path =input_rel_path ,
        disp_ids =sorted (present_disp_ids )if present_disp_ids else None ,
        fallback_folder ="UNASSIGNED",
        )

                
    return detected_records 


def run_video_dirmode (args ,inference ,yolo ,sam_predictor ,device ,*,input_rel_path :Optional [str ]=None ):
    """ID
    
    
    eIDD
     trk -> ID 
    
    : summary(dict) = { gid: { 'display_id': str, 'gender': 'M'|'F', 'male_mean': float, 'age_mean': float|None } }
    """
    import numpy as np 
    from collections import defaultdict ,deque 

    # 
    cap =cv2 .VideoCapture (args .input )
    assert cap .isOpened (),f": {args.input}"
    fps =cap .get (cv2 .CAP_PROP_FPS )
    if not fps or fps <=0 :
        print (f"[WARN] Invalid FPS detected ({fps}), fallback to 5 FPS.")
        fps =25.0 
    frame_count =cap .get (cv2 .CAP_PROP_FRAME_COUNT )
    total_frames =int (frame_count )if frame_count and frame_count >0 else ''
    w_orig =int (cap .get (cv2 .CAP_PROP_FRAME_WIDTH ))
    h_orig =int (cap .get (cv2 .CAP_PROP_FRAME_HEIGHT ))
    print (f"[INFO] Input video: FPS={fps:.2f}, frames={total_frames}, size={w_orig}x{h_orig}")
    cap .release ()

    
    print ("[INFO] Running online ReID and ID assignment...")

    # ID
    track_states ={}
    finalized_track_map ={}# { tid: { 'global_id': str, 'display_id': str, 'similarity': float } }

    save_vis =bool (getattr (args ,'save_vis',True ))
    # 
    frame_detections =[]if save_vis else None 

    
    global_to_display =getattr (inference ,'global_to_display',None )
    if global_to_display is None :
        global_to_display ={}
        setattr (inference ,'global_to_display',global_to_display )
    next_display_id =getattr (inference ,'next_display_id',1 )
    global_gender_stats =getattr (inference ,'global_gender_stats',None )
    if global_gender_stats is None :
        global_gender_stats ={}
        setattr (inference ,'global_gender_stats',global_gender_stats )
    global_age_stats =getattr (inference ,'global_age_stats',None )
    if global_age_stats is None :
        global_age_stats ={}
        setattr (inference ,'global_age_stats',global_age_stats )

        # D input_rel_path 
    candidate_ids =_candidate_ids_for_rel_path (input_rel_path )

    last_reid_update_frame =defaultdict (lambda :-10 **9 )
    update_every_n_frames =max (1 ,int (round (fps *args .sec_interval )))
    video_age_stats =defaultdict (list )
    seen_gids =set ()
    gid_reservations =_GidRangeReservation ()

    
    try :
        yolo .predictor =None 
    except Exception :
        pass 

        # 
        
    if args .verbose :
        print ("[INFO] Running YOLO detection...")
        
        
        
    results_gen =yolo .track (
    source =args .input ,
    stream =True ,
    conf =args .det_conf ,
    iou =args .det_iou ,
    tracker =args .tracker ,
    persist =True ,
    verbose =False ,
    workers =0 ,
    vid_stride =int (getattr (args ,'vid_stride',1 )or 1 ),
    )
    if args .verbose :
        print ("[INFO] Running YOLO detection...")
    frame_idx =-1 
    for result in results_gen :
        frame_idx +=1 
        
        if frame_idx %50 ==0 or (total_frames >0 and total_frames !=''and frame_idx %max (1 ,total_frames //5 )==0 ):
            progress =f"{frame_idx}/{total_frames}"if total_frames !=''else f"{frame_idx}"
            print (f"   {progress} ..")

        frame_bgr =result .orig_img 
        if frame_bgr is None :
            if frame_detections is not None :
                frame_detections .append ((None ,None ))
            continue 

            
        is_dark ,mean_bright =_is_dark_frame (frame_bgr ,args =args )
        if is_dark :
            if frame_detections is not None :
                frame_detections .append ((None ,None ))
            if getattr (args ,'verbose',False )and (frame_idx ==0 or frame_idx %max (1 ,int (round (fps )))==0 ):
                th =float (getattr (args ,'min_frame_brightness',0.0 )or 0.0 )
                print (f"   @frame{frame_idx}: filtered by mean_gray={mean_bright:.1f} < {th}")
            continue 
        frame_rgb =cv2 .cvtColor (frame_bgr ,cv2 .COLOR_BGR2RGB )

        boxes =result .boxes 
        if boxes is None or len (boxes )==0 :
            if frame_detections is not None :
                frame_detections .append ((None ,None ))
            continue 

        xyxy =boxes .xyxy .cpu ().numpy ().astype (int )
        ids =boxes .id 
        ids =ids .cpu ().numpy ().astype (int )if ids is not None else np .array ([-1 ]*len (xyxy ),dtype =int )
        if frame_detections is not None :
            frame_detections .append ((xyxy .copy (),ids .copy ()))

        current_tracks =set ()
        do_reid_now =(frame_idx %update_every_n_frames ==0 )
        batch_tensors ,batch_track_ids ,batch_crops =[],[],[]

        
        sam_ready =False 
        if do_reid_now and sam_predictor is not None :
            try :
            
                if frame_idx ==0 :
                    print (f"[INFO] SAM set_image: {frame_rgb.shape[1]}x{frame_rgb.shape[0]}")
                sam_predictor .set_image (frame_rgb )
                sam_ready =True 
                if frame_idx ==0 :
                    print (f"   AM set_image ")
            except Exception as e :
                if args .verbose :
                    print (f"[WARN] SAM set_image failed: {e}")
                sam_ready =False 

        for i ,(x1 ,y1 ,x2 ,y2 )in enumerate (xyxy ):
            tid =int (ids [i ])if i <len (ids )else -1 
            if tid <0 :
                continue 
            current_tracks .add (tid )
            if tid not in track_states :
                track_states [tid ]={
                'global_id':None ,'display_id':None ,'last_seen_frame':frame_idx ,
                'first_seen_frame':frame_idx ,'feature_buffer':[],
                'male_hist':deque (maxlen =50 ),'age_hist':deque (maxlen =50 ),
                'last_similarity':None ,'finalized':False ,
                'lq_bad':0 ,'lq_good':0 ,
                }
            else :
                track_states [tid ]['last_seen_frame']=frame_idx 

            if do_reid_now and frame_idx -last_reid_update_frame [tid ]>=update_every_n_frames :
                crop_rgb =None 
                mask =None 
                if sam_ready and sam_predictor is not None :
                    try :
                        box =np .array ([x1 ,y1 ,x2 ,y2 ],dtype =np .float32 )[None ,:]
                        masks ,scores ,_ =sam_predictor .predict (box =box ,multimask_output =True )
                        if masks is not None and len (masks )>0 :
                            best =int (np .argmax (scores ))
                            mask =masks [best ].astype (bool )
                            crop_rgb =inference ._process_mask_region_to_rectangle (frame_rgb ,mask )
                    except Exception :
                        crop_rgb =None 
                        mask =None 
                if crop_rgb is None or crop_rgb .size ==0 :
                # SAM /OI
                    crop_rgb =_crop_roi_with_expansion_numpy (frame_rgb ,(x1 ,y1 ,x2 ,y2 ),expand_ratio =0.1 )
                    mask =None 
                if crop_rgb is None or crop_rgb .size ==0 :
                    continue 

                is_bad ,reasons =_is_low_quality_roi (
                frame_rgb ,(x1 ,y1 ,x2 ,y2 ),crop_rgb ,mask =mask ,args =args 
                )
                if is_bad :
                    state =track_states .get (tid )
                    if state is not None :
                        state ['lq_bad']=int (state .get ('lq_bad',0 )or 0 )+1 
                    if getattr (args ,'verbose',False ):
                        print (f"   [trk{tid}] OI @frame{frame_idx}: {reasons}")
                    continue 
                state =track_states .get (tid )
                if state is not None :
                    state ['lq_good']=int (state .get ('lq_good',0 )or 0 )+1 

                crop_pil =Image .fromarray (crop_rgb )
                tensor =inference .transform (crop_pil ).unsqueeze (0 ).to (device )
                batch_tensors .append (tensor )
                batch_track_ids .append (tid )
                batch_crops .append (crop_rgb )

        if batch_tensors :
            with torch .no_grad ():
                inp =torch .cat (batch_tensors ,dim =0 )
                feat_after_bn ,male_probs ,age_pred =_forward_reid_and_aux (
                inference ,inp ,crops_rgb =batch_crops 
                )

            for bi ,tid in enumerate (batch_track_ids ):
                f =feat_after_bn [bi ].clone ()
                male_p =float (male_probs [bi ].item ())
                age_p =float (age_pred [bi ].item ())
                state =track_states [tid ]
                state ['feature_buffer'].append (f )
                state ['male_hist'].append (male_p )
                state ['age_hist'].append (age_p )
                last_reid_update_frame [tid ]=frame_idx 

                # D
        if frame_idx %int (fps )==0 :
            cutoff =frame_idx -int (args .max_keep_missing_sec *fps )
            to_del =[tid for tid ,st in track_states .items ()if st ['last_seen_frame']<cutoff ]
            reserved_ids =gid_reservations 

            for tid in to_del :
                state =track_states .get (tid )
                if not state or state .get ('finalized',False ):
                    continue 

                    
                skip_track ,skip_info =_should_skip_track_by_lowq (state ,args )
                if skip_track :
                    state ['finalized']=True 
                    finalized_track_map [tid ]={'skipped':True ,**(skip_info or {})}
                    if getattr (args ,'verbose',False ):
                        br =float (skip_info .get ('lq_bad_ratio',0.0 )or 0.0 )if isinstance (skip_info ,dict )else 0.0 
                        print (
                        f"   [trk{tid}] ID): reason={skip_info.get('skip_reason','-')} "
                        f"bad={skip_info.get('lq_bad',0)}/{skip_info.get('lq_obs',0)} ratio={br:.2f}"
                        )
                    continue 

                if len (state .get ('feature_buffer',[]))==0 :
                    continue 

                result_tuple =_finalize_track_id (
                state ,inference ,global_to_display ,global_gender_stats ,
                video_age_stats ,global_age_stats ,{},
                reserved_ids ,args ,verbose =args .verbose ,tid =tid ,
                known_ids =candidate_ids 
                )
                if result_tuple [0 ]is not None :
                    gid ,sim ,_ =result_tuple 
                    next_display_id =_ensure_display_id (global_to_display ,gid ,next_display_id )
                    setattr (inference ,'next_display_id',next_display_id )
                    state ['global_id']=gid 
                    state ['display_id']=global_to_display [gid ]
                    state ['last_similarity']=sim 
                    state ['finalized']=True 
                    finalized_track_map [tid ]={
                    'global_id':gid ,
                    'display_id':global_to_display [gid ],
                    'similarity':sim ,
                    'id_confidence':state .get ('last_id_confidence'),
                    }
                    seen_gids .add (gid )

            for tid in to_del :
                track_states .pop (tid ,None )
                last_reid_update_frame .pop (tid ,None )

                
    reserved_ids =gid_reservations 
    for tid ,state in track_states .items ():
        if state .get ('finalized',False ):
            gid =state .get ('global_id')
            if gid :
                finalized_track_map [tid ]={
                'global_id':gid ,
                'display_id':global_to_display .get (gid ,''),
                'similarity':state .get ('last_similarity'),
                'id_confidence':state .get ('last_id_confidence'),
                }
            continue 

            
        skip_track ,skip_info =_should_skip_track_by_lowq (state ,args )
        if skip_track :
            state ['finalized']=True 
            finalized_track_map [tid ]={'skipped':True ,**(skip_info or {})}
            if getattr (args ,'verbose',False ):
                br =float (skip_info .get ('lq_bad_ratio',0.0 )or 0.0 )if isinstance (skip_info ,dict )else 0.0 
                print (
                f"   [trk{tid}] ): reason={skip_info.get('skip_reason','-')} "
                f"bad={skip_info.get('lq_bad',0)}/{skip_info.get('lq_obs',0)} ratio={br:.2f}"
                )
            continue 
        if len (state .get ('feature_buffer',[]))==0 :
            continue 

        result_tuple =_finalize_track_id (
        state ,inference ,global_to_display ,global_gender_stats ,
        video_age_stats ,global_age_stats ,{},
        reserved_ids ,args ,verbose =args .verbose ,tid =tid ,
        known_ids =candidate_ids 
        )
        if result_tuple [0 ]is not None :
            gid ,sim ,_ =result_tuple 
            next_display_id =_ensure_display_id (global_to_display ,gid ,next_display_id )
            setattr (inference ,'next_display_id',next_display_id )
            state ['global_id']=gid 
            state ['display_id']=global_to_display [gid ]
            state ['last_similarity']=sim 
            state ['finalized']=True 
            finalized_track_map [tid ]={
            'global_id':gid ,
            'display_id':global_to_display [gid ],
            'similarity':sim ,
            'id_confidence':state .get ('last_id_confidence'),
            }
            seen_gids .add (gid )

    n_frames =len (frame_detections )if frame_detections is not None else (frame_idx +1 )
    print (f"[INFO] Processed {n_frames} frames, tracked IDs: {len(seen_gids)}")

    for gid in list (seen_gids ):
        _register_gid_source (inference ,gid ,input_rel_path or args .input ,'video')
    if frame_detections is not None :
        for xyxy ,ids in frame_detections :
            if ids is None :
                continue 
            frame_gids =set ()
            for tid_raw in ids :
                try :
                    tid =int (tid_raw )
                except Exception :
                    continue 
                info =finalized_track_map .get (tid )or track_states .get (tid )or {}
                gid =info .get ('global_id')
                if gid :
                    frame_gids .add (str (gid ))
            _register_conflict_pairs_from_gids (inference ,frame_gids )

    id_root =getattr (inference ,'id_group_root',None )if bool (getattr (inference ,'group_by_id',False ))else None 
    merge_alias_map ={}
    if bool (getattr (args ,'post_merge_enable_video',getattr (args ,'post_merge_enable',True ))):
        merge_alias_map =_consolidate_runtime_identities (
        inference =inference ,
        candidate_gids =list (getattr (inference ,'global_to_display',{}).keys ()),
        global_to_display =global_to_display ,
        global_gender_stats =global_gender_stats ,
        global_age_stats =global_age_stats ,
        args =args ,
        gid_feature_map =None ,
        id_root =id_root ,
        log_prefix =f"video-post-merge:{os.path.basename(args.input)}",
        )
    if merge_alias_map :
        for child_gid ,root_gid in list (merge_alias_map .items ()):
            if child_gid in video_age_stats :
                video_age_stats [root_gid ].extend (list (video_age_stats .pop (child_gid ,[])or []))
        seen_gids ={_remap_gid_to_root (gid ,merge_alias_map )or gid for gid in seen_gids}
        for tid ,info in finalized_track_map .items ():
            gid =info .get ('global_id')
            if gid :
                gid =_remap_gid_to_root (gid ,merge_alias_map )or gid 
                info ['global_id']=gid 
                info ['display_id']=global_to_display .get (gid ,gid )
        for tid ,state in track_states .items ():
            gid =state .get ('global_id')
            if gid :
                gid =_remap_gid_to_root (gid ,merge_alias_map )or gid 
                state ['global_id']=gid 
                state ['display_id']=global_to_display .get (gid ,gid )

    # +DID
    if getattr (inference ,'group_by_id',False )and len (seen_gids )==0 :
        if getattr (args ,'verbose',False ):
            print ("[WARN] No identities assigned in group_by_id mode.")
        return {}

    if not save_vis :
        if getattr (args ,'verbose',False ):
            print ("[WARN] save_vis=0: visualization video will not be written.")
        summary ={}
        for gid in seen_gids :
        # ID/
            forced_meta =_ID_META .get (gid )
            if forced_meta is not None :
                gender =forced_meta .get ('gender')or 'F'
                male_mean =1.0 if gender =='M'else 0.0 
                try :
                    age_mean =float (forced_meta .get ('age'))if forced_meta .get ('age')is not None else None 
                except Exception :
                    age_mean =None 
                summary [gid ]={
                'display_id':global_to_display .get (gid ,gid if _is_canonical_id_name (gid )else ''),
                'gender':gender ,
                'male_mean':male_mean ,
                'gender_conf':_gender_confidence_01 (male_mean ,stable_gender =gender ,threshold =args .gender_threshold ),
                'age_mean':age_mean ,
                }
                continue 

            gstat =global_gender_stats .get (gid )
            if not gstat :
                continue 
            total =gstat ['alpha_male']+gstat ['alpha_female']
            male_mean =float (gstat ['alpha_male']/max (1e-6 ,total ))
            gender =gstat .get ('stable_gender')or ('M'if male_mean >=args .gender_threshold else 'F')
            ages =video_age_stats .get (gid ,[])
            age_mean =float (np .mean (ages ))if len (ages )>0 else None 
            summary [gid ]={
            'display_id':global_to_display .get (gid ,''),
            'gender':gender ,
            'male_mean':male_mean ,
            'gender_conf':_gender_confidence_01 (male_mean ,stable_gender =gender ,threshold =args .gender_threshold ),
            'age_mean':age_mean ,
            }

            
        if frame_detections is not None :
            del frame_detections 
        del track_states 
        del finalized_track_map 
        return summary 

        # ====================  ====================
    print ("[INFO] Starting visualization rendering pass...")

    out_dir =os .path .dirname (args .output )
    if out_dir :
        _ensure_dir (out_dir )

    write_fps =round (fps )
    fourcc_list =['mp4v','avc1','XVID','MJPG']

    cap2 =cv2 .VideoCapture (args .input )
    assert cap2 .isOpened (),f": {args.input}"

    writer =None 
    writer_size =None 

    frame_idx =-1 
    frames_written =0 

    while True :
        ret ,frame_bgr =cap2 .read ()
        if not ret :
            break 
        frame_idx +=1 

        if writer is None :
            h_out ,w_out =frame_bgr .shape [:2 ]
            for cc in fourcc_list :
                fourcc =cv2 .VideoWriter_fourcc (*cc )
                tmp =cv2 .VideoWriter (args .output ,fourcc ,write_fps ,(w_out ,h_out ))
                if tmp .isOpened ():
                    writer =tmp 
                    writer_size =(w_out ,h_out )
                    break 
            if writer is None :
                raise RuntimeError (f": {args.output}")

        if frame_idx <len (frame_detections ):
            xyxy ,ids =frame_detections [frame_idx ]
        else :
            xyxy ,ids =None ,None 

            
        if xyxy is not None and ids is not None :
            for i ,(x1 ,y1 ,x2 ,y2 )in enumerate (xyxy ):
                tid =int (ids [i ])if i <len (ids )else -1 
                if tid <0 :
                    continue 

                track_info =finalized_track_map .get (tid )
                
                if track_info and track_info .get ('skipped'):
                    continue 
                if track_info :
                    disp_id =track_info ['display_id']
                    similarity =track_info .get ('similarity')
                    gid =track_info ['global_id']
                    box_color =_VIS_BOX_BGR # 
                else :
                    disp_id =f'trk{tid}'
                    similarity =None 
                    gid =None 
                    box_color =_VIS_BOX_BGR # 

                    
                gender_txt =None 
                male_mean =None 
                age_txt =None 

                if gid is not None :
                    forced_meta =_ID_META .get (gid )
                    if forced_meta is not None :
                        gender_txt =forced_meta .get ('gender')
                        male_mean =1.0 if gender_txt =='M'else 0.0 
                        try :
                            age_txt =float (forced_meta .get ('age'))if forced_meta .get ('age')is not None else None 
                        except Exception :
                            age_txt =None 
                    else :
                        gstat =global_gender_stats .get (gid )
                        if gstat :
                            total =gstat ['alpha_male']+gstat ['alpha_female']
                            male_mean =float (gstat ['alpha_male']/max (1e-6 ,total ))
                            gender_txt =gstat .get ('stable_gender')or ('M'if male_mean >=args .gender_threshold else 'F')

                        ages =video_age_stats .get (gid ,[])
                        if len (ages )>0 :
                            age_display =getattr (args ,'age_display','median')
                            age_txt =float (np .mean (ages ))if age_display =='mean'else float (np .median (ages ))

                            # 
                disp =[disp_id ]
                if gender_txt is not None and male_mean is not None :
                    gender_conf =_gender_confidence_01 (male_mean ,stable_gender =gender_txt ,threshold =args .gender_threshold )
                    disp .append (f"{gender_txt}_conf:{gender_conf :.2f}")
                if age_txt is not None :
                    disp .append (f"Age:{age_txt:.0f}")
                id_conf =None 
                if track_info and track_info .get ('id_confidence')is not None :
                    id_conf =float (track_info ['id_confidence'])
                elif similarity is not None :
                    id_conf =_identity_confidence_01 (similarity ,adaptive_th =getattr (args ,'base_threshold',None ),is_new =False )
                if id_conf is not None :
                    disp .append (f"id_conf:{id_conf :.2f}")
                txt =' | '.join (disp )

                # 
                font ,font_scale ,thickness ,pad_txt =_get_adaptive_text_style (
                frame_bgr ,base_scale =0.82 ,min_scale =0.62 ,max_scale =2.7 
                )
                (_tw ,th ),baseline =cv2 .getTextSize (txt ,font ,font_scale ,thickness )
                x_txt =x1 
                y_txt =max (th +baseline +pad_txt ,y1 -pad_txt )

                box_th =max (2 ,int (round (thickness *0.9 )))
                cv2 .rectangle (frame_bgr ,(x1 ,y1 ),(x2 ,y2 ),box_color ,box_th )
                _draw_text_block (
                frame_bgr ,
                txt ,
                x_txt ,
                y_txt ,
                font ,
                font_scale ,
                thickness ,
                pad_txt ,
                bg_color =_VIS_LABEL_BG_BGR ,
                text_color =_VIS_TEXT_BGR ,
                border_color =box_color ,
                )

        if writer_size and (frame_bgr .shape [1 ],frame_bgr .shape [0 ])!=writer_size :
            frame_bgr =cv2 .resize (frame_bgr ,writer_size ,interpolation =cv2 .INTER_AREA )
        writer .write (frame_bgr )
        frames_written +=1 

    if writer is not None :
        writer .release ()
    cap2 .release ()

    print (f"[INFO] Rendered {frames_written} frames at FPS={write_fps}")

    # ========== D ==========
    if getattr (inference ,'group_by_id',False ):
        id_root =getattr (inference ,'id_group_root',None )or os .path .dirname (args .output )
        print (f" trk -> ID..")
        _reorganize_folders_by_id (id_root ,finalized_track_map ,global_to_display )

        try :
            src_video =args .output 
            if input_rel_path :
                flat =_flatten_rel_path (input_rel_path )
                stem =os .path .splitext (flat )[0 ]if flat else os .path .splitext (os .path .basename (args .input ))[0 ]
                save_name =f"{stem}.mp4"
            else :
                save_name =os .path .basename (args .output )
                if not save_name .lower ().endswith ('.mp4'):
                    save_name =os .path .splitext (save_name )[0 ]+'.mp4'
            disp_ids_for_video ={
            info ['display_id']
            for info in finalized_track_map .values ()
            if info .get ('display_id')and not info .get ('skipped')
            }
            for disp_id in disp_ids_for_video :
                dst_dir =os .path .join (id_root ,disp_id )
                os .makedirs (dst_dir ,exist_ok =True )
                dst_path =os .path .join (dst_dir ,save_name )
                shutil .copy2 (src_video ,dst_path )
        except Exception as e :
            if getattr (args ,'verbose',False ):
                print (f"[WARN] Failed to dispatch grouped video outputs: {e}")
                
        try :
            if args .output and os .path .isfile (args .output ):
                os .remove (args .output )
                
            tmp_dir =os .path .dirname (args .output )if args .output else ""
            if tmp_dir and os .path .basename (tmp_dir )=='tmp_videos':
                try :
                    if len (os .listdir (tmp_dir ))==0 :
                        os .rmdir (tmp_dir )
                except Exception :
                    pass 
        except Exception as e :
            if getattr (args ,'verbose',False ):
                print (f"[WARN] : {e}")

                
    summary ={}
    for gid in seen_gids :
    # ID/
        forced_meta =_ID_META .get (gid )
        if forced_meta is not None :
            gender =forced_meta .get ('gender')or 'F'
            male_mean =1.0 if gender =='M'else 0.0 
            try :
                age_mean =float (forced_meta .get ('age'))if forced_meta .get ('age')is not None else None 
            except Exception :
                age_mean =None 
            summary [gid ]={
            'display_id':global_to_display .get (gid ,gid if _is_canonical_id_name (gid )else ''),
            'gender':gender ,
            'male_mean':male_mean ,
            'gender_conf':_gender_confidence_01 (male_mean ,stable_gender =gender ,threshold =args .gender_threshold ),
            'age_mean':age_mean ,
            }
            continue 

        gstat =global_gender_stats .get (gid )
        if not gstat :
            continue 
        total =gstat ['alpha_male']+gstat ['alpha_female']
        male_mean =float (gstat ['alpha_male']/max (1e-6 ,total ))
        gender =gstat .get ('stable_gender')or ('M'if male_mean >=args .gender_threshold else 'F')
        ages =video_age_stats .get (gid ,[])
        age_mean =float (np .mean (ages ))if len (ages )>0 else None 
        summary [gid ]={
        'display_id':global_to_display .get (gid ,''),
        'gender':gender ,
        'male_mean':male_mean ,
        'gender_conf':_gender_confidence_01 (male_mean ,stable_gender =gender ,threshold =args .gender_threshold ),
        'age_mean':age_mean ,
        }

    if not getattr (inference ,'group_by_id',False ):
        print (f" {args.output}")
    else :
        print (f"DD: {getattr(inference, 'id_group_root', os.path.dirname(args.output))}")

        
    del frame_detections 
    del track_states 
    del finalized_track_map 

    return summary 


def run ():
    args =parse_args ()
    _user_base_threshold =args .base_threshold 
    _user_twopass_threshold =getattr (args ,'twopass_threshold',None )
    _user_model_path =args .model_path

    
    
    if not args .input :
        args .input =USER_DEFAULTS .get ('input')
    if not args .output :
        args .output =USER_DEFAULTS .get ('output')
    if args .cfg is None :
        args .cfg =USER_DEFAULTS .get ('cfg')
    if args .model_path is None :
        args .model_path =USER_DEFAULTS .get ('model_path')
    if getattr (args ,'aux_model_path',None )is None :
        args .aux_model_path =USER_DEFAULTS .get ('aux_model_path')
    if args .mode is None :
        args .mode =USER_DEFAULTS .get ('mode')
    if args .similarity_threshold is None :
        args .similarity_threshold =USER_DEFAULTS .get ('similarity_threshold')
        
    if not bool (getattr (args ,'verbose',False )):
        args .verbose =bool (USER_DEFAULTS .get ('verbose',False ))

        
    if getattr (args ,'base_threshold',None )is None :
        args .base_threshold =USER_DEFAULTS .get ('base_threshold')
    if getattr (args ,'adaptive_threshold_min',None )is None :
        args .adaptive_threshold_min =USER_DEFAULTS .get ('adaptive_threshold_min')
    if getattr (args ,'adaptive_threshold_max',None )is None :
        args .adaptive_threshold_max =USER_DEFAULTS .get ('adaptive_threshold_max')
    if getattr (args ,'confidence_threshold',None )is None :
        args .confidence_threshold =USER_DEFAULTS .get ('confidence_threshold')
    if getattr (args ,'quality_threshold',None )is None :
        args .quality_threshold =USER_DEFAULTS .get ('quality_threshold')

        
    if args .gender_threshold is None :
        args .gender_threshold =USER_DEFAULTS .get ('gender_threshold')
    if getattr (args ,'gender_hysteresis',None )is None :
        args .gender_hysteresis =USER_DEFAULTS .get ('gender_hysteresis')
    if getattr (args ,'age_scope',None )is None :
        args .age_scope =USER_DEFAULTS .get ('age_scope')
    if getattr (args ,'age_display',None )is None :
        args .age_display =USER_DEFAULTS .get ('age_display')
    if getattr (args ,'sim_cosine_w',None )is None :
        args .sim_cosine_w =USER_DEFAULTS .get ('sim_cosine_w')
    if getattr (args ,'sim_euclid_w',None )is None :
        args .sim_euclid_w =USER_DEFAULTS .get ('sim_euclid_w')
    if getattr (args ,'aux_gender_penalty',None )is None :
        args .aux_gender_penalty =USER_DEFAULTS .get ('aux_gender_penalty')
    if getattr (args ,'aux_age_reweight',None )is None :
        args .aux_age_reweight =USER_DEFAULTS .get ('aux_age_reweight')
    if getattr (args ,'aux_min_age_sigma',None )is None :
        args .aux_min_age_sigma =USER_DEFAULTS .get ('aux_min_age_sigma')
    if getattr (args ,'aux_feature_fuse_weight',None )is None :
        args .aux_feature_fuse_weight =USER_DEFAULTS .get ('aux_feature_fuse_weight',0.0 )
    args .aux_feature_fuse_weight =float (min (1.0 ,max (0.0 ,float (args .aux_feature_fuse_weight ))))
    if getattr (args ,'prototype_strategy',None )is None :
        args .prototype_strategy =USER_DEFAULTS .get ('prototype_strategy','single')
    args .prototype_strategy =str (args .prototype_strategy or 'single').lower ()
    if getattr (args ,'prototype_max_slots',None )is None :
        args .prototype_max_slots =USER_DEFAULTS .get ('prototype_max_slots',4 )
    args .prototype_max_slots =int (max (1 ,int (args .prototype_max_slots )))
    if getattr (args ,'prototype_spawn_similarity',None )is None :
        args .prototype_spawn_similarity =USER_DEFAULTS .get ('prototype_spawn_similarity',0.68 )
    args .prototype_spawn_similarity =float (args .prototype_spawn_similarity )
    if getattr (args ,'prototype_update_similarity',None )is None :
        args .prototype_update_similarity =USER_DEFAULTS .get ('prototype_update_similarity',0.58 )
    args .prototype_update_similarity =float (args .prototype_update_similarity )
    if getattr (args ,'prototype_topk',None )is None :
        args .prototype_topk =USER_DEFAULTS .get ('prototype_topk',2 )
    args .prototype_topk =int (max (1 ,int (args .prototype_topk )))
    if getattr (args ,'prototype_aggregate_weight',None )is None :
        args .prototype_aggregate_weight =USER_DEFAULTS .get ('prototype_aggregate_weight',0.75 )
    args .prototype_aggregate_weight =float (max (0.0 ,min (1.0 ,float (args .prototype_aggregate_weight ))))
        
    if getattr (args ,'save_vis',None )is None :
        args .save_vis =USER_DEFAULTS .get ('save_vis',True )
    args .save_vis =bool (int (args .save_vis ))if isinstance (args .save_vis ,(int ,np .integer ))else bool (args .save_vis )
    if not getattr (args ,'group_by_id',False ):
        args .group_by_id =USER_DEFAULTS .get ('group_by_id',False )
    if getattr (args ,'image_as_roi',None )is None :
        args .image_as_roi =USER_DEFAULTS .get ('image_as_roi',None )
    if getattr (args ,'roi_vis_use_detector',None )is None :
        args .roi_vis_use_detector =USER_DEFAULTS .get ('roi_vis_use_detector',True )
    args .roi_vis_use_detector =bool (int (args .roi_vis_use_detector ))if isinstance (args .roi_vis_use_detector ,(int ,np .integer ))else bool (args .roi_vis_use_detector )
    if getattr (args ,'roi_vis_det_conf',None )is None :
        args .roi_vis_det_conf =USER_DEFAULTS .get ('roi_vis_det_conf',0.35 )
    if getattr (args ,'roi_vis_det_iou',None )is None :
        args .roi_vis_det_iou =USER_DEFAULTS .get ('roi_vis_det_iou',0.45 )
    if bool (getattr (args ,'image_as_roi',False )):
        print ("[WARN] image_as_roi=1: image pipeline bypasses YOLO+SAM cropping and uses full-image ROI embedding.")
    else :
        print ("[INFO] image_as_roi=0: image pipeline uses YOLO detection + SAM segmentation for ROI extraction.")

        
    if getattr (args ,'filter_low_quality',None )is None :
        args .filter_low_quality =USER_DEFAULTS .get ('filter_low_quality',True )
    args .filter_low_quality =bool (int (args .filter_low_quality ))if isinstance (args .filter_low_quality ,(int ,np .integer ))else bool (args .filter_low_quality )
    if getattr (args ,'min_frame_brightness',None )is None :
        args .min_frame_brightness =USER_DEFAULTS .get ('min_frame_brightness',0.0 )
    if getattr (args ,'min_bbox_area_ratio',None )is None :
        args .min_bbox_area_ratio =USER_DEFAULTS .get ('min_bbox_area_ratio',0.0 )
    if getattr (args ,'max_bbox_area_ratio',None )is None :
        args .max_bbox_area_ratio =USER_DEFAULTS .get ('max_bbox_area_ratio',0.0 )
    if getattr (args ,'bbox_border_ratio',None )is None :
        args .bbox_border_ratio =USER_DEFAULTS .get ('bbox_border_ratio',0.0 )
    if getattr (args ,'min_mask_fill_ratio',None )is None :
        args .min_mask_fill_ratio =USER_DEFAULTS .get ('min_mask_fill_ratio',0.0 )
    if getattr (args ,'min_blur_var',None )is None :
        args .min_blur_var =USER_DEFAULTS .get ('min_blur_var',0.0 )
    if getattr (args ,'min_brightness',None )is None :
        args .min_brightness =USER_DEFAULTS .get ('min_brightness',0.0 )

        
    if getattr (args ,'track_bad_ratio_threshold',None )is None :
        args .track_bad_ratio_threshold =USER_DEFAULTS .get ('track_bad_ratio_threshold',0.8 )
    if getattr (args ,'min_track_obs',None )is None :
        args .min_track_obs =USER_DEFAULTS .get ('min_track_obs',5 )
    if getattr (args ,'min_track_good',None )is None :
        args .min_track_good =USER_DEFAULTS .get ('min_track_good',2 )
    if getattr (args ,'update_existing_min_id_conf',None )is None :
        args .update_existing_min_id_conf =USER_DEFAULTS .get ('update_existing_min_id_conf',0.55 )
    if getattr (args ,'shuffle_images',None )is None :
        args .shuffle_images =USER_DEFAULTS .get ('shuffle_images',True )
    args .shuffle_images =bool (int (args .shuffle_images ))if isinstance (args .shuffle_images ,(int ,np .integer ))else bool (args .shuffle_images )
    if getattr (args ,'shuffle_seed',None )is None :
        args .shuffle_seed =USER_DEFAULTS .get ('shuffle_seed',42 )
    if getattr (args ,'image_cluster_mode',None )is None :
        args .image_cluster_mode =USER_DEFAULTS .get ('image_cluster_mode','twopass')
    args .image_cluster_mode =str (args .image_cluster_mode or 'twopass').lower ()
    if getattr (args ,'cluster_method',None )is None :
        args .cluster_method =USER_DEFAULTS .get ('cluster_method','auto')
    args .cluster_method =str (args .cluster_method or 'auto').lower ()
    if getattr (args ,'cluster_agglomerative_max_n',None )is None :
        args .cluster_agglomerative_max_n =USER_DEFAULTS .get ('cluster_agglomerative_max_n',5000 )
    args .cluster_agglomerative_max_n =int (max (1 ,int (args .cluster_agglomerative_max_n )))
    if getattr (args ,'cluster_momentum',None )is None :
        args .cluster_momentum =USER_DEFAULTS .get ('cluster_momentum',0.9 )
    args .cluster_momentum =float (args .cluster_momentum )
    if getattr (args ,'max_rois_per_image',None )is None :
        args .max_rois_per_image =USER_DEFAULTS .get ('max_rois_per_image',1 )
    args .max_rois_per_image =int (max (1 ,int (args .max_rois_per_image )))
    if _user_twopass_threshold is not None :
        args .twopass_threshold =float (_user_twopass_threshold )
    elif _user_base_threshold is not None :
        args .twopass_threshold =float (_user_base_threshold )
    else :
        args .twopass_threshold =float (USER_DEFAULTS .get ('twopass_threshold',0.17 ))
    args .twopass_threshold =float (max (0.0 ,min (1.0 ,args .twopass_threshold )))
    if getattr (args ,'twopass_threshold_auto',None )is None :
        args .twopass_threshold_auto =USER_DEFAULTS .get ('twopass_threshold_auto',True )
    args .twopass_threshold_auto =bool (int (args .twopass_threshold_auto ))if isinstance (args .twopass_threshold_auto ,(int ,np .integer ))else bool (args .twopass_threshold_auto )
    if getattr (args ,'twopass_fuse_full_image',None )is None :
        args .twopass_fuse_full_image =USER_DEFAULTS .get ('twopass_fuse_full_image',True )
    args .twopass_fuse_full_image =bool (int (args .twopass_fuse_full_image ))if isinstance (args .twopass_fuse_full_image ,(int ,np .integer ))else bool (args .twopass_fuse_full_image )
    if getattr (args ,'twopass_full_image_weight',None )is None :
        args .twopass_full_image_weight =USER_DEFAULTS .get ('twopass_full_image_weight',0.35 )
    args .twopass_full_image_weight =float (max (0.0 ,min (1.0 ,float (args .twopass_full_image_weight ))))
    if getattr (args ,'twopass_cluster_aux_reid_weight',None )is None :
        args .twopass_cluster_aux_reid_weight =USER_DEFAULTS .get ('twopass_cluster_aux_reid_weight',0.0 )
    args .twopass_cluster_aux_reid_weight =float (max (0.0 ,min (1.0 ,float (args .twopass_cluster_aux_reid_weight ))))
    if getattr (args ,'twopass_cluster_aux_gender_weight',None )is None :
        args .twopass_cluster_aux_gender_weight =USER_DEFAULTS .get ('twopass_cluster_aux_gender_weight',0.0 )
    args .twopass_cluster_aux_gender_weight =float (max (0.0 ,min (1.0 ,float (args .twopass_cluster_aux_gender_weight ))))
    if getattr (args ,'twopass_cluster_aux_age_weight',None )is None :
        args .twopass_cluster_aux_age_weight =USER_DEFAULTS .get ('twopass_cluster_aux_age_weight',0.0 )
    args .twopass_cluster_aux_age_weight =float (max (0.0 ,min (1.0 ,float (args .twopass_cluster_aux_age_weight ))))
    if getattr (args ,'twopass_auto_merge',None )is None :
        args .twopass_auto_merge =USER_DEFAULTS .get ('twopass_auto_merge',False )
    args .twopass_auto_merge =bool (int (args .twopass_auto_merge ))if isinstance (args .twopass_auto_merge ,(int ,np .integer ))else bool (args .twopass_auto_merge )
    if getattr (args ,'twopass_target_cluster_size',None )is None :
        args .twopass_target_cluster_size =USER_DEFAULTS .get ('twopass_target_cluster_size',26.0 )
    args .twopass_target_cluster_size =float (max (4.0 ,float (args .twopass_target_cluster_size )))
    if getattr (args ,'twopass_merge_min_sim',None )is None :
        args .twopass_merge_min_sim =USER_DEFAULTS .get ('twopass_merge_min_sim',0.45 )
    args .twopass_merge_min_sim =float (max (-1.0 ,min (1.0 ,float (args .twopass_merge_min_sim ))))
    if getattr (args ,'twopass_refine_split',None )is None :
        args .twopass_refine_split =USER_DEFAULTS .get ('twopass_refine_split',True )
    args .twopass_refine_split =bool (int (args .twopass_refine_split ))if isinstance (args .twopass_refine_split ,(int ,np .integer ))else bool (args .twopass_refine_split )
    if getattr (args ,'twopass_refine_min_size',None )is None :
        args .twopass_refine_min_size =USER_DEFAULTS .get ('twopass_refine_min_size',18 )
    args .twopass_refine_min_size =int (max (4 ,int (args .twopass_refine_min_size )))
    if getattr (args ,'twopass_refine_delta',None )is None :
        args .twopass_refine_delta =USER_DEFAULTS .get ('twopass_refine_delta',0.08 )
    args .twopass_refine_delta =float (max (0.0 ,float (args .twopass_refine_delta )))
    if getattr (args ,'twopass_refine_min_subcluster',None )is None :
        args .twopass_refine_min_subcluster =USER_DEFAULTS .get ('twopass_refine_min_subcluster',4 )
    args .twopass_refine_min_subcluster =int (max (2 ,int (args .twopass_refine_min_subcluster )))
    if getattr (args ,'twopass_centroid_merge',None )is None :
        args .twopass_centroid_merge =USER_DEFAULTS .get ('twopass_centroid_merge',False )
    args .twopass_centroid_merge =bool (int (args .twopass_centroid_merge ))if isinstance (args .twopass_centroid_merge ,(int ,np .integer ))else bool (args .twopass_centroid_merge )
    if getattr (args ,'twopass_centroid_image_only',None )is None :
        args .twopass_centroid_image_only =USER_DEFAULTS .get ('twopass_centroid_image_only',True )
    args .twopass_centroid_image_only =bool (int (args .twopass_centroid_image_only ))if isinstance (args .twopass_centroid_image_only ,(int ,np .integer ))else bool (args .twopass_centroid_image_only )
    if getattr (args ,'twopass_centroid_min_fragments',None )is None :
        args .twopass_centroid_min_fragments =USER_DEFAULTS .get ('twopass_centroid_min_fragments',128 )
    args .twopass_centroid_min_fragments =int (max (2 ,int (args .twopass_centroid_min_fragments )))
    if getattr (args ,'twopass_centroid_method',None )is None :
        args .twopass_centroid_method =USER_DEFAULTS .get ('twopass_centroid_method','agglomerative')
    args .twopass_centroid_method =str (args .twopass_centroid_method or 'agglomerative').lower ()
    if getattr (args ,'twopass_centroid_threshold_auto',None )is None :
        args .twopass_centroid_threshold_auto =USER_DEFAULTS .get ('twopass_centroid_threshold_auto',True )
    args .twopass_centroid_threshold_auto =bool (int (args .twopass_centroid_threshold_auto ))if isinstance (args .twopass_centroid_threshold_auto ,(int ,np .integer ))else bool (args .twopass_centroid_threshold_auto )
    if getattr (args ,'twopass_centroid_threshold',None )is None :
        args .twopass_centroid_threshold =USER_DEFAULTS .get ('twopass_centroid_threshold',0.18 )
    args .twopass_centroid_threshold =float (max (0.0 ,min (1.0 ,float (args .twopass_centroid_threshold ))))
    if getattr (args ,'twopass_centroid_fragment_group_size',None )is None :
        args .twopass_centroid_fragment_group_size =USER_DEFAULTS .get ('twopass_centroid_fragment_group_size',16.0 )
    args .twopass_centroid_fragment_group_size =float (max (1.0 ,float (args .twopass_centroid_fragment_group_size )))
    if getattr (args ,'twopass_centroid_auto_merge',None )is None :
        args .twopass_centroid_auto_merge =USER_DEFAULTS .get ('twopass_centroid_auto_merge',False )
    args .twopass_centroid_auto_merge =bool (int (args .twopass_centroid_auto_merge ))if isinstance (args .twopass_centroid_auto_merge ,(int ,np .integer ))else bool (args .twopass_centroid_auto_merge )
    if getattr (args ,'twopass_centroid_merge_min_sim',None )is None :
        args .twopass_centroid_merge_min_sim =USER_DEFAULTS .get ('twopass_centroid_merge_min_sim',0.40 )
    args .twopass_centroid_merge_min_sim =float (max (-1.0 ,min (1.0 ,float (args .twopass_centroid_merge_min_sim ))))

    if getattr (args ,'post_merge_enable',None )is None :
        args .post_merge_enable =USER_DEFAULTS .get ('post_merge_enable',True )
    args .post_merge_enable =bool (int (args .post_merge_enable ))if isinstance (args .post_merge_enable ,(int ,np .integer ))else bool (args .post_merge_enable )
    if getattr (args ,'post_merge_enable_video',None )is None :
        args .post_merge_enable_video =USER_DEFAULTS .get ('post_merge_enable_video',args .post_merge_enable )
    args .post_merge_enable_video =bool (int (args .post_merge_enable_video ))if isinstance (args .post_merge_enable_video ,(int ,np .integer ))else bool (args .post_merge_enable_video )
    if getattr (args ,'post_merge_enable_image',None )is None :
        args .post_merge_enable_image =USER_DEFAULTS .get ('post_merge_enable_image',args .post_merge_enable )
    args .post_merge_enable_image =bool (int (args .post_merge_enable_image ))if isinstance (args .post_merge_enable_image ,(int ,np .integer ))else bool (args .post_merge_enable_image )
    if getattr (args ,'post_merge_video_min_sim',None )is None :
        args .post_merge_video_min_sim =USER_DEFAULTS .get ('post_merge_video_min_sim',0.41 )
    args .post_merge_video_min_sim =float (args .post_merge_video_min_sim )
    if getattr (args ,'post_merge_image_min_sim',None )is None :
        args .post_merge_image_min_sim =USER_DEFAULTS .get ('post_merge_image_min_sim',0.56 )
    args .post_merge_image_min_sim =float (args .post_merge_image_min_sim )
    if getattr (args ,'post_merge_mixed_min_sim',None )is None :
        args .post_merge_mixed_min_sim =USER_DEFAULTS .get ('post_merge_mixed_min_sim',0.52 )
    args .post_merge_mixed_min_sim =float (args .post_merge_mixed_min_sim )
    if getattr (args ,'post_merge_gender_conf_guard',None )is None :
        args .post_merge_gender_conf_guard =USER_DEFAULTS .get ('post_merge_gender_conf_guard',0.80 )
    args .post_merge_gender_conf_guard =float (args .post_merge_gender_conf_guard )
    if getattr (args ,'post_merge_age_gap_soft',None )is None :
        args .post_merge_age_gap_soft =USER_DEFAULTS .get ('post_merge_age_gap_soft',3.0 )
    args .post_merge_age_gap_soft =float (args .post_merge_age_gap_soft )
    if getattr (args ,'post_merge_age_gap_hard',None )is None :
        args .post_merge_age_gap_hard =USER_DEFAULTS .get ('post_merge_age_gap_hard',6.0 )
    args .post_merge_age_gap_hard =float (args .post_merge_age_gap_hard )
    if getattr (args ,'post_merge_fragment_relax',None )is None :
        args .post_merge_fragment_relax =USER_DEFAULTS .get ('post_merge_fragment_relax',0.04 )
    args .post_merge_fragment_relax =float (args .post_merge_fragment_relax )
    if getattr (args ,'post_merge_tiny_image_max_size',None )is None :
        args .post_merge_tiny_image_max_size =USER_DEFAULTS .get ('post_merge_tiny_image_max_size',3 )
    args .post_merge_tiny_image_max_size =int (max (1 ,int (args .post_merge_tiny_image_max_size )))
    if getattr (args ,'post_merge_tiny_image_min_sim',None )is None :
        args .post_merge_tiny_image_min_sim =USER_DEFAULTS .get ('post_merge_tiny_image_min_sim',0.44 )
    args .post_merge_tiny_image_min_sim =float (args .post_merge_tiny_image_min_sim )

        
    if getattr (args ,'yolo_repo_root',None )is None :
        args .yolo_repo_root =USER_DEFAULTS .get ('yolo_repo_root',_YOLO26_LOCAL_VENDOR )
    if args .det_model is None :
        args .det_model =USER_DEFAULTS .get ('det_model')
    if getattr (args ,'det_conf',None )is None :
        args .det_conf =USER_DEFAULTS .get ('det_conf',0.5 )
    if getattr (args ,'det_iou',None )is None :
        args .det_iou =USER_DEFAULTS .get ('det_iou',0.5 )
    if getattr (args ,'use_sam',None )is None :
        args .use_sam =USER_DEFAULTS .get ('use_sam',True )
    args .use_sam =bool (int (args .use_sam ))if isinstance (args .use_sam ,(int ,np .integer ))else bool (args .use_sam )
    if args .tracker is None :
        args .tracker =USER_DEFAULTS .get ('tracker')
    if args .sec_interval is None :
        args .sec_interval =USER_DEFAULTS .get ('sec_interval')
    if getattr (args ,'vid_stride',None )is None :
        args .vid_stride =USER_DEFAULTS .get ('vid_stride',1 )
    args .vid_stride =int (max (1 ,int (args .vid_stride or 1 )))
    if args .max_keep_missing_sec is None :
        args .max_keep_missing_sec =USER_DEFAULTS .get ('max_keep_missing_sec')

    assert args .input ,"input path is required (or set USER_DEFAULTS['input'])"

    
    _image_exts ={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
    _video_exts ={'.mp4','.avi','.mov','.mkv','.wmv','.flv','.webm','.ts','.m4v'}

    def _detect_input_kind (input_path :str )->str :
        if not input_path :
            return 'unknown'
        if os .path .isdir (input_path ):
            has_img =False 
            has_vid =False 
            for root ,dirs ,files in os .walk (input_path ):
                for name in files :
                    ext =os .path .splitext (name )[1 ].lower ()
                    if ext in _image_exts :
                        has_img =True 
                    elif ext in _video_exts :
                        has_vid =True 
                    if has_img and has_vid :
                        return 'mixed_dir'
            if has_img :
                return 'image_dir'
            if has_vid :
                return 'video_dir'
            return 'empty_dir'
        ext =os .path .splitext (input_path )[1 ].lower ()
        if ext in _image_exts :
            return 'image_file'
        if ext in _video_exts :
            return 'video_file'
        img_try =_imread_unicode (input_path )
        if img_try is not None :
            return 'image_file'
        cap_try =cv2 .VideoCapture (input_path )
        is_video =cap_try .isOpened ()
        cap_try .release ()
        if is_video :
            return 'video_file'
        return 'unknown'

    input_kind =_detect_input_kind (args .input )
    if _user_model_path is None :
        mode_model_path =None 
        if input_kind in {'image_dir','image_file'}:
            mode_model_path =USER_DEFAULTS .get ('image_model_path')or USER_DEFAULTS .get ('model_path')
        elif input_kind in {'video_dir','video_file'}:
            mode_model_path =USER_DEFAULTS .get ('video_model_path')or USER_DEFAULTS .get ('model_path')
        if mode_model_path :
            args .model_path =mode_model_path 
            if bool (getattr (args ,'verbose',False )):
                print (f"[INFO] Auto-selected model_path for {input_kind}: {args.model_path}")

    if os .path .isdir (args .input ):
        input_dir =args .input 
        
        
        
        image_as_roi_inferred =False 
        if getattr (args ,'image_as_roi',None )is None :
            dir_flag =os .path .basename (os .path .normpath (input_dir )).lower ()
            args .image_as_roi =1 if (('roi'in dir_flag )or ('crop'in dir_flag ))else 0 
            image_as_roi_inferred =True 
        input_base =os .path .basename (os .path .normpath (input_dir ))
        #  --output
        if not args .output :
            output_dir =_ensure_dir (input_dir .rstrip ('/\\')+'_out')
        else :
            output_dir =_ensure_dir (os .path .join (args .output ,input_base ))

            
        if bool (getattr (args ,'verbose',False )):
            ts =time .strftime ('%Y%m%d_%H%M%S')
            log_dir =_ensure_dir (os .path .join (output_dir ,'video_openworld_logs'))
            log_path =os .path .join (log_dir ,f'openworld_{ts}.log')
            _enable_stdout_tee (log_path )
            print (f" verbose  {log_path}")

            
        infer_args =build_infer_args_for_inference (
        cfg_path =args .cfg ,out_root =_ensure_dir (os .path .join (output_dir ,'video_openworld_logs')),
        mode =args .mode ,sim_th =args .similarity_threshold ,verbose =args .verbose ,
        )
        
        infer_args .base_threshold =args .base_threshold 
        infer_args .adaptive_threshold_min =args .adaptive_threshold_min 
        infer_args .adaptive_threshold_max =args .adaptive_threshold_max 
        infer_args .confidence_threshold =args .confidence_threshold 
        infer_args .quality_threshold =args .quality_threshold 
        infer_args .sim_cosine_w =args .sim_cosine_w 
        infer_args .sim_euclid_w =args .sim_euclid_w 
        infer_args .aux_gender_penalty =args .aux_gender_penalty 
        infer_args .aux_age_reweight =args .aux_age_reweight 
        infer_args .aux_min_age_sigma =args .aux_min_age_sigma 

        config =get_config (infer_args )
        inference =PandaReIDInference (config =config ,model_path =args .model_path ,args =infer_args )
        setattr (inference ,'args',args )
        _attach_aux_predictor (inference ,args )
        device =inference .device 
        _patch_prototype_threshold_no_quality (inference ,verbose =bool (getattr (args ,'verbose',False )))
        
        inference .global_to_display ={}
        inference .next_display_id =1 
        inference .global_gender_stats ={}
        
        inference .group_by_id =bool (getattr (args ,'group_by_id',False ))
        if inference .group_by_id :
            try :
            
                id_group_root =_ensure_dir (output_dir )
                inference .id_group_root =id_group_root 
                print (f"[INFO] Group-by-ID output enabled: {id_group_root}")
            except Exception as e :
                print (f"[WARN] Failed to initialize group-by-id output: {e}")
                inference .group_by_id =False 

        yolo =_build_yolo_detector (args )
        sam_predictor =None 
        if not args .use_sam :
            print ("[INFO] SAM refinement disabled by --use-sam 0.")
        elif not _HAS_SAM :
            print ("[WARN] segment_anything not available, SAM refinement is disabled.")
        else :
            sam_ckpt =os .path .join (os .path .dirname (__file__ ),'SAMmodel','sam_vit_h_4b8939.pth')
            if os .path .isfile (sam_ckpt ):
                sam =sam_model_registry .get ('vit_h',None )
                if sam is not None :
                    sam_model =sam (checkpoint =sam_ckpt )
                    sam_model .to (device )
                    sam_predictor =SamPredictor (sam_model )

                    
        all_image_paths =[]
        all_video_paths =[]
        for root ,dirs ,files in os .walk (input_dir ):
        # 
            dirs [:]=[d for d in dirs if not d .endswith ('_out')and d !=os .path .basename (output_dir )]
            for name in sorted (files ):
                ext =os .path .splitext (name )[1 ].lower ()
                full_path =os .path .join (root ,name )
                if ext in _image_exts :
                    all_image_paths .append (full_path )
                elif ext in _video_exts :
                    all_video_paths .append (full_path )

        if getattr (args ,'file_list',None ):
            selected_paths =set ()
            try :
                with open (args .file_list ,'r',encoding ='utf-8')as _fl :
                    for _line in _fl :
                        _item =_line .strip ().strip ('"')
                        if not _item :
                            continue 
                        _path =_item if os .path .isabs (_item )else os .path .join (input_dir ,_item )
                        selected_paths .add (os .path .normcase (os .path .abspath (_path )))
                all_image_paths =[p for p in all_image_paths if os .path .normcase (os .path .abspath (p ))in selected_paths ]
                all_video_paths =[p for p in all_video_paths if os .path .normcase (os .path .abspath (p ))in selected_paths ]
                print (f"[INFO] Applied --file-list: {len(selected_paths)} requested, {len(all_image_paths)} images and {len(all_video_paths)} videos selected.")
            except Exception as _file_list_e :
                raise RuntimeError (f"Failed to read --file-list {args.file_list}: {_file_list_e}")

        print (f"[INFO] Found {len(all_image_paths)} images and {len(all_video_paths)} videos.")
        if args .shuffle_images and len (all_image_paths )>1 :
            rng =np .random .default_rng (int (args .shuffle_seed ))
            rng .shuffle (all_image_paths )
            print (f"[INFO] Shuffled image inference order with seed={int(args.shuffle_seed)}")

        
        if image_as_roi_inferred and len (all_video_paths )>0 :
            args .image_as_roi =0 
        if getattr (args ,'verbose',False ):
            src ="auto"if image_as_roi_inferred else "user"
            print (f"[INFO] image_as_roi={int(args.image_as_roi)} (source={src})")

            
            # detection_records: list of (rel_path, list_of_tuples)   tuple = (disp_id, gender, age)
        detection_records =[]

        
        video_summaries =[]# [(video_path, summary_dict)]
        for vid_idx ,in_path in enumerate (all_video_paths ):
            print (f"\n [{vid_idx+1}/{len(all_video_paths)}]  {os.path.basename(in_path)}")

            
            try :
                test_cap =cv2 .VideoCapture (in_path )
                if not test_cap .isOpened ():
                    print (f"  {in_path}")
                    test_cap .release ()
                    continue 
                    
                ret ,test_frame =test_cap .read ()
                test_cap .release ()
                if not ret or test_frame is None :
                    print (f"  {in_path}")
                    continue 
            except Exception as e :
                print (f" {in_path}: {e}")
                continue 

                # 
            rel_path =os .path .relpath (in_path ,input_dir )
            
            if getattr (inference ,'group_by_id',False ):
                flat =_flatten_rel_path (rel_path )
                stem =os .path .splitext (flat )[0 ]if flat else os .path .splitext (os .path .basename (in_path ))[0 ]
                if bool (getattr (args ,'save_vis',True )):
                    tmp_dir =_ensure_dir (os .path .join (output_dir ,'video_openworld_logs','tmp_videos'))
                    out_path =os .path .join (tmp_dir ,f"{stem}.mp4")
                else :
                
                    out_path =os .path .join (output_dir ,'video_openworld_logs','tmp_videos',f"{stem}.mp4")
            else :
                out_path =os .path .join (output_dir ,rel_path )
                out_dir_for_file =os .path .dirname (out_path )
                if out_dir_for_file and bool (getattr (args ,'save_vis',True )):
                    _ensure_dir (out_dir_for_file )
                    
            args .input =in_path 
            args .output =out_path 
            try :
                vid_summary =run_video_dirmode (args ,inference ,yolo ,sam_predictor ,device ,input_rel_path =rel_path )
                video_summaries .append ((in_path ,vid_summary ))
                
                if vid_summary :
                
                    vid_records =[]
                    for gid ,v in sorted (vid_summary .items (),key =lambda x :x [1 ].get ('display_id')or x [0 ]):
                        disp_id =v .get ('display_id')or gid 
                        gender =v .get ('gender','-')
                        age =v .get ('age_mean')
                        vid_records .append ((disp_id ,gender ,age ))
                    if vid_records :
                        detection_records .append ((rel_path ,vid_records ))
                print (f"[{vid_idx+1}/{len(all_video_paths)}] Processed video: {os.path.basename(in_path)}")
            except KeyboardInterrupt :
                print (f"\n ..")
                break 
            except Exception as e :
                import traceback 
                print (f"[ERROR] Failed processing {in_path}: {e}")
                traceback .print_exc ()
            finally :
            # OOM
                import gc 
                gc .collect ()
                if torch .cuda .is_available ():
                    torch .cuda .empty_cache ()
                    
                    if (vid_idx +1 )%10 ==0 :
                        mem_used =torch .cuda .memory_allocated ()/1024 **3 
                        mem_total =torch .cuda .get_device_properties (0 ).total_memory /1024 **3 
                        print (f"   [GPU] memory={mem_used:.2f}GB / {mem_total:.2f}GB")

                        # 
        gid2disp_runtime =getattr (inference ,'global_to_display',{})
        gstats_runtime =getattr (inference ,'global_gender_stats',{})
        agestats_runtime =getattr (inference ,'global_age_stats',{})
        final_merge_alias ={}
        if bool (getattr (args ,'post_merge_enable_video',getattr (args ,'post_merge_enable',True ))):
            final_merge_alias =_consolidate_runtime_identities (
            inference =inference ,
            candidate_gids =set (gid2disp_runtime .keys ())|set (gstats_runtime .keys ())|set (agestats_runtime .keys ()),
            global_to_display =gid2disp_runtime ,
            global_gender_stats =gstats_runtime ,
            global_age_stats =agestats_runtime ,
            args =args ,
            gid_feature_map =None ,
            id_root =getattr (inference ,'id_group_root',None )if getattr (inference ,'group_by_id',False )else None ,
            log_prefix ="directory-post-merge-preimage",
            )
        if final_merge_alias :
            detection_records =_remap_detection_records (detection_records ,gid2disp_runtime ,final_merge_alias )
            remapped_video_summaries =[]
            for name ,summ in video_summaries :
                grouped ={}
                for gid ,v in (summ or {}).items ():
                    root_gid =_remap_gid_to_root (gid ,final_merge_alias )or gid 
                    grouped [root_gid ]={
                    'display_id':gid2disp_runtime .get (root_gid ,v .get ('display_id')or root_gid ),
                    'gender':v .get ('gender'),
                    'male_mean':v .get ('male_mean'),
                    'gender_conf':v .get ('gender_conf'),
                    'age_mean':v .get ('age_mean'),
                    }
                remapped_video_summaries .append ((name ,grouped ))
            video_summaries =remapped_video_summaries
        total_vid_ids =sum (len (s [1 ])for s in video_summaries )
        print (f"[INFO] Video summaries: {len(video_summaries)} files, {total_vid_ids} unique IDs")
        for name ,summ in video_summaries :
            print (f"  {name}: {len(summ)} IDs")
            for gid ,v in summ .items ():
                age_str =f"{v['age_mean']:.1f}"if v ['age_mean']is not None else "-"
                disp =v .get ('display_id')or gid 
                print (f"    {disp} ({gid}): gender={v['gender']} (conf={float(v.get('gender_conf',0.0)):.2f}), age={age_str}")

                # ==========  ==========
        img_count ,img_kept =0 ,0 
        if len (all_image_paths )>0 and str (getattr (args ,'image_cluster_mode','twopass')or 'twopass').lower ()=='twopass':
            img_count ,img_kept ,image_detection_records =run_image_dirmode_twopass (
            args ,
            inference ,
            yolo ,
            sam_predictor ,
            device ,
            input_dir =input_dir ,
            output_dir =output_dir ,
            all_image_paths =all_image_paths ,
            )
            detection_records .extend (image_detection_records )
        else :
            for in_path in all_image_paths :
                rel_path =os .path .relpath (in_path ,input_dir )
                if getattr (inference ,'group_by_id',False ):
                    out_path =None 
                else :
                    out_path =os .path .join (output_dir ,rel_path )
                    out_dir_for_file =os .path .dirname (out_path )
                    if out_dir_for_file and bool (getattr (args ,'save_vis',True )):
                        _ensure_dir (out_dir_for_file )

                args .input =in_path 
                args .output =out_path 
                try :
                    detected_records =run_image (args ,inference ,yolo ,sam_predictor ,device ,input_rel_path =rel_path )
                    if detected_records :
                        detection_records .append ((rel_path ,detected_records ))
                        note =""if bool (getattr (args ,'save_vis',True ))else "()"
                        print (f"{note}): {in_path}")
                        img_kept +=1 
                    else :
                        if getattr (args ,'verbose',False ):
                            print (f"  (: {in_path}")
                except AssertionError as e :
                    print (f"  {in_path}: {e}")
                except Exception as e :
                    print (f"[ERROR] Failed processing {in_path}: {e}")
                img_count +=1 

            # D
        gid2disp_runtime =getattr (inference ,'global_to_display',{})
        gstats_runtime =getattr (inference ,'global_gender_stats',{})
        agestats_runtime =getattr (inference ,'global_age_stats',{})
        final_merge_alias ={}
        _enable_dir_post_merge =False 
        if len (all_image_paths )>0 and bool (getattr (args ,'post_merge_enable_image',getattr (args ,'post_merge_enable',True ))):
            _enable_dir_post_merge =True 
        if len (all_video_paths )>0 and bool (getattr (args ,'post_merge_enable_video',getattr (args ,'post_merge_enable',True ))):
            _enable_dir_post_merge =True 
        if _enable_dir_post_merge :
            final_merge_alias =_consolidate_runtime_identities (
            inference =inference ,
            candidate_gids =set (gid2disp_runtime .keys ())|set (gstats_runtime .keys ())|set (agestats_runtime .keys ()),
            global_to_display =gid2disp_runtime ,
            global_gender_stats =gstats_runtime ,
            global_age_stats =agestats_runtime ,
            args =args ,
            gid_feature_map =None ,
            id_root =getattr (inference ,'id_group_root',None )if getattr (inference ,'group_by_id',False )else None ,
            log_prefix ="directory-post-merge-final",
            )
        if final_merge_alias :
            detection_records =_remap_detection_records (detection_records ,gid2disp_runtime ,final_merge_alias )

        image_summary =[]
        gid2disp =getattr (inference ,'global_to_display',{})
        gstats =getattr (inference ,'global_gender_stats',{})
        agestats =getattr (inference ,'global_age_stats',{})
        for gid ,stat in gstats .items ():
            forced_meta =_ID_META .get (gid )
            if forced_meta is not None :
                gender =forced_meta .get ('gender')or 'F'
                male_mean =1.0 if gender =='M'else 0.0 
                try :
                    age_mean =float (forced_meta .get ('age'))if forced_meta .get ('age')is not None else None 
                except Exception :
                    age_mean =None 
                image_summary .append ({
                'gid':gid ,
                'display_id':gid2disp .get (gid ,gid ),
                'gender':gender ,
                'male_mean':male_mean ,
                'gender_conf':_gender_confidence_01 (male_mean ,stable_gender =gender ,threshold =args .gender_threshold ),
                'age_mean':age_mean ,
                })
                continue 

            total =stat ['alpha_male']+stat ['alpha_female']
            male_mean =float (stat ['alpha_male']/max (1e-6 ,total ))
            gender =stat .get ('stable_gender')or ('M'if male_mean >=args .gender_threshold else 'F')
            ages =agestats .get (gid ,[])
            # 
            
            
            if getattr (args ,'age_scope','video')=='global':
                age_mean =float (np .mean (ages ))if len (ages )>0 else None 
            else :
                age_mean =None 
            image_summary .append ({
            'gid':gid ,
            'display_id':gid2disp .get (gid ,''),
            'gender':gender ,
            'male_mean':male_mean ,
            'gender_conf':_gender_confidence_01 (male_mean ,stable_gender =gender ,threshold =args .gender_threshold ),
            'age_mean':age_mean ,
            })

        print (f" {img_count}  {img_kept} {output_dir}")
        print (f"[INFO] Image ID summary count: {len(image_summary)}")

        
        gid2disp =getattr (inference ,'global_to_display',{})
        gstats =getattr (inference ,'global_gender_stats',{})
        agestats =getattr (inference ,'global_age_stats',{})
        final_summary =[]
        for gid ,stat in gstats .items ():
            forced_meta =_ID_META .get (gid )
            if forced_meta is not None :
                gender =forced_meta .get ('gender')or 'F'
                male_mean =1.0 if gender =='M'else 0.0 
                try :
                    age_mean =float (forced_meta .get ('age'))if forced_meta .get ('age')is not None else None 
                except Exception :
                    age_mean =None 
                final_summary .append ({
                'gid':gid ,
                'display_id':gid2disp .get (gid ,gid ),
                'gender':gender ,
                'male_mean':male_mean ,
                'gender_conf':_gender_confidence_01 (male_mean ,stable_gender =gender ,threshold =args .gender_threshold ),
                'age_mean':age_mean ,
                })
                continue 

            total =stat ['alpha_male']+stat ['alpha_female']
            male_mean =float (stat ['alpha_male']/max (1e-6 ,total ))
            gender =stat .get ('stable_gender')or ('M'if male_mean >=args .gender_threshold else 'F')
            ages =agestats .get (gid ,[])
            
            
            if getattr (args ,'age_scope','video')=='global':
                age_mean =float (np .mean (ages ))if len (ages )>0 else None 
            else :
                age_mean =None 
            final_summary .append ({
            'gid':gid ,
            'display_id':gid2disp .get (gid ,''),
            'gender':gender ,
            'male_mean':male_mean ,
            'gender_conf':_gender_confidence_01 (male_mean ,stable_gender =gender ,threshold =args .gender_threshold ),
            'age_mean':age_mean ,
            })
        print ("[INFO] Final identity summary count: {} IDs".format (len (final_summary )))
        for it in sorted (final_summary ,key =lambda x :x ['display_id']or x ['gid']):
            age_str =f"{it['age_mean']:.1f}"if it ['age_mean']is not None else "-"
            disp =it ['display_id']or it ['gid']
            print (f"  {disp} ({it['gid']}): gender={it['gender']} (conf={it.get('gender_conf',0.0):.2f}), age={age_str}")

            
            
        detail_rows =getattr (inference ,'_last_detection_details',None )
        dir_run_summary =getattr (inference ,'_last_directory_run_summary',{})or {}
        identity_feature_summary ={}
        try :
            feat_store =_identity_gid_features_store (inference )
            gid2disp_for_feat =getattr (inference ,'global_to_display',{})or {}
            for _gid ,_feat in (feat_store or {}).items ():
                _arr =np .asarray (_feat ,dtype =np .float32 ).reshape (-1 )
                if _arr .size <=0 :
                    continue 
                _norm =float (np .linalg .norm (_arr ))
                if np .isfinite (_norm )and _norm >0 :
                    _arr =_arr /_norm 
                identity_feature_summary [str (_gid )]={
                'display_id':str (gid2disp_for_feat .get (_gid ,_gid )),
                'feature':_arr .astype (float ).tolist (),
                }
        except Exception as _feat_e :
            identity_feature_summary ={'_error':str (_feat_e )}
        try :
            json_payload =save_folder_level_metrics (
            output_dir ,
            detection_records ,
            detection_details =detail_rows ,
            extra_summary ={
            'img_count':int (img_count ),
            'img_kept':int (img_kept ),
            'image_id_summary_count':int (len (image_summary )),
            'final_identity_summary_count':int (len (final_summary )),
            'identity_feature_summary':identity_feature_summary ,
            **dir_run_summary ,
            },
            )
            folder_metrics =(json_payload .get ('folder_level_metrics',{})or {}).get ('clustering',{})or {} 
            if folder_metrics :
                print (
                f"[METRIC] assignment={float(folder_metrics.get('assignment_accuracy',0.0)):.4f}, "
                f"purity={float(folder_metrics.get('cluster_purity',0.0)):.4f}, "
                f"id_acc={float(folder_metrics.get('id_count_accuracy',0.0)):.4f}, "
                f"pred_ids={int(folder_metrics.get('predicted_id_count',0))}, true_ids={int(folder_metrics.get('true_id_count',0))}"
                )
        except Exception as e :
            print (f"[WARN] Failed writing detection_results.json: {e}")

        xlsx_output_path =os .path .join (output_dir ,'detection_results.xlsx')
        try :
            from openpyxl import Workbook 
            from openpyxl .styles import Font ,Alignment 

            wb =Workbook ()
            ws =wb .active 
            ws .title ="Summary"

            # 
            headers =['ID','','','']
            for col ,header in enumerate (headers ,1 ):
                cell =ws .cell (row =1 ,column =col ,value =header )
                cell .font =Font (bold =True )
                cell .alignment =Alignment (horizontal ='center')

                
            detection_records_sorted =sorted (detection_records ,key =lambda x :x [0 ])
            row_idx =2 
            total_records =0 
            for rel_path ,id_records in detection_records_sorted :
            
                rel_path_win =rel_path .replace ('/','\\')
                
                for disp_id ,gender ,age in sorted (id_records ,key =lambda x :x [0 ]):
                    ws .cell (row =row_idx ,column =1 ,value =disp_id )
                    ws .cell (row =row_idx ,column =2 ,value =rel_path_win )
                    ws .cell (row =row_idx ,column =3 ,value =gender if gender else '-')
                    
                    age_str =f"{age:.1f}"if age is not None else '-'
                    ws .cell (row =row_idx ,column =4 ,value =age_str )
                    row_idx +=1 
                    total_records +=1 

                    # 
            ws .column_dimensions ['A'].width =10 
            ws .column_dimensions ['B'].width =60 
            ws .column_dimensions ['C'].width =8 
            ws .column_dimensions ['D'].width =10 

            
            
            id_folders =defaultdict (set )# disp_id -> {folder1, folder2, ...}
            id_genders =defaultdict (list )# disp_id -> ['M'/'F', ...]
            id_ages =defaultdict (list )# disp_id -> [age_float, ...]
            id_counts =defaultdict (int )# disp_id -> count

            for rel_path ,id_records in detection_records_sorted :
                folder =_first_folder_of_rel_path (rel_path )
                for disp_id ,gender ,age in id_records :
                    disp_id =str (disp_id )
                    id_counts [disp_id ]+=1 
                    if folder :
                        id_folders [disp_id ].add (str (folder ))
                    if isinstance (gender ,str )and gender .strip ()in {'M','F'}:
                        id_genders [disp_id ].append (gender .strip ())
                    if age is not None :
                        try :
                            id_ages [disp_id ].append (float (age ))
                        except Exception :
                            pass 

            def _sort_disp_id_key (x :str ):
                try :
                    if _is_canonical_id_name (x ):
                        return (0 ,int (x [2 :]))
                except Exception :
                    pass 
                return (1 ,str (x ))

            def _majority_gender (vs ):
                if not vs :
                    return '-'
                m =sum (1 for v in vs if v =='M')
                f =sum (1 for v in vs if v =='F')
                if m ==f :
                    return vs [-1 ]
                return 'M'if m >f else 'F'

            ws_sum =wb .create_sheet (title ="ID")
            sum_headers =['ID','Count','Gender','Age','Samples']
            for col ,header in enumerate (sum_headers ,1 ):
                cell =ws_sum .cell (row =1 ,column =col ,value =header )
                cell .font =Font (bold =True )
                cell .alignment =Alignment (horizontal ='center')

            row =2 
            for disp_id in sorted (id_counts .keys (),key =_sort_disp_id_key ):
                folders =sorted (id_folders .get (disp_id ,set ()),key =str )
                folders_str =','.join (folders )if folders else '-'
                gender =_majority_gender (id_genders .get (disp_id ,[]))
                ages =id_ages .get (disp_id ,[])
                age_mean =float (np .mean (ages ))if len (ages )>0 else None 
                age_str =f"{age_mean:.1f}"if age_mean is not None else '-'
                ws_sum .cell (row =row ,column =1 ,value =disp_id )
                ws_sum .cell (row =row ,column =2 ,value =folders_str )
                ws_sum .cell (row =row ,column =3 ,value =gender )
                ws_sum .cell (row =row ,column =4 ,value =age_str )
                ws_sum .cell (row =row ,column =5 ,value =int (id_counts [disp_id ]))
                row +=1 

            ws_sum .column_dimensions ['A'].width =10 
            ws_sum .column_dimensions ['B'].width =20 
            ws_sum .column_dimensions ['C'].width =8 
            ws_sum .column_dimensions ['D'].width =10 
            ws_sum .column_dimensions ['E'].width =10 

            wb .save (xlsx_output_path )
            print (f" {xlsx_output_path}")
            print (f"   {total_records}  {len(detection_records)} ")
        except ImportError :
            print ("[WARN] openpyxl not installed. Install with: pip install openpyxl")
        except Exception as e :
            print (f"[WARN] Failed writing XLSX summary: {e}")

        return 

        # 
    ext =os .path .splitext (args .input )[1 ].lower ()
    is_image =ext in _image_exts 
    is_video =ext in _video_exts 
    if not (is_image or is_video ):
        img_try =_imread_unicode (args .input )
        if img_try is not None :
            is_image =True 
        else :
            cap_try =cv2 .VideoCapture (args .input )
            is_video =cap_try .isOpened ()
            cap_try .release ()


            
    args .output =_resolve_output_path (args .output ,args .input ,is_image ,is_video )

    out_root =_ensure_dir (os .path .join (os .path .dirname (os .path .abspath (args .output )),'video_openworld_logs'))

    
    if bool (getattr (args ,'verbose',False )):
        ts =time .strftime ('%Y%m%d_%H%M%S')
        log_path =os .path .join (out_root ,f'openworld_{ts}.log')
        _enable_stdout_tee (log_path )
        print (f" verbose  {log_path}")

    infer_args =build_infer_args_for_inference (
    cfg_path =args .cfg ,
    out_root =out_root ,
    mode =args .mode ,
    sim_th =args .similarity_threshold ,
    verbose =args .verbose ,
    )
    
    
    infer_args .base_threshold =args .base_threshold 
    infer_args .adaptive_threshold_min =args .adaptive_threshold_min 
    infer_args .adaptive_threshold_max =args .adaptive_threshold_max 
    infer_args .confidence_threshold =args .confidence_threshold 
    infer_args .quality_threshold =args .quality_threshold 
    infer_args .sim_cosine_w =args .sim_cosine_w 
    infer_args .sim_euclid_w =args .sim_euclid_w 
    infer_args .aux_gender_penalty =args .aux_gender_penalty 
    infer_args .aux_age_reweight =args .aux_age_reweight 
    infer_args .aux_min_age_sigma =args .aux_min_age_sigma 
    config =get_config (infer_args )

    
    inference =PandaReIDInference (config =config ,model_path =args .model_path ,args =infer_args )
    setattr (inference ,'args',args )
    _attach_aux_predictor (inference ,args )
    device =inference .device 
    _patch_prototype_threshold_no_quality (inference ,verbose =bool (getattr (args ,'verbose',False )))

    
    yolo =_build_yolo_detector (args )

    
    sam_predictor =None 

    if not args .use_sam :
        print ("[INFO] SAM refinement disabled by --use-sam 0.")
    elif not _HAS_SAM :
        print ("[WARN] segment_anything not available, SAM refinement is disabled.")
    else :
        sam_ckpt =os .path .join (os .path .dirname (__file__ ),'SAMmodel','sam_vit_h_4b8939.pth')
        if not os .path .isfile (sam_ckpt ):
            print (f"[WARN] SAM checkpoint missing: {sam_ckpt}; SAM refinement disabled.")
        else :
            sam =sam_model_registry .get ('vit_h',None )
            if sam is None :
                print ("[WARN] SAM model type not recognized; using ROI without SAM refinement.")
            else :
                sam_model =sam (checkpoint =sam_ckpt )
                sam_model .to (device )
                sam_predictor =SamPredictor (sam_model )

                
                
    if is_image and getattr (args ,'image_as_roi',None )is None :
        args .image_as_roi =0 
    if is_image :
        run_image (args ,inference ,yolo ,sam_predictor ,device )
        
        print (f" {args.input}")
        return 

        
        
    in_abs =os .path .abspath (args .input )
    out_abs =os .path .abspath (args .output )
    if in_abs ==out_abs :
        raise ValueError (f"Output path ({args.output}) must be different from input path ({args.input}).")

    cap =cv2 .VideoCapture (args .input )
    assert cap .isOpened (),f": {args.input}"
    fps =cap .get (cv2 .CAP_PROP_FPS )
    frame_count =cap .get (cv2 .CAP_PROP_FRAME_COUNT )
    cap .release ()# capYOLO

    
    if not fps or fps <=0 :
        print (f"[WARN] Invalid FPS detected ({fps}), fallback to 5 FPS.")
        fps =25.0 
        expected_total_frames =None # 
    else :
        expected_total_frames =int (frame_count )if frame_count and frame_count >0 else None 
        total_frames =expected_total_frames if expected_total_frames is not None else ''
        expected_duration =frame_count /fps if frame_count and fps else ''
        if isinstance (expected_duration ,float ):
            expected_duration =f"{expected_duration:.1f}s"
        print (f"[INFO] Input video: FPS={fps:.2f}, frames={total_frames}, duration={expected_duration}")
        

    out_dir =os .path .dirname (args .output )
    if out_dir :
        _ensure_dir (out_dir )
    writer =None 
    writer_size =None 
    
    
    
    write_fps =round (fps )

    # 5) ID
    # ========== D ==========
    # track_id -> {
    
    
    #   'last_seen_frame': int,          # 
    #   'first_seen_frame': int,         # 
    #   'feature_buffer': list,          # 
    #   'male_hist': deque,              # 
    #   'age_hist': deque,               # 
    #   'last_similarity': float|None,   # 
    #   'finalized': bool,               # D
    # }
    track_states ={}
    gid_reservations =_GidRangeReservation ()
    next_display_id =1 

    # ID -> (ID1/ID2...)
    global_to_display ={}

    
    global_gender_stats ={}

    
    global_age_stats ={}
    video_age_stats =defaultdict (list )

    # ID
    # finalized_tracks: { track_id: { 'global_id': str, 'display_id': str, 'similarity': float } }
    finalized_tracks ={}

    # 
    last_reid_update_frame =defaultdict (lambda :-10 **9 )
    update_every_n_frames =max (1 ,int (round (fps *args .sec_interval )))

    # 6) D
    # 
    save_vis =bool (getattr (args ,'save_vis',True ))
    frame_detections =[]if save_vis else None 

    print ("[INFO] Running online ReID and ID assignment...")
    frame_idx =-1 

    # D
    try :
        yolo .predictor =None 
    except Exception :
        pass 

        
    results_gen =yolo .track (
    source =args .input ,
    stream =True ,
    conf =args .det_conf ,
    iou =args .det_iou ,
    tracker =args .tracker ,
    persist =True ,
    verbose =False ,
    workers =0 ,
    vid_stride =int (getattr (args ,'vid_stride',1 )or 1 ),
    )

    for result in results_gen :
        frame_idx +=1 
        
        if frame_idx %100 ==0 or (total_frames >0 and frame_idx %max (1 ,total_frames //10 )==0 ):
            progress =f"{frame_idx}/{total_frames}"if total_frames >0 else f"{frame_idx}"
            print (f"   {progress} ..")

        frame_bgr =result .orig_img # BGR
        if frame_bgr is None :
            if frame_detections is not None :
                frame_detections .append ((None ,None ))
            continue 

            
        is_dark ,mean_bright =_is_dark_frame (frame_bgr ,args =args )
        if is_dark :
            if frame_detections is not None :
                frame_detections .append ((None ,None ))
            if getattr (args ,'verbose',False )and (frame_idx ==0 or frame_idx %max (1 ,int (round (fps )))==0 ):
                th =float (getattr (args ,'min_frame_brightness',0.0 )or 0.0 )
                print (f"   @frame{frame_idx}: filtered by mean_gray={mean_bright:.1f} < {th}")
            continue 
        frame_rgb =cv2 .cvtColor (frame_bgr ,cv2 .COLOR_BGR2RGB )


        # 
        boxes =result .boxes 
        if boxes is None or len (boxes )==0 :
        # 
            if frame_detections is not None :
                frame_detections .append ((None ,None ))
            continue 

        xyxy =boxes .xyxy .cpu ().numpy ().astype (int )# (N,4)
        ids =boxes .id 
        ids =ids .cpu ().numpy ().astype (int )if ids is not None else np .array ([-1 ]*len (xyxy ),dtype =int )

        
        if frame_detections is not None :
            frame_detections .append ((xyxy .copy (),ids .copy ()))

            
        current_tracks =set ()

        # eID
        do_reid_now =(frame_idx %update_every_n_frames ==0 )

        # ReIDOI
        batch_tensors =[]
        batch_track_ids =[]
        batch_crops =[]

        # ReIDAM
        if do_reid_now and sam_predictor is not None :
            try :
                sam_predictor .set_image (frame_rgb )
            except Exception :
                sam_predictor =None # 

        for i ,(x1 ,y1 ,x2 ,y2 )in enumerate (xyxy ):
            tid =int (ids [i ])if i <len (ids )else -1 
            if tid <0 :
            
                continue 
            current_tracks .add (tid )

            
            if tid not in track_states :
                track_states [tid ]={
                'global_id':None ,# None
                'display_id':None ,
                'last_seen_frame':frame_idx ,
                'first_seen_frame':frame_idx ,
                'feature_buffer':[],# 
                'male_hist':deque (maxlen =50 ),# 
                'age_hist':deque (maxlen =50 ),
                'last_similarity':None ,
                'finalized':False ,# D
                'lq_bad':0 ,'lq_good':0 ,
                }
            else :
                track_states [tid ]['last_seen_frame']=frame_idx 

                
            if do_reid_now and frame_idx -last_reid_update_frame [tid ]>=update_every_n_frames :
                crop_rgb =None 
                mask =None 
                #  SAM 
                if sam_predictor is not None :
                    try :
                        box =np .array ([x1 ,y1 ,x2 ,y2 ],dtype =np .float32 )[None ,:]
                        masks ,scores ,_ =sam_predictor .predict (box =box ,multimask_output =True )
                        if masks is not None and len (masks )>0 :
                            best =int (np .argmax (scores ))
                            mask =masks [best ].astype (bool )
                            crop_rgb =inference ._process_mask_region_to_rectangle (frame_rgb ,mask )
                    except Exception :
                    
                        crop_rgb =None 
                        mask =None 

                        
                if crop_rgb is None or crop_rgb .size ==0 :
                    crop_rgb =_crop_roi_with_expansion_numpy (frame_rgb ,(x1 ,y1 ,x2 ,y2 ))
                    mask =None 
                    if crop_rgb is None or crop_rgb .size ==0 :
                        continue 

                is_bad ,reasons =_is_low_quality_roi (
                frame_rgb ,(x1 ,y1 ,x2 ,y2 ),crop_rgb ,mask =mask ,args =args 
                )
                if is_bad :
                    state =track_states .get (tid )
                    if state is not None :
                        state ['lq_bad']=int (state .get ('lq_bad',0 )or 0 )+1 
                    if getattr (args ,'verbose',False ):
                        print (f"   [trk{tid}] OI @frame{frame_idx}: {reasons}")
                    continue 
                state =track_states .get (tid )
                if state is not None :
                    state ['lq_good']=int (state .get ('lq_good',0 )or 0 )+1 

                    
                crop_pil =Image .fromarray (crop_rgb )
                tensor =inference .transform (crop_pil ).unsqueeze (0 ).to (device )
                batch_tensors .append (tensor )
                batch_track_ids .append (tid )
                batch_crops .append (crop_rgb )

                # 
                # ========== ID ==========
        if batch_tensors :
            with torch .no_grad ():
                inp =torch .cat (batch_tensors ,dim =0 )# [B,3,H,W]
                feat_after_bn ,male_probs ,age_pred =_forward_reid_and_aux (
                inference ,inp ,crops_rgb =batch_crops 
                )

                # uffer/
            for bi ,tid in enumerate (batch_track_ids ):
                f =feat_after_bn [bi ].clone ()# 
                male_p =float (male_probs [bi ].item ())
                age_p =float (age_pred [bi ].item ())
                state =track_states [tid ]

                
                state ['feature_buffer'].append (f )
                state ['male_hist'].append (male_p )
                state ['age_hist'].append (age_p )
                last_reid_update_frame [tid ]=frame_idx 

                
                

                # 
                # ========== ID ==========-
        if frame_idx %int (fps )==0 :
            cutoff =frame_idx -int (args .max_keep_missing_sec *fps )
            to_del =[tid for tid ,st in track_states .items ()if st ['last_seen_frame']<cutoff ]

            
            
            
            reserved_ids =gid_reservations 

            for tid in to_del :
                state =track_states .get (tid )
                if not state or state .get ('finalized',False ):
                    continue 

                    
                skip_track ,skip_info =_should_skip_track_by_lowq (state ,args )
                if skip_track :
                    state ['finalized']=True 
                    state ['skipped']=True 
                    finalized_tracks [tid ]={'skipped':True ,**(skip_info or {})}
                    if getattr (args ,'verbose',False ):
                        br =float (skip_info .get ('lq_bad_ratio',0.0 )or 0.0 )if isinstance (skip_info ,dict )else 0.0 
                        print (
                        f"   [trk{tid}] ): reason={skip_info.get('skip_reason','-')} "
                        f"bad={skip_info.get('lq_bad',0)}/{skip_info.get('lq_obs',0)} ratio={br:.2f}"
                        )
                    continue 

                if len (state .get ('feature_buffer',[]))==0 :
                    continue 

                result_tuple =_finalize_track_id (
                state ,inference ,global_to_display ,global_gender_stats ,
                video_age_stats ,global_age_stats ,finalized_tracks ,
                reserved_ids ,args ,verbose =args .verbose ,tid =tid ,event ='',
                )
                if result_tuple [0 ]is None :
                    continue 
                gid ,sim_for_log ,_ =result_tuple 

                if gid not in global_to_display :
                    global_to_display [gid ]=f"ID{next_display_id}"
                    next_display_id +=1 

                finalized_tracks [tid ]={
                'global_id':gid ,
                'display_id':global_to_display [gid ],
                'similarity':sim_for_log ,
                'id_confidence':state .get ('last_id_confidence'),
                'male_hist':list (state .get ('male_hist',[])),
                'age_hist':list (state .get ('age_hist',[])),
                }

                
                state ['global_id']=gid 
                state ['display_id']=global_to_display [gid ]
                state ['last_similarity']=sim_for_log 
                state ['finalized']=True 

                
                
            for tid in to_del :
                last_reid_update_frame .pop (tid ,None )

    n_frames =len (frame_detections )if frame_detections is not None else (frame_idx +1 )
    print (f"[INFO] Total processed frames: {n_frames}")

    

    print (f"[INFO] Active track states before finalize: {len(track_states)}")
    
    
    reserved_ids =gid_reservations 

    for tid ,state in track_states .items ():
        if state .get ('finalized',False ):
        # ID
            continue 

            
        skip_track ,skip_info =_should_skip_track_by_lowq (state ,args )
        if skip_track :
            state ['finalized']=True 
            state ['skipped']=True 
            finalized_tracks [tid ]={'skipped':True ,**(skip_info or {})}
            if getattr (args ,'verbose',False ):
                br =float (skip_info .get ('lq_bad_ratio',0.0 )or 0.0 )if isinstance (skip_info ,dict )else 0.0 
                print (
                f"   [trk{tid}] ): reason={skip_info.get('skip_reason','-')} "
                f"bad={skip_info.get('lq_bad',0)}/{skip_info.get('lq_obs',0)} ratio={br:.2f}"
                )
            continue 

        if len (state .get ('feature_buffer',[]))==0 :
            continue 

        result_tuple =_finalize_track_id (
        state ,inference ,global_to_display ,global_gender_stats ,
        video_age_stats ,global_age_stats ,finalized_tracks ,
        reserved_ids ,args ,verbose =args .verbose ,tid =tid ,event ='',
        )
        if result_tuple [0 ]is None :
            continue 
        gid ,sim_for_log ,_ =result_tuple 

        if gid not in global_to_display :
            global_to_display [gid ]=f"ID{next_display_id}"
            next_display_id +=1 

        finalized_tracks [tid ]={
        'global_id':gid ,
        'display_id':global_to_display [gid ],
        'similarity':sim_for_log ,
        'id_confidence':state .get ('last_id_confidence'),
        'male_hist':list (state .get ('male_hist',[])),
        'age_hist':list (state .get ('age_hist',[])),
        }

        
        state ['global_id']=gid 
        state ['display_id']=global_to_display [gid ]
        state ['last_similarity']=sim_for_log 
        state ['finalized']=True 

        # 
    unique_gids =set ()
    for ft_info in finalized_tracks .values ():
        if ft_info .get ('global_id'):
            unique_gids .add (ft_info ['global_id'])
    print (f"[INFO] Unique IDs: {len(unique_gids)}, finalized tracks: {len(finalized_tracks)}")

    if not save_vis :
        if getattr (args ,'verbose',False ):
            print ("[WARN] save_vis=0: visualization video will not be written.")
        return 

        # ==========  ==========
    print ("[INFO] Starting visualization rendering pass...")

    cap2 =cv2 .VideoCapture (args .input )
    if not cap2 .isOpened ():
    
        raise RuntimeError (f"Cannot open input video: {args.input}")

        
    h_out =int (cap2 .get (cv2 .CAP_PROP_FRAME_HEIGHT ))
    w_out =int (cap2 .get (cv2 .CAP_PROP_FRAME_WIDTH ))

    
    
    import subprocess 

    
    ffmpeg_available =False 
    try :
        result =subprocess .run (['ffmpeg','-version'],capture_output =True ,timeout =5 )
        ffmpeg_available =(result .returncode ==0 )
    except Exception :
        pass 

    use_ffmpeg_pipe =False 
    ffmpeg_process =None 
    writer =None 

    if ffmpeg_available :
    #  FFmpeg 
        use_ffmpeg_pipe =True 
        ffmpeg_cmd =[
        'ffmpeg','-y',# 
        '-f','rawvideo',
        '-vcodec','rawvideo',
        '-pix_fmt','bgr24',
        '-s',f'{w_out}x{h_out}',
        '-r',str (write_fps ),# 
        '-i','-',
        '-c:v','libx264',#  H.264 
        '-preset','fast',
        '-crf','18',
        '-pix_fmt','yuv420p',# 
        args .output 
        ]
        print (f"[INFO] FFmpeg output settings: {w_out}x{h_out}, FPS={write_fps}")
        try :
            ffmpeg_process =subprocess .Popen (
            ffmpeg_cmd ,
            stdin =subprocess .PIPE ,
            stderr =subprocess .PIPE ,# 
            bufsize =10 **8 # 
            )
        except Exception as e :
            print (f"[WARN] FFmpeg initialization failed: {e}. Fallback to OpenCV VideoWriter")
            use_ffmpeg_pipe =False 
            ffmpeg_process =None 

    if not use_ffmpeg_pipe :
    
        print (f"[WARN] OpenCV VideoWriter initialization failed for output video.")
        out_ext =os .path .splitext (args .output )[1 ].lower ()
        if out_ext =='.avi':
            codec_list =['MJPG','XVID']
        else :
            codec_list =['mp4v','avc1','XVID','MJPG']

        for cc in codec_list :
            fourcc =cv2 .VideoWriter_fourcc (*cc )
            tmp =cv2 .VideoWriter (args .output ,fourcc ,write_fps ,(w_out ,h_out ))
            if tmp .isOpened ():
                writer =tmp 
                print (f"[INFO] VideoWriter initialized: codec={cc}, FPS={write_fps}, size={w_out}x{h_out}")
                break 
        if writer is None :
            raise RuntimeError (f": {args.output}")

    frames_written =0 
    frame_idx =0 

    while True :
        ret ,frame_bgr =cap2 .read ()
        if not ret :
            break 

            
        if frame_idx <len (frame_detections ):
            xyxy ,ids =frame_detections [frame_idx ]
        else :
            xyxy ,ids =None ,None 

            
        if xyxy is not None and ids is not None :
            for i ,(x1 ,y1 ,x2 ,y2 )in enumerate (xyxy ):
                tid =int (ids [i ])if i <len (ids )else -1 
                if tid <0 :
                    continue 

                    
                ft_info =finalized_tracks .get (tid )
                state =track_states .get (tid )
                
                if (ft_info and ft_info .get ('skipped'))or (state and state .get ('skipped')):
                    continue 

                    
                color =_VIS_BOX_BGR 
                cv2 .rectangle (frame_bgr ,(x1 ,y1 ),(x2 ,y2 ),color ,2 )

                # 
                disp =[]

                # ID
                if ft_info and ft_info .get ('display_id'):
                    disp .append (ft_info ['display_id'])
                elif state and state .get ('display_id'):
                    disp .append (state ['display_id'])
                else :
                    disp .append (f"ID{tid}")

                    
                gid =None 
                if ft_info :
                    gid =ft_info .get ('global_id')
                elif state :
                    gid =state .get ('global_id')

                if gid and gid in global_gender_stats :
                    gstat =global_gender_stats [gid ]
                    total =gstat ['alpha_male']+gstat ['alpha_female']
                    male_mean =float (gstat ['alpha_male']/max (1e-6 ,total ))
                    gender ='M'if male_mean >=args .gender_threshold else 'F'
                    gender_conf =_gender_confidence_01 (male_mean ,stable_gender =gender ,threshold =args .gender_threshold )
                    disp .append (f"{gender}_conf:{gender_conf :.2f}")
                elif state and len (state .get ('male_hist',[]))>0 :
                    mp =float (np .mean (state ['male_hist']))
                    gender ='M'if mp >=args .gender_threshold else 'F'
                    gender_conf =_gender_confidence_01 (mp ,stable_gender =gender ,threshold =args .gender_threshold )
                    disp .append (f"{gender}_conf:{gender_conf :.2f}")

                    # 
                age_txt =None 
                if gid and gid in global_age_stats and len (global_age_stats [gid ])>0 :
                    ages =global_age_stats [gid ]
                    age_txt =float (np .median (ages ))
                elif state and len (state .get ('age_hist',[]))>0 :
                    ages =state ['age_hist']
                    age_txt =float (np .median (ages ))

                if age_txt is not None :
                    disp .append (f"Age:{age_txt:.0f}")

                    
                sim_val =None 
                if ft_info and ft_info .get ('similarity')is not None :
                    sim_val =ft_info ['similarity']
                elif state and state .get ('last_similarity')is not None :
                    sim_val =state ['last_similarity']

                id_conf =None 
                if ft_info and ft_info .get ('id_confidence')is not None :
                    id_conf =float (ft_info ['id_confidence'])
                elif state and state .get ('last_id_confidence')is not None :
                    id_conf =float (state ['last_id_confidence'])
                elif sim_val is not None :
                    id_conf =_identity_confidence_01 (sim_val ,adaptive_th =getattr (args ,'base_threshold',None ),is_new =False )

                if id_conf is not None :
                    disp .append (f"id_conf:{id_conf :.2f}")

                txt =' | '.join (disp )

                
                font ,font_scale ,thickness ,pad_txt =_get_adaptive_text_style (
                frame_bgr ,base_scale =0.82 ,min_scale =0.62 ,max_scale =2.7 
                )
                (_tw ,th ),baseline =cv2 .getTextSize (txt ,font ,font_scale ,thickness )
                x_txt =x1 
                y_txt =max (th +baseline +pad_txt ,y1 -pad_txt )
                _draw_text_block (
                frame_bgr ,
                txt ,
                x_txt ,
                y_txt ,
                font ,
                font_scale ,
                thickness ,
                pad_txt ,
                bg_color =_VIS_LABEL_BG_BGR ,
                text_color =_VIS_TEXT_BGR ,
                border_color =color ,
                )

                
        if use_ffmpeg_pipe and ffmpeg_process is not None :
        #  FFmpeg 
            try :
                ffmpeg_process .stdin .write (frame_bgr .tobytes ())
                frames_written +=1 
            except Exception as e :
                print (f"  {frame_idx}  {e}")
        else :
        #  OpenCV VideoWriter
            write_ok =writer .write (frame_bgr )
            if not write_ok :
                print (f"  {frame_idx} ")
            frames_written +=1 
        frame_idx +=1 

    cap2 .release ()

    
    if use_ffmpeg_pipe and ffmpeg_process is not None :
    #  FFmpeg 
        try :
            ffmpeg_process .stdin .close ()
            ffmpeg_process .wait (timeout =120 )#  FFmpeg 
            if ffmpeg_process .returncode ==0 :
                print (f"FFmpeg ")
            else :
                stderr_output =ffmpeg_process .stderr .read ().decode ('utf-8',errors ='ignore')if ffmpeg_process .stderr else ''
                print (f"[WARN] FFmpeg returned non-zero code: {ffmpeg_process.returncode}")
                if stderr_output :
                    print (f"[WARN] FFmpeg stderr tail: {stderr_output[-500:]}")
        except Exception as e :
            print (f"  FFmpeg  {e}")
    elif writer is not None :
        writer .release ()

        
    if os .path .exists (args .output ):
        try :
            verify_cap =cv2 .VideoCapture (args .output )
            if verify_cap .isOpened ():
                actual_frame_count =int (verify_cap .get (cv2 .CAP_PROP_FRAME_COUNT ))
                actual_fps =verify_cap .get (cv2 .CAP_PROP_FPS )
                actual_duration =actual_frame_count /actual_fps if actual_fps >0 else 0 
                verify_cap .release ()
                print (f"[INFO] Output video stats: frames={actual_frame_count}, fps={actual_fps:.2f}, duration={actual_duration:.1f}s")
                if actual_frame_count !=frames_written :
                    print (f"[WARN] Written frames ({frames_written}) != output video frames ({actual_frame_count})")
                    print ("   This may happen due to encoder buffering or dropped frames.")
        except Exception as e :
            print (f"[WARN] Failed to verify output video: {e}")

            #  frames_written 
            
    expected_duration =frames_written /write_fps if write_fps else 0 
    if 'expected_total_frames'in locals ()and isinstance (expected_total_frames ,int )and expected_total_frames >0 :
        if expected_total_frames !=frames_written :
            print (f"[WARN] Expected total frames {expected_total_frames}, but wrote {frames_written}.")
            print ("   Check OpenCV backend/codec consistency if mismatch is large.")
    print (f"[INFO] Final output: frames={frames_written}, fps={write_fps}, duration={expected_duration:.1f}s")
    print (f" {args.output}")
    print (f"[INFO] Final unique IDs: {len(unique_gids)}, finalized tracks: {len(finalized_tracks)}")

if __name__ =='__main__':
    run ()










