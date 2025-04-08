import argparse

def get_flags_train():
    parser = argparse.ArgumentParser()
    # ImVoteNet related options
    parser.add_argument('--use_imvotenet', action='store_true', help='Use ImVoteNet (instead of VoteNet) with RGB.')
    parser.add_argument('--max_imvote_per_pixel', type=int, default=3, help='Maximum number of image votes per pixel [default: 3]')
    parser.add_argument('--tower_weights', default='0.3,0.3,0.4', help='Tower weights for img_only, pc_only and pc_img [default: 0.3,0.3,0.4]')
    # Shared options with VoteNet
    parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
    parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
    parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
    parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
    parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
    parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
    parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
    parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training [default: 8]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
    parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
    parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
    parser.add_argument('--lr_decay_steps', default='80,120,160', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
    parser.add_argument('--max_epoch', type=int, default=180, help='Epoch to run [default: 180]')
    parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
    parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
    parser.add_argument('--dataset', default='sunrgbd', help='choose dataset(sunrgbd,scannet,lvis)[default: sunrgbd]')
    
    parser.add_argument('--if_inference_stage_box_filter', default=True, help='box filter during inference stage')
    parser.add_argument('--inference_stage_box_filter_thr', type=float, default=0.05, help='box filter during inference stage')
    
    parser.add_argument('--dump_results', action='store_true', help='Dump results.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of works for loading training data [default: 4]')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--dist_url", default="tcp://localhost:12955", type=str)
    parser.add_argument('--if_wandb', action='store_true', help='Dump results.')

    parser.add_argument('--output_distill_weight', type=float, default=1.0,
                       help='Weight for output distillation loss')
    parser.add_argument('--feat_distill_weight', type=float, default=0.5,
                       help='Weight for feature distillation loss')
    parser.add_argument('--pseudo_label_weight', type=float, default=0.5,
                       help='Weight for pseudo_label distillation loss')

    # 支持 'sim'（vote-to-text），'ema'（EMA伪标签），或 'fused'（置信度融合）
    parser.add_argument('--ema_pseudo_source', type=str, default='fused',
                        choices=['sim', 'ema', 'fused'], help='Pseudo label source: sim | ema | fused')

    parser.add_argument('--student_tower_weights', default='0.0,1.0,0.0',
                        help='Tower weights for img_only, pc_only, and pc_img in student network [default: 0.3,0.3,0.4]')
    parser.add_argument('--teacher_tower_weights', default='0.3,0.3,0.4',
                        help='Tower weights for img_only, pc_only, and pc_img in teacher network [default: 0.4,0.3,0.3]')

    FLAGS = parser.parse_args()
    return FLAGS

def get_flags_eval():
    parser = argparse.ArgumentParser()
    # ImVoteNet related options
    parser.add_argument('--use_imvotenet', action='store_true', help='Use ImVoteNet (instead of VoteNet) with RGB.')
    parser.add_argument('--max_imvote_per_pixel', type=int, default=3, help='Maximum number of image votes per pixel [default: 3]')
    # Shared options with VoteNet
    parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
    parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
    parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
    parser.add_argument('--num_target', type=int, default=256, help='Point Number [default: 256]')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
    parser.add_argument('--vote_factor', type=int, default=1, help='Number of votes generated from each seed [default: 1]')
    parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
    parser.add_argument('--ap_iou_thresholds', default='0.25,0.5', help='A list of AP IoU thresholds [default: 0.25,0.5]')
    parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_3d_nms', action='store_true', help='Use 3D NMS instead of 2D NMS.')
    parser.add_argument('--use_cls_nms', action='store_true', help='Use per class NMS.')
    parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
    parser.add_argument('--per_class_proposal', action='store_true', help='Duplicate each proposal num_class times.')
    parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
    parser.add_argument('--conf_thresh', type=float, default=0.05, help='Filter out predictions with obj prob less than it. [default: 0.05]')
    parser.add_argument('--dataset', default='sunrgbd', help='choose dataset(sunrgbd,scannet,lvis)[default: sunrgbd]')
    
    parser.add_argument('--if_inference_stage_box_filter', default=True, help='box filter during inference stage')
    parser.add_argument('--inference_stage_box_filter_thr', type=float, default=0.05, help='box filter during inference stage')
    
    parser.add_argument('--faster_eval', action='store_true', help='Faster evaluation by skippling empty bounding box removal.')
    parser.add_argument('--shuffle_dataset', action='store_true', help='Shuffle the dataset (random order).')

    FLAGS = parser.parse_args()
    return FLAGS

global_flag = None

def get_flags(flag_train):
    global global_flag 
    global_flag = flag_train
    if global_flag==True:
        FLAGS = get_flags_train()
    elif global_flag==False: 
        FLAGS = get_flags_eval()
    else:
        print("Undefined behavior")
        exit()
    return FLAGS
