import os
import sys
import numpy as np
from datetime import datetime
import argparse
from config import get_flags

FLAGS = get_flags(flag_train=True)
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler
from utils.dist import init_distributed, is_distributed, is_primary, get_rank, barrier
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import wandb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pytorch_utils import BNMomentumScheduler
from ap_helper import APCalculator, parse_predictions, parse_groundtruths

from torch.nn.parallel import DistributedDataParallel as DDP


import torch.nn.functional as F  # 引用包报错添加




# GLOBAL CONFIG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
FLAGS.learning_rate = FLAGS.learning_rate * torch.cuda.device_count()
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert (len(LR_DECAY_STEPS) == len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
FLAGS.DUMP_DIR = DUMP_DIR
# Setting tower weights

# 解析 tower 权重
student_weights = [float(x) for x in FLAGS.student_tower_weights.split(',')]
teacher_weights = [float(x) for x in FLAGS.teacher_tower_weights.split(',')]

STUDENT_TOWER_WEIGHTS = {
    'img_only_weight': student_weights[0],
    'pc_only_weight': student_weights[1],
    'pc_img_weight': student_weights[2]
}
TEACHER_TOWER_WEIGHTS = {
    'img_only_weight': teacher_weights[0],
    'pc_only_weight': teacher_weights[1],
    'pc_img_weight': teacher_weights[2]
}


if FLAGS.use_imvotenet:
    KEY_PREFIX_LIST = ['img_only_', 'pc_only_', 'pc_img_']
    weights = [float(x) for x in FLAGS.tower_weights.split(',')]
    TOWER_WEIGHTS = {'img_only_weight': weights[0], 'pc_only_weight': weights[1], 'pc_img_weight': weights[2]}
    print('Tower weights123', TOWER_WEIGHTS)
else:
    KEY_PREFIX_LIST = ['pc_only_']
    TOWER_WEIGHTS = {'pc_only_weight': 1.0}

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)' % (LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s' % (LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

if FLAGS.dataset == 'sunrgbd':
    # Create Dataset and Dataloader
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset as DetectionVotesDataset
    from model_util_sunrgbd import SunrgbdDatasetConfig

    DATASET_CONFIG = SunrgbdDatasetConfig()
elif FLAGS.dataset == 'scannet':
    # Create Dataset and Dataloader
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import scannetDetectionVotesDataset as DetectionVotesDataset
    from model_util_scannet import scannetDatasetConfig

    DATASET_CONFIG = scannetDatasetConfig()

elif FLAGS.dataset == 'lvis':
    # Create Dataset and Dataloader
    sys.path.append(os.path.join(ROOT_DIR, 'lvis'))
    from lvis_detection_dataset import lvisDetectionVotesDataset as DetectionVotesDataset
    from model_util_lvis import lvisDatasetConfig

    DATASET_CONFIG = lvisDatasetConfig()
else:
    raise ValueError("No dataset specified")


# Init the some of optimzier


def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i, lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Used for AP calculation
CONFIG_DICT = {'remove_empty_box': False, 'use_3d_nms': True,
               'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
               'per_class_proposal': True, 'conf_thresh': 0.05,
               'dataset_config': DATASET_CONFIG,
               'if_inference_stage_box_filter': FLAGS.if_inference_stage_box_filter,
               'inference_stage_box_filter_thr': FLAGS.inference_stage_box_filter_thr
               }

CONFIG_DICT_LIST = [CONFIG_DICT]
DATASET_CONFIG_LIST = [DATASET_CONFIG]


def load_teacher_model(MODEL, TRAIN_DATASET, FLAGS, DATASET_CONFIG, num_input_channel):
    """加载教师模型（固定结构，不依赖FLAGS.tower_weights）"""
    teacher_model = MODEL.ImVoteNet(  # 已修改MODEL.路径问题
        num_class=DATASET_CONFIG.num_class,
        num_heading_bin=DATASET_CONFIG.num_heading_bin,
        num_size_cluster=DATASET_CONFIG.num_size_cluster,
        mean_size_arr=DATASET_CONFIG.mean_size_arr,
        num_proposal=FLAGS.num_target,
        input_feature_dim=num_input_channel,
        vote_factor=FLAGS.vote_factor,
        sampling=FLAGS.cluster_sampling,
        max_imvote_per_pixel=FLAGS.max_imvote_per_pixel,
        image_feature_dim=getattr(TRAIN_DATASET, 'image_feature_dim', 18)  # 安全访问
    ).cuda()  # 学生模型原文：net = net.cuda(local_rank)#
    '''
    checkpoint = torch.load('checkpoint_48.tar', map_location='cuda') #记得手动改写
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model.eval()
    '''
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model


# 当前维度已对齐，暂未使用
def compute_distillation_loss(student_end_points, teacher_end_points, dataset_config):
    print("Student logits :", student_end_points['pc_img_sem_cls_scores'].shape)
    print("Teacher logits :", teacher_end_points['pc_img_sem_cls_scores'].shape)

    # 确保教师和学生logits维度匹配
    student_logits = student_end_points['pc_img_sem_cls_scores']  # [B,N,23]

    # 获取教师logits中对应数据集类别的部分
    eval_class_ids = torch.tensor(
        sorted(list(dataset_config.class2type_eval.keys())),
        device=student_logits.device
    )
    teacher_logits = teacher_end_points['pc_img_sem_cls_scores'].index_select(
        -1, eval_class_ids)  # [B,N,23]

    # 安全计算KL散度
    return F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits.detach(), dim=-1),
        reduction='batchmean'
    )


def train_one_epoch(net, MODEL, criterion, optimizer, bnm_scheduler, TRAIN_DATALOADER,
                    student_tower_weights, teacher_tower_weights,
                    teacher_model=None, ema_model=None, strategy='fused'):
    if is_primary():
        print(">>>>> ema_model received:", type(ema_model))
        if ema_model is not None:
            print(">>>>> ema_model device:", next(ema_model.parameters()).device)

    stat_dict = {}
    net.train()

    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        # ------- 将 batch 数据移到 GPU 上 -------
        for key in batch_data_label:
            if isinstance(batch_data_label[key], list):
                batch_data_label[key] = [item.cuda(non_blocking=True) for item in batch_data_label[key]]
            else:
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        inputs = batch_data_label
        optimizer.zero_grad()

        # ------- 学生模型前向传播 -------
        if teacher_model is not None or ema_model is not None:
            student_end_points, student_features = net(inputs, return_features=True)
        else:
            student_end_points = net(inputs)
            student_features = None

        # ------- 将标签补充到 end_points 中 -------
        for key in batch_data_label:
            if key not in student_end_points:
                student_end_points[key] = batch_data_label[key]

        # ------- 原始任务损失（使用学生权重）-------
        original_loss, student_end_points = criterion(
            student_end_points, DATASET_CONFIG, KEY_PREFIX_LIST, student_tower_weights
        )

        # ======= 蒸馏损失计算 START =======
        distill_loss = torch.tensor(0.0).cuda()
        feat_distill_loss = torch.tensor(0.0).cuda()
        pseudo_label_loss = torch.tensor(0.0).cuda()

        # ------- 教师蒸馏 -------
        if teacher_model is not None:
            with torch.no_grad():
                teacher_end_points, teacher_features = teacher_model(inputs, return_features=True)
                for key in batch_data_label:
                    if key not in teacher_end_points:
                        teacher_end_points[key] = batch_data_label[key]
                # 使用教师模型权重计算 teacher_end_points（用于 soft label 对齐）
                _ = criterion(teacher_end_points, DATASET_CONFIG, KEY_PREFIX_LIST, teacher_tower_weights)

            # 输出蒸馏损失（分类 KL）
            distill_loss = F.kl_div(
                F.log_softmax(student_end_points['pc_img_sem_cls_scores'], dim=-1),
                F.softmax(teacher_end_points['pc_img_sem_cls_scores'], dim=-1),
                reduction='batchmean'
            )

            # 特征蒸馏损失（MSE + Cosine）
            feat_distill_loss += F.mse_loss(
                student_features['backbone_features'],
                teacher_features['backbone_features']
            )
            feat_distill_loss += 1 - F.cosine_similarity(
                student_features['vote_features'],
                teacher_features['vote_features'],
                dim=1
            ).mean()

        # ------- EMA伪标签蒸馏 -------
        if ema_model is not None:
            with torch.no_grad():
                _, ema_features = ema_model(inputs, return_features=True)

            vote_feats = student_features['vote_features']  # [B, C, N]
            B, C, N = vote_feats.shape
            vote_feats = vote_feats.permute(0, 2, 1).reshape(-1, C)
            vote_feats = F.normalize(vote_feats, dim=1)

            # 获取文本特征向量
            text_feats = F.normalize((net.module if isinstance(net, DDP) else net).text_feats_2Dsemantic, dim=1)
            text_feats = text_feats.float()
            logits = torch.matmul(vote_feats, text_feats.T).view(B, N, -1)

            # 获取伪标签
            pseudo_labels_sim = student_features['pseudo_labels']
            pseudo_labels_ema = ema_features['pseudo_labels']
            similarity_logits_ema = ema_features['similarity_logits']

            # 选择伪标签融合策略
            if strategy == 'sim':
                fused_pseudo_labels = pseudo_labels_sim
            elif strategy == 'ema':
                fused_pseudo_labels = pseudo_labels_ema
            elif strategy == 'fused':
                sim_conf = F.softmax(logits, dim=-1).max(dim=-1).values
                ema_conf = F.softmax(similarity_logits_ema, dim=-1).max(dim=-1).values
                fused_pseudo_labels = pseudo_labels_sim.clone()
                mask_use_ema = ema_conf > sim_conf
                fused_pseudo_labels[mask_use_ema] = pseudo_labels_ema[mask_use_ema]

            # 计算伪标签蒸馏损失
            pseudo_label_loss = F.cross_entropy(logits.permute(0, 2, 1), fused_pseudo_labels)

        # ======= 总损失组合 =======
        total_loss = (
                original_loss +
                FLAGS.output_distill_weight * distill_loss +
                FLAGS.feat_distill_weight * feat_distill_loss +
                FLAGS.pseudo_label_weight * pseudo_label_loss
        )

        # 反向传播 + 参数更新
        total_loss.backward()
        optimizer.step()
        bnm_scheduler.step()

        # ------- 日志记录 -------
        stat_dict['total_loss'] = stat_dict.get('total_loss', 0.0) + total_loss.item()
        stat_dict['original_loss'] = stat_dict.get('original_loss', 0.0) + original_loss.item()
        if teacher_model:
            stat_dict['distill_loss'] = stat_dict.get('distill_loss', 0.0) + distill_loss.item()
            stat_dict['feat_distill_loss'] = stat_dict.get('feat_distill_loss', 0.0) + feat_distill_loss.item()
        if ema_model:
            stat_dict['pseudo_label_loss'] = stat_dict.get('pseudo_label_loss', 0.0) + pseudo_label_loss.item()

        # ------- 控制台打印 -------
        if is_primary() and (batch_idx + 1) % 1 == 0:
            log_str = f'[Batch {batch_idx + 1:03d}] total_loss: {total_loss.item():.4f}, original: {original_loss.item():.4f} '
            if teacher_model:
                log_str += f'| out_distill: {distill_loss.item():.4f}, feat_distill: {feat_distill_loss.item():.4f} '
            if ema_model:
                log_str += f'| pseudo_label_loss: {pseudo_label_loss.item():.4f}'
            log_string(log_str)

        # ------- 更新 EMA 模型 -------
        if ema_model is not None:
            update_ema_model(student_model=net.module if isinstance(net, DDP) else net,
                             ema_model=ema_model,
                             alpha=0.999)

    # Save checkpoint logic here:
    barrier()

    # #__________________________________
    # # if is_primary():
    # # Save the model checkpoint
    # save_dict = {
    #     'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': total_loss.item(),
    # }
    # try:
    #     save_dict['model_state_dict'] = net.module.state_dict()
    # except:
    #     save_dict['model_state_dict'] = net.state_dict()
    #
    # print(LOG_DIR)
    # print(save_dict, os.path.join(LOG_DIR, f'checkpoint.tar'))
    #
    # torch.save(save_dict, os.path.join(FLAGS.log_dir, f'checkpoint.tar'))
    #
    # if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9:  # Eval every 10 epochs
    #     if not os.path.exists(LOG_DIR):
    #         os.makedirs(LOG_DIR)
    #     save_path = os.path.join(LOG_DIR, f'checkpoint_{EPOCH_CNT}.tar')
    #     print(f"Saving checkpoint to: {save_path}")
    #     torch.save(save_dict, save_path)
    #
    # for i, mAP in enumerate(mAP_LIST):
    #     if mAP > max_mAP[i]:
    #         max_mAP[i] = mAP
    #         if not os.path.exists(LOG_DIR):
    #             os.makedirs(LOG_DIR)
    #         save_path = os.path.join(LOG_DIR, f'checkpoint_best_mAP_dataset_in_{FLAGS.dataset}.tar')
    #         print(f"Saving checkpoint to: {save_path}")
    #         torch.save(save_dict, save_path)
    #
    # return stat_dict


def evaluate_one_epoch(net, MODEL, criterion, optimizer, TRAIN_DATALOADER, TEST_DATALOADER, epoch):
    mAP_LIST = []
    if FLAGS.use_imvotenet:
        KEY_PREFIX_LST = KEY_PREFIX_LIST[2:]
    else:
        KEY_PREFIX_LST = KEY_PREFIX_LIST

    for DATASET_idx, DATASET_ITEM in enumerate([TEST_DATALOADER]):
        print(FLAGS.dataset, DATASET_idx, DATASET_CONFIG_LIST[DATASET_idx].class2type_eval)

        stat_dict = {}  # collect statistics
        ap_calculator_dict = {}
        for key_prefix in KEY_PREFIX_LST:
            ap_calculator_dict[key_prefix + 'ap_calculator'] = APCalculator(ap_iou_thresh=FLAGS.ap_iou_thresh,
                                                                            class2type_map=DATASET_CONFIG_LIST[
                                                                                DATASET_idx].class2type_eval)
        device = next(net.parameters()).device
        net.eval()  # set model to eval mode (for bn and dp)
        barrier()
        for batch_idx, batch_data_label in enumerate(DATASET_ITEM):
            if batch_idx % 10 == 0:
                print('Eval batch: %d' % (batch_idx))
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(device)

            # Forward pass
            inputs = {'point_clouds': batch_data_label['point_clouds']}
            if FLAGS.use_imvotenet:
                inputs.update({'scale': batch_data_label['scale'],
                               'calib_K': batch_data_label['calib_K'],
                               'calib_Rtilt': batch_data_label['calib_Rtilt'],
                               'cls_score_feats': batch_data_label['cls_score_feats'],
                               'full_img_votes_1d': batch_data_label['full_img_votes_1d'],
                               'full_img_1d': batch_data_label['full_img_1d'],
                               'full_img_width': batch_data_label['full_img_width'],
                               })
            with torch.no_grad():
                if FLAGS.use_imvotenet:
                    end_points = net(inputs, joint_only=True)
                else:
                    end_points = net(inputs)

            # Compute loss
            for key in batch_data_label:
                if key not in end_points:
                    end_points[key] = batch_data_label[key]

            loss, end_points = criterion(end_points, DATASET_CONFIG_LIST[DATASET_idx], KEY_PREFIX_LST, TOWER_WEIGHTS)

            # Accumulate statistics and print out
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'ratio' in key:
                    if key not in stat_dict: stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()

            for key_prefix in KEY_PREFIX_LST:
                batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT_LIST[DATASET_idx], key_prefix)
                batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT_LIST[DATASET_idx])
                ap_calculator_dict[key_prefix + 'ap_calculator'].step(batch_pred_map_cls, batch_gt_map_cls)

            barrier()
        if is_primary():
            for key in sorted(stat_dict.keys()):
                log_string(f'{FLAGS.dataset}' + '_eval mean %s: %f' % (key, stat_dict[key] / (float(batch_idx + 1))))

            # Evaluate average precision
            for key_prefix in KEY_PREFIX_LST:
                metrics_dict = ap_calculator_dict[key_prefix + 'ap_calculator'].compute_metrics()
                for key in metrics_dict:
                    log_string(f'{FLAGS.dataset}' + '_eval %s: %f' % (key, metrics_dict[key]))

                    if key != 'mAP':
                        if FLAGS.if_wandb:
                            wandb.log({f"dataset_{FLAGS.dataset}/{key}": metrics_dict[key]}, step=epoch)
                    if key == 'mAP':
                        if FLAGS.if_wandb:
                            wandb.log({f"test/mAP_dataset_{FLAGS.dataset}": metrics_dict[key]}, step=epoch)
                        mAP_LIST.append(metrics_dict[key])

        mean_loss = stat_dict['loss'] / float(batch_idx + 1)
    return mean_loss, mAP_LIST


def train_or_evaluate(start_epoch, net, MODEL, net_no_ddp, criterion, optimizer, bnm_scheduler, train_sampler,
                      TRAIN_DATALOADER, TEST_DATALOADER, ema_model, teacher_model=None, strategy='fused', FLAGS=None):
    global EPOCH_CNT
    loss = 0
    max_mAP = [0.0]  # Initialize max_mAP to a small value

    # ----------- 新增：解析 tower 权重 -----------
    student_weights = [float(x) for x in FLAGS.student_tower_weights.split(',')]
    teacher_weights = [float(x) for x in FLAGS.teacher_tower_weights.split(',')]
    student_tower_weights = {
        'img_only_weight': student_weights[0],
        'pc_only_weight': student_weights[1],
        'pc_img_weight': student_weights[2]
    }
    teacher_tower_weights = {
        'img_only_weight': teacher_weights[0],
        'pc_only_weight': teacher_weights[1],
        'pc_img_weight': teacher_weights[2]
    }

    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        if is_distributed():
            train_sampler.set_epoch(EPOCH_CNT)
        if is_primary():
            log_string('**** EPOCH %03d ****' % (epoch))
            log_string('Current learning rate: %f' % (get_current_lr(epoch)))
            log_string('Current BN decay momentum: %f' % (bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
            log_string(str(datetime.now()))
            log_string(str(ema_model is None))

        # Reset numpy seed.
        np.random.seed()

        # ----------- 调用 train_one_epoch 并传入 tower 权重 -----------
        stat_dict_loss = train_one_epoch(
            net, MODEL, criterion, optimizer, bnm_scheduler, TRAIN_DATALOADER,
            student_tower_weights, teacher_tower_weights,
            teacher_model=teacher_model,
            ema_model=ema_model,
            strategy=strategy
        )

        if is_primary() and FLAGS.if_wandb:
            for key_prefix in KEY_PREFIX_LIST:
                if key_prefix in stat_dict_loss:
                    average = sum(stat_dict_loss[key_prefix]) / len(stat_dict_loss[key_prefix])
                    wandb.log({f"train/{key_prefix}_loss": average}, step=epoch)


        # Save checkpoint
        save_dict = {
            'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        try:
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()

        print(LOG_DIR)
        print(save_dict, os.path.join(LOG_DIR, f'checkpoint.tar'))
        if is_primary():
            torch.save(save_dict, os.path.join(LOG_DIR, f'checkpoint.tar'))
            if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9:  # Eval every 10 epochs
                if not os.path.exists(LOG_DIR):
                    os.makedirs(LOG_DIR)
                save_path = os.path.join(LOG_DIR, f'checkpoint.tar')
                print(f"Saving checkpoint to: {save_path}")
                torch.save(save_dict, os.path.join(LOG_DIR, f'checkpoint_{EPOCH_CNT}.tar'))
            for i, mAP in enumerate(mAP_LIST):
                if mAP > max_mAP[i]:
                    max_mAP[i] = mAP
                    if not os.path.exists(LOG_DIR):
                        os.makedirs(LOG_DIR)
                    save_path = os.path.join(LOG_DIR, f'checkpoint.tar')
                    print(f"Saving checkpoint to: {save_path}")
                    torch.save(save_dict, os.path.join(LOG_DIR, f'checkpoint_best_mAP_dataset_in_{FLAGS.dataset}.tar'))



# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def setup_student_network(local_rank, FLAGS, DATASET_CONFIG, TRAIN_DATASET):
    # 获取学生网络的tower_weights
    student_tower_weights = list(map(float, FLAGS.student_tower_weights.split(',')))

    num_input_channel = int(FLAGS.use_color) * 3 + int(not FLAGS.no_height) * 1

    if FLAGS.use_imvotenet:
        MODEL = importlib.import_module('imvotenet')
        net = MODEL.ImVoteNet(
            num_class=DATASET_CONFIG.num_class,
            num_heading_bin=DATASET_CONFIG.num_heading_bin,
            num_size_cluster=DATASET_CONFIG.num_size_cluster,
            mean_size_arr=DATASET_CONFIG.mean_size_arr,
            num_proposal=FLAGS.num_target,
            input_feature_dim=num_input_channel,
            vote_factor=FLAGS.vote_factor,
            sampling=FLAGS.cluster_sampling,
            max_imvote_per_pixel=FLAGS.max_imvote_per_pixel,
            image_feature_dim=TRAIN_DATASET.image_feature_dim,

        )
    else:
        MODEL = importlib.import_module('votenet')
        net = MODEL.VoteNet(
            num_class=DATASET_CONFIG.num_class,
            num_heading_bin=DATASET_CONFIG.num_heading_bin,
            num_size_cluster=DATASET_CONFIG.num_size_cluster,
            mean_size_arr=DATASET_CONFIG.mean_size_arr,
            num_proposal=FLAGS.num_target,
            input_feature_dim=num_input_channel,
            vote_factor=FLAGS.vote_factor,
            sampling=FLAGS.cluster_sampling,

        )

    # 冻结与CLIP相关的参数
    for name, param in net.named_parameters():
        if "model_clip" in name or "model_clip_2Dsemantic" in name:
            param.requires_grad = False

    # 将网络移动到GPU
    net = net.cuda(local_rank)

    return net


def setup_teacher_network(MODEL, TRAIN_DATASET, FLAGS, DATASET_CONFIG, num_input_channel):
    # 获取教师网络的tower_weights
    teacher_tower_weights = list(map(float, FLAGS.teacher_tower_weights.split(',')))

    # 初始化教师模型（如果是主节点）
    teacher_model = None
    if is_primary():
        teacher_model = load_teacher_model(MODEL, TRAIN_DATASET, FLAGS, DATASET_CONFIG, num_input_channel)
        # 将教师的tower_weights传递给模型
        teacher_model.set_tower_weights(teacher_tower_weights)

    return teacher_model


def setup_ema_model(net_no_ddp):
    # 初始化EMA模型
    ema_model = init_ema_model(net_no_ddp)
    return ema_model



def main_dist(local_rank, FLAGS):
    start_epoch = 0
    init_distributed(
        local_rank,
        global_rank=local_rank,
        world_size=FLAGS.ngpus,
        dist_url=FLAGS.dist_url,
        dist_backend="nccl",
    )

    if is_primary():
        print(f"Called with args: {FLAGS}")

    if FLAGS.if_wandb and is_primary():
        wandb.init(project="ImOV3D", config=vars(FLAGS))

    torch.cuda.set_device(local_rank)
    np.random.seed(FLAGS.seed + get_rank())
    torch.manual_seed(FLAGS.seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FLAGS.seed + get_rank())

    # 加载训练和测试数据集
    TRAIN_DATASET = DetectionVotesDataset('train', num_points=NUM_POINT, augment=True,
                                          use_color=FLAGS.use_color,
                                          use_height=(not FLAGS.no_height),
                                          use_imvote=FLAGS.use_imvotenet,
                                          max_imvote_per_pixel=FLAGS.max_imvote_per_pixel)

    TEST_DATASET = DetectionVotesDataset('val', num_points=NUM_POINT, augment=False,
                                         use_color=FLAGS.use_color,
                                         use_height=(not FLAGS.no_height),
                                         use_imvote=FLAGS.use_imvotenet,
                                         max_imvote_per_pixel=FLAGS.max_imvote_per_pixel)

    # 设置学生网络（正在训练的网络）
    net = setup_student_network(local_rank, FLAGS, DATASET_CONFIG, TRAIN_DATASET)

    # 设置教师模型（如果是主节点）
    teacher_model = setup_teacher_network(importlib.import_module('imvotenet'), TRAIN_DATASET, FLAGS, DATASET_CONFIG, int(FLAGS.use_color) * 3 + int(not FLAGS.no_height) * 1)


    # 设置EMA模型
    ema_model = setup_ema_model(net)

    # 如果是分布式训练，使用DistributedDataParallel
    if is_distributed():
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

        # 同步教师和EMA模型的BN（尽管他们不训练，但一些任务可能需要）
        if teacher_model is not None:
            teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
        if ema_model is not None:
            ema_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ema_model)

    # 损失函数
    criterion = importlib.import_module('imvotenet').get_loss
    train_sampler = DistributedSampler(TRAIN_DATASET)

    # 数据加载器
    TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
                                  sampler=train_sampler, num_workers=FLAGS.num_workers,
                                  worker_init_fn=my_worker_init_fn)

    TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
                                 num_workers=FLAGS.num_workers, worker_init_fn=my_worker_init_fn)

    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

    # 恢复模型
    if FLAGS.resume and CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        log_string(f"-> loaded checkpoint {CHECKPOINT_PATH} (epoch: {start_epoch})")
        torch.cuda.empty_cache()

    # 初始化BatchNorm Momentum调度器
    BN_MOMENTUM_INIT = 0.5
    BN_MOMENTUM_MAX = 0.001
    bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE ** (int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
    bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch - 1)


    if hasattr(FLAGS, 'student_tower_weights'):
        print(f"📚 [学生模型] tower_weights: {FLAGS.student_tower_weights}")

    # 如果 FLAGS 中定义了 teacher_tower_weights 且 teacher_model 不为 None，则打印教师模型的 tower_weights
    if teacher_model is not None and hasattr(FLAGS, 'teacher_tower_weights'):
        print(f"🎓 [教师模型] tower_weights: {FLAGS.teacher_tower_weights}")

    train_or_evaluate(
        start_epoch,
        net,
        importlib.import_module('imvotenet'),
        net,
        criterion,
        optimizer,
        bnm_scheduler,
        train_sampler,
        TRAIN_DATALOADER,
        TEST_DATALOADER,
        ema_model=ema_model,
        teacher_model=teacher_model,
        strategy='fused',  # 'fused' 结合教师模型和EMA蒸馏
        FLAGS=FLAGS  # 传递 FLAGS 参数
    )


def launch_distributed(FLAGS):
    if torch.cuda.device_count() > 1:
        log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
    FLAGS.ngpus = torch.cuda.device_count()
    world_size = FLAGS.ngpus
    if world_size == 1:
        main_dist(local_rank=0, FLAGS=FLAGS)
    else:
        torch.multiprocessing.spawn(main_dist, nprocs=world_size, args=(FLAGS,))


def init_ema_model(student_model, decay=0.999):
    import copy
    ema_model = copy.deepcopy(student_model)
    for param in ema_model.parameters():
        param.requires_grad = False
    ema_model.eval()
    return ema_model


@torch.no_grad()
def update_ema_model(student_model, ema_model, alpha=0.999):
    for ema_param, student_param in zip(ema_model.parameters(), student_model.parameters()):
        ema_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)


if __name__ == "__main__":

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    launch_distributed(FLAGS)
