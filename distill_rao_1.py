import os
import sys
import numpy as np
from datetime import datetime
import argparse
from config import get_flags
#

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
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pytorch_utils import BNMomentumScheduler
from ap_helper import APCalculator, parse_predictions, parse_groundtruths

from torch.nn.parallel import DistributedDataParallel as DDP

import torch.nn.functional as F  # 引用包报错添加


import wandb

wandb.login(key="1729dc06a1cea72bca2d6044ead8af11a2555e1a")

counter = 0

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

def custom_pseudo_label_loss(vote_feats, fused_text_feats, tau=0.07):
    """
    vote_feats: [B, C, N] - 每个样本 N 个 vote 的 3D 特征
    fused_text_feats: [B, N] - 每个样本的伪标签 text 特征（不是 C 维度）
    """
    B, C, N = vote_feats.shape  # [B, C, N]

    # L2 normalize vote features and text features

    vote_feats = F.normalize(vote_feats, dim=1)  # [B, C, N]
    text_feats = F.normalize(fused_text_feats.float(), dim=1)  # [B, N]

    # 把 text_feats reshape 成 [B, 1, N]，方便和 vote_feats 对齐计算
    text_feats = text_feats.unsqueeze(1)  # [B, 1, N]

    # 计算每个 vote 与 text_feat 的点积：[B, C, N] * [B, 1, N] -> [B, N]
    sim_logits = torch.sum(vote_feats * text_feats, dim=1)  # [B, N]

    # 每个样本中对 vote 做 softmax，分母是所有 vote
    sim_logits = sim_logits / tau
    log_probs = sim_logits - torch.logsumexp(sim_logits, dim=1, keepdim=True)  # log softmax

    # 分子是正确的 vote 与 text 的匹配 logit，对角线（正样本）
    loss = -log_probs.mean()  # 所有样本 N 个 vote 的平均
    return loss


def RelationDistillationLoss(student_features, teacher_features, lambda_relation=1.0):
    """
    计算关系特征蒸馏损失 (Relation Distillation Loss).

    输入:
        student_features: 学生模型的特征字典，包含关系特征
        teacher_features: 教师模型的特征字典，包含关系特征
        lambda_relation: 控制关系蒸馏损失的权重系数

    输出:
        relation_loss: 关系特征蒸馏损失
    """

    # 从学生和教师模型中提取关系特征
    student_relation = student_features['fp2_relation']  # 这里选择 'fp2_relation'，可以根据需要修改
    teacher_relation = teacher_features['fp2_relation']  # 同上，使用教师模型的关系特征

    # 计算关系特征蒸馏损失：采用L2范数（欧式距离）作为蒸馏损失
    relation_loss = F.mse_loss(student_relation, teacher_relation)

    # 返回损失乘以一个系数
    return lambda_relation * relation_loss

def kl_div_3d(tensor1, tensor2, epsilon=1e-8):
    """
    计算两个[8, 256, 1024]张量间的KL散度
    返回: [8, 256] 的KL散度矩阵
    """
    # 归一化为概率分布（沿最后一个维度）
    p = F.softmax(tensor1 + epsilon, dim=2)
    q = F.softmax(tensor2 + epsilon, dim=2)
    
    # 计算KL散度并返回标量
    return F.kl_div(
        input=p.log(),
        target=q,
        reduction='batchmean'  # 关键参数！自动计算批平均标量
    )


def train_one_epoch(net, MODEL, criterion, optimizer, bnm_scheduler, TRAIN_DATALOADER,
                    student_tower_weights, teacher_tower_weights,
                    teacher_model=None, ema_model=None, strategy='fused', adapter_net = None ,epoch=None):
    global counter  # 添加这一行
    
    stat_dict = {}
    net.train()
    adapter_net.train()
    

    # 使用 tqdm 显示进度
    progress = tqdm(total=len(TRAIN_DATALOADER),
                    desc=f"Epoch {epoch}" if epoch is not None else "Training",
                    disable=not is_primary())  # 仅主进程显示

    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        # ------- 将 batch 数据移到 GPU 上 -------
        # if batch_idx >= 5:  # 🔧 只跑前5个 batch 用于 debug
        #   break
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
        adapter_loss = torch.tensor(0.0).cuda()

        
        # ------- adapter -------
        adapter_end_points = adapter_net(student_features['img_features'])
        adapter_loss = kl_div_3d(adapter_end_points, student_features['pc_features'])
        #print(adapter_loss)
        #print(student_features['pc_features'])
        #print(adapter_end_points)
        # ------- 教师蒸馏 -------
        if teacher_model is not None:
            with torch.no_grad():
                teacher_end_points, teacher_features = teacher_model(inputs, return_features=True)
                for key in batch_data_label:
                    if key not in teacher_end_points:
                        teacher_end_points[key] = batch_data_label[key]
                # 使用教师模型权重计算 teacher_end_points（用于 soft label 对齐）
                _ = criterion(teacher_end_points, DATASET_CONFIG, KEY_PREFIX_LIST, teacher_tower_weights)

            # 输出蒸馏损失（分类 KL） 等待增强
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

        vote_feats = student_features['vote_features']  # [B, C, N]
        B, C, N = vote_feats.shape
        # 将 vote_feats 变成 [B, C * N]
        # vote_feats = vote_feats.permute(0, 2, 1).reshape(B, -1)  # [B, C * N]
        # vote_feats = F.normalize(vote_feats, dim=1)

        # # 获取文本特征向量
        # text_feats = F.normalize((net.module if isinstance(net, DDP) else net).text_feats_2Dsemantic, dim=1)
        # text_feats = text_feats.float()
        # logits = torch.matmul(vote_feats, text_feats.T).view(B, N, -1)

        # 获取伪标签
        pseudo_labels_sim = student_features['sim_pseudo_labels']  # sim的伪标签
        pseudo_labels_ema = student_features['ema_pseudo_labels']  # EMA 模型的伪标签
        similarity_logits = student_features['similarity_logits']
        # print("similarity_logits", similarity_logits.shape)
        ema_similarity_logits = student_features['ema_similarity_logits']  # EMA 模型的相似度 logits

        fused_pseudo_labels=None

        # 选择伪标签融合策略
        if strategy == 'sim':
            fused_pseudo_labels = pseudo_labels_sim
        elif strategy == 'ema':
            fused_pseudo_labels = pseudo_labels_ema
        elif strategy == 'fused':
            fused_pseudo_labels = pseudo_labels_sim
            sim_conf = F.softmax(similarity_logits, dim=-1).max(dim=-1).values
            ema_conf = F.softmax(ema_similarity_logits, dim=-1).max(dim=-1).values
            fused_pseudo_labels = pseudo_labels_sim.clone()
            if ema_conf.max() > sim_conf.max():
                fused_pseudo_labels = pseudo_labels_ema


        # 计算伪标签蒸馏损失
        # pseudo_label_loss = F.cross_entropy(vote_feats, fused_pseudo_labels)

        pseudo_label_loss = custom_pseudo_label_loss(vote_feats, fused_pseudo_labels)
        # 直接转化为数值（float），不再是张量
        pseudo_label_loss = pseudo_label_loss.item() if pseudo_label_loss is not None else 0.0
        # print("pseudo_label_loss", pseudo_label_loss)



        # ======= 关系特征蒸馏损失计算 =======
        relation_distill_loss = torch.tensor(0.0).cuda()

        if teacher_model is not None:
            # 计算教师模型和学生模型的关系特征蒸馏损失
            relation_distill_loss = RelationDistillationLoss(student_features, teacher_features)

        # ======= 总损失组合 =======
        total_loss = (
                original_loss +
                FLAGS.output_distill_weight * distill_loss +
                FLAGS.feat_distill_weight * feat_distill_loss +
                FLAGS.pseudo_label_weight * pseudo_label_loss +
                FLAGS.relation_distill_weight * relation_distill_loss+  # 关系特征蒸馏损失
                adapter_loss   #adapter损失
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
            stat_dict['pseudo_label_loss'] = stat_dict.get('pseudo_label_loss', 0.0) + pseudo_label_loss
            stat_dict['relation_distill_loss'] = stat_dict.get('relation_distill_loss',
                                                               0.0) + relation_distill_loss.item()

        if is_primary() and (batch_idx + 1) % 1 == 0:
            log_str = f'[Batch {batch_idx + 1:03d}] total_loss: {total_loss.item():.4f}, original: {original_loss.item():.4f} '
            if teacher_model:
                log_str += f'| out_distill: {distill_loss.item():.4f}, feat_distill: {feat_distill_loss.item():.4f} '
                log_str += f'| pseudo_label_loss: {pseudo_label_loss:.4f} '
            log_str += f'| relation_distill_loss: {relation_distill_loss.item():.4f}'
            # ✅ 用 tqdm.write 替代 log_string，避免破坏进度条 (ok的)
            tqdm.write(log_str)

        counter = counter + 1
        if is_primary() and FLAGS.if_wandb:
            wandb.log({
                f"train/{k}": v for k, v in {
                    "total_loss": total_loss.item(),
                    "original_loss": original_loss.item(),
                    "distill_loss": distill_loss.item() if teacher_model else 0.0,
                    "feat_distill_loss": feat_distill_loss.item() if teacher_model else 0.0,
                    "pseudo_label_loss": pseudo_label_loss,
                    "relation_distill_loss": relation_distill_loss.item()
                }.items()
            }, step=counter)
        # 更新进度条
        progress.update(1)
        progress.set_postfix({"loss": total_loss.item()})

    progress.close()


    # Save checkpoint logic here:
    barrier()

    return stat_dict


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
            # if batch_idx >= 5:
            #   break
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
                      TRAIN_DATALOADER, TEST_DATALOADER, ema_model, teacher_model=None, strategy='fused',adapter_net=None, FLAGS=None):
    global EPOCH_CNT
    loss = 0
    max_mAP = [0.0]  # Initialize max_mAP to a small value
    mAP_LIST = []  # 初始化 mAP_LIST以防止其报错

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
            strategy=strategy,
            adapter_net=adapter_net,
            epoch=epoch
        )

        LOSS_KEYS = [
            "total_loss",
            "original_loss",
            "distill_loss",
            "feat_distill_loss",
            "pseudo_label_loss",
            "relation_distill_loss"
        ]

        # if is_primary() and FLAGS.if_wandb:
        #     for loss_name in LOSS_KEYS:
        #         if loss_name in stat_dict_loss:
        #             average = sum(stat_dict_loss[loss_name]) / len(stat_dict_loss[loss_name])
        #             wandb.log({f"train/{loss_name}": average}, step=epoch)

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

        # # 更改了eval的顺序，先检查ckpt是否能够保存，然后再检查是否能够成功完成对应的验证
        # loss, mAP_LIST = evaluate_one_epoch(net, MODEL, criterion, optimizer, TRAIN_DATALOADER, TEST_DATALOADER, epoch)

        if is_primary():
            torch.save(save_dict, os.path.join(LOG_DIR, f'checkpoint.tar'))
            if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9:  # Eval every 10 epochs
                if not os.path.exists(LOG_DIR):
                    os.makedirs(LOG_DIR)
                save_path = os.path.join(LOG_DIR, f'checkpoint.tar')
                print(f"Saving checkpoint to: {save_path}")
                torch.save(save_dict, os.path.join(LOG_DIR, f'checkpoint_{EPOCH_CNT}.tar'))
            # for i, mAP in enumerate(mAP_LIST):
            #     if mAP > max_mAP[i]:
            #         max_mAP[i] = mAP
            #         if not os.path.exists(LOG_DIR):
            #             os.makedirs(LOG_DIR)
            #         save_path = os.path.join(LOG_DIR, f'checkpoint.tar')
            #         print(f"Saving checkpoint to: {save_path}")
            #         torch.save(save_dict, os.path.join(LOG_DIR, f'checkpoint_best_mAP_dataset_in_{FLAGS.dataset}.tar'))


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

def setup_adapter_network(local_rank, FLAGS, DATASET_CONFIG, TRAIN_DATASET):
    # 单纯模仿def setup_student_network，不确定是否能运用至分布式，我先这么写着吧


    MODEL = importlib.import_module('adapter_net')
    net = MODEL.PCFeatureMLP(
        input_dim=256,
        hidden_dim=256,
        output_dim=256,
        )


    # 将网络移动到GPU
    net = net.cuda(local_rank)
    
    return net




def setup_teacher_network(MODEL, TRAIN_DATASET, FLAGS, DATASET_CONFIG, num_input_channel):
    teacher_tower_weights = list(map(float, FLAGS.teacher_tower_weights.split(',')))

    # 所有进程都加载教师模型
    teacher_model = load_teacher_model(MODEL, TRAIN_DATASET, FLAGS, DATASET_CONFIG, num_input_channel)
    teacher_model.set_tower_weights(teacher_tower_weights)

    # teacher_model.eval()
    # for p in teacher_model.parameters():
    #     p.requires_grad = False

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
    teacher_model = setup_teacher_network(importlib.import_module('imvotenet'), TRAIN_DATASET, FLAGS, DATASET_CONFIG,
                                          int(FLAGS.use_color) * 3 + int(not FLAGS.no_height) * 1)



    # 如果是分布式训练，使用DistributedDataParallel
    if is_distributed():
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

        # 同步教师和EMA模型的BN（尽管他们不训练，但一些任务可能需要）
        if teacher_model is not None:
            teacher_model = teacher_model.cuda()  # 搬上 GPU 即可

            # 只有在有梯度参数时才使用 DDP
            if any(p.requires_grad for p in teacher_model.parameters()):
                teacher_model = torch.nn.parallel.DistributedDataParallel(
                    teacher_model,
                    device_ids=[local_rank],
                    find_unused_parameters=True
                )


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
    #optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)  原有位置，后置是为了可以独立加载adapter_net的权重（尚未设置）

    # 恢复模型
    if FLAGS.resume and CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        log_string(f"-> loaded checkpoint {CHECKPOINT_PATH} (epoch: {start_epoch})")
        torch.cuda.empty_cache()

    if FLAGS.finetune and CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
        print("finetune!!!")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
        old_state_dict = checkpoint['model_state_dict']
        keys_to_skip = [
            'img_only_pnet.conv3.weight', 'img_only_pnet.conv3.bias',
            'pc_only_pnet.conv3.weight', 'pc_only_pnet.conv3.bias',
            'pc_img_pnet.conv3.weight', 'pc_img_pnet.conv3.bias',
            'image_mlp.img_feat_conv1.weight', 'image_mlp.img_feat_conv1.bias'
        ]
        # ===== 新增代码开始 =====
        teacher_checkpoint = torch.load('/root/autodl-tmp/ImOV3D/checkpoint_48.tar',
                                        map_location=torch.device("cpu"))  # 这里我也没在cfg中更改,直接在文件中实现了#有点弄混是用哪个checkpoint了
        teacher_old_state_dict = teacher_checkpoint['model_state_dict']
        # keys_to_skip不变
        teacher_new_state_dict = {k: v for k, v in teacher_old_state_dict.items() if k not in keys_to_skip}
        teacher_model.load_state_dict(teacher_new_state_dict, strict=False)
        log_string("-> loaded finetune teacher checkpoint ")
        # ===== 新增代码结束 =====
        new_state_dict = {k: v for k, v in old_state_dict.items() if k not in keys_to_skip}
        net.load_state_dict(new_state_dict, strict=False)

        log_string("-> loaded finetune checkpoint %s" % (CHECKPOINT_PATH))
        torch.cuda.empty_cache()
    adapter_net=setup_adapter_network(local_rank, FLAGS, DATASET_CONFIG, TRAIN_DATASET)
    
    optimizer = optim.Adam(list(net.parameters()) + list(adapter_net.parameters()), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)
    
    
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
        ema_model=None,
        teacher_model=teacher_model,
        strategy='fused',  # 'fused' 结合教师模型和EMA蒸馏
        adapter_net=adapter_net, #传递adapter_net
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