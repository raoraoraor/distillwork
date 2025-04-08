import os
import sys
import numpy as np
from datetime import datetime
import argparse
from config import get_flags
FLAGS = get_flags(flag_train = True)
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

import torch.nn.functional as F  #引用包报错添加


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
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
FLAGS.DUMP_DIR = DUMP_DIR
# Setting tower weights


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
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)'%(LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s'%(LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
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
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Used for AP calculation
CONFIG_DICT = {'remove_empty_box':False, 'use_3d_nms':True,
    'nms_iou':0.25, 'use_old_type_nms':False, 'cls_nms':True,
    'per_class_proposal': True, 'conf_thresh':0.05,
    'dataset_config':DATASET_CONFIG,
    'if_inference_stage_box_filter': FLAGS.if_inference_stage_box_filter,
    'inference_stage_box_filter_thr': FLAGS.inference_stage_box_filter_thr
    }


CONFIG_DICT_LIST = [CONFIG_DICT]
DATASET_CONFIG_LIST = [DATASET_CONFIG]

def load_teacher_model(MODEL,TRAIN_DATASET,FLAGS, DATASET_CONFIG, num_input_channel):
    """加载教师模型（固定结构，不依赖FLAGS.tower_weights）"""
    teacher_model = MODEL.ImVoteNet( #已修改MODEL.路径问题
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
    ).cuda()  #学生模型原文：net = net.cuda(local_rank)#
    '''
    checkpoint = torch.load('checkpoint_48.tar', map_location='cuda') #记得手动改写
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model.eval()
    '''
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model

#当前维度已对齐，暂未使用
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

def train_one_epoch(net,MODEL,criterion,optimizer,bnm_scheduler,TRAIN_DATALOADER,teacher_model=None):
    stat_dict = {} # collect statistics
    stat_dict_loss = {} # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step() # decay BN momentum
    device = next(net.parameters()).device
    net.train() # set model to training mode
    barrier()
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)
        
        # Forward pass
        optimizer.zero_grad()
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
        # 仅第一个batch计算教师输出
        #if batch_idx == 0 and teacher_model is not None:
        with torch.no_grad():
            teacher_end_points = teacher_model(inputs)
        
        student_end_points = net(inputs)
        
        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            if key not in student_end_points:
                student_end_points[key] = batch_data_label[key]
        print("Student logits :", student_end_points['pc_img_sem_cls_scores'].shape)
        print("Student len:",len(student_end_points))
        original_loss, student_end_points = criterion(student_end_points, DATASET_CONFIG, KEY_PREFIX_LIST, TOWER_WEIGHTS)   #尚未更改
        print("Student len:",len(student_end_points))
        print("Student logits :", student_end_points['pc_img_sem_cls_scores'].shape)
        # ======= 插入蒸馏损失计算 START ======= 
        distill_loss = 0
        #if teacher_model is not None and batch_idx == 0:  # 仅第一个batch计算   #我们暂时先不做这一步的优化吧
        with torch.no_grad():  # 确保不计算教师梯度
            teacher_end_points = teacher_model(inputs)
            for key in batch_data_label:
                if key not in teacher_end_points:
                    teacher_end_points[key] = batch_data_label[key]
            print("Teacher logits :", teacher_end_points['pc_img_sem_cls_scores'].shape)
            print("Teacher len:",len(teacher_end_points))
            teacher_loss, teacher_end_points = criterion(teacher_end_points, DATASET_CONFIG, KEY_PREFIX_LIST, TOWER_WEIGHTS)   #尚未更改
            print("Teacher len:",len(teacher_end_points))
            print("Teacher logits :", teacher_end_points['pc_img_sem_cls_scores'].shape)
        
        distill_loss = F.kl_div(
            F.log_softmax(student_end_points['pc_img_sem_cls_scores'], dim=-1),
            F.softmax(teacher_end_points['pc_img_sem_cls_scores'], dim=-1),
            reduction='batchmean'
        )#公式倒时可以再改改
        
        #每一次训练学生模型会变，教师模型不变，所以不能把distill_loss放入只有首次的if支线中
        stat_dict['distill_loss'] = distill_loss.item()
        
        total_loss = original_loss + 0.5 * distill_loss  # 组合损失
        # ======= 插入蒸馏损失计算 END =======
        
        total_loss.backward()
        optimizer.step()

        # Accumulate statistics and print out
        for key in student_end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += student_end_points[key].item()

        batch_interval = 10
        if is_primary() and (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            output_str = "batch id: %d " % batch_idx
            for key_prefix in KEY_PREFIX_LIST:
                output_str += '%s: %f '%(key_prefix+'loss',
                                         stat_dict[key_prefix+'loss']/batch_interval)
                if key_prefix not in stat_dict_loss:
                    stat_dict_loss[key_prefix] = []
                stat_dict_loss[key_prefix].append(stat_dict[key_prefix+'loss']/batch_interval)
            log_string(output_str)
            for key in sorted(stat_dict.keys()):
                stat_dict[key] = 0
        barrier()
    return stat_dict_loss

        

    
def evaluate_one_epoch(net,MODEL,criterion,optimizer,TRAIN_DATALOADER,TEST_DATALOADER,epoch):
    mAP_LIST = []
    if FLAGS.use_imvotenet:
        KEY_PREFIX_LST = KEY_PREFIX_LIST[2:]
    else:
        KEY_PREFIX_LST = KEY_PREFIX_LIST

    for DATASET_idx, DATASET_ITEM in enumerate([TEST_DATALOADER]):
        print(FLAGS.dataset, DATASET_idx,DATASET_CONFIG_LIST[DATASET_idx].class2type_eval)
        
        stat_dict = {} # collect statistics
        ap_calculator_dict = {}
        for key_prefix in  KEY_PREFIX_LST:
            ap_calculator_dict[key_prefix+'ap_calculator'] = APCalculator(ap_iou_thresh=FLAGS.ap_iou_thresh,
                                                                        class2type_map=DATASET_CONFIG_LIST[DATASET_idx].class2type_eval)
        device = next(net.parameters()).device
        net.eval() # set model to eval mode (for bn and dp)
        barrier()
        for batch_idx, batch_data_label in enumerate(DATASET_ITEM):
            if batch_idx % 10 == 0:
                print('Eval batch: %d'%(batch_idx))
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
                    end_points = net(inputs,joint_only=True)
                else:
                    end_points = net(inputs)

            # Compute loss
            for key in batch_data_label:
                if key not in end_points:
                    end_points[key] = batch_data_label[key]
                    
            loss, end_points = criterion(end_points, DATASET_CONFIG_LIST[DATASET_idx],  KEY_PREFIX_LST, TOWER_WEIGHTS)

            # Accumulate statistics and print out
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'ratio' in key:
                    if key not in stat_dict: stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()

            for key_prefix in  KEY_PREFIX_LST:
                batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT_LIST[DATASET_idx], key_prefix)
                batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT_LIST[DATASET_idx]) 
                ap_calculator_dict[key_prefix+'ap_calculator'].step(batch_pred_map_cls, batch_gt_map_cls)
            
            barrier()
        if is_primary():
            for key in sorted(stat_dict.keys()):
                log_string(f'{FLAGS.dataset}'+'_eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))
        
            # Evaluate average precision
            for key_prefix in KEY_PREFIX_LST:
                metrics_dict = ap_calculator_dict[key_prefix+'ap_calculator'].compute_metrics()
                for key in metrics_dict:
                    log_string(f'{FLAGS.dataset}'+'_eval %s: %f'%(key, metrics_dict[key]))
                    
                        
                    if key != 'mAP':
                        if FLAGS.if_wandb:
                            wandb.log({f"dataset_{FLAGS.dataset}/{key}": metrics_dict[key]}, step=epoch)
                    if key =='mAP':
                        if FLAGS.if_wandb:
                            wandb.log({f"test/mAP_dataset_{FLAGS.dataset}": metrics_dict[key]}, step=epoch)
                        mAP_LIST.append(metrics_dict[key])

                

        mean_loss = stat_dict['loss']/float(batch_idx+1)
    return mean_loss,mAP_LIST

def train_or_evaluate(start_epoch,net,MODEL,net_no_ddp,criterion,optimizer,bnm_scheduler,train_sampler,TRAIN_DATALOADER,TEST_DATALOADER, teacher_model=None):
    global EPOCH_CNT 
    loss = 0
    max_mAP = [0.0]  # Initialize max_mAP to a small value
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        if is_distributed():
            train_sampler.set_epoch(EPOCH_CNT)
        if is_primary():
            log_string('**** EPOCH %03d ****' % (epoch))
            log_string('Current learning rate: %f'%(get_current_lr(epoch)))
            log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
            log_string(str(datetime.now()))
        # Reset numpy seed.
        np.random.seed()
        # REF: https://github.com/pytorch/pytorch/issues/5059
        stat_dict_loss = train_one_epoch(net,MODEL,criterion,optimizer,bnm_scheduler,TRAIN_DATALOADER,teacher_model)
        if is_primary() and FLAGS.if_wandb:
            for key_prefix in KEY_PREFIX_LIST:
                if key_prefix in stat_dict_loss:
                    average = sum(stat_dict_loss[key_prefix]) / len(stat_dict_loss[key_prefix])
                    wandb.log({f"train/{key_prefix}_loss": average }, step=epoch)
        
    
        loss,mAP_LIST = evaluate_one_epoch(net,MODEL,criterion,optimizer,TRAIN_DATALOADER,TEST_DATALOADER,epoch)
        
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        if is_primary():
            torch.save(save_dict, os.path.join(LOG_DIR, f'checkpoint.tar'))
            if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
                torch.save(save_dict, os.path.join(LOG_DIR, f'checkpoint_{EPOCH_CNT}.tar'))
            for i, mAP in enumerate(mAP_LIST):
                if mAP > max_mAP[i]:
                    max_mAP[i] = mAP
                    torch.save(save_dict, os.path.join(LOG_DIR, f'checkpoint_best_mAP_dataset_in_{FLAGS.dataset}.tar'))
        


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
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
        wandb.init(
            # set the wandb project where this run will be logged
            project="ImOV3D",
            
            # track hyperparameters and run metadata
            config=vars(FLAGS)  # Pass the parsed argparse arguments directly
        )
    torch.cuda.set_device(local_rank)
    np.random.seed(FLAGS.seed + get_rank())
    torch.manual_seed(FLAGS.seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FLAGS.seed + get_rank())
    
    TRAIN_DATASET = DetectionVotesDataset('train', 
                                            num_points=NUM_POINT,
                                            augment=True,
                                            use_color=FLAGS.use_color, 
                                            use_height=(not FLAGS.no_height),
                                            use_imvote=FLAGS.use_imvotenet,
                                            max_imvote_per_pixel=FLAGS.max_imvote_per_pixel,
                                           )

    TEST_DATASET = DetectionVotesDataset('val', 
                                                num_points=NUM_POINT,
                                                augment=False,
                                                use_color=FLAGS.use_color, 
                                                use_height=(not FLAGS.no_height),
                                                use_imvote=FLAGS.use_imvotenet,
                                                max_imvote_per_pixel=FLAGS.max_imvote_per_pixel,
                                                )      

    num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1
    if FLAGS.use_imvotenet:
        MODEL = importlib.import_module('imvotenet')
        net = MODEL.ImVoteNet(num_class=DATASET_CONFIG.num_class,
                            num_heading_bin=DATASET_CONFIG.num_heading_bin,
                            num_size_cluster=DATASET_CONFIG.num_size_cluster,
                            mean_size_arr=DATASET_CONFIG.mean_size_arr,
                            num_proposal=FLAGS.num_target,
                            input_feature_dim=num_input_channel,
                            vote_factor=FLAGS.vote_factor,
                            sampling=FLAGS.cluster_sampling,
                            max_imvote_per_pixel=FLAGS.max_imvote_per_pixel,
                            image_feature_dim=TRAIN_DATASET.image_feature_dim)

    else:
        MODEL = importlib.import_module('votenet')
        net = MODEL.VoteNet(num_class=DATASET_CONFIG.num_class,
                            num_heading_bin=DATASET_CONFIG.num_heading_bin,
                            num_size_cluster=DATASET_CONFIG.num_size_cluster,
                            mean_size_arr=DATASET_CONFIG.mean_size_arr,
                            num_proposal=FLAGS.num_target,
                            input_feature_dim=num_input_channel,
                            vote_factor=FLAGS.vote_factor,
                            sampling=FLAGS.cluster_sampling)
    # if is_primary():
    #     print("net",net)
    if is_primary():
        if FLAGS.if_wandb:
	        wandb.watch(net)
        print("dataset",len(TRAIN_DATASET), len(TEST_DATASET))
    for name, param in net.named_parameters():
        if "model_clip" in name:
            param.requires_grad=False
    for name, param in net.named_parameters():
        if "model_clip_2Dsemantic" in name:
            param.requires_grad=False  

      
                   
    net = net.cuda(local_rank)
    net_no_ddp = net
    
    # ===== 新增代码开始 =====
    # 加载教师模型（仅主进程）
    teacher_model = load_teacher_model(MODEL,TRAIN_DATASET,FLAGS, DATASET_CONFIG, num_input_channel) if is_primary() else None
    
    # 强制使用纯3D分支配置（覆盖FLAGS设置）
    #global TOWER_WEIGHTS 
    #TOWER_WEIGHTS = {'img_only_weight': 0.0, 'pc_only_weight': 1.0, 'pc_img_weight': 0.0}  #直接在计算教师endpoints那里改就是了

    print('Tower weights', TOWER_WEIGHTS)
    # ===== 新增代码结束 =====
        
    if is_distributed():
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[local_rank]#,find_unused_parameters=True
        )
        
        teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
        '''
        teacher_model = torch.nn.parallel.DistributedDataParallel(
            teacher_model, device_ids=[local_rank]#,find_unused_parameters=True
        )#cpoy   #已经被冻结的参数这里就直接跳过吧
        '''
    criterion = MODEL.get_loss
   
    train_sampler = DistributedSampler(TRAIN_DATASET)
    #test_sampler = DistributedSampler(TEST_DATASET)

    
    TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
        sampler=train_sampler, num_workers=FLAGS.num_workers, worker_init_fn=my_worker_init_fn)

    TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
        num_workers=FLAGS.num_workers, worker_init_fn=my_worker_init_fn)
 
    optimizer = optim.Adam(net_no_ddp.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)
    if  FLAGS.resume and CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
        print("resume!!!")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
        net_no_ddp.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))
        torch.cuda.empty_cache()
    if  FLAGS.finetune and CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
        print("finetune!!!")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
        old_state_dict = checkpoint['model_state_dict']
        keys_to_skip = [
            'img_only_pnet.conv3.weight', 'img_only_pnet.conv3.bias',
            'pc_only_pnet.conv3.weight', 'pc_only_pnet.conv3.bias',
            'pc_img_pnet.conv3.weight', 'pc_img_pnet.conv3.bias',
            'image_mlp.img_feat_conv1.weight','image_mlp.img_feat_conv1.bias'
        ]
        # ===== 新增代码开始 =====
        teacher_checkpoint = torch.load('checkpoint_99.tar', map_location=torch.device("cpu")) #这里我也没在cfg中更改,直接在文件中实现了#有点弄混是用哪个checkpoint了
        teacher_old_state_dict = teacher_checkpoint['model_state_dict']
        #keys_to_skip不变
        teacher_new_state_dict = {k: v for k, v in teacher_old_state_dict.items() if k not in keys_to_skip}
        teacher_model.load_state_dict(teacher_new_state_dict, strict=False)
        log_string("-> loaded finetune teacher checkpoint ")
        # ===== 新增代码结束 =====
        new_state_dict = {k: v for k, v in old_state_dict.items() if k not in keys_to_skip}
        net_no_ddp.load_state_dict(new_state_dict, strict=False)

        log_string("-> loaded finetune checkpoint %s"%(CHECKPOINT_PATH))
        torch.cuda.empty_cache()
        
    it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    # Decay Batchnorm momentum from 0.5 to 0.999
    # note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
    BN_MOMENTUM_INIT = 0.5
    BN_MOMENTUM_MAX = 0.001
    bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
    bnm_scheduler = BNMomentumScheduler(net_no_ddp, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)
        
    train_or_evaluate(start_epoch,
                      net,
                      MODEL,
                      net_no_ddp,
                      criterion,
                      optimizer,
                      bnm_scheduler,
                      train_sampler,
                      TRAIN_DATALOADER,
                      TEST_DATALOADER,
                      teacher_model
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



if __name__ == "__main__":

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    launch_distributed(FLAGS)

