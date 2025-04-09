# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" ImVoteNet for 3D object detection with RGB-D.

Author: Charles R. Qi, Xinlei Chen and Or Litany
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from proposal_module import ProposalModule
from image_feature_module import ImageFeatureModule, ImageMLPModule, append_img_feat
from dump_helper import dump_results
from loss_helper import get_loss

from config import get_flags, global_flag

import torch.nn.functional as F  # Add this line with other imports

FLAGS = get_flags(global_flag)
if FLAGS.dataset == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    import sunrgbd_utils as dataset_utils
elif FLAGS.dataset == 'scannet':
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    import scannet_utils as dataset_utils

elif FLAGS.dataset == 'lvis':
    sys.path.append(os.path.join(ROOT_DIR, 'lvis'))
    import lvis_utils as dataset_utils

import clip


def sample_valid_seeds(mask, num_sampled_seed=1024):
    """
    (TODO) write doc for this function
    """
    mask = mask.cpu().detach().numpy()  # B,N
    all_inds = np.arange(mask.shape[1])  # 0,1,,,,N-1
    batch_size = mask.shape[0]
    sample_inds = np.zeros((batch_size, num_sampled_seed))
    for bidx in range(batch_size):
        valid_inds = np.nonzero(mask[bidx, :])[0]  # return index of non zero elements
        if len(valid_inds) < num_sampled_seed:
            assert (num_sampled_seed <= 1024)
            rand_inds = np.random.choice(list(set(np.arange(1024)) - set(np.mod(valid_inds, 1024))),
                                         num_sampled_seed - len(valid_inds),
                                         replace=False)
            cur_sample_inds = np.concatenate((valid_inds, rand_inds))
        else:
            cur_sample_inds = np.random.choice(valid_inds, num_sampled_seed, replace=False)
        sample_inds[bidx, :] = cur_sample_inds
    sample_inds = torch.from_numpy(sample_inds).long()
    return sample_inds


class ImVoteNet(nn.Module):
    r"""
        ImVoteNet module.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
        max_imvote_per_pixel: (default: 3)
            Maximum number of image votes per pixel.
        image_feature_dim: (default: 18)
            Total number of dimensions for image features, geometric + semantic + texture
        image_hidden_dim: (default: 256)
            Hidden dimensions for the image based VoteNet, default same as point based VoteNet
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps',
                 max_imvote_per_pixel=3, image_feature_dim=18, image_hidden_dim=256,output_distill_weight=0.5, feat_distill_weight=1.0, use_distillation=True):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.max_imvote_per_pixel = max_imvote_per_pixel
        self.image_feature_dim = image_feature_dim

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Image feature extractor
        self.image_feature_extractor = ImageFeatureModule(max_imvote_per_pixel=self.max_imvote_per_pixel)
        # MLP on image features before fusing with point features
        self.image_mlp = ImageMLPModule(image_feature_dim, image_hidden_dim=image_hidden_dim)

        # Hough voting modules
        self.img_only_vgen = VotingModule(self.vote_factor, image_hidden_dim)
        self.pc_only_vgen = VotingModule(self.vote_factor, 256)
        self.pc_img_vgen = VotingModule(self.vote_factor, image_hidden_dim + 256)

        #
        self.text_2Dsemantic = ["a photo of " + str(item) for item in dataset_utils.type2class.keys()]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_clip_2Dsemantic, _ = clip.load("ViT-B/32", device=device)
        text_2Dsemantic = clip.tokenize(self.text_2Dsemantic).to(device)
        self.text_feats_2Dsemantic = self.batch_encode_text(text_2Dsemantic)

        # Vote aggregation and detection
        self.img_only_pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
                                            mean_size_arr, num_proposal, sampling, seed_feat_dim=image_hidden_dim,
                                            key_prefix='img_only_')
        self.pc_only_pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
                                           mean_size_arr, num_proposal, sampling, seed_feat_dim=256,
                                           key_prefix='pc_only_')
        self.pc_img_pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
                                          mean_size_arr, num_proposal, sampling, seed_feat_dim=image_hidden_dim + 256,
                                          key_prefix='pc_img_')

        self.output_distill_weight = output_distill_weight
        self.feat_distill_weight = feat_distill_weight
        self.use_distillation = use_distillation

    # def initialize_ema_model(self):
    #     """ 使用 deepcopy 初始化 EMA 模型 """
    #     self.ema_model = copy.deepcopy(self)
    #     self.ema_model.eval()
    #     for param in self.ema_model.parameters():
    #         param.requires_grad = False
    #
    # def update_ema_model(self, alpha=None):
    #     """ EMA 参数更新 """
    #     if self.ema_model is None:
    #         return
    #
    #     if alpha is None:
    #         alpha = self.ema_decay
    #
    #     for ema_param, param in zip(self.ema_model.parameters(), self.parameters()):
    #         ema_param.data.mul_(alpha).add_(param.data * (1 - alpha))

    def batch_encode_text(self, text):
        batch_size = 20

        text_num = text.shape[0]
        cur_start = 0
        cur_end = 0

        all_text_feats = []
        while cur_end < text_num:
            # print(cur_end)
            cur_start = cur_end
            cur_end += batch_size
            if cur_end >= text_num:
                cur_end = text_num

            cur_text = text[cur_start:cur_end, :]
            cur_text_feats = self.model_clip_2Dsemantic.encode_text(cur_text).detach()
            all_text_feats.append(cur_text_feats)

        all_text_feats = torch.cat(all_text_feats, dim=0)
        # print(all_text_feats.shape)
        return all_text_feats

    def set_tower_weights(self, tower_weights):
        # 将传入的tower_weights保存为类的成员变量
        self.tower_weights = tower_weights

    def forward(self, inputs, joint_only=False,return_features=False):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)

                (TODO) write doc for this function
        Returns:
            end_points: dict
        """
        end_points = {}
        end_points.update(inputs)

        end_points = self.backbone_net(inputs['point_clouds'], end_points)
        img_feat_list = self.image_feature_extractor(end_points)
        assert len(img_feat_list) == self.max_imvote_per_pixel
        xyz, features, seed_inds = append_img_feat(img_feat_list, end_points, self.text_feats_2Dsemantic)
        seed_sample_inds = sample_valid_seeds(features[:, -1, :], 1024).cuda()
        features = torch.gather(features, -1, seed_sample_inds.unsqueeze(1).repeat(1, features.shape[1], 1))
        xyz = torch.gather(xyz, 1, seed_sample_inds.unsqueeze(-1).repeat(1, 1, 3))
        seed_inds = torch.gather(seed_inds, 1, seed_sample_inds)

        end_points['seed_inds'] = seed_inds
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features

        pc_features = features[:, :256, :]
        img_features = features[:, 256:, :]
        img_features = self.image_mlp(img_features)
        joint_features = torch.cat((pc_features, img_features), 1)

        # 1. Backbone特征提取
        end_points = self.backbone_net(inputs['point_clouds'], end_points)
        backbone_features = end_points['fp2_features']  # [B, 256, N]

        # 2. 图像特征处理 (保留原有流程)
        img_feat_list = self.image_feature_extractor(end_points)
        xyz, features, seed_inds = append_img_feat(img_feat_list, end_points, self.text_feats_2Dsemantic)
        seed_sample_inds = sample_valid_seeds(features[:, -1, :], 1024).cuda()
        features = torch.gather(features, -1, seed_sample_inds.unsqueeze(1).repeat(1, features.shape[1], 1))

        # 3. 特征分离与融合
        pc_features = features[:, :256, :]  # 点云特征 [B, 256, N]
        img_features = features[:, 256:, :]  # 图像特征
        img_features = self.image_mlp(img_features)
        joint_features = torch.cat((pc_features, img_features), 1)  # [B, 256+hidden_dim, N]

        # 4. 投票特征生成
        vote_xyz, vote_features = self.pc_img_vgen(end_points['seed_xyz'], joint_features)
        vote_features = F.normalize(vote_features, p=2, dim=1)

        if not joint_only:
            # --------- IMAGE-ONLY TOWER ---------
            prefix = 'img_only_'
            xyz, features = self.img_only_vgen(end_points['seed_xyz'], img_features)
            features_norm = torch.norm(features, p=2, dim=1)
            features = features.div(features_norm.unsqueeze(1))
            end_points[prefix + 'vote_xyz'] = xyz
            end_points[prefix + 'vote_features'] = features
            end_points = self.img_only_pnet(xyz, features, end_points)

            # --------- POINTS-ONLY TOWER ---------
            prefix = 'pc_only_'
            xyz, features = self.pc_only_vgen(end_points['seed_xyz'], pc_features)
            features_norm = torch.norm(features, p=2, dim=1)
            features = features.div(features_norm.unsqueeze(1))
            end_points[prefix + 'vote_xyz'] = xyz
            end_points[prefix + 'vote_features'] = features
            end_points = self.pc_only_pnet(xyz, features, end_points)

        # --------- JOINT TOWER ---------
        prefix = 'pc_img_'
        xyz, features = self.pc_img_vgen(end_points['seed_xyz'], joint_features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points[prefix + 'vote_xyz'] = xyz
        end_points[prefix + 'vote_features'] = features
        end_points = self.pc_img_pnet(xyz, features, end_points)

        end_points = self.pc_img_pnet(vote_xyz, vote_features, end_points)

        if return_features:
            # 创建一个字典存储特征信息
            feature_dict = {
                'backbone_features': end_points['fp2_features'],  # 上采样后的 backbone 特征
                'vote_features': vote_features,  # 投票后的联合特征
            }

            # 添加各层的 R^a 关系特征
            relation_keys = ['sa1_relation', 'sa2_relation', 'sa3_relation', 'sa4_relation', 'fp2_relation']
            for k in relation_keys:
                feature_dict[k] = end_points[k]  # 直接传递 R^a 张量

            # 计算基于相似度的伪标签
            B, C, N = vote_features.shape  # 获取 batch size (B), 类别数 (C), 投票点数量 (N)
            # print("vote_features",vote_features.shape)
            # 获取文本特征并进行归一化
            text_feats = self.text_feats_2Dsemantic  # 文本特征（预训练好的 2D 语义特征）
            text_feats = F.normalize(text_feats, dim=1)  # 按行进行归一化


            # 展平 vote_features，形状为 [B*N, C]，用于后续计算相似度
            vote_features_flat = vote_features.permute(0, 2, 1).reshape(-1, C)  # 转置并展平
            vote_features_flat = F.normalize(vote_features_flat, dim=1)  # 对投票特征进行归一化

            # 计算投票特征与文本特征的相似度（内积计算）
            sim_logits = torch.matmul(vote_features_flat, text_feats.float().T)  # 计算相似度 logits
            sim_logits = sim_logits.view(B, N, -1)  # 重塑形状为 [B, N, num_classes]

            # 生成基于相似度的伪标签（取相似度最大的类别索引）
            pseudo_labels_sim = sim_logits.argmax(dim=-1)  # 通过 argmax 获取最大相似度对应的类别标签
            feature_dict['similarity_logits'] = sim_logits  # 存储相似度 logits
            
            feature_dict['sim_pseudo_labels'] = pseudo_labels_sim  # 存储基于相似度的伪标签

            # # 生成基于 EMA 模型的伪标签
            # ema_vote_features = self.ema_model.pc_img_vgen(end_points['seed_xyz'], joint_features)[1]  # 从 EMA 模型生成投票特征
            # ema_vote_features = F.normalize(ema_vote_features, p=2, dim=1)  # 对 EMA 投票特征进行归一化
            #
            # # 展平 EMA 投票特征并进行归一化
            # ema_vote_features_flat = ema_vote_features.permute(0, 2, 1).reshape(-1, C)  # 转置并展平
            # ema_vote_features_flat = F.normalize(ema_vote_features_flat, dim=1)  # 对 EMA 投票特征进行归一化
            #
            # # 计算 EMA 投票特征与文本特征的相似度（内积计算）
            # ema_sim_logits = torch.matmul(ema_vote_features_flat, text_feats.float().T)  # 计算 EMA 相似度 logits
            # ema_sim_logits = ema_sim_logits.view(B, N, -1)  # 重塑形状为 [B, N, num_classes]
            #
            # # 生成基于 EMA 的伪标签（取相似度最大的类别索引）
            # ema_pseudo_labels = ema_sim_logits.argmax(dim=-1)  # 通过 argmax 获取最大相似度对应的类别标签
            # feature_dict['ema_similarity_logits'] = ema_sim_logits  # 存储 EMA 相似度 logits
            # feature_dict['ema_pseudo_labels'] = ema_pseudo_labels  # 存储基于 EMA 的伪标签

            return end_points, feature_dict  # 返回 end_points 和特征字典
        return end_points


# 返回的 feature_dict 结构说明：
#
# Key                    Shape / 类型       说明
# ----------------------------------------------------------------------------
# backbone_features       (B, 256, N)        上采样后的 backbone 特征
#                                             B: batch size, 256: 特征维度, N: 点云中投票的数量
#
# vote_features           (B, C, N)          投票后的联合特征
#                                             C: 特征维度, N: 点云中投票的数量
#
# sa1_relation            (B, N1, 3+C)       第1层关系特征
#                                             B: batch size, N1: 第1层特征点数量,
#                                             3: 相对空间位置 (x, y, z)，C: 特征维度
#
# sa2_relation            (B, N2, 3+C)       第2层关系特征
#                                             B: batch size, N2: 第2层特征点数量,
#                                             3: 相对空间位置 (x, y, z)，C: 特征维度
#
# sa3_relation            (B, N3, 3+C)       第3层关系特征
#                                             B: batch size, N3: 第3层特征点数量,
#                                             3: 相对空间位置 (x, y, z)，C: 特征维度
#
# sa4_relation            (B, N4, 3+C)       第4层关系特征
#                                             B: batch size, N4: 第4层特征点数量,
#                                             3: 相对空间位置 (x, y, z)，C: 特征维度
#
# fp2_relation            (B, N, 3+C)        上采样层后的关系特征
#                                             B: batch size, N: 上采样后的特征点数量,
#                                             3: 相对空间位置 (x, y, z)，C: 特征维度
#
# pseudo_labels           (B, N)             每个 vote 的伪标签
#                                             B: batch size, N: 点云中投票的数量
#
# similarity_logits       (B, N, num_classes) vote 与文本类别的相似度
#                                             B: batch size, N: 点云中投票的数量,
#                                             num_classes: 类别数，表示每个投票与各个类别之间的相似度
