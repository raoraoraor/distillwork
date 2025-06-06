# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        if not end_points:
            end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features
        end_points['sa1_relation'] = compute_relation_features(xyz, features)

        xyz, features, fps_inds = self.sa2(xyz, features)
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features
        end_points['sa2_relation'] = compute_relation_features(xyz, features)

        xyz, features, fps_inds = self.sa3(xyz, features)
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features
        end_points['sa3_relation'] = compute_relation_features(xyz, features)

        xyz, features, fps_inds = self.sa4(xyz, features)
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features
        end_points['sa4_relation'] = compute_relation_features(xyz, features)

        # --------- 2 FEATURE UPSAMPLING LAYERS ---------
        fp1_features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'],
                                end_points['sa3_features'], end_points['sa4_features'])
        end_points['fp1_features'] = fp1_features
        end_points['fp1_xyz'] = end_points['sa3_xyz']
        end_points['fp1_relation'] = compute_relation_features(end_points['fp1_xyz'], fp1_features)

        fp2_features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'],
                                end_points['sa2_features'], fp1_features)
        end_points['fp2_features'] = fp2_features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:, 0:num_seed]
        end_points['fp2_relation'] = compute_relation_features(end_points['fp2_xyz'], fp2_features)

        return end_points


def compute_relation_features(xyz, features, k=16):
    """
    xyz: (B, N, 3)
    features: (B, C, N)
    return: (B, N, 3 + C)
    """
    B, N, _ = xyz.shape
    C = features.shape[1]
    features = features.permute(0, 2, 1)  # (B, N, C)

    # Compute pairwise distance for knn
    dist = torch.cdist(xyz, xyz)  # (B, N, N)
    knn_idx = dist.topk(k=k+1, largest=False)[1][:, :, 1:]  # remove self

    idx_base = torch.arange(0, B, device=xyz.device).view(-1, 1, 1) * N
    knn_idx = (knn_idx + idx_base).view(-1)

    xyz_flat = xyz.reshape(B * N, -1)  # Changed view to reshape
    neighbor_xyz = xyz_flat[knn_idx].view(B, N, k, 3)
    center_xyz = xyz.unsqueeze(2).expand(-1, -1, k, -1)
    delta_xyz = neighbor_xyz - center_xyz  # (B, N, k, 3)

    feat_flat = features.reshape(B * N, -1)  # Changed view to reshape
    neighbor_feat = feat_flat[knn_idx].view(B, N, k, C)
    center_feat = features.unsqueeze(2).expand(-1, -1, k, -1)
    delta_feat = neighbor_feat - center_feat  # (B, N, k, C)

    relation = torch.cat([delta_xyz, delta_feat], dim=-1)  # (B, N, k, 3+C)
    R = relation.mean(dim=2)  # (B, N, 3+C)
    return R


if __name__=='__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16,20000,6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)


# Input pointcloud (B, N, 3+C)
#        |
#        v
# +--------------------------+
# |      SA1 (2048 pts)      |
# +--------------------------+
#        |
#        v
# +--------------------------+
# |      SA2 (1024 pts)      |
# +--------------------------+
#        |
#        v
# +--------------------------+
# |      SA3 (512 pts)       |
# +--------------------------+
#        |
#        v
# +--------------------------+
# |      SA4 (256 pts)       |
# +--------------------------+
#        |
#        v
# +--------------------------+
# |    FP1 (256 → 512 pts)   |
# +--------------------------+
#        |
#        v
# +--------------------------+
# |   FP2 (512 → 1024 pts)   |
# +--------------------------+
#        |
#        v
#  Output: end_points

# 层名	采样数量	邻域半径	特征维度输出
# sa1	2048	0.2	128
# sa2	1024	0.4	256
# sa3	512	0.8	256
# sa4	256	1.2	256
