# 所需的最小导入集
import torch
import torch.nn as nn

# 如果还需要使用其他功能（如激活函数）
import torch.nn.functional as F  # 可选，当前实现未直接使用

class PCFeatureMLP(nn.Module):
    """MLP for processing point cloud features
    Input:  [B, 256, N]
    Output: [B, 512, N]
    """
    def __init__(self, input_dim=256, hidden_dim=384, output_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, pc_features):
        return self.mlp(pc_features)

#预期输入：img_features
#预期输出：pc_features
#img_features维度: torch.Size([8, 256, 1024])
#pc_features维度: torch.Size([8, 256, 1024])