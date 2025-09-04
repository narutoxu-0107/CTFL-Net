import torch
import torch.nn as nn
import torch.nn.functional as F

"""
1. 加权损失函数
动机：在一些回归任务中，不同范围的预测误差可能对实际应用的影响不同。对于 LAI 回归，低值和高值可能需要不同的精度。
实现方法：为不同的 LAI 范围设置权重，使模型在关键范围内（如低值或高值的 LAI）更注重精度。例如
"""

class weighted_mse_loss(nn.Module):
    def __init__(self):
        super(weighted_mse_loss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight:torch.Tensor=0.5) -> torch.Tensor:

        target = target.type(torch.float).view(-1,1)
        try:
            loss = weight * (pred - target) ** 2
        except:
            loss = weight * (pred.logits - target) ** 2
        return loss


"""
2. Huber Loss（平滑的L1损失）
动机：Huber Loss 在对抗异常值时比 MSE 更具鲁棒性，但在误差较小时又比 L1 更为平滑。适合 LAI 回归时处理异常的预测值。
实现方法：使用 torch.nn.SmoothL1Loss 来代替传统的 MSE：
"""
class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        target = target.type(torch.float).view(-1,1)
        loss_fn = torch.nn.SmoothL1Loss()
        try:
            loss = loss_fn(pred, target)
        except:
            loss = loss_fn(pred.logits, target)
        return loss

"""
4. 基于物理意义的损失
动机：将 LAI 估计任务中的物理或生物学知识纳入损失函数，确保预测值更符合实际的物理规律。
实现方法：例如，添加一个正则项来惩罚不合理的 LAI 预测（如不应有负值）
"""
class physically_constrained_loss(nn.Module):
    def __init__(self):
        super(physically_constrained_loss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, lambda_ = 0.5) -> torch.Tensor:

        target = target.type(torch.float).view(-1,1)
        try:
            loss = F.mse_loss(pred, target)
            # 物理约束：LAI 应为正
            constraint_loss = torch.mean(torch.relu(-pred))  # 惩罚负值
        except:
            loss = F.mse_loss(pred.logits, target)
            # 物理约束：LAI 应为正
            constraint_loss = torch.mean(torch.relu(-pred))  # 惩罚负值
        return loss+ lambda_ * constraint_loss

"""
6. 不对称损失函数
动机：有时候，预测值偏高和偏低的影响可能不同。针对这种情况，可以设计不对称的损失函数，特别关注预测偏高或偏低的情况。
实现方法：对偏高或偏低的预测给予不同的惩罚
"""
class asymmetric_loss(nn.Module):
    def __init__(self):
        super(asymmetric_loss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, delta=1.0) -> torch.Tensor:

        target = target.type(torch.float).view(-1,1)
        try:
            diff = pred - target
            loss = torch.where(diff > 0, delta * diff ** 2, diff ** 2)
        except:
            diff = pred.logits - target
            loss = torch.where(diff > 0, delta * diff ** 2, diff ** 2)
        return loss.mean()


class HuberLoss(nn.Module):
    def __init__(self):
        super(HuberLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, delta=1.0) -> torch.Tensor:

        target = target.type(torch.float).view(-1,1)
        loss_fn = nn.HuberLoss(reduction="sum", delta=delta)
        try:
            loss = loss_fn(pred , target)
        except:
            loss = loss_fn(pred.logits , target)
        return loss

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        target = target.type(torch.float).view(-1,1)
        loss_fn = F.l1_loss
        try:
            loss = loss_fn(pred , target)
        except:
            loss = loss_fn(pred.logits , target)
        return loss

class HuberLoss_L1(nn.Module):
    def __init__(self):
        super(HuberLoss_L1, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,feat_local: torch.Tensor,feat_global: torch.Tensor, delta=1.0) -> torch.Tensor:

        target = target.type(torch.float).view(-1,1)
        loss_main = nn.HuberLoss(reduction="sum", delta=delta)
        loss_sub = F.l1_loss
        try:
            loss = loss_main(pred , target)+0.1*loss_sub(feat_local, feat_global)
        except:
            loss = loss_main(pred.logits , target)+0.1*loss_sub(feat_local, feat_global)
        return loss

class HuberLoss_L1_v1(nn.Module):
    def __init__(self):
        super(HuberLoss_L1_v1, self).__init__()
        self.sub_weight = nn.Parameter(torch.tensor(0.2), requires_grad=True)
    def forward(self, pred: torch.Tensor, target: torch.Tensor,feat_local: torch.Tensor,feat_global: torch.Tensor, delta=1.0) -> torch.Tensor:

        target = target.type(torch.float).view(-1,1)
        loss_main = nn.HuberLoss(reduction="sum", delta=delta)
        loss_sub = F.l1_loss

        try:
            loss = loss_main(pred , target)+self.sub_weight*loss_sub(feat_local, feat_global)
        except:
            loss = loss_main(pred.logits , target)+self.sub_weight*loss_sub(feat_local, feat_global)
        return loss



# 定义互信息损失函数
class MutualInformationLoss(nn.Module):
    def __init__(self):
        super(MutualInformationLoss, self).__init__()

    def forward(self, local_feat, global_feat):
        # 计算全局特征和局部特征之间的互信息
        N, C, H, W = global_feat.size()
        global_feat_flat = global_feat.view(N, C, -1)
        local_feat_flat = local_feat.view(N, C, -1)

        # 计算特征的相关性矩阵
        global_norm = global_feat_flat / (global_feat_flat.norm(dim=-1, keepdim=True) + 1e-6)
        local_norm = local_feat_flat / (local_feat_flat.norm(dim=-1, keepdim=True) + 1e-6)

        # 计算互信息损失，鼓励全局和局部特征的差异性
        mutual_information = torch.einsum('nct,nct->nc', global_norm, local_norm)
        loss = -torch.mean(mutual_information)  # 最大化差异性，最小化相似性
        return loss


# 定义新的 Loss 结合互信息损失和 MSE loss
class MiLoss(nn.Module):
    def __init__(self,sub_weight=0.1):
        super(MiLoss, self).__init__()
        self.loss_main = nn.HuberLoss(reduction="sum", delta=1)
        self.sub_loss = MutualInformationLoss()
        self.sub_weight = sub_weight

    def forward(self, pred, target, local_feat,global_feat):

        target = target.type(torch.float).view(-1, 1)
        # MSE loss for regression task
        main_loss = self.loss_main(pred, target)

        # Mutual Information Loss to encourage difference
        mi_loss = self.sub_loss(local_feat, global_feat)

        # Combine the losses
        total_loss = main_loss + self.sub_weight* mi_loss  # 权重可以根据实际需要调整

        return total_loss

if __name__ == '__main__':
    loss = MutualInformationLoss()
    x1 = torch.randn(16,768,7,7)
    x2 = torch.randn(16, 768,7,7)
    print(loss(x1,x2))





