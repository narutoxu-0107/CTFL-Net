import torch
import torch.nn as nn


class AsymmetricLossMultiLabel(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossMultiLabel, self).__init__()

        # 非对称损失参数
        self.gamma_neg = gamma_neg  # 负样本的聚焦因子
        self.gamma_pos = gamma_pos  # 正样本的聚焦因子
        self.clip = clip    # 非对称裁剪的阈值
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss  # 是否禁用focal loss的梯度
        self.eps = eps  # 防止对数函数取对数时的数值稳定性

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        """
        前向传播函数
        x: 输入的logits
        y: 目标（多标签二进制向量）
        """

        # Calculating Probabilities
        # 计算sigmoid激活后的值
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid      # 计算sigmoid激活后的值
        xs_neg = 1 - x_sigmoid  # 计算sigmoid激活后的值

        # Asymmetric Clipping
        # 应用非对称裁剪
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        # 基本的交叉熵损失计算
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps)) # 正样本损失
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))   # 负样本损失
        loss = los_pos + los_neg

        # Asymmetric Focusing
        # 应用非对称聚焦机制
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)   # 禁用梯度计算
            pt0 = xs_pos * y    # 正样本的预测概率
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p  # 负样本的预测概率
            pt = pt0 + pt1  # 总的预测概率
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y) # 非对称因子
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)    # 非对称权重
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)    # 重新启用梯度计算
            loss *= one_sided_w # 应用非对称权重

        return -loss.sum()


class AsymmetricLossSingleLabel(nn.Module):
    def __init__(self, gamma_pos=1, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(AsymmetricLossSingleLabel, self).__init__()

        # 非对称损失参数
        self.eps = eps  # 非对称损失参数
        self.logsoftmax = nn.LogSoftmax(dim=-1) # 非对称损失参数
        self.targets_classes = []  # prevent gpu repeated memory allocation # 非对称损失参数
        self.gamma_pos = gamma_pos  # 非对称损失参数
        self.gamma_neg = gamma_neg  # 负样本的聚焦因子
        self.reduction = reduction  # 损失的缩减方式，'mean'或'sum'

    def forward(self, inputs, target, reduction=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (1-hot vector)
        """

        num_classes = inputs.size()[-1] # 类别数量
        log_preds = self.logsoftmax(inputs) # 应用LogSoftmax
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)  # 创建目标类别的掩码

        # ASL weights
        # 计算非对称损失的权重
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)   # 正样本的预测概率
        xs_neg = 1 - xs_pos             # 负样本的预测概率
        xs_pos = xs_pos * targets       # 调整为仅正样本的概率
        xs_neg = xs_neg * anti_targets  # 调整为仅负样本的概率
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)  # 非对称权重
        log_preds = log_preds * asymmetric_w    # 应用权重

        if self.eps > 0:  # label smoothing# 应用标签平滑
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation  # 计算损失
        loss = - self.targets_classes.mul(log_preds)    # 交叉熵损失

        loss = loss.sum(dim=-1) # 对类别维度求和
        if self.reduction == 'mean':    # 如果指定了缩减为'mean'
            loss = loss.mean()          # 计算平均损失

        return loss
