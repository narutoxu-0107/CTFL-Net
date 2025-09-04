import torch
import torch.nn as nn
import torch.nn.functional as F

class MMD_Loss(nn.Module):
    def __init__(self, attn_pool=None, global_pool=None,fc_norm=nn.Identity(), head_drop=nn.Identity()):
        super(MMD_Loss, self).__init__()
        self.attn_pool = attn_pool
        self.global_pool = global_pool
        self.fc_norm = fc_norm
        self.head_drop = head_drop

    def flatten(self, x):
        if self.attn_pool is not None:
            # 如果模型中定义了 attn_pool（注意力池化层），则使用它来处理输入 x。
            # 注意力池化是一种特征聚合方法，可以增强模型对输入数据中重要部分的敏感性。
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            # 如果没有定义 attn_pool，但全局池化（global_pool）设置为 'avg'，
            # 则对输入 x 从 self.num_prefix_tokens 索引之后的部分进行平均池化。
            # 这通常用于减少特征维度，同时保留序列的重要信息。
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            # 如果全局池化定义了但不是 'avg'，
            # 则选择序列的第一个元素（通常是类别标记，即 class token）作为代表整个序列的特征。
            x = x[:, 0]  # class token
        # 对经过池化操作的特征 x 进行归一化处理（fc_norm），这有助于稳定后续全连接层的训练过程。
        x = self.fc_norm(x)
        # 应用 head_drop 方法对特征进行随机丢弃，这是一种正则化手段，用于减少过拟合。
        x = self.head_drop(x)
        return x

    def forward(self, source, target):
        source = self.flatten(source)
        target = self.flatten(target)
        delta = source.mean(0) - target.mean(0)
        loss = delta.dot(delta)
        return loss

def main():
    X = torch.randn(16,768)
    Y = torch.randn(16,768)
    loss = MMD_Loss()
    Loss = loss(X,Y)
    print(Loss)

if __name__ == '__main__':
    main()