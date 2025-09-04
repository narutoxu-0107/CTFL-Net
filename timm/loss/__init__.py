from .asymmetric_loss import AsymmetricLossMultiLabel, AsymmetricLossSingleLabel
from .binary_cross_entropy import BinaryCrossEntropy
from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from .jsd import JsdCrossEntropy
from .mse import MSELoss
from .mmd_loss import MMD_Loss
from .custom_loss import SmoothL1Loss, HuberLoss
