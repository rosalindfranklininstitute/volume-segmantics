"""
    Some parts of this are taken from Adrian Wolny's pytorch-3dunet repository found at
    https://github.com/wolny/pytorch-3dunet
"""

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import numpy as np

from volume_segmantics.utilities.pytorch3dunet_utils import expand_as_one_hot

from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

from typing import Optional, Tuple



class BoundaryDoUDiceLoss(nn.Module):
    """Linear combination of Boundary DoU and Dice losses"""

    def __init__(self, alpha, beta):
        super(BoundaryDoUDiceLoss, self).__init__()
        self.alpha = alpha
        self.bdou = BoundaryDoULoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bdou(input, target) + self.beta * self.dice(input, target)



def _make_2d_contour_table(spacing: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """
    Marching-squares contour length contribution for each of the 16
    configurations of a 2x2 pixel square.

    Bit layout (matches power kernel reshape(1,1,2,2) with arange(0,4)):
        bit0 = top-left,  bit1 = top-right
        bit2 = bottom-left, bit3 = bottom-right

    Boundary segments connect midpoints of edges where pixel values differ.
    Saddle cases (k=6, k=9) are resolved as two separate diagonal segments.
    """
    sy, sx = spacing
    half_diag = 0.5 * np.sqrt(sx ** 2 + sy ** 2)

    def seg_length(tl, tr, bl, br):
        crossings = []
        if tl != tr:
            crossings.append('top')
        if bl != br:
            crossings.append('bot')
        if tl != bl:
            crossings.append('left')
        if tr != br:
            crossings.append('right')

        n = len(crossings)
        if n == 0:
            return 0.0
        if n == 2:
            pair = frozenset(crossings)
            if pair == {'top', 'bot'}:
                return sy
            if pair == {'left', 'right'}:
                return sx
            return half_diag      # corner case: diagonal segment
        if n == 4:
            return 2.0 * half_diag  # saddle: two diagonal segments

        raise ValueError(
            f"Unexpected crossing count {n} for ({tl},{tr},{bl},{br})"
        )

    table = np.zeros(16, dtype=np.float32)
    for k in range(16):
        tl = (k >> 0) & 1
        tr = (k >> 1) & 1
        bl = (k >> 2) & 1
        br = (k >> 3) & 1
        table[k] = seg_length(tl, tr, bl, br)
    return table


class FastSurfaceDiceLoss2D(nn.Module):
    """
    Differentiable surface (boundary) Dice loss for 2D binary or multi-class
    segmentation.

    Approximates the NSD metric at zero tolerance using a marching-squares
    contour length table and overlapping 2x2 sliding windows (stride 1).

    Expected shapes
    ---------------
    preds   : (B, C, H, W)  -- raw logits
    targets : (B, C, H, W)  -- float binary masks, one channel per class

    For binary segmentation pass C=1. For multi-class, each channel is treated
    as an independent binary problem and the per-channel losses are averaged.

    Parameters
    ----------
    spacing : (sy, sx) pixel spacing used for contour length computation.
              Set to your actual voxel size if anisotropic.
    eps     : numerical stability term.
    """

    def __init__(
        self,
        spacing: Tuple[float, float] = (1.0, 1.0),
        eps: float = 1e-5,
    ):
        super().__init__()

        # Power kernel: encodes each 2x2 pixel square into a 4-bit integer
        power = (2 ** np.arange(0, 4)).reshape(1, 1, 2, 2).astype(np.float32)
        contour_table = _make_2d_contour_table(spacing)

        self.register_buffer('power', torch.from_numpy(power))
        self.register_buffer('kernel', torch.ones(1, 1, 2, 2))
        self.register_buffer('contour', torch.from_numpy(contour_table))
        self.eps = eps

    def _surface_dice_binary(
        self,
        logits: torch.Tensor,   # (B, 1, H, W)
        target: torch.Tensor,   # (B, 1, H, W)  float {0, 1}
    ) -> torch.Tensor:
        B = logits.shape[0]

        # logsigmoid + all-ones conv2d accumulates log-probs over the 2x2 window;
        # exp gives P(all 4 pixels in the square are foreground/background).
        fg_prob = F.conv2d(F.logsigmoid(logits),  self.kernel).exp().flatten(1)
        bg_prob = F.conv2d(F.logsigmoid(-logits), self.kernel).exp().flatten(1)
        surf_prob = (1.0 - fg_prob - bg_prob).clamp(min=0.0)

        with torch.no_grad():
            # Encode each 2x2 window as a 4-bit integer (0-15)
            sq_code = F.conv2d(target, self.power).to(torch.int32)
            gt_contour = self.contour[sq_code.reshape(-1)].reshape(B, -1)
            gt_fg   = (sq_code == 15).float().reshape(B, -1)  # all foreground
            gt_bg   = (sq_code ==  0).float().reshape(B, -1)  # all background
            gt_surf = (gt_contour > 0).float()

        fg_dice = (
            (2.0 * (fg_prob * gt_fg).sum(-1) + self.eps)
            / (fg_prob.sum(-1) + gt_fg.sum(-1) + self.eps)
        )
        bg_dice = (
            (2.0 * (bg_prob * gt_bg).sum(-1) + self.eps)
            / (bg_prob.sum(-1) + gt_bg.sum(-1) + self.eps)
        )
        # Surface dice weighted by ground-truth contour length
        surf_dice = (
            (2.0 * (surf_prob * gt_contour).sum(-1) + self.eps)
            / ((surf_prob + gt_surf) * gt_contour).sum(-1).add(self.eps)
        )

        dice = (fg_dice + bg_dice + surf_dice) / 3.0
        return 1.0 - dice.mean()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds   : (B, C, H, W) logits
        targets : (B, C, H, W) binary float masks
        """
        assert preds.shape == targets.shape, (
            f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
        )
        C = preds.shape[1]
        loss = sum(
            self._surface_dice_binary(
                preds[:, c:c + 1],
                targets[:, c:c + 1].float(),
            )
            for c in range(C)
        )
        return loss / C

class BoundaryLoss(nn.Module):
    # def __init__(self, **kwargs):
    def __init__(self, classes=1) -> None:
        super().__init__()
        # # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        # self.idc: List[int] = kwargs["idc"]
        self.idx = [i for i in range(classes)]

    def compute_sdf1_1(self, img_gt, out_shape):
        """
        compute the normalized signed distance map of binary mask
        input: segmentation, shape = (batch_size, x, y, z)
        output: the Signed Distance Map (SDM)
        sdf(x) = 0; x in segmentation boundary
                -inf|x-y|; x in segmentation
                +inf|x-y|; x out of segmentation
        normalize sdf to [-1, 1]
        """
        img_gt = img_gt.cpu().numpy().astype(np.uint8)

        normalized_sdf = np.zeros(out_shape)
        for b in range(out_shape[0]):
            for c in range(0, out_shape[1]):
                posmask = img_gt[b][c].astype(bool)
                if posmask.any():
                    negmask = ~posmask
                    posdis = distance(posmask)
                    negdis = distance(negmask)
                    boundary = skimage_seg.find_boundaries(
                        posmask, mode="inner"
                    ).astype(np.uint8)
                    #plt.figure()
                    #plt.imshow(boundary)
                    #plt.savefig('boundary_t1.png')
                    sdf = (negdis - np.min(negdis)) / (
                        np.max(negdis) - np.min(negdis)
                    ) - (posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis))
                    sdf[boundary == 1] = 0
                    normalized_sdf[b][c] = sdf

        return normalized_sdf

    def forward(self, outputs, gt):
        """
        Compute boundary loss for binary segmentation.
        outputs: sigmoid results,  shape=(b, 2, x, y, z)
        gt:      ground truth mask; shape=(b, 2, x, y, z)
        """
        outputs_soft = outputs.sigmoid()
        gt_sdf = self.compute_sdf1_1(gt, outputs_soft.shape)
        pc = outputs_soft[:, self.idx, ...]
        # Use outputs_soft.device instead of hardcoded .cuda()
        dc = torch.from_numpy(gt_sdf[:, self.idx, ...]).to(outputs_soft.device)
        multipled = torch.einsum("bxyz, bxyz->bxyz", pc, dc)
        bd_loss = multipled.mean()
        return bd_loss


class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes=1):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        device = target.device
        kernel = torch.tensor(
            [[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],
            dtype=torch.float32,
            device=device,
        )

        # Batched conv2d: (B, 1, H, W) -> (B, 1, H, W) -> (B, H, W)
        Y = F.conv2d(
            target.unsqueeze(1).float(),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=1,
        ).squeeze(1)

        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = min(float(2 * alpha - 1), 0.8)

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (
            z_sum + y_sum - (1 + alpha) * intersect + smooth
        )
        return loss

    def forward(self, inputs, target):
        # inputs = torch.softmax(inputs, dim=1)
        inputs = inputs.sigmoid()
        # target = self._one_hot_encoder(target)

        assert (
            inputs.size() == target.size()
        ), "predict {} & target {} shape do not match".format(
            inputs.size(), target.size()
        )

        # return self._adaptive_size(inputs, target)
        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes


class BoundaryDoULossV2(nn.Module):
    def __init__(self, n_classes=1, allowed_outlier_fraction=0.25):
        super(BoundaryDoULossV2, self).__init__()
        self.n_classes = n_classes
        self.allowed_outlier_fraction = allowed_outlier_fraction

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        # Replace per-sample loop with a single batched F.conv2d call
        device = target.device
        kernel = torch.tensor(
            [[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],
            dtype=torch.float32,
            device=device,
        )

        # Batched conv2d: (B, 1, H, W) -> (B, 1, H, W) -> (B, H, W)
        Y = F.conv2d(
            target.unsqueeze(1).float(),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=1,
        ).squeeze(1)

        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = min(float(2 * alpha - 1), 0.8)

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (
            z_sum + y_sum - (1 + alpha) * intersect + smooth
        )
        return loss

    def forward(self, inputs, target):
        inputs = inputs.sigmoid()

        assert (
            inputs.size() == target.size()
        ), "predict {} & target {} shape do not match".format(
            inputs.size(), target.size()
        )

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])

        # Outlier fraction term -- binary, applied to channel 0
        # inputs is already a probability; do NOT apply sigmoid again.
        # Use log(1-p) directly instead of logsigmoid(-logit).
        output = inputs[:, 0]
        target_f = target[:, 0].float()

        neg_mask = target_f.eq(0.0)

        pt = output.clamp(1e-6, 1 - 1e-6)
        neg_loss = -torch.pow(pt, 2) * torch.log(1.0 - pt) * neg_mask

        if self.allowed_outlier_fraction < 1:
            neg_loss = neg_loss.flatten()
            M = neg_loss.numel()
            num_elements_to_keep = int(M * (1 - self.allowed_outlier_fraction))
            neg_loss, _ = torch.topk(
                neg_loss, k=num_elements_to_keep, largest=False, sorted=False
            )

        neg_loss_reduced = neg_loss.sum() / (neg_mask.sum() + 1e-6)

        return (loss + neg_loss_reduced) / (self.n_classes + 1)


class TverskyLoss(nn.Module):
    """
    Tversky loss: T(P,G) = TP / (TP + alpha*FP + beta*FN)

    alpha controls FP penalty, beta controls FN penalty.
    For imbalanced biomedical segmentation where missing structure is costly,
    prefer alpha < beta (e.g. alpha=0.3, beta=0.7).

    By default the background class (index 0) is excluded from the sum;
    set include_background=True to include it.
    """

    def __init__(
        self,
        classes: int,
        alpha: float = 0.3,
        beta: float = 0.7,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.alpha = alpha
        self.beta = beta
        self.include_background = include_background

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _tversky_index(
        self, score: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        EPS = 1e-5
        tp = torch.sum(score * target)
        fp = self.alpha * torch.sum(score * (1.0 - target))
        fn = self.beta  * torch.sum((1.0 - score) * target)
        return tp / (tp + fp + fn + EPS)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = torch.softmax(y_pred, dim=1)
        y_true = self._one_hot_encoder(y_true)

        start_class = 0 if self.include_background else 1
        n_terms = self.classes - start_class
        if n_terms <= 0:
            raise ValueError(
                f"No classes to evaluate: classes={self.classes}, "
                f"include_background={self.include_background}"
            )
        loss = sum(
            self._tversky_index(y_pred[:, i, ...], y_true[:, i, ...])
            for i in range(start_class, self.classes)
        )
        return 1.0 - loss / n_terms


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    #print(input.size(), target.size())
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(_MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, 'ignore_index cannot be None'
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False

        # mask out input/target so that the gradient is zero where on the mask
        input = input * mask
        target = target * mask

        # forward masked input and target to the loss
        return self.loss(input, target)


class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target):
        assert target.size(1) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'

        # skips last target channel if needed
        target = target[:, :-1, ...]

        if self.squeeze_channel:
            # squeeze channel dimension if singleton
            target = torch.squeeze(target, dim=1)
        return self.loss(input, target)


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)
        # expand weights
        weights = weights.unsqueeze(1)
        weights = weights.expand_as(input)

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().to(input.device)
        else:
            class_weights = self.class_weights

        # resize class_weights to be broadcastable into the weights
        class_weights = class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        weights = class_weights * weights

        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()


class WeightedSmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, threshold, initial_weight, apply_below_threshold=True):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1[mask] = l1[mask] * self.weight

        return l1.mean()


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config['loss']
    name = loss_config.pop('name')

    ignore_index = loss_config.pop('ignore_index', None)
    skip_last_target = loss_config.pop('skip_last_target', False)
    weight = loss_config.pop('weight', None)

    if weight is not None:
        # convert to cuda tensor if necessary
        weight = torch.tensor(weight).to(config['device'])

    pos_weight = loss_config.pop('pos_weight', None)
    if pos_weight is not None:
        # convert to cuda tensor if necessary
        pos_weight = torch.tensor(pos_weight).to(config['device'])

    loss = _create_loss(name, loss_config, weight, ignore_index, pos_weight)

    if not (ignore_index is None or name in ['CrossEntropyLoss', 'WeightedCrossEntropyLoss']):
        # use MaskingLossWrapper only for non-cross-entropy losses, since CE losses allow specifying 'ignore_index' directly
        loss = _MaskingLossWrapper(loss, ignore_index)

    if skip_last_target:
        loss = SkipLastTargetChannelWrapper(loss, loss_config.get('squeeze_channel', False))

    return loss


#######################################################################################################################

def _create_loss(name, loss_config, weight, ignore_index, pos_weight):
    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == 'BCEDiceLoss':
        alpha = loss_config.get('alpha', 1.)
        beta = loss_config.get('beta', 1.)
        return BCEDiceLoss(alpha, beta)
    elif name == 'CrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'WeightedCrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return WeightedCrossEntropyLoss(ignore_index=ignore_index)
    elif name == 'PixelWiseCrossEntropyLoss':
        return PixelWiseCrossEntropyLoss(class_weights=weight, ignore_index=ignore_index)
    elif name == 'GeneralizedDiceLoss':
        normalization = loss_config.get('normalization', 'sigmoid')
        return GeneralizedDiceLoss(normalization=normalization)
    elif name == 'DiceLoss':
        normalization = loss_config.get('normalization', 'sigmoid')
        return DiceLoss(weight=weight, normalization=normalization)
    elif name == 'MSELoss':
        return MSELoss()
    elif name == 'SmoothL1Loss':
        return SmoothL1Loss()
    elif name == 'L1Loss':
        return L1Loss()
    elif name == 'WeightedSmoothL1Loss':
        return WeightedSmoothL1Loss(threshold=loss_config['threshold'],
                                    initial_weight=loss_config['initial_weight'],
                                    apply_below_threshold=loss_config.get('apply_below_threshold', True))
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")
