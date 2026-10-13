import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS

from mmdet.structures.bbox import bbox_overlaps
from mmdet.models import weighted_loss


def separate_sigmoid_focal_loss(pred,
                                target,
                                weight=None,
                                gamma=2.0,
                                alpha=0.25,
                                reduction='mean',
                                avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)

    pos_inds = target.eq(1)
    neg_inds = target.lt(1)

    pos_weights = weight[pos_inds]

    pos_pred = pred_sigmoid[pos_inds]
    neg_pred = pred_sigmoid[neg_inds]

    pos_loss = -torch.log(pos_pred) * torch.pow(1 -
                                                pos_pred, gamma) * pos_weights * alpha
    neg_loss = -torch.log(1 - neg_pred) * \
        torch.pow(neg_pred, gamma) * (1 - alpha)

    if pos_pred.nelement() == 0:
        loss = neg_loss.sum() / avg_factor
    else:
        loss = pos_loss.sum() / pos_weights.sum() + neg_loss.sum() / avg_factor

    return loss

@weighted_loss
def giou_loss_oldver(pred, target, eps=1e-7):
    """
    Generalized Intersection over Union: A Metric and A Loss for
    Bounding Box Regression
    https://arxiv.org/abs/1902.09630

    code refer to:
    https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py#L36

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt + 1).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1)
    ag = (target[:, 2] - target[:, 0] + 1) * (target[:, 3] - target[:, 1] + 1)
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + eps

    # GIoU
    gious = ious - (enclose_area - union) / enclose_area
    loss = 1 - gious
    return loss

@MODELS.register_module()
@weighted_loss
def quality_focal_loss(pred, target, beta=2.0, background_label=None):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    if background_label is None:
        bg_class_ind = pred.size(1)
        pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    else:
        pos = (label!=background_label).nonzero().squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    if len(pos) == 0:
        pass
    else:
        scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
        loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
            pred[pos, pos_label], score[pos],
            reduction='none') * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss


@MODELS.register_module()
class RPDQualityFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(RPDQualityFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.requires_box=True

    def forward(self,
                pred,
                target,
                weight=None,
                bbox_pred=None,
                bbox_target=None,
                bbox_weight=None,
                background_label=0,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        #compute iou score
        iou_score = bbox_target.new_zeros(target.size()).squeeze()
        pos_inds = (target!=background_label).nonzero().squeeze(1)
        if len(pos_inds)>0:
            score=bbox_overlaps(bbox_pred[pos_inds],bbox_target[pos_inds],is_aligned=True)
            #print(score)
            iou_score[pos_inds]=score.detach()
        iou_score = iou_score.detach()

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * quality_focal_loss(
                pred,
                (target,iou_score),
                weight,
                beta=self.beta,
                background_label=background_label,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls

@MODELS.register_module()
class RPDGIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(RPDGIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss_oldver(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

@MODELS.register_module()
class SEPFocalLoss(nn.Module):

    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(SEPFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * separate_sigmoid_focal_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls

@weighted_loss
def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss

@MODELS.register_module()
class LegacyGaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(LegacyGaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_reg = self.loss_weight * gaussian_focal_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg