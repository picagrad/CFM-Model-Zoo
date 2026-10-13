
import torch
from mmdet.models.task_modules import AssignResult, BaseAssigner, build_iou_calculator
from mmdet.registry import TASK_UTILS
import random 
@TASK_UTILS.register_module()
class PointAssignerV2(BaseAssigner):
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    """

    def __init__(self, scale=4, pos_num=3, mask_center_sample=False, use_center=False):
        self.scale = scale
        self.pos_num = pos_num
        self.mask_center_sample = mask_center_sample
        self.use_center = use_center

    def assign(self, points, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None, gt_bit_masks=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every point, each bbox
        will be assigned with  0, or a positive number.
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every points to 0
        2. for each gt box, we find the k most closest points to the
            box center and assign the gt bbox to those points, we also record
            the minimum distance from each point to the closest gt box. When we
            assign the bbox to the points, we check whether its distance to the
            points is closest.

        Args:
            points (Tensor): points to be assigned, shape(n, 3) while last
                dimension stands for (x, y, stride).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 1e8
        num_gts, num_points = gt_bboxes.shape[0], points.shape[0]

        if num_gts == 0 or num_points == 0:
            # If no truth assign everything to the background
            assigned_gt_inds = points.new_full((num_points, ),
                                               0,
                                               dtype=torch.long)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = points.new_full((num_points, ),
                                                  -1,
                                                  dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(points_stride).int()  # [3...,4...,5...,6...,7...]
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()

        # assign gt box
        gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        # assign gt box
        if self.mask_center_sample:
            _, _h, _w = gt_bit_masks.size()

            _ys = torch.arange(0, _h, dtype=torch.float32, device=gt_bit_masks.device)
            _xs = torch.arange(0, _w, dtype=torch.float32, device=gt_bit_masks.device)
            m00 = gt_bit_masks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (gt_bit_masks * _xs).sum(dim=-1).sum(dim=-1)
            m01 = (gt_bit_masks * _ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_xs = m10 / m00
            center_ys = m01 / m00
            gt_center_xy = torch.stack((center_xs, center_ys), dim=-1)
        else:
            gt_center_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
            print('b: ',gt_bboxes.device)


        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)
        scale = self.scale
        gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) +
                          torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)
        

        if self.use_center:
            distances = ((points_xy[:, None, :] - gt_center_xy[None, :, :]) / gt_bboxes_wh[None, :, :]).norm(dim=2)
        else:
            distances = ((points_xy[:, None, :] - gt_bboxes_xy[None, :, :]) / gt_bboxes_wh[None, :, :]).norm(dim=2)
        distances[points_lvl[:, None] != gt_bboxes_lvl[None, :]] = INF

        # stores the assigned gt index of each point
        assigned_gt_inds = points.new_zeros((num_points, ), dtype=torch.long)

        min_dist, min_dist_index = torch.topk(distances, self.pos_num, dim=0, largest=False)

        distances_inf = torch.full_like(distances, INF)
        distances_inf[min_dist_index, torch.arange(num_gts)] = min_dist

        min_dist, min_dist_index = distances_inf.min(dim=1)
        assigned_gt_inds[min_dist != INF] = min_dist_index[min_dist != INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_points, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)

@TASK_UTILS.register_module()
class ATSSAssignerV2(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
        mask_center_sample: used in mask reppoints v2
    ##: a little difference in dealing with the situation when the number of bbox per level is smaller than topk
    """

    def __init__(self,
                 topk,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1,
                 mask_center_sample=False):
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr
        self.mask_center_sample = mask_center_sample

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               gt_bit_masks=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt


        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        if self.mask_center_sample:
            _, _h, _w = gt_bit_masks.size()

            _ys = torch.arange(0, _h, dtype=torch.float32, device=gt_bit_masks.device)
            _xs = torch.arange(0, _w, dtype=torch.float32, device=gt_bit_masks.device)

            m00 = gt_bit_masks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (gt_bit_masks * _xs).sum(dim=-1).sum(dim=-1)
            m01 = (gt_bit_masks * _ys[:, None]).sum(dim=-1).sum(dim=-1)
            gt_cx = m10 / m00
            gt_cy = m01 / m00
        else:
            gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
            gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            if bboxes_per_level<self.topk:
                provided_idxs = list(range(start_idx,end_idx))
                if len(provided_idxs) >= self.topk-bboxes_per_level:
                    pesudo_idxs_per_level = provided_idxs + random.sample(provided_idxs,self.topk-bboxes_per_level)
                else:
                    pesudo_idxs_per_level = provided_idxs + [random.choice(provided_idxs) for _ in range(self.topk-bboxes_per_level)]
                pesudo_idxs_per_level = distances.new_tensor(pesudo_idxs_per_level).long().unsqueeze(1).expand(self.topk,gt_points.size(0))
                candidate_idxs.append(pesudo_idxs_per_level)
            else:
                distances_per_level = distances[start_idx:end_idx, :]
                selectable_k = min(self.topk, bboxes_per_level)
                _, topk_idxs_per_level = distances_per_level.topk(
                    selectable_k, dim=0, largest=False)
                candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

@TASK_UTILS.register_module()
class PointHMAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    """

    def __init__(self, gaussian_bump=False, gaussian_iou=0.7):
        self.gaussian_bump = gaussian_bump
        self.gaussian_iou = gaussian_iou

    def assign(self, points, gt_bboxes, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every point, each bbox
        will be assigned with  0, or a positive number.
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every points to 0
        2. for each gt box, we find the k most closest points to the
            box center and assign the gt bbox to those points, we also record
            the minimum distance from each point to the closest gt box. When we
            assign the bbox to the points, we check whether its distance to the
            points is closest.

        Args:
            points (Tensor): points to be assigned, shape(n, 3) while last
                dimension stands for (x, y, stride).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 1e8
        num_gts, num_points = gt_bboxes.shape[0], points.shape[0]

        if self.gaussian_bump:
            dtype = torch.float32
        else:
            dtype = torch.long
        if num_points == 0 or num_gts == 0:
            assigned_gt_hm_tl = points.new_zeros((num_points,), dtype=dtype)
            assigned_gt_hm_br = points.new_zeros((num_points,), dtype=dtype)
            # stores the assigned gt dist (to this point) of each point
            assigned_gt_offset_tl = points.new_zeros((num_points, 2), dtype=torch.float32)
            assigned_gt_offset_br = points.new_zeros((num_points, 2), dtype=torch.float32)

            pos_inds_tl = torch.nonzero(assigned_gt_hm_tl == 1, as_tuple=False).squeeze(-1).unique()
            pos_inds_br = torch.nonzero(assigned_gt_hm_br == 1, as_tuple=False).squeeze(-1).unique()
            neg_inds_tl = torch.nonzero(assigned_gt_hm_tl < 1, as_tuple=False).squeeze(-1).unique()
            neg_inds_br = torch.nonzero(assigned_gt_hm_br < 1, as_tuple=False).squeeze(-1).unique()

            return assigned_gt_hm_tl, assigned_gt_offset_tl, pos_inds_tl, neg_inds_tl, \
                   assigned_gt_hm_br, assigned_gt_offset_br, pos_inds_br, neg_inds_br
        points_range = torch.arange(num_points, device = gt_bboxes[0].device)
        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(points_stride).int()  # [3...,4...,5...,6...,7...]
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()

        # assign gt box
        gt_bboxes_xtl, gt_bboxes_ytl, gt_bboxes_xbr, gt_bboxes_ybr = torch.chunk(gt_bboxes, 4, dim=1)
        gt_bboxes_xytl = torch.cat([gt_bboxes_xtl, gt_bboxes_ytl], -1)
        gt_bboxes_xybr = torch.cat([gt_bboxes_xbr, gt_bboxes_ybr], -1)
        if self.gaussian_bump:
            gt_bboxes_w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
            gt_bboxes_h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
            radius = gaussian_radius((gt_bboxes_h, gt_bboxes_w), self.gaussian_iou)
            diameter = 2 * radius + 1
            sigma = diameter / 6
        else:
            radius = None

        distances_tl = (points_xy[:, None, :] - gt_bboxes_xytl[None, :, :]).norm(dim=2)
        distances_br = (points_xy[:, None, :] - gt_bboxes_xybr[None, :, :]).norm(dim=2)

        # stores the assigned gt heatmap of each point
        assigned_gt_hm_tl = points.new_zeros((num_points,), dtype=dtype)
        assigned_gt_hm_br = points.new_zeros((num_points,), dtype=dtype)
        # stores the assigned gt dist (to this point) of each point
        assigned_gt_offset_tl = points.new_zeros((num_points, 2), dtype=torch.float32)
        assigned_gt_offset_br = points.new_zeros((num_points, 2), dtype=torch.float32)

        lvls = torch.arange(lvl_min, lvl_max + 1, dtype=points_lvl.dtype, device=points_lvl.device)
        for gt_lvl in lvls:
            lvl_idx = gt_lvl == points_lvl
            points_index = points_range[lvl_idx]
            lvl_points = points_xy[lvl_idx, :]

            downscale_factor = torch.pow(2, gt_lvl)
            lvl_distances_tl = distances_tl[lvl_idx, :]
            lvl_distances_br = distances_br[lvl_idx, :]

            _, min_dist_index_tl = lvl_distances_tl.min(dim=0)
            min_dist_points_index_tl = points_index[min_dist_index_tl]

            assigned_gt_offset_tl[min_dist_points_index_tl, :] = \
                (gt_bboxes_xytl - lvl_points[min_dist_index_tl, :]) / downscale_factor

            _, min_dist_index_br = lvl_distances_br.min(dim=0)
            min_dist_points_index_br = points_index[min_dist_index_br]
            assigned_gt_offset_br[min_dist_points_index_br, :] = \
                (gt_bboxes_xybr - lvl_points[min_dist_index_br, :]) / downscale_factor
            if self.gaussian_bump:
                out_index_tl = lvl_distances_tl >= radius[None, :]
                lvl_gaussian_tl = torch.exp(-torch.pow(lvl_distances_tl, 2) / (2 * sigma * sigma)[None, :])
                lvl_gaussian_tl[out_index_tl] = -INF
                max_gaussian_tl, _ = lvl_gaussian_tl.max(dim=1)
                assigned_gt_hm_tl[points_index[max_gaussian_tl != -INF]] = max_gaussian_tl[max_gaussian_tl != -INF]

                out_index_br = lvl_distances_br >= radius[None, :]
                lvl_gaussian_br = torch.exp(-torch.pow(lvl_distances_br, 2) / (2 * sigma * sigma)[None, :])
                lvl_gaussian_br[out_index_br] = -INF
                max_gaussian_br, _ = lvl_gaussian_br.max(dim=1)
                assigned_gt_hm_br[points_index[max_gaussian_br != -INF]] = max_gaussian_br[max_gaussian_br != -INF]
            assigned_gt_hm_tl[min_dist_points_index_tl] = 1
            assigned_gt_hm_br[min_dist_points_index_br] = 1

        pos_inds_tl = torch.nonzero(assigned_gt_hm_tl == 1, as_tuple=False).squeeze(-1).unique()
        pos_inds_br = torch.nonzero(assigned_gt_hm_br == 1, as_tuple=False).squeeze(-1).unique()
        neg_inds_tl = torch.nonzero(assigned_gt_hm_tl < 1, as_tuple=False).squeeze(-1).unique()
        neg_inds_br = torch.nonzero(assigned_gt_hm_br < 1, as_tuple=False).squeeze(-1).unique()

        return assigned_gt_hm_tl, assigned_gt_offset_tl, pos_inds_tl, neg_inds_tl, \
               assigned_gt_hm_br, assigned_gt_offset_br, pos_inds_br, neg_inds_br


def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)

    r = torch.stack([r1, r2, r3], dim=1)
    return torch.min(r, dim=1)[0]