# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
from collections import deque

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.motion.cmc.sof import SOF
from boxmot.motion.kalman_filters.xywh_kf import KalmanFilterXYWH
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.utils.matching import (embedding_distance, fuse_score,
                                   iou_distance, linear_assignment)
from boxmot.utils.ops import xywh2xyxy, xyxy2xywh
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils import PerClassDecorator
from density.Rect_density import smooth_gaussian_rect
from sklearn.neighbors import KDTree


class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, det, feat=None, feat_history=50, max_obs=50):
        # wait activate
        self.xywh = xyxy2xywh(det[0:4])  # (x1, y1, x2, y2) --> (xc, yc, w, h)
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]
        self.max_obs = max_obs
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(self.cls, self.conf)
        self.history_observations = deque([], maxlen=self.max_obs)

        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

        self.density = 0.0

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, conf):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += conf
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, conf])
                self.cls = cls
        else:
            self.cls_hist.append([cls, conf])
            self.cls = cls

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id,density):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.xywh)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.density =density

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.next_id()
        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind
        # 更新密度值为新轨迹的密度
        self.density = new_track.density
        self.update_cls(new_track.cls, new_track.conf)

    def update(self, new_track, frame_id, density):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.history_observations.append(self.xyxy)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh
        )

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.density = density

        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind
        self.update_cls(new_track.cls, new_track.conf)

    @property
    def xyxy(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        if self.mean is None:
            ret = self.xywh.copy()  # (xc, yc, w, h)
        else:
            ret = self.mean[:4].copy()  # kf (xc, yc, w, h)
        ret = xywh2xyxy(ret)
        return ret


class Wbotsort(BaseTracker):
    def __init__(
            self,
            model_weights,
            device,
            fp16,
            per_class=False,
            track_high_thresh: float = 0.5,
            track_low_thresh: float = 0.1,
            new_track_thresh: float = 0.6,
            track_buffer: int = 30,
            match_thresh: float = 0.8,
            proximity_thresh: float = 0.5,
            appearance_thresh: float = 0.25,
            cmc_method: str = "sof",
            frame_rate=30,
            fuse_first_associate: bool = False,
            with_reid: bool = True,
    ):
        super().__init__()
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.per_class = per_class
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.kalman_filter = KalmanFilterXYWH()

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh

        self.with_reid = with_reid
        if self.with_reid:
            self.model = ReidAutoBackend(
                weights=model_weights, device=device, half=fp16
            ).model

        self.cmc = SOF()
        self.fuse_first_associate = fuse_first_associate

        ##新增参数
        self.density_sigma = 1.5  # 高斯核宽度系数(相对目标尺寸)
        self.min_density = 1.5  # 触发外观权重提升的最小密度阈值

    def _compute_density(self, detections):
        """计算检测框之间的相互密度"""
        if len(detections) < 2:
            return np.zeros(len(detections))

        # 使用检测框中心构建KDTree
        det_points = np.array([det.xywh[:2] for det in detections])
        kdtree = KDTree(det_points)

        densities = []
        for i, det in enumerate(detections):
            cx, cy, w, h = det.xywh
            sigma = self.density_sigma * (w + h) / 2  # 尺寸自适应的sigma

            # 查询半径根据目标尺寸动态调整
            neighbors = kdtree.query_radius(
                [[cx, cy]],
                r=2 * sigma
            )[0]

            density = 0.0
            for j in neighbors:
                if j == i: continue  # 排除自身
                nbr = detections[j]
                density += smooth_gaussian_rect(
                    cx, cy,
                    nbr.xywh[0], nbr.xywh[1],
                    nbr.xywh[2], nbr.xywh[3],
                    sigma
                )

            # 添加自身密度项 (保证最小密度>0)
            self_density = smooth_gaussian_rect(
                cx, cy, cx, cy,
                w, h, sigma
            )
            densities.append(density + self_density)

        return np.array(densities)

    """
    def _compute_density(self, tracks, detections):
        """"""
        if not tracks:
            return np.zeros(len(detections))

        # 构建轨迹KDTree
        track_points = np.array([t.mean[:2] for t in tracks])
        kdtree = KDTree(track_points)

        densities = []
        for det in detections:
            cx, cy, w, h = det.xywh
            points = np.array([
                [cx, cy],
                [cx - w / 2, cy - h / 2],
                [cx + w / 2, cy - h / 2],
                [cx - w / 2, cy + h / 2],
                [cx + w / 2, cy + h / 2]
            ])

            sigma = self.density_sigma * min(w, h)
            neighbor_indices = kdtree.query_radius(points, r=2 * sigma)

            point_densities = []
            for i, pt in enumerate(points):
                density = 0.0
                for idx in neighbor_indices[i]:
                    track = tracks[idx]
                    tw, th = track.mean[2], track.mean[3]
                    density += smooth_gaussian_rect(
                        pt[0], pt[1], track.mean[0], track.mean[1],
                        tw, th, self.density_sigma
                    )
                point_densities.append(density)

            avg_density = np.mean(point_densities)
            densities.append(avg_density)
            det.density = avg_density  # 将密度存入STrack对象

        return np.array(densities)
    """

    def _dynamic_fusion(self, ious_dists, emb_dists, densities):
        """"""
        if len(densities) == 0:
            return ious_dists
        log_dens = np.log1p(densities + 1e-6)
        density_ratio = (log_dens - np.median(log_dens)) / (np.std(log_dens) + 1e-6)
        alpha = 1 / (1 + np.exp(-density_ratio))  # 密度越大alpha越接近1

        alpha = alpha.reshape(1, -1)  # 转换为(1, ndet)用于广播

        # 线性融合
        fused_dists = alpha * emb_dists + (1 - alpha) * ious_dists
        print(alpha)

        # 应用掩码
        # mask = (ious_dists > self.proximity_thresh) | (emb_dists * 2.0 > self.appearance_thresh)
        # fused_dists[mask] = 1.0
        print(fused_dists)

        return fused_dists


    @PerClassDecorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:

        self.check_inputs(dets, img)

        self.frame_count += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])

        # Remove bad detections
        confs = dets[:, 4]

        # find second round association detections
        second_mask = np.logical_and(confs > self.track_low_thresh, confs < self.track_high_thresh)
        dets_second = dets[second_mask]

        # find first round association detections
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]

        """Extract embeddings """
        # appearance descriptor extraction
        if self.with_reid:
            if embs is not None:
                features_high = embs
            else:
                # (Ndets x X) [512, 1024, 2048]
                features_high = self.model.get_features(dets_first[:, 0:4], img)

        if len(dets) > 0:
            """Detections"""
            if self.with_reid:
                detections = [STrack(det, f, max_obs=self.max_obs) for (det, f) in zip(dets_first, features_high)]
            else:
                detections = [STrack(det, max_obs=self.max_obs) for (det) in np.array(dets_first)]
        else:
            detections = []

        """ Add newly detected tracklets to active_tracks"""
        unconfirmed = []
        active_tracks = []  # type: list[STrack]
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)

        """ Step 2: First association, with high conf detection boxes"""
        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.cmc.apply(img, dets_first)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high conf detection boxes
        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, detections)

        # if self.with_reid:
        #    emb_dists = embedding_distance(strack_pool, detections) / 2.0
        #    emb_dists[emb_dists > self.appearance_thresh] = 1.0
        #    emb_dists[ious_dists_mask] = 1.0
        #    dists = np.minimum(ious_dists, emb_dists)
        # else:
        #    dists = ious_dists

        densities = self._compute_density(detections)

        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections)

            dists = self._dynamic_fusion(ious_dists, emb_dists, densities)
        else:
            dists = ious_dists

        #dists = np.minimum(dists, 1.0)

        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count ,det.density)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low conf detection boxes"""
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [STrack(dets_second, max_obs=self.max_obs) for dets_second in dets_second]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count,det.density)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        ious_dists = iou_distance(unconfirmed, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh

        ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            ## iou和emb结合的地方
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count , detections[idet].density)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.conf < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_count , track.density)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_age:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_starcks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )

        output_stracks = [track for track in self.active_tracks if track.is_activated]
        outputs = []
        for t in output_stracks:
            output = []
            output.extend(t.xyxy)
            output.append(t.id)
            output.append(t.conf)
            output.append(t.cls)
            output.append(t.det_ind)
            output.append(t.density)
            outputs.append(output)

        outputs = np.asarray(outputs)
        return outputs


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.id] = t
    for t in tlistb:
        tid = t.id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
