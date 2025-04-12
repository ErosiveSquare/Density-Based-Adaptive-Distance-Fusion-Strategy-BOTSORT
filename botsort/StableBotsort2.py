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


class STrack_history(BaseTrack):
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, det, feat=None, feat_history=150, max_obs=50):

        self.history = {}  # 保存frame_id到bbox的映射

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

    # 在STrack_history类中添加以下方法
    def record_history(self, frame_id):
        """记录当前帧的bbox到历史"""
        self.history[frame_id] = self.xyxy



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
            multi_mean, multi_covariance = STrack_history.shared_kalman.multi_predict(
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

    def activate(self, kalman_filter, frame_id):
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

        self.update_cls(new_track.cls, new_track.conf)

    def update(self, new_track, frame_id):

        self.history[frame_id] = self.xyxy  # 记录当前帧的bbox

        self.record_history(frame_id)  # 记录历史

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


class Stablebotsort2(BaseTracker):
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
            appearance_thresh: float = 0.5,
            cmc_method: str = "sof",
            frame_rate=30,
            fuse_first_associate: bool = False,
            with_reid: bool = True,
    ):
        super().__init__()

        # 新增参数
        self.stable_check_interval = 10  # 稳定检测间隔帧数
        self.stable_history_length = 150  # 需要检查的历史长度
        self.stable_iou_thresh = 0.8  # IOU阈值
        self.match_iou_thresh = 0.6  # 匹配IOU阈值
        self.stable_tracks = []  # 持续保存稳定轨迹
        self.frame_count = 0  # 初始化帧计数
        self.stale_age = 300  # 稳定轨迹的最大保留时间（帧数）

        self.lost_stracks = []  # type: list[STrack_history]
        self.removed_stracks = []  # type: list[STrack_history]
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



    ## -------------------------2025-1-26-----------------------------
    def calculate_iou(self, box1, box2):
        """计算两个bbox的IOU（xyxy格式）"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union != 0 else 0

    def update_stable_tracks(self):
        """更新稳定轨迹池"""
        current_frame = self.frame_count
        updated_stable = []

        # 1. 添加新稳定轨迹
        candidates = self.active_tracks + self.lost_stracks
        for track in candidates:
            if self.is_stable_track(track, current_frame):
                # 检查是否已经存在
                exists = any(t.id == track.id for t in self.stable_tracks)
                if not exists:
                    # 克隆轨迹以避免后续更新影响稳定记录
                    cloned = self._clone_track(track)
                    cloned.stable_frame = current_frame
                    self.stable_tracks.append(cloned)
                else:
                    # 更新已存在稳定轨迹的时间戳
                    for stable in self.stable_tracks:
                        if stable.id == track.id:
                            stable.stable_frame = current_frame

        # 2. 清理过期轨迹
        for track in self.stable_tracks:
            # 保留条件：最近被更新或在时间窗口内
            if (current_frame - track.stable_frame) <= self.stale_age:
                updated_stable.append(track)
        self.stable_tracks = updated_stable

    def _clone_track(self, track):
        """克隆轨迹对象用于稳定轨迹池"""
        cloned = STrack_history(
            det=np.zeros(7),  # 伪数据，实际不会使用
            feat=track.smooth_feat.copy() if track.smooth_feat is not None else None,
            feat_history=track.features.maxlen
        )
        cloned.id = track.id
        cloned.xywh = track.xywh.copy()
        cloned.cls = track.cls
        cloned.smooth_feat = track.smooth_feat.copy() if track.smooth_feat is not None else None
        cloned.history = track.history.copy()  # 复制历史记录
        return cloned

    def match_against_stable(self, detections):
        """将检测与稳定轨迹匹配"""
        if len(detections) == 0 or len(self.stable_tracks) == 0:
            return [], detections

        # IOU匹配
        ious = iou_distance(self.stable_tracks, detections)
        iou_matches = np.where(ious < self.match_iou_thresh)  # 使用配置的IOU阈值

        # 特征匹配
        valid_matches = []
        matched_det_indices = set()
        for s_idx, d_idx in zip(*iou_matches):
            stable = self.stable_tracks[s_idx]
            det = detections[d_idx]

            # 计算特征相似度
            if self.with_reid and stable.smooth_feat is not None and det.curr_feat is not None:
                emb_dist = embedding_distance([stable], [det])[0][0]
                if emb_dist < self.appearance_thresh:
                    valid_matches.append((s_idx, d_idx))
                    matched_det_indices.add(d_idx)

        # 处理有效匹配
        matched_pairs = []
        for s_idx, d_idx in valid_matches:
            stable = self.stable_tracks[s_idx]
            det = detections[d_idx]

            # 继承ID并更新稳定记录
            det.id = stable.id
            det.update_features(stable.smooth_feat)  # 使用稳定特征
            matched_pairs.append((det, stable))

        # 返回未匹配的检测
        remaining_dets = [d for i, d in enumerate(detections) if i not in matched_det_indices]
        return matched_pairs, remaining_dets


    def is_stable_track1(self, track, current_frame):
        """判断轨迹是否稳定"""
        # 生成需要检查的帧序列
        required_frames = [current_frame - i * self.stable_check_interval
                           for i in range(1, self.stable_history_length // self.stable_check_interval + 1)]

        # 检查是否所有需要的帧都存在
        if any(f not in track.history for f in required_frames):
            return False

        # 收集所有bbox并计算IOU差异
        bboxes = [track.history[f] for f in required_frames]

        # 计算所有bbox对的最大IOU差异
        min_iou = 1.0
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                iou = self.calculate_iou(bboxes[i], bboxes[j])
                if iou < min_iou:
                    min_iou = iou
                    if min_iou < self.stable_iou_thresh:
                        return False
        return True

    def is_stable_track(self, track, current_frame):
        """增强版稳定性判断"""
        # 基础条件：存在足够历史记录
        if len(track.history) < self.stable_history_length:
            return False

        # 检查周期性帧的位移
        check_points = sorted(track.history.keys())[-self.stable_history_length:]
        positions = [track.history[f] for f in check_points]

        # 计算平均位移
        total_displacement = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            total_displacement += np.sqrt(dx ** 2 + dy ** 2)

        avg_displacement = total_displacement / (len(positions) - 1)

        # 同时满足低位移和高IOU稳定性
        return avg_displacement < 5 and self.is_stable_track1(track, current_frame)

    def get_stable_tracks(self):
        """获取当前所有稳定轨迹"""
        current_frame = self.frame_count
        stable_tracks = []
        for track in self.active_tracks + self.lost_stracks:
            if self.is_stable_track(track, current_frame):
                stable_tracks.append(track)
        return stable_tracks

    ## -------------------------2025-1-26-----------------------------
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
                detections = [STrack_history(det, f, max_obs=self.max_obs) for (det, f) in
                              zip(dets_first, features_high)]
            else:
                detections = [STrack_history(det, max_obs=self.max_obs) for (det) in np.array(dets_first)]
        else:
            detections = []

        """ Add newly detected tracklets to active_tracks"""
        unconfirmed = []
        active_tracks = []  # type: list[STrack_history]
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)

        """ Step 2: First association, with high conf detection boxes"""
        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        # Predict the current location with KF
        STrack_history.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.cmc.apply(img, dets_first)
        STrack_history.multi_gmc(strack_pool, warp)
        STrack_history.multi_gmc(unconfirmed, warp)

        # Associate with high conf detection boxes
        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low conf detection boxes"""
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [STrack_history(dets_second, max_obs=self.max_obs) for dets_second in dets_second]
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
                track.update(det, self.frame_count)
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
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks 2025.1.26修改后  0207"""
        """ Step 4: Init new stracks """
        # 先与稳定轨迹匹配
        matched_pairs, remaining_dets = self.match_against_stable([detections[i] for i in u_detection])

        # 处理匹配到的稳定轨迹
        for det, stable in matched_pairs:
            # 使用稳定轨迹的特征和ID
            det.activate(self.kalman_filter, self.frame_count)
            det.id = stable.id  # 继承ID
            activated_starcks.append(det)

            # 更新稳定记录的时间戳
            stable.stable_frame = self.frame_count

        # 处理剩余未匹配的检测
        for det in remaining_dets:
            if det.conf < self.new_track_thresh:
                continue
            det.activate(self.kalman_filter, self.frame_count)
            activated_starcks.append(det)

        """ 新增步骤5: 更新稳定轨迹池 """
        self.update_stable_tracks()

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
