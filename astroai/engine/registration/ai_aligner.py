"""ONNX CNN-based noise-robust frame registration (FR-2.2, SuperPoint-style)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter

if TYPE_CHECKING:
    import onnxruntime as ort

__all__ = ["AIAligner", "AlignmentResult"]

logger = logging.getLogger(__name__)

_PATCH_RADIUS = 4   # half-size → 8×8 = 64-dim patch descriptor
_MIN_MATCHES = 6    # minimum correspondences to attempt RANSAC


@dataclass
class AlignmentResult:
    """Per-frame alignment result with quality metrics."""

    aligned: np.ndarray
    transform: np.ndarray       # 3×3 homography (ref → target)
    confidence: float           # 0.0 … 1.0  (inliers / matched pairs)
    inlier_count: int
    keypoints_matched: int


class AIAligner:
    """Noise-robust AI-based frame aligner using keypoint matching + RANSAC homography.

    Uses an ONNX model (SuperPoint protocol) when ``model_path`` is provided;
    falls back to a deterministic DoG / patch-descriptor extractor otherwise.
    """

    def __init__(
        self,
        model_path: str | None = None,
        inlier_threshold: float = 3.0,
        ransac_iterations: int = 1000,
        min_inliers: int = 6,
        quality_threshold: float = 0.3,
        max_keypoints: int = 256,
        rng_seed: int | None = None,
    ) -> None:
        self.inlier_threshold = inlier_threshold
        self.ransac_iterations = ransac_iterations
        self.min_inliers = min_inliers
        self.quality_threshold = quality_threshold
        self.max_keypoints = max_keypoints
        self._rng = np.random.default_rng(rng_seed)
        self._session: ort.InferenceSession | None = None
        if model_path is not None:
            self._session = _load_onnx_session(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(self, reference: np.ndarray, target: np.ndarray) -> AlignmentResult:
        ref_gray = _to_grayscale(reference)
        tgt_gray = _to_grayscale(target)

        kp_ref, desc_ref = self._extract_features(ref_gray)
        kp_tgt, desc_tgt = self._extract_features(tgt_gray)

        if len(kp_ref) < _MIN_MATCHES or len(kp_tgt) < _MIN_MATCHES:
            logger.debug(
                "Too few keypoints (ref=%d tgt=%d); returning identity",
                len(kp_ref), len(kp_tgt),
            )
            return _identity_result(target)

        idx_ref, idx_tgt = self._match_descriptors(desc_ref, desc_tgt)
        if len(idx_ref) < _MIN_MATCHES:
            logger.debug("Too few matches (%d); returning identity", len(idx_ref))
            return _identity_result(target, keypoints_matched=len(idx_ref))

        # Convert internal (y, x) keypoints → (x, y) for homography
        src_xy = kp_ref[idx_ref, ::-1]
        dst_xy = kp_tgt[idx_tgt, ::-1]

        H, _, inlier_count = self._ransac_homography(src_xy, dst_xy)
        confidence = inlier_count / max(len(idx_ref), 1)

        logger.debug(
            "AI align: %d matches, %d inliers, conf=%.3f",
            len(idx_ref), inlier_count, confidence,
        )

        aligned = self._warp_perspective(target, H)
        return AlignmentResult(
            aligned=aligned,
            transform=H,
            confidence=float(confidence),
            inlier_count=inlier_count,
            keypoints_matched=len(idx_ref),
        )

    def align_batch(
        self,
        reference: np.ndarray,
        targets: list[np.ndarray],
    ) -> list[AlignmentResult]:
        return [self.align(reference, t) for t in targets]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(
        self, gray: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._session is not None:
            return _onnx_extract(self._session, gray, self.max_keypoints)
        return _dog_extract(gray, self.max_keypoints)

    def _match_descriptors(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cross-check nearest-neighbour matching on L2-normalised descriptors."""
        if len(desc1) == 0 or len(desc2) == 0:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)

        n1 = desc1 / (np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-8)
        n2 = desc2 / (np.linalg.norm(desc2, axis=1, keepdims=True) + 1e-8)
        sim = n1 @ n2.T                         # (N1, N2) cosine similarity
        fwd = np.argmax(sim, axis=1)            # best in desc2 for each in desc1
        bwd = np.argmax(sim, axis=0)            # best in desc1 for each in desc2
        idx1 = np.arange(len(desc1))
        mask = bwd[fwd] == idx1
        return idx1[mask], fwd[mask]

    def _ransac_homography(
        self,
        src: np.ndarray,
        dst: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """RANSAC homography from (N,2) (x,y) correspondences.

        Returns ``(H, inlier_mask, inlier_count)``.
        """
        n = len(src)
        best_H = np.eye(3, dtype=np.float64)
        best_mask = np.zeros(n, dtype=bool)
        best_count = 0

        if n < 4:
            return best_H, best_mask, best_count

        src_h = np.column_stack([src, np.ones(n, dtype=np.float64)])

        for _ in range(self.ransac_iterations):
            idx = self._rng.choice(n, 4, replace=False)
            try:
                H = _dlt_homography(src[idx], dst[idx])
            except (np.linalg.LinAlgError, ValueError):
                continue

            proj = (H @ src_h.T).T          # (N, 3)
            w = proj[:, 2]
            safe_w = np.where(np.abs(w) < 1e-8, 1.0, w)
            proj_xy = proj[:, :2] / safe_w[:, None]
            dists = np.linalg.norm(proj_xy - dst, axis=1)
            mask = dists < self.inlier_threshold
            count = int(mask.sum())

            if count > best_count:
                best_count, best_H, best_mask = count, H, mask

        # Refit on all inliers for better accuracy
        if best_count >= 4:
            try:
                best_H = _dlt_homography_lstsq(src[best_mask], dst[best_mask])
            except (np.linalg.LinAlgError, ValueError):
                pass

        return best_H, best_mask, best_count

    def _warp_perspective(self, image: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Warp image with H (ref→target): output[p] = image[H @ p]."""
        h, w = image.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w]
        coords_out = np.stack(
            [xs.ravel(), ys.ravel(), np.ones(h * w, dtype=np.float64)], axis=0
        )
        coords_in = H @ coords_out              # sample positions in target space
        wc = coords_in[2]
        safe_wc = np.where(np.abs(wc) < 1e-8, 1.0, wc)
        x_in = coords_in[0] / safe_wc
        y_in = coords_in[1] / safe_wc

        if image.ndim == 2:
            return map_coordinates(
                image.astype(np.float64), [y_in, x_in],
                order=1, mode="constant", cval=0.0,
            ).reshape(h, w).astype(image.dtype)

        channels = [
            map_coordinates(
                image[..., c].astype(np.float64), [y_in, x_in],
                order=1, mode="constant", cval=0.0,
            ).reshape(h, w)
            for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=-1).astype(image.dtype)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _identity_result(
    target: np.ndarray,
    keypoints_matched: int = 0,
) -> AlignmentResult:
    return AlignmentResult(
        aligned=target.copy(),
        transform=np.eye(3, dtype=np.float64),
        confidence=0.0,
        inlier_count=0,
        keypoints_matched=keypoints_matched,
    )


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float64)
    weights = np.array([0.2989, 0.5870, 0.1140])
    return np.dot(frame[..., :3], weights).astype(np.float64)


def _dog_extract(
    gray: np.ndarray,
    max_keypoints: int,
) -> tuple[np.ndarray, np.ndarray]:
    """DoG keypoints + normalised 8×8 patch descriptors (no model required)."""
    lo = gaussian_filter(gray, sigma=1.0)
    hi = gaussian_filter(gray, sigma=2.0)
    dog = lo - hi

    abs_dog = np.abs(dog)
    local_max = maximum_filter(abs_dog, size=7) == abs_dog
    threshold = max(abs_dog.mean() * 2.0, 1e-10)
    mask = local_max & (abs_dog > threshold)

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 64), dtype=np.float32)

    pr = _PATCH_RADIUS
    h, w = gray.shape
    ok = (ys >= pr) & (ys < h - pr) & (xs >= pr) & (xs < w - pr)
    ys, xs = ys[ok], xs[ok]
    if len(ys) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 64), dtype=np.float32)

    responses = abs_dog[ys, xs]
    order = np.argsort(-responses)[:max_keypoints]
    ys, xs = ys[order], xs[order]

    descs = []
    for y, x in zip(ys, xs):
        patch = gray[y - pr: y + pr, x - pr: x + pr].copy()
        std = patch.std()
        patch = (patch - patch.mean()) / (std if std > 1e-8 else 1.0)
        descs.append(patch.ravel().astype(np.float32))

    return (
        np.column_stack([ys, xs]).astype(np.float64),
        np.array(descs, dtype=np.float32),
    )


def _load_onnx_session(model_path: str) -> "ort.InferenceSession":
    import onnxruntime as ort
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def _onnx_extract(
    session: "ort.InferenceSession",
    gray: np.ndarray,
    max_keypoints: int,
) -> tuple[np.ndarray, np.ndarray]:
    """SuperPoint-protocol ONNX inference: outputs[0]=(N,2) (x,y), outputs[1]=(N,D)."""
    inp = gray.astype(np.float32)[None, None]
    inp_name = session.get_inputs()[0].name
    outputs = session.run(None, {inp_name: inp})
    kpts_xy = outputs[0][:max_keypoints]
    descs = outputs[1][:max_keypoints].astype(np.float32)
    keypoints = kpts_xy[:, ::-1].astype(np.float64)  # (x,y) → (y,x)
    return keypoints, descs


def _dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute homography from exactly 4 correspondences (DLT, (x,y) inputs)."""
    A: list[list[float]] = []
    for (x, y), (xp, yp) in zip(src, dst):
        A.extend([
            [-x, -y, -1.0,  0.0,  0.0,  0.0, xp * x, xp * y, xp],
            [ 0.0,  0.0,  0.0, -x, -y, -1.0, yp * x, yp * y, yp],
        ])
    _, _, Vt = np.linalg.svd(np.array(A, dtype=np.float64))
    H = Vt[-1].reshape(3, 3)
    if abs(H[2, 2]) < 1e-10:
        raise ValueError("Degenerate homography")
    return H / H[2, 2]


def _dlt_homography_lstsq(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares DLT homography from N ≥ 4 correspondences."""
    n = len(src)
    A = np.zeros((2 * n, 9), dtype=np.float64)
    for i, ((x, y), (xp, yp)) in enumerate(zip(src, dst)):
        A[2 * i]     = [-x, -y, -1.0,  0.0,  0.0,  0.0, xp * x, xp * y, xp]
        A[2 * i + 1] = [ 0.0,  0.0,  0.0, -x,  -y, -1.0, yp * x, yp * y, yp]
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    if abs(H[2, 2]) < 1e-10:
        raise ValueError("Degenerate homography")
    return H / H[2, 2]
