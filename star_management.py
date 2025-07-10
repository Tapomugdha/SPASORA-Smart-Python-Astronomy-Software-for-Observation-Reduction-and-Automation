import cv2
import numpy as np
from star_detection import detect_stars  # Use your robust detection function

class StarManager:
    def __init__(self, min_brightness=60, max_brightness=250, max_saturation=245):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_saturation = max_saturation
        self.selected_star = None
        self.secondary_stars = []

    def ensure_grayscale(self, img):
        """Convert to grayscale if image is color or has more than 2 dimensions."""
        if img.ndim == 3:
            if img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 1:
                return img[:, :, 0]
            else:
                raise ValueError(f"Unsupported channel number: {img.shape[2]}")
        elif img.ndim == 2:
            return img
        else:
            raise ValueError(f"Input image must be 2D or 3D with 1 or 3 channels, got shape {img.shape}")

    def find_candidate_stars(self, frame, threshold_sigma=5, fwhm=3.0, min_brightness=None, max_brightness=None):
        """
        Find candidate stars in the given frame using sigma-based thresholding.
        Returns a list of dicts: {'centroid': (x, y), 'mean': ..., 'max': ..., 'area': ...}
        """
        img = frame.astype(np.float32)
        img = self.ensure_grayscale(img)

        # Use robust star detection function (photutils or OpenCV fallback)
        centroids = detect_stars(img, threshold_sigma=threshold_sigma, fwhm=fwhm)

        # Use provided min/max brightness if given, else use object defaults
        min_bright = self.min_brightness if min_brightness is None else min_brightness
        max_bright = self.max_brightness if max_brightness is None else max_brightness

        candidates = []
        for (cx, cy) in centroids:
            x, y = int(round(cx)), int(round(cy))
            roi_size = 7
            x1 = max(x - roi_size // 2, 0)
            y1 = max(y - roi_size // 2, 0)
            x2 = min(x + roi_size // 2 + 1, img.shape[1])
            y2 = min(y + roi_size // 2 + 1, img.shape[0])
            roi = img[y1:y2, x1:x2]
            mean_val = np.mean(roi)
            max_val = np.max(roi)
            area = roi.size
            if (
                min_bright < mean_val < max_bright
                and max_val < self.max_saturation
                and area > 2
            ):
                candidates.append({'centroid': (cx, cy), 'mean': mean_val, 'max': max_val, 'area': area})
        return candidates

    def select_best_star(self, candidates):
        """
        Select the best star (primary) from the candidates.
        Returns the centroid (x, y) of the best star.
        """
        if not candidates:
            self.selected_star = None
            return None
        # Example: select the candidate with the highest mean brightness
        best = max(candidates, key=lambda c: c['mean'])
        self.selected_star = best['centroid']
        return self.selected_star

    def select_secondary_stars(self, candidates, primary, max_secondary=8):
        """
        Select up to max_secondary stars as secondary guide stars, excluding the primary.
        Returns a list of centroids.
        """
        if not candidates or primary is None:
            self.secondary_stars = []
            return []
        # Exclude the primary star
        secondaries = [c['centroid'] for c in candidates if c['centroid'] != primary]
        # Sort by mean brightness, descending
        secondaries = sorted(
            secondaries,
            key=lambda c: -next(x['mean'] for x in candidates if x['centroid'] == c)
        )
        self.secondary_stars = secondaries[:max_secondary]
        return self.secondary_stars

    def is_star_lost(self, frame, centroid, threshold_sigma=5, loss_radius=10, fwhm=3.0, min_brightness=None, max_brightness=None):
        """
        Check if the star is still near the expected centroid.
        """
        img = frame.astype(np.float32)
        img = self.ensure_grayscale(img)
        candidates = self.find_candidate_stars(
            img,
            threshold_sigma=threshold_sigma,
            fwhm=fwhm,
            min_brightness=min_brightness,
            max_brightness=max_brightness
        )
        for star in candidates:
            sx, sy = star['centroid']
            if np.hypot(sx - centroid[0], sy - centroid[1]) < loss_radius:
                return False  # Star is still there
        return True  # Star lost

    def auto_reacquire(self, frame, threshold_sigma=5, fwhm=3.0, min_brightness=None, max_brightness=None):
        """
        Auto-select the best star in the current frame.
        """
        img = frame.astype(np.float32)
        img = self.ensure_grayscale(img)
        candidates = self.find_candidate_stars(
            img,
            threshold_sigma=threshold_sigma,
            fwhm=fwhm,
            min_brightness=min_brightness,
            max_brightness=max_brightness
        )
        return self.select_best_star(candidates)

    def manual_select_star(self, click_x, click_y, frame, threshold_sigma=5, max_distance=20, fwhm=3.0, min_brightness=None, max_brightness=None):
        """
        Manually select the star closest to the clicked coordinates.
        Returns the centroid (x, y) of the selected star, or None if no star within max_distance.
        """
        img = frame.astype(np.float32)
        img = self.ensure_grayscale(img)
        candidates = self.find_candidate_stars(
            img,
            threshold_sigma=threshold_sigma,
            fwhm=fwhm,
            min_brightness=min_brightness,
            max_brightness=max_brightness
        )
        if not candidates:
            return None
        centroids = [c['centroid'] for c in candidates]
        distances = [np.hypot(cx - click_x, cy - click_y) for (cx, cy) in centroids]
        min_dist = min(distances)
        if min_dist <= max_distance:
            idx = distances.index(min_dist)
            self.selected_star = centroids[idx]
            return centroids[idx]
        else:
            return None
