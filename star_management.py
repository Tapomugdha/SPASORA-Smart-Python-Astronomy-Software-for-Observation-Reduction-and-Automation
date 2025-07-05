import cv2
import numpy as np

class StarManager:
    def __init__(self, min_brightness=60, max_brightness=250, max_saturation=245):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_saturation = max_saturation
        self.selected_star = None
        self.secondary_stars = []

    def find_candidate_stars(self, frame, threshold_percent=10):
        """
        Find candidate stars in the given frame using Otsu's thresholding.
        Returns a list of dicts: {'centroid': (x, y), 'mean': ..., 'max': ..., 'area': ...}
        """
        img = frame.astype(np.float32)
        # Only convert to grayscale if needed (i.e., if image has 3 channels)
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(img.astype(np.uint8), 3)
        # Use Otsu's thresholding for robust star detection
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            mask = np.zeros(img.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_val = cv2.mean(img, mask=mask)[0]
            max_val = np.max(img[mask == 255])
            area = cv2.contourArea(cnt)
            if (
                self.min_brightness < mean_val < self.max_brightness
                and max_val < self.max_saturation
                and area > 2
            ):
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
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
        secondaries = sorted(secondaries, key=lambda c: -next(x['mean'] for x in candidates if x['centroid'] == c))
        self.secondary_stars = secondaries[:max_secondary]
        return self.secondary_stars

    def is_star_lost(self, frame, centroid, threshold_percent=10, loss_radius=10):
        """
        Check if the star is still near the expected centroid.
        """
        candidates = self.find_candidate_stars(frame, threshold_percent=threshold_percent)
        for star in candidates:
            sx, sy = star['centroid']
            if np.hypot(sx - centroid[0], sy - centroid[1]) < loss_radius:
                return False  # Star is still there
        return True  # Star lost

    def auto_reacquire(self, frame, threshold_percent=10):
        """
        Auto-select the best star in the current frame.
        """
        candidates = self.find_candidate_stars(frame, threshold_percent=threshold_percent)
        return self.select_best_star(candidates)

    def manual_select_star(self, click_x, click_y, frame, threshold_percent=10, max_distance=20):
        """
        Manually select the star closest to the clicked coordinates.
        Returns the centroid (x, y) of the selected star, or None if no star within max_distance.
        """
        candidates = self.find_candidate_stars(frame, threshold_percent=threshold_percent)
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
