# polar_alignment.py
import time
import numpy as np

class PolarAlignmentAssistant:
    def __init__(self, camera, detect_star, weighted_centroid, roi_size, noise_threshold):
        self.camera = camera
        self.detect_star = detect_star
        self.weighted_centroid = weighted_centroid
        self.roi_size = roi_size
        self.noise_threshold = noise_threshold

    def measure_drift(self, duration_sec=120, interval_sec=5):
        print("Starting drift measurement...")
        frame = self.camera.get_frame()
        star = self.detect_star(frame)
        ref_centroid = self.weighted_centroid(frame, star, self.roi_size, self.noise_threshold)
        drifts = []
        times = []
        t0 = time.time()
        while time.time() - t0 < duration_sec:
            time.sleep(interval_sec)
            frame2 = self.camera.get_frame()
            star2 = self.detect_star(frame2)
            centroid2 = self.weighted_centroid(frame2, star2, self.roi_size, self.noise_threshold)
            if centroid2:
                dx = centroid2[0] - ref_centroid[0]
                dy = centroid2[1] - ref_centroid[1]
                drifts.append((dx, dy))
                times.append(time.time() - t0)
                print(f"Time: {times[-1]:.1f}s, Drift: dx={dx:.2f}, dy={dy:.2f} pixels")
        return times, drifts

    def analyze_drift(self, times, drifts, plate_scale):
        # Fit a line to dx and dy over time to get drift rates (pixels/sec)
        if not drifts:
            print("No drift data collected.")
            return
        dxs, dys = zip(*drifts)
        dx_rate = np.polyfit(times, dxs, 1)[0]  # pixels/sec
        dy_rate = np.polyfit(times, dys, 1)[0]
        print(f"Drift rate: RA={dx_rate*plate_scale:.3f} arcsec/sec, DEC={dy_rate*plate_scale:.3f} arcsec/sec")
        return dx_rate * plate_scale, dy_rate * plate_scale
