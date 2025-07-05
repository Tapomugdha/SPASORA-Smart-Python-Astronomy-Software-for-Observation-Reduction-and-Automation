# calibration.py
import time
import json

class CalibrationRoutine:
    def __init__(self, mount, camera, detect_star, weighted_centroid, roi_size, noise_threshold):
        self.mount = mount
        self.camera = camera
        self.detect_star = detect_star
        self.weighted_centroid = weighted_centroid
        self.roi_size = roi_size
        self.noise_threshold = noise_threshold

    def calibrate_axis(self, direction, pulse_ms=1000, settle_time=2.0):
        print(f"Calibrating {direction} axis with {pulse_ms} ms pulse...")
        frame = self.camera.get_frame()
        star = self.detect_star(frame)
        ref_centroid = self.weighted_centroid(frame, star, self.roi_size, self.noise_threshold)
        # Send pulse
        self.mount.pulse_guide(direction, pulse_ms)
        time.sleep(settle_time)  # Wait for mount to settle
        frame2 = self.camera.get_frame()
        star2 = self.detect_star(frame2)
        new_centroid = self.weighted_centroid(frame2, star2, self.roi_size, self.noise_threshold)
        if ref_centroid and new_centroid:
            dx = new_centroid[0] - ref_centroid[0]
            dy = new_centroid[1] - ref_centroid[1]
            print(f"Star moved: dx={dx:.2f}, dy={dy:.2f} pixels")
            return dx, dy
        else:
            print("Calibration failed: could not detect star.")
            return None, None

    def full_calibration(self, pulse_ms=1000, settle_time=2.0):
        results = {}
        for direction in ['east', 'west', 'north', 'south']:
            dx, dy = self.calibrate_axis(direction, pulse_ms, settle_time)
            results[direction] = {'dx': dx, 'dy': dy}
        # Save calibration data
        with open("calibration_data1.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Calibration complete. Results saved to calibration_data1.json.")
        return results
