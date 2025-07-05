import os
import cv2
import numpy as np

from camera_module import Camera
from star_detection import detect_star, weighted_centroid
from plate_scale import calculate_plate_scale
from mount_control import MountController
from guiding_control import GuidingController
from calibration import CalibrationRoutine
from logging_utils import GuidingLogger
from star_management import StarManager
from dithering import DitherManager
from polar_alignment import PolarAlignmentAssistant
from plate_solving import plate_solve_with_astrometrynet
from seeing_quality import measure_fwhm, measure_snr

# --- Configuration ---
FOCAL_LENGTH_MM = 430
PIXEL_SIZE_MICRONS = 2.4
THRESHOLD = 50
NOISE_THRESHOLD = 20
ROI_SIZE = 15
PLATE_SCALE_ARCSEC_PER_PIXEL = calculate_plate_scale(FOCAL_LENGTH_MM, PIXEL_SIZE_MICRONS)
IMAGE_FOLDER = r"C:\Users\Tapomugdha Mandal\Desktop\Indigenous"
API_KEY = "dsvcytsmnzxfnuwx"

# --- Initialize modules ---
os.makedirs(IMAGE_FOLDER, exist_ok=True)
cam = Camera()
mount = MountController()
guider = GuidingController(mount, arcsec_per_pixel=PLATE_SCALE_ARCSEC_PER_PIXEL, p_gain=100)
logger = GuidingLogger()
star_manager = StarManager()
dither_manager = DitherManager(max_dither_pixels=5)
polar_aligner = PolarAlignmentAssistant(cam, detect_star, weighted_centroid, ROI_SIZE, NOISE_THRESHOLD)
reference_pos = None
dither_interval = 50
frame_count = 0

window_title = "Star Detection & Guiding (Modular)"
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_title, cam.width, cam.height)

print("Press 'q' to quit, 'r' to reset reference star, 'c' to calibrate, 'p' for polar alignment, 's' to plate solve, 'd' to dither.")

while True:
    frame = cam.get_frame()
    if frame is None or frame.size == 0:
        print("Warning: Blank or invalid frame received from camera.")
        # Show a gray frame to indicate error
        frame = (128 * np.ones((cam.height, cam.width, 3), dtype=np.uint8)
                 if hasattr(cam, 'height') and hasattr(cam, 'width')
                 else 128 * np.ones((480, 640, 3), dtype=np.uint8))
        cv2.putText(frame, "No camera frame!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(window_title, frame)
        key = cv2.waitKey(100)
        if key == ord('q'):
            print('Exiting...')
            break
        continue

    candidates = star_manager.find_candidate_stars(frame, threshold=THRESHOLD)
    if reference_pos is None or star_manager.is_star_lost(frame, reference_pos):
        reference_pos = star_manager.select_best_star(candidates)
        dither_manager.reset()

    refined = weighted_centroid(frame, reference_pos, roi_size=ROI_SIZE, noise_threshold=NOISE_THRESHOLD)
    if refined:
        frame_count += 1
        # Dithering
        if frame_count % dither_interval == 0:
            dither_manager.random_dither()
            reference_pos = dither_manager.apply_dither(reference_pos)

        dx = refined[0] - reference_pos[0]
        dy = refined[1] - reference_pos[1]
        dx_arcsec = dx * PLATE_SCALE_ARCSEC_PER_PIXEL
        dy_arcsec = dy * PLATE_SCALE_ARCSEC_PER_PIXEL

        # Draw overlays
        cv2.circle(frame, (int(round(refined[0])), int(round(refined[1]))), 4, (0, 0, 255), 2)
        cv2.circle(frame, (int(round(reference_pos[0])), int(round(reference_pos[1]))), 6, (0, 255, 0), 2)
        cv2.putText(frame, f"Pixel Offset: ({dx:.2f}, {dy:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Angular Offset: ({dx_arcsec:.2f}\", {dy_arcsec:.2f}\")", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Guiding
        guider.guide(dx, dy)

        # Logging
        logger.log(dx, dy, dx_arcsec, dy_arcsec, 0, 0)  # You can add actual pulse durations if desired

        # Seeing/Quality metrics
        try:
            fwhm = measure_fwhm(frame, refined, roi_size=ROI_SIZE)
        except Exception as e:
            print(f"FWHM calculation failed: {e}")
            fwhm = None
        try:
            snr = measure_snr(frame, refined, roi_size=ROI_SIZE)
        except Exception as e:
            print(f"SNR calculation failed: {e}")
            snr = None

        if fwhm is not None and snr is not None:
            cv2.putText(frame, f"FWHM: {fwhm:.2f} px, SNR: {snr:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "FWHM: ---, SNR: ---", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "No star detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "FWHM: ---, SNR: ---", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow(window_title, frame)
    key = cv2.waitKey(10)
    if key == ord('q'):
        print('Exiting...')
        break
    elif key == ord('r'):
        reference_pos = None
    elif key == ord('c'):
        print("Starting calibration...")
        calibration = CalibrationRoutine(mount, cam, detect_star, weighted_centroid, ROI_SIZE, NOISE_THRESHOLD)
        calibration.full_calibration()
    elif key == ord('p'):
        print("Starting polar alignment drift test...")
        times, drifts = polar_aligner.measure_drift(duration_sec=120, interval_sec=5)
        polar_aligner.analyze_drift(times, drifts, PLATE_SCALE_ARCSEC_PER_PIXEL)
    elif key == ord('s'):
        print("Starting plate solving...")
        image_path = os.path.join(IMAGE_FOLDER, f"plate_solve_frame_{frame_count:04d}.png")
        os.makedirs(IMAGE_FOLDER, exist_ok=True)
        success = cv2.imwrite(image_path, frame)
        if success:
            print(f"Saved image: {image_path}")
            result = plate_solve_with_astrometrynet(image_path, API_KEY)
        else:
            print("Failed to save image for plate solving.")
    elif key == ord('d'):
        print("Manual dither triggered.")
        dither_manager.random_dither()
        reference_pos = dither_manager.apply_dither(reference_pos)

try:
    cam.close()
except Exception:
    pass
try:
    mount.disconnect()
except Exception as e:
    print(f"Warning: Could not disconnect mount cleanly: {e}")
try:
    logger.close()
except Exception:
    pass
cv2.destroyAllWindows()
