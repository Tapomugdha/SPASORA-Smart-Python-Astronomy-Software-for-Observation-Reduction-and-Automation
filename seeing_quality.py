# seeing_quality.py
import numpy as np
from scipy.optimize import curve_fit

def gaussian_2d(xy_tuple, amp, xo, yo, sigma_x, sigma_y, offset):
    (x, y) = xy_tuple
    g = offset + amp * np.exp(
        -(((x - xo) ** 2) / (2 * sigma_x ** 2) + ((y - yo) ** 2) / (2 * sigma_y ** 2))
    )
    return g.ravel()

def measure_fwhm(frame, centroid, roi_size=15):
    cx, cy = int(round(centroid[0])), int(round(centroid[1]))
    x1 = max(cx - roi_size // 2, 0)
    y1 = max(cy - roi_size // 2, 0)
    x2 = min(cx + roi_size // 2 + 1, frame.shape[1])
    y2 = min(cy + roi_size // 2 + 1, frame.shape[0])
    roi = frame[y1:y2, x1:x2]
    y, x = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    initial_guess = (roi.max() - roi.min(), roi.shape[1]/2, roi.shape[0]/2, 2, 2, roi.min())
    try:
        popt, _ = curve_fit(gaussian_2d, (x, y), roi.ravel(), p0=initial_guess)
        sigma_x, sigma_y = popt[3], popt[4]
        fwhm_x = 2.3548 * sigma_x
        fwhm_y = 2.3548 * sigma_y
        fwhm = (fwhm_x + fwhm_y) / 2
        return fwhm
    except Exception as e:
        print("FWHM fitting failed:", e)
        return None

def measure_snr(frame, centroid, roi_size=15):
    cx, cy = int(round(centroid[0])), int(round(centroid[1]))
    x1 = max(cx - roi_size // 2, 0)
    y1 = max(cy - roi_size // 2, 0)
    x2 = min(cx + roi_size // 2 + 1, frame.shape[1])
    y2 = min(cy + roi_size // 2 + 1, frame.shape[0])
    roi = frame[y1:y2, x1:x2]
    signal = roi.max()
    background = np.median(roi)
    noise = np.std(roi)
    snr = (signal - background) / (noise if noise > 0 else 1)
    return snr
