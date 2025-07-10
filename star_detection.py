import numpy as np
import cv2

# Try to import photutils for robust star detection
try:
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats
    PHOTUTILS_AVAILABLE = True
except ImportError:
    PHOTUTILS_AVAILABLE = False

def ensure_grayscale(img):
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

def detect_stars(frame, threshold_sigma=5, fwhm=3.0):
    """
    Detect all stars in an image.

    Parameters:
        frame: 2D or 3D numpy array (image)
        threshold_sigma: Detection threshold in sigma above background (e.g. 5 for 5-sigma)
        fwhm: FWHM for DAOStarFinder (in pixels)

    Returns:
        List of centroids [(x, y), ...]
    """
    img = frame.astype(np.float32)
    img = ensure_grayscale(img)

    centroids = []

    if PHOTUTILS_AVAILABLE:
        # Estimate background statistics
        mean, median, std = sigma_clipped_stats(img, sigma=3.0)
        abs_threshold = threshold_sigma * std
        daofind = DAOStarFinder(fwhm=fwhm, threshold=abs_threshold)
        sources = daofind(img - median)
        if sources is not None and len(sources) > 0:
            for row in sources:
                centroids.append((row['xcentroid'], row['ycentroid']))
    else:
        # Fallback: OpenCV-based detection
        blurred = cv2.medianBlur(img.astype(np.uint8), 3)
        thresh_val = int((threshold_sigma / 10.0) * img.max())  # GUI slider: 10 ~ 10% of max
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))
    return centroids

def detect_star(frame, threshold_sigma=5, fwhm=3.0):
    """
    Detect the brightest star in an image.

    Parameters:
        frame: 2D or 3D numpy array (image)
        threshold_sigma: Detection threshold in sigma above background (e.g. 5 for 5-sigma)
        fwhm: FWHM for DAOStarFinder (in pixels)

    Returns:
        (x, y) centroid of the brightest star, or None if not found.
    """
    img = frame.astype(np.float32)
    img = ensure_grayscale(img)

    if PHOTUTILS_AVAILABLE:
        mean, median, std = sigma_clipped_stats(img, sigma=3.0)
        abs_threshold = threshold_sigma * std
        daofind = DAOStarFinder(fwhm=fwhm, threshold=abs_threshold)
        sources = daofind(img - median)
        if sources is not None and len(sources) > 0:
            brightest_idx = np.argmax(sources['flux'])
            return (sources['xcentroid'][brightest_idx], sources['ycentroid'][brightest_idx])
        else:
            return None
    else:
        blurred = cv2.medianBlur(img.astype(np.uint8), 3)
        thresh_val = int((threshold_sigma / 10.0) * img.max())
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        star_centroid = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    star_centroid = (cx, cy)
                    max_area = area
        return star_centroid

def weighted_centroid(frame, centroid, roi_size=15, noise_threshold=20):
    if centroid is None:
        return None
    cx, cy = centroid
    x1 = int(max(cx - roi_size // 2, 0))
    y1 = int(max(cy - roi_size // 2, 0))
    x2 = int(min(cx + roi_size // 2 + 1, frame.shape[1]))
    y2 = int(min(cy + roi_size // 2 + 1, frame.shape[0]))
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    roi = roi.squeeze()
    roi_height, roi_width = roi.shape
    ys, xs = np.mgrid[0:roi_height, 0:roi_width]
    abs_xs = xs + x1
    abs_ys = ys + y1
    mask = roi > noise_threshold
    weights = roi[mask] - noise_threshold
    xs_roi = abs_xs[mask]
    ys_roi = abs_ys[mask]
    if np.sum(weights) > 0:
        cx_weighted = np.sum(xs_roi * weights) / np.sum(weights)
        cy_weighted = np.sum(ys_roi * weights) / np.sum(weights)
        return (cx_weighted, cy_weighted)
    else:
        return centroid

def manual_select_star(click_x, click_y, candidates, max_distance=20):
    """
    Select the star closest to the clicked coordinates.

    Parameters:
        click_x, click_y: Coordinates of the mouse click.
        candidates: List of (x, y) star centroids.
        max_distance: Maximum distance in pixels to consider a valid selection.

    Returns:
        (x, y) of the selected star, or None if no star within max_distance.
    """
    if not candidates:
        return None
    distances = [np.hypot(cx - click_x, cy - click_y) for (cx, cy) in candidates]
    min_dist = min(distances)
    if min_dist <= max_distance:
        idx = distances.index(min_dist)
        return candidates[idx]
    else:
        return None
