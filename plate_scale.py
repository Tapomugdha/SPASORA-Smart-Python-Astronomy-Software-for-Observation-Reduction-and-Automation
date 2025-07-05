def calculate_plate_scale(focal_length_mm, pixel_size_microns):
    """
    Returns plate scale in arcseconds per pixel.
    """
    return (206.265 * pixel_size_microns) / focal_length_mm
