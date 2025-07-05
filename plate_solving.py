from astroquery.astrometry_net import AstrometryNet

def plate_solve_with_astrometrynet(image_path, api_key):
    ast = AstrometryNet()
    ast.api_key = api_key
    wcs_header = ast.solve_from_image(image_path)
    if wcs_header:
        print("Plate solving successful.")
        return wcs_header
    else:
        print("Plate solving failed.")
        return None

