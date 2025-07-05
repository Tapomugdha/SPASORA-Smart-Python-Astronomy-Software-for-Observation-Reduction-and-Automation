# guiding_control.py

class GuidingController:
    def __init__(self, mount, arcsec_per_pixel, p_gain=100):
        self.mount = mount
        self.arcsec_per_pixel = arcsec_per_pixel
        self.p_gain = p_gain  # ms per arcsec, tune as needed

    def guide(self, dx_pixels, dy_pixels):
        dx_arcsec = dx_pixels * self.arcsec_per_pixel
        dy_arcsec = dy_pixels * self.arcsec_per_pixel

        # Calculate pulse durations (ms), tune scaling for your mount
        pulse_ra = int(abs(dx_arcsec) * self.p_gain)
        pulse_dec = int(abs(dy_arcsec) * self.p_gain)

        # Only send pulse if offset is significant
        if pulse_ra > 0:
            if dx_arcsec > 0:
                self.mount.pulse_guide('west', pulse_ra)
            else:
                self.mount.pulse_guide('east', pulse_ra)
        if pulse_dec > 0:
            if dy_arcsec > 0:
                self.mount.pulse_guide('south', pulse_dec)
            else:
                self.mount.pulse_guide('north', pulse_dec)
