# logging_utils.py
import csv
import matplotlib.pyplot as plt

class GuidingLogger:
    def __init__(self, filename="guiding_log.csv"):
        self.filename = filename
        self.file = open(self.filename, "w", newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['frame', 'dx_pixels', 'dy_pixels', 'dx_arcsec', 'dy_arcsec', 'pulse_ra_ms', 'pulse_dec_ms'])
        self.frame_count = 0
        # For plotting
        self.dx_history = []
        self.dy_history = []

    def log(self, dx_pixels, dy_pixels, dx_arcsec, dy_arcsec, pulse_ra_ms, pulse_dec_ms):
        self.writer.writerow([self.frame_count, dx_pixels, dy_pixels, dx_arcsec, dy_arcsec, pulse_ra_ms, pulse_dec_ms])
        self.dx_history.append(dx_arcsec)
        self.dy_history.append(dy_arcsec)
        self.frame_count += 1

    def close(self):
        self.file.close()

    def plot(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.dx_history, label='RA Error (arcsec)')
        plt.plot(self.dy_history, label='DEC Error (arcsec)')
        plt.xlabel('Frame')
        plt.ylabel('Guiding Error (arcsec)')
        plt.title('Guiding Error Over Time')
        plt.legend()
        plt.grid()
        plt.show()
