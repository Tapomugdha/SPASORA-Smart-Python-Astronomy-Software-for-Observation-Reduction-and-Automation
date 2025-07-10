import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import cv2
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Import your modules
from camera_module import Camera
from star_detection import detect_star, detect_stars, weighted_centroid, manual_select_star
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

FOCAL_LENGTH_MM = 430
PIXEL_SIZE_MICRONS = 2.4
ROI_SIZE = 15
PLATE_SCALE_ARCSEC_PER_PIXEL = calculate_plate_scale(FOCAL_LENGTH_MM, PIXEL_SIZE_MICRONS)
IMAGE_FOLDER = r"C:\Users\Tapomugdha Mandal\Desktop\Indigenous"
API_KEY = "dsvcytsmnzxfnuwx"

def show_splash():
    splash_root = tk.Tk()
    splash_root.overrideredirect(True)
    splash_root.configure(bg="#222244")
    width, height = 500, 300
    screen_width = splash_root.winfo_screenwidth()
    screen_height = splash_root.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    splash_root.geometry(f"{width}x{height}+{x}+{y}")

    label = tk.Label(
        splash_root,
        text="SPASORA",
        font=("Helvetica", 48, "bold"),
        fg="#FFD700",
        bg="#222244"
    )
    label.pack(expand=True)

    subtitle_text = (
        "Tapo's Creation:\n"
        "Smart Python Astronomy Software for Observation\n"
        "and Realtime Automation"
    )
    subtitle = tk.Label(
        splash_root,
        text=subtitle_text,
        font=("Helvetica", 13, "italic"),
        fg="#CCCCFF",
        bg="#222244",
        wraplength=440,
        justify="center"
    )
    subtitle.pack(pady=(0, 40))

    def close_splash():
        splash_root.destroy()

    splash_root.after(2000, close_splash)
    splash_root.mainloop()

class AstroGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SPASORA - Modular Astronomy Control")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.geometry("1250x700")  # Large enough for all widgets

        # Initialize modules
        self.cam = Camera()
        self.mount = MountController()
        self.guider = GuidingController(self.mount, arcsec_per_pixel=PLATE_SCALE_ARCSEC_PER_PIXEL, p_gain=100)
        self.logger = GuidingLogger()
        self.star_manager = StarManager()
        self.dither_manager = DitherManager(max_dither_pixels=5)
        self.polar_aligner = PolarAlignmentAssistant(self.cam, detect_star, weighted_centroid, ROI_SIZE, 20)
        self.reference_pos = None
        self.dither_interval = 50
        self.frame_count = 0
        self.guiding = False

        # Store RA/Dec error histories
        self.ra_error_history = []
        self.dec_error_history = []

        # Zoom factor for live feed display
        self.zoom_factor = 1.0

        # Fullscreen flag
        self.fullscreen = False

        # Exposure time and threshold variables
        self.exposure_ms = tk.IntVar(value=100)
        self.threshold_sigma = tk.DoubleVar(value=5.0)  # Now sigma-based

        # New: FWHM, min/max brightness
        self.fwhm = tk.DoubleVar(value=5.0)
        self.min_brightness = tk.DoubleVar(value=60.0)
        self.max_brightness = tk.DoubleVar(value=250.0)

        # Looping (live view) flag
        self.looping = tk.BooleanVar(value=True)

        # For manual star selection
        self.manual_select_mode = False

        self.calibrated = False  # <-- Added for green square/circle logic

        self.create_widgets()
        self.update_frame()

    def create_widgets(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.img_label = tk.Label(self, bg="black")
        self.img_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.img_label.bind("<Button-1>", self.on_image_click)

        controls_frame = ttk.Frame(self)
        controls_frame.grid(row=0, column=1, sticky="ns", padx=5, pady=5)
        controls_frame.grid_propagate(False)
        controls_frame.config(width=320)

        row = 0

        self.looping_btn = ttk.Checkbutton(
            controls_frame, text="Looping (Live View)", variable=self.looping, command=self.toggle_looping
        )
        self.looping_btn.grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Button(controls_frame, text="Auto-Detect Star", command=self.auto_detect_star).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Button(controls_frame, text="Manual Select Star", command=self.enable_manual_select).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Button(controls_frame, text="Start Guiding", command=self.start_guiding_with_popup).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1
        ttk.Button(controls_frame, text="Stop Guiding", command=self.stop_guiding).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Button(controls_frame, text="Reset Reference", command=self.reset_reference).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Button(controls_frame, text="Calibrate", command=self.calibrate_with_popup).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1
        ttk.Button(controls_frame, text="Polar Align", command=self.polar_align_with_popup).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Button(controls_frame, text="Plate Solve", command=self.plate_solve).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1
        ttk.Button(controls_frame, text="Dither", command=self.dither).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Button(controls_frame, text="Connect Mount", command=self.connect_mount).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1
        ttk.Button(controls_frame, text="Disconnect Mount", command=self.disconnect_mount).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Button(controls_frame, text="Toggle Fullscreen", command=self.toggle_fullscreen).grid(row=row, column=0, sticky="ew", pady=10)
        row += 1
        ttk.Button(controls_frame, text="Exit", command=self.on_close).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Label(controls_frame, text="Exposure (ms):").grid(row=row, column=0, sticky="w", pady=(20, 2))
        row += 1
        exposure_slider = ttk.Scale(controls_frame, from_=50, to=5000, variable=self.exposure_ms,
                                    orient='horizontal', command=self.on_exposure_change, length=180)
        exposure_slider.grid(row=row, column=0, sticky="ew")
        row += 1
        self.exposure_label = ttk.Label(controls_frame, text=f"{self.exposure_ms.get()} ms")
        self.exposure_label.grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Label(controls_frame, text="Detection Threshold (σ):").grid(row=row, column=0, sticky="w", pady=(20, 2))
        row += 1
        threshold_slider = ttk.Scale(controls_frame, from_=2, to=10, variable=self.threshold_sigma,
                                     orient='horizontal', command=self.on_threshold_change, length=180)
        threshold_slider.grid(row=row, column=0, sticky="ew")
        row += 1
        self.threshold_label = ttk.Label(controls_frame, text=f"{self.threshold_sigma.get():.1f} σ")
        self.threshold_label.grid(row=row, column=0, sticky="ew")
        row += 1

        # FWHM slider
        ttk.Label(controls_frame, text="Star FWHM (px):").grid(row=row, column=0, sticky="w", pady=(20, 2))
        row += 1
        fwhm_slider = ttk.Scale(controls_frame, from_=2, to=10, variable=self.fwhm,
                                orient='horizontal', command=self.on_fwhm_change, length=180)
        fwhm_slider.grid(row=row, column=0, sticky="ew")
        row += 1
        self.fwhm_label = ttk.Label(controls_frame, text=f"{self.fwhm.get():.1f} px")
        self.fwhm_label.grid(row=row, column=0, sticky="ew")
        row += 1

        # Min brightness slider
        ttk.Label(controls_frame, text="Min Star Brightness:").grid(row=row, column=0, sticky="w", pady=(20, 2))
        row += 1
        min_bright_slider = ttk.Scale(controls_frame, from_=0, to=200, variable=self.min_brightness,
                                      orient='horizontal', command=self.on_min_brightness_change, length=180)
        min_bright_slider.grid(row=row, column=0, sticky="ew")
        row += 1
        self.min_brightness_label = ttk.Label(controls_frame, text=f"{self.min_brightness.get():.0f}")
        self.min_brightness_label.grid(row=row, column=0, sticky="ew")
        row += 1

        # Max brightness slider
        ttk.Label(controls_frame, text="Max Star Brightness:").grid(row=row, column=0, sticky="w", pady=(20, 2))
        row += 1
        max_bright_slider = ttk.Scale(controls_frame, from_=100, to=255, variable=self.max_brightness,
                                      orient='horizontal', command=self.on_max_brightness_change, length=180)
        max_bright_slider.grid(row=row, column=0, sticky="ew")
        row += 1
        self.max_brightness_label = ttk.Label(controls_frame, text=f"{self.max_brightness.get():.0f}")
        self.max_brightness_label.grid(row=row, column=0, sticky="ew")
        row += 1

        zoom_frame = ttk.Frame(controls_frame)
        zoom_frame.grid(row=row, column=0, pady=(20, 0), sticky="ew")
        ttk.Label(zoom_frame, text="Zoom:").pack(side="left")
        ttk.Button(zoom_frame, text="+", width=3, command=self.zoom_in).pack(side="left", padx=2)
        ttk.Button(zoom_frame, text="-", width=3, command=self.zoom_out).pack(side="left", padx=2)
        ttk.Button(zoom_frame, text="Reset", width=5, command=self.zoom_reset).pack(side="left", padx=2)
        row += 1

        self.fig, self.ax = plt.subplots(figsize=(4.5, 2.2))
        self.ra_line, = self.ax.plot([], [], label="RA Error (px)", color="tab:blue")
        self.dec_line, = self.ax.plot([], [], label="Dec Error (px)", color="tab:orange")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Error (px)")
        self.ax.legend()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=1, column=2, rowspan=15, sticky="ew", padx=5, pady=5)

    def on_exposure_change(self, val):
        value = int(float(val))
        self.exposure_label.config(text=f"{value} ms")
        self.cam.set_exposure(value * 1000)  # ms to us

    def on_threshold_change(self, val):
        value = float(val)
        self.threshold_label.config(text=f"{value:.1f} σ")

    def on_fwhm_change(self, val):
        value = float(val)
        self.fwhm_label.config(text=f"{value:.1f} px")

    def on_min_brightness_change(self, val):
        value = float(val)
        self.min_brightness_label.config(text=f"{value:.0f}")

    def on_max_brightness_change(self, val):
        value = float(val)
        self.max_brightness_label.config(text=f"{value:.0f}")

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor * 1.25, 5.0)

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor / 1.25, 0.2)

    def zoom_reset(self):
        self.zoom_factor = 1.0

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.attributes("-fullscreen", self.fullscreen)

    def toggle_looping(self):
        if not self.looping.get():
            self.after_cancel(self._after_id)
        else:
            self.update_frame()

    def auto_detect_star(self):
        frame = self.cam.get_frame()
        candidates = self.star_manager.find_candidate_stars(
            frame,
            threshold_sigma=self.threshold_sigma.get(),
            fwhm=self.fwhm.get(),
            min_brightness=self.min_brightness.get(),
            max_brightness=self.max_brightness.get()
        )
        best = self.star_manager.select_best_star(candidates)
        if best is not None:
            self.reference_pos = best
            messagebox.showinfo("Auto-Detect", "Best star auto-selected.")
        else:
            messagebox.showwarning("Auto-Detect", "No suitable star found.")

    def enable_manual_select(self):
        self.manual_select_mode = True
        messagebox.showinfo("Manual Star Selection", "Click on a star in the image to select it.")

    def on_image_click(self, event):
        if not self.manual_select_mode:
            return
        frame = self.cam.get_frame()
        height, width = frame.shape[:2]
        disp_width = int(width * self.zoom_factor)
        disp_height = int(height * self.zoom_factor)
        x = int(event.x / disp_width * width)
        y = int(event.y / disp_height * height)
        candidates = self.star_manager.find_candidate_stars(
            frame,
            threshold_sigma=self.threshold_sigma.get(),
            fwhm=self.fwhm.get(),
            min_brightness=self.min_brightness.get(),
            max_brightness=self.max_brightness.get()
        )
        selected = self.star_manager.manual_select_star(
            x, y, frame,
            threshold_sigma=self.threshold_sigma.get(),
            fwhm=self.fwhm.get(),
            min_brightness=self.min_brightness.get(),
            max_brightness=self.max_brightness.get()
        )
        if selected is not None:
            self.reference_pos = selected
            messagebox.showinfo("Manual Selection", f"Star selected at {selected}")
        else:
            messagebox.showwarning("Manual Selection", "No star found near click position.")
        self.manual_select_mode = False

    def update_frame(self):
        if not self.looping.get():
            return
        frame = self.cam.get_frame()
        if frame is None or frame.size == 0:
            frame = 128 * np.ones((self.cam.height, self.cam.width, 3), dtype=np.uint8)
            cv2.putText(frame, "No camera frame!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        threshold_value = self.threshold_sigma.get()
        fwhm_value = self.fwhm.get()
        min_brightness_value = self.min_brightness.get()
        max_brightness_value = self.max_brightness.get()
        candidates = self.star_manager.find_candidate_stars(
            frame,
            threshold_sigma=threshold_value,
            fwhm=fwhm_value,
            min_brightness=min_brightness_value,
            max_brightness=max_brightness_value
        )
        if self.reference_pos is None or self.star_manager.is_star_lost(
            frame, self.reference_pos,
            threshold_sigma=threshold_value,
            fwhm=fwhm_value,
            min_brightness=min_brightness_value,
            max_brightness=max_brightness_value
        ):
            self.reference_pos = self.star_manager.select_best_star(candidates)
            self.dither_manager.reset()

        refined = weighted_centroid(frame, self.reference_pos, roi_size=ROI_SIZE, noise_threshold=20)
        if refined:
            self.frame_count += 1
            dx = refined[0] - self.reference_pos[0]
            dy = refined[1] - self.reference_pos[1]
            dx_arcsec = dx * PLATE_SCALE_ARCSEC_PER_PIXEL
            dy_arcsec = dy * PLATE_SCALE_ARCSEC_PER_PIXEL
            cv2.circle(frame, (int(round(refined[0])), int(round(refined[1]))), 4, (0, 0, 255), 2)

            # --- Green circle before calibration, green square after calibration ---
            ref_x, ref_y = int(round(self.reference_pos[0])), int(round(self.reference_pos[1]))
            if self.calibrated:
                size = 8
                top_left = (ref_x - size, ref_y - size)
                bottom_right = (ref_x + size, ref_y + size)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            else:
                cv2.circle(frame, (ref_x, ref_y), 6, (0, 255, 0), 2)
            # ----------------------------------------------------------------------

            cv2.putText(frame, f"Pixel Offset: ({dx:.2f}, {dy:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Angular Offset: ({dx_arcsec:.2f}\", {dy_arcsec:.2f}\")", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if self.guiding:
                self.guider.guide(dx, dy)
                self.logger.log(dx, dy, dx_arcsec, dy_arcsec, 0, 0)
                self.ra_error_history.append(dx)
                self.dec_error_history.append(dy)
                if len(self.ra_error_history) > 100:
                    self.ra_error_history.pop(0)
                if len(self.dec_error_history) > 100:
                    self.dec_error_history.pop(0)
                self.update_plot()

            try:
                fwhm = measure_fwhm(frame, refined, roi_size=ROI_SIZE)
            except Exception:
                fwhm = None
            try:
                snr = measure_snr(frame, refined, roi_size=ROI_SIZE)
            except Exception:
                snr = None

            if fwhm is not None and snr is not None:
                cv2.putText(frame, f"FWHM: {fwhm:.2f} px, SNR: {snr:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "FWHM: ---, SNR: ---", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No star detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "FWHM: ---, SNR: ---", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        height, width = frame.shape[:2]
        new_width = int(width * self.zoom_factor)
        new_height = int(height * self.zoom_factor)
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.img_label.imgtk = imgtk
        self.img_label.configure(image=imgtk)

        self._after_id = self.after(30, self.update_frame)

    def update_plot(self):
        x_vals = range(len(self.ra_error_history))
        self.ra_line.set_data(x_vals, self.ra_error_history)
        self.dec_line.set_data(x_vals, self.dec_error_history)
        self.ax.set_xlim(0, max(100, len(self.ra_error_history)))
        all_errors = self.ra_error_history + self.dec_error_history
        y_max = max(1, max([abs(e) for e in all_errors]) if all_errors else 1)
        self.ax.set_ylim(-y_max, y_max)
        self.canvas.draw()

    def start_guiding_with_popup(self):
        messagebox.showinfo("Guiding", "Guiding started.")
        self.start_guiding()
        messagebox.showinfo("Guiding", "Guiding complete.")

    def start_guiding(self):
        self.guiding = True

    def stop_guiding(self):
        self.guiding = False

    def reset_reference(self):
        self.reference_pos = None
        self.calibrated = False  # <-- Reset calibration state

    def calibrate_with_popup(self):
        messagebox.showinfo("Calibration", "Calibration started.")
        threading.Thread(target=self._calibrate_thread, daemon=True).start()

    def _calibrate_thread(self):
        calibration = CalibrationRoutine(self.mount, self.cam, detect_star, weighted_centroid, ROI_SIZE, 20)
        calibration.full_calibration()
        self.calibrated = True  # <-- Set calibration state after calibration
        messagebox.showinfo("Calibration", "Calibration complete.")

    def polar_align_with_popup(self):
        messagebox.showinfo("Polar Alignment", "Polar alignment started.")
        threading.Thread(target=self._polar_align_thread, daemon=True).start()

    def _polar_align_thread(self):
        times, drifts = self.polar_aligner.measure_drift(duration_sec=120, interval_sec=5)
        self.polar_aligner.analyze_drift(times, drifts, PLATE_SCALE_ARCSEC_PER_PIXEL)
        messagebox.showinfo("Polar Alignment", "Polar alignment drift test complete.")

    def plate_solve(self):
        threading.Thread(target=self._plate_solve_thread, daemon=True).start()

    def _plate_solve_thread(self):
        image_path = os.path.join(IMAGE_FOLDER, f"plate_solve_frame_{self.frame_count:04d}.png")
        frame = self.cam.get_frame()
        cv2.imwrite(image_path, frame)
        result = plate_solve_with_astrometrynet(image_path, API_KEY)
        if result:
            messagebox.showinfo("Plate Solving", "Plate solving successful.")
        else:
            messagebox.showwarning("Plate Solving", "Plate solving failed.")

    def dither(self):
        self.dither_manager.random_dither()
        self.reference_pos = self.dither_manager.apply_dither(self.reference_pos)

    def connect_mount(self):
        try:
            self.mount = MountController()
            messagebox.showinfo("Mount", "Mount connected.")
        except Exception as e:
            messagebox.showerror("Mount", f"Failed to connect: {e}")

    def disconnect_mount(self):
        try:
            self.mount.disconnect()
            messagebox.showinfo("Mount", "Mount disconnected.")
        except Exception as e:
            messagebox.showwarning("Mount", f"Could not disconnect mount: {e}")

    def on_close(self):
        self.stop_guiding()
        try:
            self.cam.close()
        except Exception:
            pass
        try:
            self.mount.disconnect()
        except Exception:
            pass
        try:
            self.logger.close()
        except Exception:
            pass
        self.destroy()

if __name__ == "__main__":
    show_splash()
    app = AstroGUI()
    app.mainloop()
