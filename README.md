# SPASORA-Smart-Python-Astronomy-Software-for-Observation-Reduction-and-Automation-
Modular, open-source platform for astronomical imaging, guiding, and automation. Designed for both amateur and professional astronomers, SPASORA provides a robust, user-friendly interface for controlling cameras, mounts, and guiding systems, with advanced features for star detection, calibration, logging, and more.

## Features
Live Camera Feed: Real-time display and control of your astronomy camera.

Star Detection & Centroiding: Automatic and manual star selection with robust centroid algorithms.

Guiding Control: High-precision guiding with real-time error tracking in RA and Dec.

Calibration: Automated calibration routines for accurate mount response.

Dithering: Random and programmable dithering for improved image quality.

Plate Solving: Integration with Astrometry.net for field recognition and plate scale calculation.

Polar Alignment Assistant: Drift analysis and visual feedback for accurate polar alignment.

Seeing Quality Metrics: Real-time FWHM and SNR measurements for image quality assessment.

Comprehensive Logging: Detailed guiding logs with pixel and arcsecond errors.

User-Friendly GUI: Intuitive interface built with Tkinter and OpenCV.

Modular Design: Easily extend or replace modules for camera, mount, star detection, and more.

## Prerequisites
Python 3.8+
Required packages: opencv-python, numpy, matplotlib, Pillow, tkinter, and others as specified in requirements.txt
Compatible camera and mount hardware (see documentation for supported devices)

## Installation
bash
git clone https://github.com/yourusername/spasora.git
cd spasora
pip install -r requirements.txt

## Running SPASORA
bash
python gui.py

## GUI State Visualization

| State             | Reference Star Shape | Color |
|-------------------|---------------------|-------|
| Pre-calibration   | Circle              | Green |
| Post-calibration  | Square              | Green |
| Guiding Active    | Square              | Green |
| Current Star      | Circle              | Red   |


## Modular Functions
spasora/
├── gui.py                # Main GUI application
├── camera_module.py      # Camera interface
├── star_detection.py     # Star detection and centroiding
├── mount_control.py      # Mount control logic
├── guiding_control.py    # Guiding algorithms
├── calibration.py        # Calibration routines
├── logging_utils.py      # Logging and data export
├── star_management.py    # Star selection and management
├── dithering.py          # Dithering logic
├── polar_alignment.py    # Polar alignment assistant
├── plate_scale.py        # Plate scale calculations
├── plate_solving.py      # Plate solving integration
├── seeing_quality.py     # FWHM and SNR measurement
├── requirements.txt      # Python dependencies
└── README.md             # This file

## Acknowledgments
OpenCV, NumPy, Matplotlib, Pillow, and the Python scientific community,
Astrometry.net for plate solving integration,
All contributors and testers

Happy observing!
