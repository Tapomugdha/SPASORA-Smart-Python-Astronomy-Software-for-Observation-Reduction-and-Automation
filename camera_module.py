import pyPOACamera
import numpy as np
import time

class Camera:
    def __init__(
        self,
        width=512,
        height=512,
        exposure_us=1000000,  # 1s default, typical for PHD2
        gain=250,             # Typical PHD2 gain for Player One
        color_mode='RAW8',    # 'RAW8' for mono, 'RGB24' for color
        binning=1
    ):
        self.width = width
        self.height = height
        self.exposure_us = exposure_us
        self.gain = gain
        self.color_mode = color_mode
        self.binning = binning
        self.camera_id = None
        self.bufArray = None
        self.imgFormat = None
        self.imgSize = None
        self._init_camera()

    def _init_camera(self):
        count = pyPOACamera.GetCameraCount()
        if count == 0:
            raise RuntimeError("No camera detected.")
        err, props = pyPOACamera.GetCameraProperties(0)
        self.camera_id = props.cameraID
        pyPOACamera.OpenCamera(self.camera_id)
        pyPOACamera.InitCamera(self.camera_id)

        # Set image format as PHD2 would (RAW8 for mono, RGB24 for color)
        if self.color_mode.upper() == 'RGB24':
            fmt = pyPOACamera.POAImgFormat.POA_RGB24
        else:
            fmt = pyPOACamera.POAImgFormat.POA_RAW8
        pyPOACamera.SetImageFormat(self.camera_id, fmt)

        # Set binning (PHD2 default is 1x1)
        pyPOACamera.SetImageBin(self.camera_id, self.binning)

        # Set image size
        pyPOACamera.SetImageSize(self.camera_id, self.width, self.height)

        # Set exposure and gain
        pyPOACamera.SetExp(self.camera_id, self.exposure_us, False)
        pyPOACamera.SetGain(self.camera_id, self.gain, True)

        self.imgFormat = pyPOACamera.GetImageFormat(self.camera_id)[1]
        self.imgSize = pyPOACamera.ImageCalcSize(self.height, self.width, self.imgFormat)
        self.bufArray = np.zeros(self.imgSize, dtype=np.uint8)

        pyPOACamera.StartExposure(self.camera_id, False)

    def set_exposure(self, exposure_us):
        """Set a new exposure time in microseconds and apply it to the camera."""
        self.exposure_us = int(exposure_us)
        pyPOACamera.SetExp(self.camera_id, self.exposure_us, False)
        pyPOACamera.StopExposure(self.camera_id)
        pyPOACamera.StartExposure(self.camera_id, False)

    def set_gain(self, gain):
        """Set a new gain value and apply it to the camera."""
        self.gain = int(gain)
        pyPOACamera.SetGain(self.camera_id, self.gain, True)

    def set_binning(self, binning):
        """Set binning (e.g., 1 for 1x1, 2 for 2x2) and re-initialize camera."""
        self.binning = int(binning)
        pyPOACamera.StopExposure(self.camera_id)
        pyPOACamera.SetImageBin(self.camera_id, self.binning)
        pyPOACamera.SetImageSize(self.camera_id, self.width, self.height)
        self.imgSize = pyPOACamera.ImageCalcSize(self.height, self.width, self.imgFormat)
        self.bufArray = np.zeros(self.imgSize, dtype=np.uint8)
        pyPOACamera.StartExposure(self.camera_id, False)

    def set_color_mode(self, color_mode):
        """Set color mode ('RAW8' or 'RGB24') and re-initialize camera."""
        self.color_mode = color_mode
        pyPOACamera.StopExposure(self.camera_id)
        if self.color_mode.upper() == 'RGB24':
            fmt = pyPOACamera.POAImgFormat.POA_RGB24
        else:
            fmt = pyPOACamera.POAImgFormat.POA_RAW8
        pyPOACamera.SetImageFormat(self.camera_id, fmt)
        self.imgFormat = pyPOACamera.GetImageFormat(self.camera_id)[1]
        self.imgSize = pyPOACamera.ImageCalcSize(self.height, self.width, self.imgFormat)
        self.bufArray = np.zeros(self.imgSize, dtype=np.uint8)
        pyPOACamera.StartExposure(self.camera_id, False)

    def get_frame(self, timeout=10):
        """Get a frame, with timeout in seconds (as in PHD2)."""
        start_time = time.time()
        while True:
            _, ready = pyPOACamera.ImageReady(self.camera_id)
            if ready:
                break
            if (time.time() - start_time) > timeout:
                raise RuntimeError("Timeout waiting for camera frame.")
            time.sleep(0.01)
        pyPOACamera.GetImageData(self.camera_id, self.bufArray, 1000)
        img = pyPOACamera.ImageDataConvert(self.bufArray, self.height, self.width, self.imgFormat)
        return img.copy()

    def close(self):
        pyPOACamera.StopExposure(self.camera_id)
        pyPOACamera.CloseCamera(self.camera_id)
