# mount_control.py
import win32com.client

class MountController:
    """
    ASCOM-compliant mount controller for telescope guiding and slewing.
    Uses COM interface (native ASCOM driver, as PHD2 does).
    """

    DIR_MAP = {'north': 0, 'south': 1, 'east': 2, 'west': 3}

    def __init__(self, driver_id=None):
        """
        Initialize and connect to the mount via ASCOM.
        If driver_id is None, show the ASCOM chooser dialog.
        """
        self.telescope = None
        if driver_id is None:
            chooser = win32com.client.Dispatch("ASCOM.Utilities.Chooser")
            chooser.DeviceType = 'Telescope'
            driver_id = chooser.Choose(None)
            if not driver_id:
                raise RuntimeError("No ASCOM telescope driver selected.")
        self.telescope = win32com.client.Dispatch(driver_id)
        self.telescope.Connected = True
        self.telescope.Tracking = True

    @property
    def connected(self):
        return self.telescope.Connected

    @property
    def tracking(self):
        return self.telescope.Tracking

    def set_tracking(self, tracking=True):
        self.telescope.Tracking = tracking

    def pulse_guide(self, direction, duration_ms):
        """
        Pulse guide in the specified direction for the given duration (ms).
        direction: 'north', 'south', 'east', 'west'
        """
        if direction not in self.DIR_MAP:
            raise ValueError(f"Invalid direction '{direction}'. Must be one of {list(self.DIR_MAP.keys())}.")
        if not self.connected:
            raise RuntimeError("Mount not connected.")
        self.telescope.PulseGuide(self.DIR_MAP[direction], duration_ms)

    def slew_to_coordinates(self, ra, dec):
        """
        Slew the mount to the given RA/Dec coordinates (in hours and degrees).
        """
        if not self.connected:
            raise RuntimeError("Mount not connected.")
        self.telescope.SlewToCoordinates(ra, dec)

    def park(self):
        """
        Park the telescope (if supported).
        """
        if not self.connected:
            raise RuntimeError("Mount not connected.")
        self.telescope.Park()

    def unpark(self):
        """
        Unpark the telescope (if supported).
        """
        if not self.connected:
            raise RuntimeError("Mount not connected.")
        self.telescope.Unpark()

    def get_position(self):
        """
        Return the current (RA, Dec) in hours and degrees.
        """
        if not self.connected:
            raise RuntimeError("Mount not connected.")
        return self.telescope.RightAscension, self.telescope.Declination

    def disconnect(self):
        """
        Disconnect from the mount safely.
        """
        if self.telescope is not None and self.telescope.Connected:
            self.telescope.Connected = False
            self.telescope = None
