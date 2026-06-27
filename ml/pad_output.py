"""
Controller output backend for the self-drivers — virtual Xbox pad OR vJoy wheel.

The X360 pad works but AC applies its gamepad steering filters (speed sensitivity,
gamma, deadzone) which make it feel vague. A vJoy device shows up in AC as a WHEEL /
joystick, which AC drives linearly (1:1, no gamepad filtering) — far more precise.

Both backends expose the SAME methods the drivers already call, so switching is just a
flag. open_pad(use_vjoy) returns (pad, up_btn).

vJoy setup (one-time):
  1. Install the vJoy driver (https://github.com/njz3/vJoy or the classic vJoy installer).
  2. In "Configure vJoy": enable device 1 with axes X, Y, Z and at least 1 button.
  3. pip install pyvjoy
  4. In Assetto Corsa -> Controls, pick the vJoy device as a WHEEL and bind:
        Steering -> X axis      (set steering lock/range to match; gamma 1, no filter)
        Throttle -> Y axis
        Brake    -> Z axis
        Gear Up  -> Button 1
  Then run a driver with  --vjoy.
"""


class _Vgamepad:
    """Thin wrapper around vgamepad so both backends share one interface."""
    def __init__(self):
        import vgamepad as vg
        self.pad = vg.VX360Gamepad()
        self.UP = vg.XUSB_BUTTON.XUSB_GAMEPAD_A
        self.DOWN = vg.XUSB_BUTTON.XUSB_GAMEPAD_B    # bind 'Previous gear' to B for reverse-unstick
        self.kind = "xbox-pad"

    def left_joystick_float(self, x_value_float=0.0, y_value_float=0.0):
        self.pad.left_joystick_float(x_value_float=x_value_float, y_value_float=y_value_float)

    def right_trigger_float(self, value_float=0.0):
        self.pad.right_trigger_float(value_float=value_float)

    def left_trigger_float(self, value_float=0.0):
        self.pad.left_trigger_float(value_float=value_float)

    def press_button(self, button=None):
        self.pad.press_button(button=self.UP)

    def release_button(self, button=None):
        self.pad.release_button(button=self.UP)

    def gear_down(self, on):
        (self.pad.press_button if on else self.pad.release_button)(button=self.DOWN)

    def update(self):
        self.pad.update()


class _VJoy:
    """vJoy virtual wheel — linear, high-resolution (15-bit per axis), no AC pad filtering."""
    AXMIN, AXMAX = 1, 32768
    MID = (AXMIN + AXMAX) // 2

    def __init__(self, rid=1):
        import pyvjoy
        self._vj = pyvjoy
        self.dev = pyvjoy.VJoyDevice(rid)
        self.kind = "vjoy-wheel"
        self._steer = 0.0; self._thr = 0.0; self._brk = 0.0; self._up = False

    def _ax(self, frac01):                       # 0..1 -> vJoy axis range
        v = int(self.AXMIN + frac01 * (self.AXMAX - self.AXMIN))
        return max(self.AXMIN, min(self.AXMAX, v))

    def left_joystick_float(self, x_value_float=0.0, y_value_float=0.0):
        self._steer = max(-1.0, min(1.0, x_value_float))

    def right_trigger_float(self, value_float=0.0):
        self._thr = max(0.0, min(1.0, value_float))

    def left_trigger_float(self, value_float=0.0):
        self._brk = max(0.0, min(1.0, value_float))

    def press_button(self, button=None):
        self._up = True

    def release_button(self, button=None):
        self._up = False

    def gear_down(self, on):
        self._down = on

    def update(self):
        v = self._vj
        self.dev.set_axis(v.HID_USAGE_X, self._ax((self._steer + 1.0) / 2.0))   # steer (centered)
        self.dev.set_axis(v.HID_USAGE_Y, self._ax(self._thr))                    # throttle
        self.dev.set_axis(v.HID_USAGE_Z, self._ax(self._brk))                    # brake
        self.dev.set_button(1, 1 if self._up else 0)                            # gear up
        self.dev.set_button(2, 1 if getattr(self, "_down", False) else 0)       # gear down (reverse)


def _try_vjoy(rid):
    pad = _VJoy(rid)
    print(f"  output: vJoy WHEEL (device {rid}) — precise, linear. "
          f"Bind Steer->X, Throttle->Y, Brake->Z, Gear Up->Button 1 in AC.")
    return pad, None


def _try_pad():
    pad = _Vgamepad()
    print("  output: virtual Xbox pad. (For more precision, set AC steering "
          "gamma=1 / speed-sensitivity=0 / deadzone=0, or run with --vjoy.)")
    return pad, pad.UP


def open_pad(use_vjoy=False, rid=1):
    """Return (pad, up_btn) for the REQUESTED backend ONLY — no silent cross-fallback.

    Falling back to the other device was a trap: the two use DIFFERENT AC bindings (Gear Up is
    Button 1 on vJoy, the A button on the pad), so a silent switch left the car stuck in neutral
    and steering through the wrong device. So: ask for the pad, you get the pad (or a clear error)."""
    import time as _t
    if use_vjoy:
        return _try_vjoy(rid)                     # vJoy wheel only
    # Xbox pad (default). Retry a couple of times — ViGEmBus can transiently fail right after a
    # previous run if the old virtual pad hasn't been released yet.
    last = None
    for attempt in range(3):
        try:
            return _try_pad()
        except Exception as e:
            last = e
            if attempt < 2:
                print("  Xbox pad busy (ViGEmBus settling), retrying...", flush=True)
                _t.sleep(0.7)
    raise RuntimeError(
        "Xbox pad (vgamepad/ViGEmBus) unavailable: %s\n"
        "   - If you JUST closed a previous run, wait 2s and re-run (the virtual pad is still releasing).\n"
        "   - Or reinstall the latest Nefarius ViGEmBus (an older/conflicting one, e.g. bundled with\n"
        "     Omen Gaming Hub, breaks vgamepad).\n"
        "   - Or use the vJoy wheel instead:  python ml/deploy_ac.py --vjoy" % last)
