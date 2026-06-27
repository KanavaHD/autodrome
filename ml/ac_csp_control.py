"""
CSP direct-control bridge — drive the AC car through CSP's official ac.overrideCarControls API,
NOT vJoy/virtual gamepad and NOT memory injection. Our policy calls CspController.set(steer, gas,
brake); this writes the NN Drive CSP app's cmd.txt, which the Lua app applies in-engine each frame.

    steer: -1..1   gas: 0..1   brake: 0..1

Stable across game launches and CSP updates (it's the supported API), and it's how the AI itself
gets its inputs into the physics — exactly the "proprietary, no fake gamepad" path.
"""
import os, time

CSP_APP = os.environ.get(
    "NNDRIVE_DIR",
    r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\apps\lua\nndrive")
CMD = os.path.join(CSP_APP, "cmd.txt")


class CspController:
    def __init__(self, path=CMD):
        self.path = path
        self.tmp = path + ".tmp"
        self.counter = 0
        self.available = os.path.isdir(os.path.dirname(path))

    def set(self, steer=0.0, gas=0.0, brake=0.0, enabled=True):
        """Write one control frame. The Lua app reads it ~every frame and applies it to the car."""
        self.counter += 1
        s = max(-1.0, min(1.0, float(steer)))
        g = max(0.0, min(1.0, float(gas)))
        b = max(0.0, min(1.0, float(brake)))
        line = "%d %.4f %.4f %.4f %d\n" % (self.counter, s, g, b, 1 if enabled else 0)
        try:
            with open(self.tmp, "w") as f:
                f.write(line)
            os.replace(self.tmp, self.path)      # atomic — Lua never reads a half-written file
            return True
        except Exception:
            return False

    def release(self):
        """Hand control back to the user/AI."""
        self.set(0.0, 0.0, 0.0, enabled=False)


if __name__ == "__main__":
    # quick self-test: sweep steering left/right + a throttle blip (open the NN Drive app in AC to see it)
    c = CspController()
    print("writing to", c.path, "| dir exists:", c.available)
    print("enable the 'NN Drive' app window in AC, then watch the car...")
    for steer in (0.0, 0.4, 0.0, -0.4, 0.0):
        c.set(steer=steer, gas=0.2, brake=0.0)
        print("  steer=%+.2f gas=0.2" % steer)
        time.sleep(0.8)
    c.release()
    print("released control.")
