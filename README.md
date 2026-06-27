# autodrome

A self-driving agent for Assetto Corsa that teaches itself to drive, the way Sony's GT Sophy did. It
runs thousands of cars in parallel on the GPU, each one practicing a real track, and trains a
reinforcement-learning policy from all of them at once. Once the policy is quick and clean, the same
network drives the car inside the game.

The sim the agent learns in is built from the track's actual mesh, so it sees the real walls, kerbs and
grass instead of a hand-drawn approximation. Car physics (mass, power, gearing, downforce, grip) come
from the car's own data files. Nothing about the track or the car is faked.

You need your own copy of Assetto Corsa. No game files ship with this repo; the code reads them from
your install.

## How it works

There are three stages.

**Ingest.** `kn5.py` parses the track's binary mesh and `voxelize.py` sorts every surface into road,
kerb, grass, sand, wall or pit by reading the material names. That gives the sim a real collision grid
plus an exact 3D surface you can inspect. `car_ingest.py`, with `car_acd.py` (which decrypts the car's
`data.acd`), pulls the car's physics: engine torque curve, gear ratios, drivetrain, aero and tyre grip.

**Train.** `car_sim_gpu.py` is a vectorised driving sim that runs entirely on the GPU, 20,000 cars at a
time. The agent is QR-SAC, a distributional version of Soft Actor-Critic (`sac.py`). It learns from 36
simulated distance sensors, the car's own dynamics, and a look-ahead of the corners coming up.
`train_sim_gpu.py` is the entry point. The PIT WALL web app (`app/`) shows the cars learning in real
time: reward, lap times, a 3D view of the track built from the mesh, and the fastest car's steering wheel.

**Drive.** `deploy_ac.py` reads the live car state out of Assetto Corsa, rebuilds the exact observation
the policy trained on, and feeds its steering, throttle and brake back into the game through a Custom
Shaders Patch app (`csp/nndrive`). No virtual gamepad. The control goes straight into the physics, the
same path the game's own AI uses.

## Requirements

- Python 3.10 or newer
- PyTorch built with CUDA, and an NVIDIA GPU. It falls back to CPU, but training is much slower.
- NumPy
- Assetto Corsa, plus Custom Shaders Patch if you want to deploy into the game
- Pillow, only if you want the preview images

```
pip install -r requirements.txt
```

## Quickstart

The code defaults to the standard Steam path for Assetto Corsa. If yours is elsewhere, set the
`AC_ROOT` environment variable.

Train a car on a track:

```
python ml/train_sim_gpu.py --car ks_ferrari_sf70h --track imola --cars 20000
```

Or open the control panel and pick a car and track from there:

```
python app/server.py
# then open http://127.0.0.1:8077
```

The first run ingests the track and car from your AC files, then starts training. Watch the reward
climb and the lap times come down. When you are happy with it, drive it in the game:

```
python ml/deploy_ac.py --csp --car ks_ferrari_sf70h --track imola
```

Open the NN Drive app inside AC, turn the built-in AI off, and let the policy take the wheel.

## Notes

This is a research project, not a finished product. It is built for single-player, offline use with
your own car and track, and you should expect to tune things.

The agent learns a fast, clean line on the tracks it trains on. It does not transfer to a track it has
never seen, and each car and track pair gets its own policy.

## License

MIT. See [LICENSE](LICENSE).
