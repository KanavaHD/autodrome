"""
GPU car simulator — the ENTIRE vectorised sim (raycasting + bicycle physics) on CUDA in PyTorch.

Why: the numpy sim is CPU-bound, so the 1080 Ti sat at 5% and 20k cars crawled at 0.04x real time.
Moving the sim itself onto the GPU makes the GPU the workload: 20k+ cars run massively parallel,
the card actually gets used, and learning speeds up because the policy, the sim, AND the replay
buffer all live on the GPU — no per-step CPU<->GPU transfers at all.

Mirrors car_sim.VecCarSim exactly (same reward, termination, dynamics) — just in torch tensors.
The one-time track build (wall grid, phantom-wall pruning, wall-centred line) is reused from
car_sim.Track on the CPU, then uploaded to the GPU once.

  python ml/car_sim_gpu.py        # self-test: throughput + that it matches behaviour
"""
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from car_sim import (Track, RAY_MAX, FAN, DT, NSUB, V_MAX, MAX_OFF, CRASH_RAY, CRASH_MARGIN, CAR_EXTENT,
                      SPAWN_LAT, LAP_TARGET_S,
                      M, IZ, LF, LR, L, HCG, G, MU0, F_DRIVE, F_BRAKE, STIFF, C_ROLL, C_AERO, V_EPS,
                      STEER_LOCK, DR_GRIP, DR_PWR, DR_LOCK, DR_BAL,
                      TYRE_DY0_F, TYRE_DY1_F, TYRE_FZ0_F, TYRE_DY0_R, TYRE_DY1_R, TYRE_FZ0_R,
                      TYRE_LSEXPY, ALPHA_PEAK, FALLOFF_LEVEL, FALLOFF_SPEED, THR_GAMMA, BRK_GAMMA,
                      RW_FWD, RW_WALL, RW_WALL_KEEP, RW_SMOOTH, RW_STRAIGHT_STR, RW_ALIGN, RW_ALIGN_LOOK,
                      RW_GRIP, RW_OVERGRIP, RW_STRAIGHT_SPD, RW_CRASH)
from wall_sensor import H_BELOW, H_ABOVE

RW_OFFROAD = 0.08            # per-step reward cost for being off the racing surface (grass/sand) — keeps
                            # the car on the road now that walls are real/sparse, not phantom corridors
OFF_MARGIN = 22.0           # m of runoff PAST the drivable corridor before "off track" is terminal (real-
                            # wall tracks). Lets a wide line / gravel excursion survive; barriers = wall_term


class GPUBuffer:
    """Replay buffer that lives entirely on the GPU (no CPU transfers on add/sample)."""
    def __init__(self, obs_dim, act_dim, size, device):
        self.o = torch.zeros((size, obs_dim), device=device)
        self.o2 = torch.zeros((size, obs_dim), device=device)
        self.a = torch.zeros((size, act_dim), device=device)
        self.r = torch.zeros(size, device=device)
        self.d = torch.zeros(size, device=device)
        self.ptr = 0; self.n = 0; self.size = size; self.dev = device

    def add_batch(self, O, A, R, O2, D):
        m = O.shape[0]; i = self.ptr
        if i + m <= self.size:
            sl = slice(i, i + m)
            self.o[sl] = O; self.a[sl] = A; self.r[sl] = R; self.o2[sl] = O2; self.d[sl] = D
        else:                                              # wrap around the ring
            f = self.size - i
            self.o[i:] = O[:f]; self.a[i:] = A[:f]; self.r[i:] = R[:f]; self.o2[i:] = O2[:f]; self.d[i:] = D[:f]
            g = m - f
            self.o[:g] = O[f:]; self.a[:g] = A[f:]; self.r[:g] = R[f:]; self.o2[:g] = O2[f:]; self.d[:g] = D[f:]
        self.ptr = (i + m) % self.size; self.n = min(self.n + m, self.size)

    def sample(self, bs):
        idx = torch.randint(0, self.n, (bs,), device=self.dev)
        return self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx]

    def __len__(self):
        return self.n


class GPUCarSim:
    def __init__(self, n=20000, device="cuda", seed=0, geom=None, car=None):
        self.n = n; self.dev = torch.device(device)
        torch.manual_seed(seed)
        tr = Track(geom=geom)                              # geom=ingested AC track, else the car self-map
        dev = self.dev
        self.cell = float(tr.cell)
        self.gmin0 = int(tr.gmin[0]); self.gmin1 = int(tr.gmin[1])
        self.H0 = int(tr.H0); self.H1 = int(tr.H1)
        # HEIGHT-AWARE (multi-level): keep the wall height grids, not a flat boolean
        self.ymin = torch.as_tensor(tr.ymin, dtype=torch.float32, device=dev)       # (H0,H1)
        self.ymax = torch.as_tensor(tr.ymax, dtype=torch.float32, device=dev)
        self.center = torch.as_tensor(tr.center, dtype=torch.float32, device=dev)  # (K,2) xz
        self.center_y = torch.as_tensor(tr.center_y, dtype=torch.float32, device=dev)  # (K,) height
        self.length = float(tr.length)                                            # lap length (m)
        self.tangent = torch.as_tensor(tr.tangent, dtype=torch.float32, device=dev)
        self.tang_ang = torch.atan2(self.tangent[:, 1], self.tangent[:, 0])           # (K,)
        self.curve_ahead = torch.as_tensor(tr.curve_ahead, dtype=torch.float32, device=dev)  # (K,nH)
        self.corner_vmax = torch.as_tensor(tr.corner_vmax, dtype=torch.float32, device=dev)  # (K,nH)
        self.halfwidth = torch.as_tensor(tr.halfwidth, dtype=torch.float32, device=dev)
        self.spawn_idx = torch.as_tensor(tr.spawn_idx, dtype=torch.long, device=dev)
        self.K = int(tr.K)
        self.track = tr                                    # keep for wall_points() (viz)
        # ── per-CAR physics: default = the calibrated car; overridden by an ingested CarParams ──
        self.M=M; self.F_DRIVE=F_DRIVE; self.F_BRAKE=F_BRAKE; self.vmax=V_MAX; self.lock=STEER_LOCK
        self.LF=LF; self.LR=LR; self.L=L; self.HCG=HCG; self.IZ=IZ; self.C_AERO=C_AERO; self.C_ROLL=C_ROLL
        self.car_grip=1.0; self.car=car
        self.k_aero=0.0; self.aero_f=0.5                       # downforce: N per (m/s)^2, front fraction
        if car:
            self.k_aero=float(car.get('k_aero', 0.0)); self.aero_f=float(car.get('aero_f', 0.5))
            self.M=float(car['mass']); self.vmax=float(car['vmax_ms'])
            self.F_DRIVE=self.M*float(car['a_accel']); self.F_BRAKE=self.M*float(car['a_brake'])
            self.lock=float(car['steer_lock']); self.L=float(car['L'])
            self.LF=float(car['LF']); self.LR=float(car['LR']); self.HCG=float(car['HCG'])
            self.IZ=float(car['IZ']); self.C_AERO=float(car['C_aero'])
            self.C_ROLL=C_ROLL*(self.M/215.0); self.car_grip=float(car['grip'])/1.49
        # DRIVE DISTRIBUTION + TRACTIVE CURVE (real engine torque + gears). Defaults: rear-drive, flat
        # force at F_DRIVE (car unchanged). A car with an ingested torque curve gets speed-dependent
        # drive force and front/rear split (FWD puts drive through the front tyres -> power understeer).
        self.drive_f = 0.0; self.drive_r = 1.0
        if car and car.get('tractive'):
            self.tract = torch.tensor(car['tractive'], dtype=torch.float32, device=dev)
            self.tract_dv = float(car['tract_dv']); self.tract_n = self.tract.numel()
            self.drive_f = float(car.get('drive_f', 0.0)); self.drive_r = float(car.get('drive_r', 1.0))
        else:
            self.tract = torch.full((2,), float(self.F_DRIVE), dtype=torch.float32, device=dev)
            self.tract_dv = max(self.vmax, 1.0); self.tract_n = 2
        ar = np.radians(np.array(FAN, float))
        self.rc = torch.as_tensor(np.cos(ar), dtype=torch.float32, device=dev)     # (R,)
        self.rs = torch.as_tensor(np.sin(ar), dtype=torch.float32, device=dev)
        self.car_extent = torch.as_tensor(CAR_EXTENT, dtype=torch.float32, device=dev)  # footprint (R,)
        self.n_rays = len(FAN); self.n_dyn = 10
        self.n_dir = 2; self.n_ahead = self.curve_ahead.shape[1]; self.n_pred = self.n_ahead
        self.obs_dim = self.n_dyn + self.n_dir + self.n_ahead + self.n_pred + self.n_rays
        self.act_dim = 2
        step = self.cell * 0.6
        self.ray_steps = torch.arange(0.45, RAY_MAX, step, device=dev, dtype=torch.float32)  # (S,)
        z = lambda: torch.zeros(n, device=dev)
        self.pos = torch.zeros((n, 2), device=dev); self.psi = z()
        self.vx = z(); self.vy = z(); self.r = z()
        self.lonG = z(); self.latG = z(); self.grip = z()
        self.slip_f = z(); self.slip_r = z(); self.slip_b = z()
        self.mu = z(); self.pwr = z(); self.steer_lock = z(); self.gbal = z()
        self.prog = torch.zeros(n, dtype=torch.long, device=dev)   # progress index (floor-safe)
        self.cy = z()                                              # current height -> height-aware rays
        self.prev_steer = z()                                     # for the smoothness reward
        self.lap_dist = z(); self.lap_steps = torch.zeros(n, dtype=torch.long, device=dev)  # lap tracking
        self.last_lap_times = torch.zeros(0, device=dev)
        self._offs = torch.arange(-18, 19, device=dev)            # localise search window
        self.PI2 = float(np.pi / 2)
        self.n_wall = 0; self.n_off = 0
        self.crash_pos = torch.zeros((0, 2), device=dev)
        # OFF-TRACK GRIP: with the REAL (sparse) walls from the voxel mesh, the car can run off the
        # racing surface onto grass. We make that cost grip + reward — but the drivable corridor is
        # defined by the fast_lane line (off < halfwidth), NOT the mesh surface classes. The line is
        # always inside its own corridor, so the racing surface is guaranteed full grip on every track
        # (the mesh's road/grass tags are inconsistent per-track and left grip holes on the line).
        self.has_surf = getattr(tr, "surf", None) is not None    # True => a voxel-wall track
        self.surf_mul = torch.ones(n, device=dev)                # grip mult, set each step from corridor
        self.off_road_margin = 1.08                              # off > halfwidth*this = off the surface
        self.reset(torch.arange(n, device=dev))

    @property
    def head(self):
        return torch.stack([torch.cos(self.psi), torch.sin(self.psi)], 1)

    @property
    def spd(self):
        return torch.sqrt(self.vx ** 2 + self.vy ** 2)

    def _rand(self, m, lo, hi):
        return torch.rand(m, device=self.dev) * (hi - lo) + lo

    def reset(self, idx):
        m = idx.numel()
        if m == 0:
            return
        k = self.spawn_idx[torch.randint(0, self.spawn_idx.numel(), (m,), device=self.dev)]
        C = self.center[k]; T = self.tangent[k]
        nrm = torch.stack([-T[:, 1], T[:, 0]], 1)
        hw = self.halfwidth[k]
        lat = torch.randn(m, 1, device=self.dev) * torch.clamp(hw[:, None] * 0.4, 0.2, SPAWN_LAT)
        self.pos[idx] = C + nrm * lat
        self.psi[idx] = torch.atan2(T[:, 1], T[:, 0])
        self.prog[idx] = k; self.cy[idx] = self.center_y[k]        # spawned on a known floor
        self.vx[idx] = self._rand(m, 0, 6); self.vy[idx] = 0.0; self.r[idx] = 0.0
        self.lonG[idx] = 0; self.latG[idx] = 0; self.grip[idx] = 0
        self.slip_f[idx] = 0; self.slip_r[idx] = 0; self.slip_b[idx] = 0
        self.mu[idx] = self.car_grip * self._rand(m, *DR_GRIP)                  # per-car grip SCALE (base mu from tyre model)
        self.pwr[idx] = self._rand(m, *DR_PWR)
        self.steer_lock[idx] = self.lock * self._rand(m, *DR_LOCK)
        self.gbal[idx] = self._rand(m, -DR_BAL, DR_BAL)          # front/rear grip balance
        self.prev_steer[idx] = 0.0
        self.surf_mul[idx] = 1.0                                 # respawns on the racing line = full grip
        self.lap_dist[idx] = 0.0; self.lap_steps[idx] = 0

    def raycast(self, pos, head, cy):
        """HEIGHT-AWARE rays: a cell blocks only if its wall span overlaps [cy-H_BELOW, cy+H_ABOVE],
        so lower-floor rays pass under the upper floor (multi-level)."""
        c = self.cell
        steps = self.ray_steps                                              # (S,)
        dirx = head[:, 0:1] * self.rc[None, :] - head[:, 1:2] * self.rs[None, :]   # (n,R)
        dirz = head[:, 0:1] * self.rs[None, :] + head[:, 1:2] * self.rc[None, :]
        px = pos[:, 0:1, None] + dirx[:, :, None] * steps[None, None, :]      # (n,R,S)
        pz = pos[:, 1:2, None] + dirz[:, :, None] * steps[None, None, :]
        gx = torch.clamp((px / c).round().long() - self.gmin0 + 1, 0, self.H0 - 1)
        gz = torch.clamp((pz / c).round().long() - self.gmin1 + 1, 0, self.H1 - 1)
        wmax = self.ymax[gx, gz]; wmin = self.ymin[gx, gz]                   # (n,R,S)
        lo = (cy - H_BELOW)[:, None, None]; hi = (cy + H_ABOVE)[:, None, None]
        blocked = (wmax >= lo) & (wmin <= hi)
        has = blocked.any(2)
        first = blocked.float().argmax(2)                                    # (n,R)
        dist = steps[first]
        return torch.where(has, dist, torch.full_like(dist, RAY_MAX))

    def localize(self, pos, prog):
        """Floor-safe: search only NEAR current progress so a car can't jump floors at a stack."""
        n = pos.shape[0]
        idx = (prog[:, None] + self._offs[None, :]) % self.K                 # (n, W)
        cx = self.center[idx, 0]; cz = self.center[idx, 1]
        d2 = (cx - pos[:, 0:1]) ** 2 + (cz - pos[:, 1:2]) ** 2
        j = d2.argmin(1)
        ar = torch.arange(n, device=self.dev)
        return idx[ar, j], torch.sqrt(d2[ar, j])

    def _obs(self, rays):
        dyn = torch.stack([
            self.vx / self.vmax, self.vy / 10.0, self.r / 3.0, self.slip_b / 0.6,
            self.lonG / (1.5 * G), self.latG / (1.5 * G), self.grip,
            torch.clamp(1.0 - self.grip, 0, 1),                        # grip MARGIN left
            self.slip_f / 0.4, self.slip_r / 0.4], 1)
        err = self.tang_ang[self.prog] - self.psi                       # direction: track vs car
        direction = torch.stack([torch.cos(err), torch.sin(err)], 1)
        ahead = self.curve_ahead[self.prog] / self.PI2                  # lookahead: bends ahead
        excess = torch.clamp(self.vx[:, None] / self.corner_vmax[self.prog] - 1.0, -1.5, 1.5)  # turn timing
        return torch.cat([torch.clamp(dyn, -3, 3), direction, torch.clamp(ahead, -3, 3),
                          excess, torch.clamp(rays / RAY_MAX, 0, 1.2)], 1)

    def reset_all(self):
        self.reset(torch.arange(self.n, device=self.dev))
        return self._obs(self.raycast(self.pos, self.head, self.cy))

    def _tyre_frac(self, alpha):
        """AC peak-then-falloff lateral grip fraction vs slip angle (torch). Mirrors car_sim.tyre_force_frac."""
        x = torch.abs(alpha) / ALPHA_PEAK
        rising = torch.sin(0.5 * float(np.pi) * torch.clamp(x, 0.0, 1.0))
        falloff = FALLOFF_LEVEL + (1.0 - FALLOFF_LEVEL) * torch.exp(-torch.clamp(x - 1.0, min=0.0) * FALLOFF_SPEED)
        return torch.sign(alpha) * torch.where(x <= 1.0, rising, falloff)

    def _tractive_at(self, v):
        """Drive force available at speed v, interpolated from the real torque-curve/gear envelope."""
        f = torch.clamp(v, min=0.0) / self.tract_dv
        i0 = torch.clamp(f.long(), 0, self.tract_n - 2)
        frac = torch.clamp(f - i0.float(), 0.0, 1.0)
        t0 = self.tract[i0]; t1 = self.tract[i0 + 1]
        return t0 + (t1 - t0) * frac

    def _dynamics(self, steer, longi):
        h = DT / NSUB
        delta = steer * self.steer_lock
        thr = torch.clamp(longi, 0, 1) ** THR_GAMMA * self.pwr   # AC throttle gamma (softer partial)
        brk = torch.clamp(-longi, 0, 1) ** BRK_GAMMA
        cd, sd = torch.cos(delta), torch.sin(delta)
        fx_drive = thr * self._tractive_at(self.vx)             # speed-dependent drive force (real engine)
        for _ in range(NSUB):
            vxs = torch.clamp(self.vx, min=V_EPS)
            af = torch.atan2(self.vy + self.LF * self.r, vxs) - delta
            ar_ = torch.atan2(self.vy - self.LR * self.r, vxs)
            fx_brk = brk * self.F_BRAKE * torch.sign(self.vx + 1e-6)
            fx_f = self.drive_f * fx_drive - 0.4 * fx_brk       # FWD/AWD send drive to the front axle
            fx_r = self.drive_r * fx_drive - 0.6 * fx_brk
            drag = self.C_ROLL * self.vx + self.C_AERO * self.vx * torch.abs(self.vx)
            ax_est = (fx_f + fx_r - drag) / self.M
            fzf = torch.clamp(self.M * (G * self.LR - ax_est * self.HCG) / self.L, min=50.0)
            fzr = torch.clamp(self.M * (G * self.LF + ax_est * self.HCG) / self.L, min=50.0)
            aero = self.k_aero * self.vx * self.vx              # DOWNFORCE (measured: ~k*v^2 N) ->
            fzf = fzf + self.aero_f * aero                      # extra vertical load -> more grip with
            fzr = fzr + (1.0 - self.aero_f) * aero              # speed (the F1's hard-cornering secret)
            # LOAD-SENSITIVE peak grip per axle (AC tyres.ini): mu drops as load rises
            nf_f = torch.clamp(fzf / TYRE_FZ0_F, 0.1, 6.0)
            nf_r = torch.clamp(fzr / TYRE_FZ0_R, 0.1, 6.0)
            mu_f = (TYRE_DY0_F + TYRE_DY1_F * (nf_f - 1.0)) * nf_f ** (TYRE_LSEXPY - 1.0)
            mu_r = (TYRE_DY0_R + TYRE_DY1_R * (nf_r - 1.0)) * nf_r ** (TYRE_LSEXPY - 1.0)
            # surf_mul (1.0 on road, <1 on grass/sand) scrubs grip off the racing surface
            cap_f = mu_f * self.mu * self.surf_mul * (1.0 + self.gbal) * fzf   # self.mu = per-car grip SCALE
            cap_r = mu_r * self.mu * self.surf_mul * (1.0 - self.gbal) * fzr
            fxf_c = torch.minimum(torch.maximum(fx_f, -cap_f), cap_f)
            fxr_c = torch.minimum(torch.maximum(fx_r, -cap_r), cap_r)
            fy_cap_f = torch.sqrt(torch.clamp(cap_f ** 2 - fxf_c ** 2, min=0.0))
            fy_cap_r = torch.sqrt(torch.clamp(cap_r ** 2 - fxr_c ** 2, min=0.0))
            fyf = -fy_cap_f * self._tyre_frac(af)               # AC peak-then-falloff slip curve
            fyr = -fy_cap_r * self._tyre_frac(ar_)
            Fx = fxr_c + fxf_c * cd - fyf * sd - drag
            Fy = fyr + fyf * cd + fxf_c * sd
            ax = Fx / self.M; ay = Fy / self.M
            vx_dot = ax + self.vy * self.r
            vy_dot = ay - self.vx * self.r
            r_dot = (self.LF * (fyf * cd + fxf_c * sd) - self.LR * fyr) / self.IZ
            self.vx = torch.clamp(self.vx + vx_dot * h, -4.0, self.vmax)
            self.vy = self.vy + vy_dot * h
            self.r = self.r + r_dot * h
            self.psi = self.psi + self.r * h
            cc, ss = torch.cos(self.psi), torch.sin(self.psi)
            self.pos[:, 0] = self.pos[:, 0] + (self.vx * cc - self.vy * ss) * h
            self.pos[:, 1] = self.pos[:, 1] + (self.vx * ss + self.vy * cc) * h
        self.lonG = ax; self.latG = ay
        self.grip = torch.maximum(torch.sqrt(fxf_c ** 2 + fyf ** 2) / (cap_f + 1e-6),
                                  torch.sqrt(fxr_c ** 2 + fyr ** 2) / (cap_r + 1e-6))
        self.slip_f = af; self.slip_r = ar_
        self.slip_b = torch.atan2(self.vy, torch.clamp(self.vx, min=0.1))

    def step(self, action):
        a = torch.clamp(action, -1, 1)
        prev = self.pos.clone()
        # _dynamics uses self.surf_mul carried from the previous step's corridor check (1-step lag, 0.1s)
        self._dynamics(a[:, 0], a[:, 1])
        # floor-safe localise -> height -> height-aware rays (multi-level)
        i, off = self.localize(self.pos, self.prog)
        self.prog = i; self.cy = self.center_y[i]
        # off-track = beyond the drivable corridor (fast_lane halfwidth). Sets next step's grip + a
        # per-step reward cost so the policy keeps to the racing surface without phantom side-walls.
        onroad = (off < self.halfwidth[i] * self.off_road_margin) if self.has_surf \
            else torch.ones_like(off, dtype=torch.bool)
        if self.has_surf:
            self.surf_mul = torch.where(onroad, torch.ones_like(off), torch.full_like(off, 0.6))
        rays = self.raycast(self.pos, self.head, self.cy)
        nearest_wall = (rays - self.car_extent[None, :]).min(1).values   # BODY clearance (footprint-aware)
        tang = self.tangent[i]
        fwd = ((self.pos - prev) * tang).sum(1)
        spd = self.spd
        steer = a[:, 0]
        # SMART REWARD (mirror of car_sim): progress - wall-margin - jerk - needless-steer + align + limit
        reward = RW_FWD * fwd
        # WALL MARGIN: squared + wider keep-clear -> real buffer, no scraping (the AC wall-scratch fix)
        near = torch.clamp((RW_WALL_KEEP - nearest_wall) / RW_WALL_KEEP, 0.0, 1.0)
        reward = reward - RW_WALL * near * near * (0.4 + 0.6 * spd / self.vmax)
        # SMOOTHNESS: penalise steering CHANGE (squared) -> deliberate, no twitch
        reward = reward - RW_SMOOTH * (steer - self.prev_steer) ** 2
        self.prev_steer = steer.clone()
        straight = 1.0 - torch.clamp(torch.abs(self.curve_ahead[i, 1]) / 0.4, 0, 1)
        # ANTICIPATION + NO NEEDLESS STEER: reward facing where the track GOES next (turn in on time;
        # don't steer on a straight). look_ang = tangent a few points ahead.
        look_ang = self.tang_ang[(i + RW_ALIGN_LOOK) % self.K]
        reward = reward + RW_ALIGN * (torch.cos(look_ang - self.psi) - 1.0)
        reward = reward - RW_STRAIGHT_STR * straight * steer ** 2
        # DRIVE-THE-LIMIT: use grip, don't exceed it; get on the power where straight
        reward = reward + RW_GRIP * torch.clamp(self.grip, 0, 1.0) - RW_OVERGRIP * torch.clamp(self.grip - 1.05, 0, 1.0)
        reward = reward + RW_STRAIGHT_SPD * straight * (self.vx / self.vmax)
        # OFF-ROAD penalty: with REAL (sparse) walls the car CAN run onto grass/sand, so a small
        # per-step cost (scaled by speed) teaches it to keep to the racing surface — the job the
        # phantom side-walls used to do, but without the false barriers that caused the oversteer.
        if self.has_surf:
            reward = reward - RW_OFFROAD * (~onroad).float() * (0.3 + 0.7 * spd / self.vmax)
        # LAP COMPLETION reward — covers a full lap -> big, speed-scaled bonus (chases the target time)
        self.lap_steps += 1
        self.lap_dist += fwd
        lapped = self.lap_dist >= self.length
        if lapped.any():
            lt = self.lap_steps[lapped].float() * DT
            self.last_lap_times = lt.detach().clone()
            bonus = torch.clamp(8.0 * (LAP_TARGET_S / torch.clamp(lt, min=1.0)), 2.0, 20.0)
            reward = reward.clone()
            reward[lapped] += bonus
            self.lap_dist[lapped] -= self.length
            self.lap_steps[lapped] = 0
        else:
            self.last_lap_times = torch.zeros(0, device=self.dev)
        wall_term = nearest_wall < CRASH_MARGIN               # the car BODY contacts a wall
        # OFF-TRACK: on a real-wall track the car CAN run wide onto grass/gravel (grip + reward already
        # cost it) — only terminate when it's truly lost (well past the corridor + a runoff margin), so
        # a slightly-wide line through a corner no longer instakills it before it can complete a lap.
        # Hitting an actual barrier is caught by wall_term. (The narrow car track keeps the tight MAX_OFF.)
        off_term = off > (self.halfwidth[i] + OFF_MARGIN) if self.has_surf else (off > MAX_OFF)
        done = wall_term | off_term
        reward = reward - RW_CRASH * done.float()
        self.n_wall = int(wall_term.sum().item()); self.n_off = int(off_term.sum().item())
        dmask = torch.nonzero(done, as_tuple=False).squeeze(1)
        self.crash_pos = prev[dmask][:120].clone() if dmask.numel() else torch.zeros((0, 2), device=self.dev)
        obs = self._obs(rays)
        if dmask.numel():
            self.reset(dmask)
            obs = self._obs(self.raycast(self.pos, self.head, self.cy))
        return obs, reward, done


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("building GPU sim on", dev, "...")
    sim = GPUCarSim(n=20000, device=dev)
    obs = sim.reset_all()
    torch.cuda.synchronize() if dev == "cuda" else None
    t0 = time.time(); steps = 60
    for _ in range(steps):
        a = torch.empty(sim.n, 2, device=dev).uniform_(-1, 1)
        obs, r, d = sim.step(a)
    torch.cuda.synchronize() if dev == "cuda" else None
    dt = time.time() - t0
    print("20000 cars: %d steps in %.2fs -> %.1f steps/s | %d transitions/s | %.2fx real-time"
          % (steps, dt, steps / dt, int(steps * sim.n / dt), steps / dt * DT))
    print("obs", tuple(obs.shape), "finite", bool(torch.isfinite(obs).all().item()),
          "| speed mean %.1f max %.1f" % (sim.spd.mean().item(), sim.spd.max().item()))
