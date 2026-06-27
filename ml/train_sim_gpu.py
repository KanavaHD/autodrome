"""
Train SAC with the ALL-GPU car sim — sim, policy, and replay buffer all live on the 1080 Ti.

This is the version that actually uses your GPU and makes 20,000 cars fast: ~260k transitions/sec
(vs ~8k on the CPU numpy sim). Same reward/dynamics/viz as train_sim.py; only the engine changed.

  python ml/train_sim_gpu.py                 # 20000 cars on the GPU (resumes ml/sim_sac.pt)
  python ml/train_sim_gpu.py --cars 8000    # fewer cars
  python ml/train_sim_gpu.py --steps 20000   # stop after N sim-steps
"""
import os, sys, time, math, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from car_sim import V_MAX, LAP_TARGET_S
from car_sim_gpu import GPUCarSim, GPUBuffer
from sac import SAC

CKPT = os.path.join(HERE, "sim_sac.pt")
LIVE = os.path.join(HERE, "sim_live.json")
WALLS = os.path.join(HERE, "sim_walls.json")
REG_DIR = os.path.join(HERE, "policies")   # per-policy stat sidecars for the Garage
WARMUP_STEPS = 50
UPDATES_PER_STEP = 8
BATCH = 4096
HID = 512
NSTEP = 3                     # n-step returns: propagate reward 3 steps back per update (faster learn)
VIZ_CARS = 600
HEAT_RES = 60
HEAT_DECAY = 0.95


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    cars = 20000; total = None
    if "--cars" in sys.argv:
        try: cars = int(sys.argv[sys.argv.index("--cars") + 1])
        except Exception: pass
    if "--steps" in sys.argv:
        try: total = int(sys.argv[sys.argv.index("--steps") + 1])
        except Exception: pass
    # which car/track this stint is training (recorded into the live feed so the app can show it)
    car_id = _arg("--car", "ks_ferrari_sf70h"); track_id = _arg("--track", "imola")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # load the track's ingested geometry (fast_lane.ai -> centreline + corridor); the .kn5 voxeliser
    # supplies the real walls. Auto-ingests from your AC install on first use.
    from track_ingest import load_geom, build, OUT_DIR
    key = track_id.replace("/", "__")
    if not os.path.exists(os.path.join(OUT_DIR, key + ".npz")):
        base, _, lay = track_id.partition("/"); build(base, lay)
    geom = load_geom(key)
    print("track: %s - %d-pt centreline, %.0f m" % (track_id, len(geom["center"]), float(geom["length"])))

    # per-(track, car) checkpoint
    ckpt = os.path.join(HERE, "sim_sac__%s__%s.pt" % (track_id.replace("/", "__"), car_id))

    # policy registry: stable key + carry-forward of total training time across resumes
    reg_key = "%s__%s" % (track_id.replace("/", "__"), car_id)
    prev_secs = 0.0
    try:
        prev_secs = float(json.load(open(os.path.join(REG_DIR, reg_key + ".json"))).get("train_seconds", 0.0))
    except Exception:
        pass
    run_t0 = time.time()

    # CAR physics: use the chosen car's ingested dynamics (mass, power, top speed, grip, drivetrain,
    # engine torque curve, downforce), decoded from your AC install by car_ingest / car_acd.
    car_params = None
    try:
        from car_ingest import load as load_car
        car_params = load_car(car_id)
        print("car: %s - %.0f kg, %d bhp, top %.0f km/h, grip %.2f, %s"
              % (car_params["name"], car_params["mass"], car_params["bhp"],
                 car_params["vmax_ms"] * 3.6, car_params["grip"], car_params["drive"]))
    except Exception as ex:
        print("could not load car '%s' (%s) - using baseline physics" % (car_id, type(ex).__name__))

    sim = GPUCarSim(n=cars, device=dev, geom=geom, car=car_params)
    agent = SAC(sim.obs_dim, sim.act_dim, hid=HID, buf=1, alpha_min=0.1, target_entropy=-1.0, device=dev)
    buf = GPUBuffer(sim.obs_dim, sim.act_dim, 1_500_000, torch.device(dev))   # GPU replay buffer
    agent.nstep_gamma = agent.gamma ** NSTEP        # n-step bootstrap discount (faster credit-assign)
    print("device: %s | net %d-wide | batch %d | %d cars | %d-step returns (ALL-GPU sim)"
          % (agent.device, HID, BATCH, cars, NSTEP))
    step0 = 0
    if os.path.exists(ckpt):
        try:
            d = agent.load(ckpt); step0 = d.get("sim_steps", 0)
            print("resumed: %d sim-steps" % step0)
        except Exception as ex:
            print("could not resume (%s) — fresh" % type(ex).__name__)

    center = [[round(float(x), 1), round(float(z), 1)] for x, z in sim.track.center[::2]]
    wp = sim.track.wall_points(maxpts=3000)
    try:
        json.dump({"walls": [[round(float(x), 1), round(float(z), 1)] for x, z in wp],
                   "center": center, "vmax": round(sim.vmax, 2)}, open(WALLS, "w"))
    except Exception:
        pass
    hbminx, hbmaxx = float(wp[:, 0].min()), float(wp[:, 0].max())
    hbminz, hbmaxz = float(wp[:, 1].min()), float(wp[:, 1].max())
    heat = np.zeros((HEAT_RES, HEAT_RES), np.float64)

    obs = sim.reset_all()
    last_a = torch.zeros((cars, sim.act_dim), device=dev)
    t = step0; t_log = time.time(); rwin = []; hist = []; shist = []; chist = []
    last_pub = 0.0; tpub = time.time(); spub = t
    laps_total = 0; lap_best = 999.0; lap_win = []; lap_hist = []   # lap-completion stats
    w_o = []; w_a = []; w_r = []; w_d = []                          # n-step rolling window
    GAM = agent.gamma

    def write_registry():
        """Persist this policy's stats for the Garage (fast JSON sidecar — no torch load to list)."""
        try:
            os.makedirs(REG_DIR, exist_ok=True)
            rm = float(torch.stack(rwin).mean().item()) if rwin else 0.0
            la = float(np.mean(lap_win[-120:])) if lap_win else 0.0
            rate = (float(np.mean(chist[-60:])) / max(cars, 1)) if chist else 0.0
            reg = {"key": reg_key, "car": car_id, "track": track_id, "agents": cars,
                   "sim_steps": t, "updates": int(agent.updates),
                   "best_lap": round(lap_best, 1) if lap_best < 900 else 0.0,
                   "avg_lap": round(la, 1), "reward": round(rm, 4), "crash_rate": round(rate, 5),
                   "calibrated": bool(car_params and car_params.get("from_data") == "calibrated"),
                   "vmax_kmh": round(sim.vmax * 3.6), "trained_at": time.time(),
                   "train_seconds": round(prev_secs + (time.time() - run_t0), 1),
                   "ckpt": os.path.basename(ckpt)}
            json.dump(reg, open(os.path.join(REG_DIR, reg_key + ".json"), "w"))
        except Exception:
            pass

    write_registry()                                    # show up in the Garage immediately
    print("\ntraining on GPU... watch the card work (and the reward climb).\n")
    try:
        while total is None or (t - step0) < total:
            if t < step0 + WARMUP_STEPS:
                a = torch.empty(cars, sim.act_dim, device=dev).uniform_(-1, 1)
            else:
                a = agent.act_tensor(obs)
            last_a = a
            nobs, r, done = sim.step(a)
            # N-STEP RETURNS: buffer the window, then push the oldest transition with its discounted
            # n-step return + bootstrap from nobs (masked if any done occurred in the window).
            w_o.append(obs); w_a.append(a); w_r.append(r); w_d.append(done.float())
            if len(w_o) == NSTEP:
                ret = torch.zeros(cars, device=dev); disc = 1.0
                alive = torch.ones(cars, device=dev); done_n = torch.zeros(cars, device=dev)
                for j in range(NSTEP):
                    ret = ret + alive * disc * w_r[j]
                    done_n = torch.maximum(done_n, alive * w_d[j])
                    alive = alive * (1 - w_d[j]); disc *= GAM
                buf.add_batch(w_o[0], w_a[0], ret, nobs, done_n)
                w_o.pop(0); w_a.pop(0); w_r.pop(0); w_d.pop(0)
            obs = nobs
            rwin.append(r.mean().detach()); rwin = rwin[-200:]   # keep on GPU — no per-step sync
            if sim.last_lap_times.numel():                       # cars that finished a lap this step
                lts = sim.last_lap_times.cpu().numpy()
                laps_total += len(lts); lap_win.extend(lts.tolist()); lap_win = lap_win[-400:]
                lap_best = min(lap_best, float(lts.min()))
            if len(buf) >= max(BATCH, WARMUP_STEPS * cars):
                for _ in range(UPDATES_PER_STEP):
                    agent.update_core(*buf.sample(BATCH))
            t += 1
            now = time.time()
            if now - t_log > 3.0:
                rmean = float(torch.stack(rwin).mean().item())
                print("sim-step %6d | buf %9d | avg reward/step %+.4f | alpha %.2f"
                      % (t, len(buf), rmean, float(agent.alpha.detach())), flush=True)
                t_log = now
            if now - last_pub > 0.25:
                sps = int((t - spub) * cars / max(now - tpub, 1e-3)); spub = t; tpub = now
                rmean = float(torch.stack(rwin).mean().item()) if rwin else 0.0
                hist.append(round(rmean, 4)); hist = hist[-240:]
                ks = min(VIZ_CARS, cars)
                pos = sim.pos[:ks].detach().cpu().numpy(); psi = sim.psi[:ks].detach().cpu().numpy()
                spdv = sim.spd; spd_ks = spdv[:ks].detach().cpu().numpy()
                grip_ks = sim.grip[:ks].detach().cpu().numpy()
                cars_xy = [[round(float(pos[i, 0]), 1), round(float(pos[i, 1]), 1),
                             round(float(math.degrees(psi[i])), 1), round(float(spd_ks[i]), 1),
                             round(float(grip_ks[i]), 2)] for i in range(ks)]
                L = int(torch.argmax(spdv).item())
                g = lambda x: float(x[L].item())
                lead = {"vx": round(g(sim.vx), 1), "spd": round(g(sim.spd) * 3.6, 1),
                        "slip": round(math.degrees(g(sim.slip_b)), 1), "yaw": round(g(sim.r), 2),
                        "lonG": round(g(sim.lonG) / 9.81, 2), "latG": round(g(sim.latG) / 9.81, 2),
                        "grip": round(g(sim.grip), 2), "slipF": round(g(sim.slip_f), 2),
                        "slipR": round(g(sim.slip_r), 2), "steer": round(float(last_a[L, 0].item()), 2),
                        "thr": round(float(max(0.0, last_a[L, 1].item())), 2),
                        "brk": round(float(max(0.0, -last_a[L, 1].item())), 2),
                        "x": round(g(sim.pos[:, 0]), 1), "z": round(g(sim.pos[:, 1]), 1)}
                try:                                          # LIVE BRAIN: lead car's net activations
                    brain = agent.brain(obs[L].detach().cpu().numpy())
                except Exception:
                    brain = None
                allpos = sim.pos.detach().cpu().numpy()
                gxh = np.clip(((allpos[:, 0] - hbminx) / (hbmaxx - hbminx + 1e-9) * (HEAT_RES - 1)).astype(int), 0, HEAT_RES - 1)
                gzh = np.clip(((allpos[:, 1] - hbminz) / (hbmaxz - hbminz + 1e-9) * (HEAT_RES - 1)).astype(int), 0, HEAT_RES - 1)
                heat *= HEAT_DECAY; np.add.at(heat, (gxh, gzh), 1.0)
                heat_pub = (heat / (heat.max() + 1e-6) * 255).astype(int).tolist()
                avgspd = round(float(spdv.mean().item()) * 3.6, 1); ncrash = int(done.sum().item())
                shist.append(avgspd); shist = shist[-240:]; chist.append(ncrash); chist = chist[-240:]
                crash = [[round(float(x), 1), round(float(z), 1)] for x, z in sim.crash_pos.detach().cpu().numpy()[:60]]
                lap_avg = round(float(np.mean(lap_win[-120:])), 1) if lap_win else 0.0   # recent avg lap
                lap_hist.append(lap_avg); lap_hist = lap_hist[-240:]
                try:
                    payload = {"running": True, "step": t, "warmup": t < step0 + WARMUP_STEPS,
                               "car": car_id, "track": track_id,
                               "avgspeed": avgspd, "topspeed": round(float(spdv.max().item()) * 3.6, 1),
                               "reward": round(rmean, 4), "crashes": ncrash,
                               "n_wall": sim.n_wall, "n_off": sim.n_off,
                               "alpha": round(float(agent.alpha.detach()), 3), "buf": len(buf),
                               "updates": int(agent.updates), "tps": sps, "cars_total": cars,
                               # LAP STATS — the real objective: completions, recent avg, best lap
                               "laps_total": laps_total, "lap_avg": lap_avg,
                               "lap_best": round(lap_best, 1) if lap_best < 900 else 0.0,
                               "lap_target": LAP_TARGET_S, "lap_hist": lap_hist,
                               "vmax": round(sim.vmax, 2), "history": hist, "shist": shist, "chist": chist,
                               "cars": cars_xy, "lead": lead, "brain": brain, "crash": crash,
                               "heat": heat_pub, "heatbounds": [round(hbminx, 1), round(hbmaxx, 1),
                               round(hbminz, 1), round(hbmaxz, 1)], "heatres": HEAT_RES, "ts": now}
                    json.dump(payload, open(LIVE + ".tmp", "w")); os.replace(LIVE + ".tmp", LIVE)
                except Exception:
                    pass
                last_pub = now
            if (t - step0) % 1000 == 0 and t > step0:
                agent.save(ckpt, extra={"sim_steps": t}); write_registry()
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        agent.save(ckpt, extra={"sim_steps": t}); write_registry()
        try:
            d = json.load(open(LIVE)); d["running"] = False; json.dump(d, open(LIVE, "w"))
        except Exception:
            pass
        print("saved %s (%d sim-steps)" % (ckpt, t))


if __name__ == "__main__":
    main()
