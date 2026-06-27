-- NN Drive (CSP Lua app)
-- 1) DRIVES the car: reads steer/gas/brake from cmd.txt (written by our Python policy) and applies
--    them DIRECTLY each frame via CSP's ac.overrideCarControls(0). No vJoy, no virtual gamepad.
-- 2) PUBLISHES live car state to state.json each frame (pos/look/yaw/vel/speed/collision) so the
--    Python policy localises on the REAL position. This makes NN Drive self-sufficient — you no
--    longer need the separate Car GT Logger app, and there's no stale-file trap.
-- cmd.txt format:  "<counter> <steer> <gas> <brake> <enabled>"   (steer -1..1, gas/brake 0..1)

local steer, gas, brake = 0.0, 0.0, 0.0     -- target values from the policy
local enabled = false
local lastCounter = -1
local lastApplied = -1
local overrideActive = false
local status = "waiting for cmd.txt"
local pubFrame = 0
local pubStatus = "state: starting"

-- SMOOTHING: the policy writes discrete steering targets (~14 Hz). We follow them with a
-- critically-damped SmoothDamp filter at AC's full render framerate -> velocity-continuous motion
-- (no stair-steps / choppiness), eases in AND out, converges exactly on target (no overshoot, so
-- it's precise), and tracks fast targets without the lag of a constant-rate slew. Pedals stay instant.
local STEER_SMOOTH_TIME = 0.06                -- response time (s): ~90%% in 120ms, peak ~7/s, smoother
                                              -- AND snappier than the old 6/s slew. LOWER = sharper
                                              -- (0.045 = very direct), HIGHER = softer (0.10 = floaty).
local appliedSteer = 0.0
local steerVel = 0.0                          -- SmoothDamp internal velocity (kept across frames)

-- Unity-style SmoothDamp: unconditionally stable, critically damped (no oscillation/overshoot).
local function smoothDamp(current, target, vel, smoothTime, dt)
  smoothTime = math.max(smoothTime, 1e-4)
  local omega = 2.0 / smoothTime
  local x = omega * dt
  local expf = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
  local change = current - target
  local temp = (vel + omega * change) * dt
  vel = (vel - omega * temp) * expf
  local out = target + (change + temp) * expf
  if ((target - current) > 0) == (out > target) then   -- clamp any overshoot -> precise settle
    out = target
    vel = (out - target) / dt
  end
  return out, vel
end

-- safe scalar field read: CSP's car userdata THROWS on an unknown field instead of returning nil,
-- so guard each one independently -> a missing field defaults to d, never aborts the write.
local function sget(car, key, d)
  local ok, v = pcall(function() return car[key] end)
  if ok and type(v) == "number" then return v end
  return d
end

local function readCmd()
  local content = io.load(__dirname .. '/cmd.txt', '')
  if content == nil or content == '' then return end
  local c, s, g, b, en = content:match("(%-?%d+)%s+(%-?[%d%.]+)%s+(%-?[%d%.]+)%s+(%-?[%d%.]+)%s*(%d*)")
  if c == nil then status = "bad cmd: " .. content; return end
  lastCounter = tonumber(c) or lastCounter
  steer = math.clamp(tonumber(s) or 0, -1, 1)
  gas   = math.clamp(tonumber(g) or 0, 0, 1)
  brake = math.clamp(tonumber(b) or 0, 0, 1)
  enabled = (en == '' or en == '1')
end

-- Publish the car's REAL state so the policy can localise + read heading/yaw (world frame, same as
-- the ingested track tangents). Generic: ac.getCar(0) + world position/look work for ANY car/track.
local function publishState()
  local car = ac.getCar(0)
  if car == nil then pubStatus = "state: waiting for car"; return end
  local pos = car.position
  local look = car.look
  if pos == nil or look == nil then pubStatus = "state: waiting for pose"; return end
  local vx, vy, vz = 0, 0, 0
  pcall(function() local v = car.velocity; if v ~= nil then vx, vy, vz = v.x, v.y, v.z end end)
  local yaw = 0.0                                  -- AC's REAL yaw rate (body frame) — clean, no diff noise
  pcall(function() local av = car.localAngularVelocity; if av ~= nil then yaw = av.y end end)
  local speedKmh = sget(car, "speedKmh", 0)
  local coll = sget(car, "collisionDepth", 0)
  local spline = sget(car, "normalizedSplinePosition", 0)
  pubFrame = pubFrame + 1
  local s = string.format(
    '{"frame":%d,"pos":[%.3f,%.3f,%.3f],"vel":[%.3f,%.3f,%.3f],"look":[%.4f,%.4f,%.4f],' ..
    '"speedKmh":%.2f,"collision":%.3f,"spline":%.5f,"yaw":%.4f}',
    pubFrame, pos.x, pos.y, pos.z, vx, vy, vz, look.x, look.y, look.z, speedKmh, coll, spline, yaw)
  local ok, err = pcall(function() io.save(__dirname .. '/state.json', s) end)
  pubStatus = ok and ("state: writing frame " .. pubFrame) or ("state WRITE ERR: " .. tostring(err))
end

-- Apply must run every frame so AC physics keeps seeing a fresh override.
local function apply(dt)
  pcall(publishState)                           -- always publish, even when control is disabled
  readCmd()
  if not enabled then
    status = "disabled (enabled flag = 0)"; appliedSteer = steer; steerVel = 0.0; return
  end
  -- critically-damped follow toward the policy's target (smooth + precise, framerate-correct)
  appliedSteer, steerVel = smoothDamp(appliedSteer, steer, steerVel, STEER_SMOOTH_TIME, math.max(dt or 0.016, 1e-4))
  local cc = ac.overrideCarControls(0)         -- 0 = player/first car
  if cc == nil then status = "ERROR: overrideCarControls returned nil"; return end
  cc.steer = appliedSteer                       -- smoothed full override (-1..1)
  cc.gas   = gas                                -- max(original, gas)
  cc.brake = brake                              -- max(original, brake)
  overrideActive = cc:active()
  lastApplied = lastCounter
  status = overrideActive and "DRIVING (override read by physics)" or "set, awaiting physics read"
end

function script.update(dt)
  apply(dt)
end

function script.windowMain(dt)
  apply(dt)
  ui.text("NN DRIVE  ·  direct CSP control")
  ui.separator()
  ui.text(string.format("steer  %+.3f -> %+.3f", steer, appliedSteer))
  ui.text(string.format("gas    %.3f", gas))
  ui.text(string.format("brake  %.3f", brake))
  ui.separator()
  ui.text("cmd #" .. tostring(lastCounter) .. "  applied #" .. tostring(lastApplied))
  ui.text("override active: " .. tostring(overrideActive))
  ui.textColored(status, overrideActive and rgbm(0.2, 1, 0.6, 1) or rgbm(1, 0.8, 0.3, 1))
  ui.separator()
  ui.textColored(pubStatus, pubStatus:find("ERR") and rgbm(1, 0.4, 0.4, 1) or rgbm(0.6, 0.8, 1, 1))
end
