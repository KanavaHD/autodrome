"""
Soft Actor-Critic (SAC) — the deep-RL algorithm GT Sophy is built on (it used QR-SAC, a
distributional cousin). Model-FREE: the agent learns to drive by actually driving and optimising
reward, no world model. This is the right tool — what the linear CEM policy fundamentally lacked
is here: deep value networks (critics) that learn what's good LONG-term, and entropy auto-tuning
that keeps exploration sane (so it doesn't get stuck stalling).

Standard, correct SAC: squashed-Gaussian actor, twin critics + targets, automatic temperature.
Small nets on CPU — fine, because the bottleneck is AC running in real time, not the maths.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hid=256):
        super().__init__()
        self.body = mlp([obs_dim, hid, hid])
        self.mu = nn.Linear(hid, act_dim)
        self.log_std = nn.Linear(hid, act_dim)

    def forward(self, o):
        h = F.relu(self.body(o))
        return self.mu(h), self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, o):
        mu, log_std = self(o)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        a = torch.tanh(x)
        logp = dist.log_prob(x).sum(-1) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp

    def act(self, o, deterministic=False):
        mu, log_std = self(o)
        if deterministic:
            return torch.tanh(mu)
        x = torch.distributions.Normal(mu, log_std.exp()).rsample()
        return torch.tanh(x)

    def activations(self, o):
        """Layer activations for the 'brain' visual: hidden-1, hidden-2 (both 512), output (steer,thr)."""
        z1 = self.body[1](self.body[0](o))    # relu(Linear1(o))  -> hidden layer 1
        h2 = F.relu(self.body[2](z1))         # relu(Linear2(z1)) -> hidden layer 2
        out = torch.tanh(self.mu(h2))         # action
        return z1, h2, out


class Critic(nn.Module):
    """Distributional (QR) twin critic — the GT-Sophy upgrade. Instead of one scalar Q (the AVERAGE
    return), each critic outputs N quantiles of the RETURN DISTRIBUTION. That lets the policy 'see'
    that a late-braking line is e.g. 90% brilliant / 10% small-loss and commit to it, instead of a
    mean-critic shying away from anything risky. This is what produces braver, faster lines."""
    def __init__(self, obs_dim, act_dim, hid=256, n_quantiles=32):
        super().__init__()
        self.n = n_quantiles
        self.q1 = mlp([obs_dim + act_dim, hid, hid, n_quantiles])
        self.q2 = mlp([obs_dim + act_dim, hid, hid, n_quantiles])

    def forward(self, o, a):
        x = torch.cat([o, a], -1)
        return self.q1(x), self.q2(x)        # each (B, n_quantiles)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=300000):
        self.o = np.zeros((size, obs_dim), np.float32)
        self.o2 = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros(size, np.float32)
        self.d = np.zeros(size, np.float32)
        self.ptr = 0; self.n = 0; self.size = size

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.o[i] = o; self.a[i] = a; self.r[i] = r; self.o2[i] = o2; self.d[i] = d
        self.ptr = (i + 1) % self.size; self.n = min(self.n + 1, self.size)

    def add_batch(self, O, A, R, O2, D):
        m = len(O); i = self.ptr
        if i + m <= self.size:
            sl = slice(i, i + m)
            self.o[sl] = O; self.a[sl] = A; self.r[sl] = R; self.o2[sl] = O2; self.d[sl] = D
        else:
            for j in range(m):
                self.add(O[j], A[j], R[j], O2[j], D[j])
            return
        self.ptr = (i + m) % self.size; self.n = min(self.n + m, self.size)

    def sample(self, bs):
        idx = np.random.randint(0, self.n, bs)
        t = lambda x: torch.as_tensor(x)
        return t(self.o[idx]), t(self.a[idx]), t(self.r[idx]), t(self.o2[idx]), t(self.d[idx])

    def __len__(self):
        return self.n


class SAC:
    def __init__(self, obs_dim, act_dim, gamma=0.99, tau=0.005, lr=3e-4, hid=256, buf=300000,
                 alpha_min=0.05, target_entropy=None, device=None, n_quantiles=32, kappa=1.0):
        self.obs_dim = obs_dim; self.act_dim = act_dim; self.alpha_min = alpha_min
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.nq = n_quantiles; self.kappa = kappa
        self.actor = Actor(obs_dim, act_dim, hid).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hid, n_quantiles).to(self.device)
        self.targ = Critic(obs_dim, act_dim, hid, n_quantiles).to(self.device)
        self.targ.load_state_dict(self.critic.state_dict())
        for p in self.targ.parameters():
            p.requires_grad = False
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr)
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr)
        # quantile midpoints tau_i = (i+0.5)/N — the target fractions each quantile estimates
        self.tau_hat = ((torch.arange(n_quantiles, device=self.device).float() + 0.5) / n_quantiles)
        # target entropy: standard is -act_dim, but that lets the policy go near-deterministic and
        # collapse into a timid optimum. A less-negative target keeps it exploring (driving) longer.
        self.target_entropy = float(target_entropy) if target_entropy is not None else -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.a_opt = torch.optim.Adam([self.log_alpha], lr)
        self.gamma = gamma; self.tau = tau
        self.nstep_gamma = gamma          # bootstrap discount (= gamma**n for n-step returns)
        self.buf = ReplayBuffer(obs_dim, act_dim, buf)
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def brain(self, obs_row, h_bins=24):
        """Lead car's network activations for the live brain panel: input (56), 2 pooled hidden
        layers (h_bins each), output (steer, throttle). Pooled so the 512-wide layers are renderable."""
        with torch.no_grad():
            o = torch.as_tensor(obs_row, dtype=torch.float32, device=self.device).reshape(1, -1)
            z1, h2, out = self.actor.activations(o)
        import numpy as _np
        def pool(t):
            a = t.squeeze(0).detach().cpu().numpy()
            e = _np.linspace(0, len(a), h_bins + 1).astype(int)
            return [round(float(a[e[i]:e[i + 1]].mean()), 3) for i in range(h_bins)]
        return {"in": [round(float(x), 3) for x in _np.asarray(obs_row).ravel()],
                "h1": pool(z1), "h2": pool(h2),
                "out": [round(float(out[0, 0]), 3), round(float(out[0, 1]), 3)]}

    def select_action(self, obs, deterministic=False):
        o = torch.as_tensor(np.asarray(obs, np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a = self.actor.act(o, deterministic)
        return a.squeeze(0).cpu().numpy()

    def select_actions(self, obs_batch, deterministic=False):
        o = torch.as_tensor(np.asarray(obs_batch, np.float32)).to(self.device)
        with torch.no_grad():
            a = self.actor.act(o, deterministic)
        return a.cpu().numpy()

    def act_tensor(self, obs, deterministic=False):
        """Action for a GPU obs tensor -> GPU action tensor (no CPU round-trip). For the GPU sim."""
        with torch.no_grad():
            return self.actor.act(obs, deterministic)

    def update(self, bs=256):
        o, a, r, o2, d = self.buf.sample(bs)
        dev = self.device
        return self.update_core(o.to(dev), a.to(dev), r.to(dev), o2.to(dev), d.to(dev))

    def _quantile_huber(self, pred, target):
        """QR loss: pred (B,N) quantiles vs target (B,N) samples. Pinball-weighted Huber over the
        full NxN pairwise difference — this is what fits the whole return DISTRIBUTION, not the mean."""
        # u[b,i,j] = target_j - pred_i
        u = target.unsqueeze(1) - pred.unsqueeze(2)                  # (B, N_pred, N_target)
        k = self.kappa
        huber = torch.where(u.abs() <= k, 0.5 * u.pow(2), k * (u.abs() - 0.5 * k))
        tau = self.tau_hat.view(1, -1, 1)                           # (1, N_pred, 1)
        loss = (torch.abs(tau - (u.detach() < 0).float()) * huber / k).mean(2).sum(1)
        return loss.mean()

    def update_core(self, o, a, r, o2, d):
        """The QR-SAC gradient step on already-on-device tensors (used by the all-GPU trainer)."""
        # distributional critics: bootstrap the target RETURN DISTRIBUTION
        with torch.no_grad():
            a2, logp2 = self.actor.sample(o2)
            z1t, z2t = self.targ(o2, a2)                            # (B,N) quantiles each
            zt = torch.min(z1t, z2t) - self.alpha * logp2.unsqueeze(1)   # entropy-adjusted
            y = r.unsqueeze(1) + self.nstep_gamma * (1 - d).unsqueeze(1) * zt  # (B,N) n-step target
        z1, z2 = self.critic(o, a)
        q_loss = self._quantile_huber(z1, y) + self._quantile_huber(z2, y)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()
        # actor: maximise the MEAN of the (min) return distribution (mean over quantiles = the value)
        ap, logp = self.actor.sample(o)
        z1p, z2p = self.critic(o, ap)
        qp = torch.min(z1p.mean(1), z2p.mean(1))
        pi_loss = (self.alpha.detach() * logp - qp).mean()
        self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()
        # temperature (auto-tuned) — but FLOOR it so exploration can't collapse to ~0 and trap the
        # policy in a timid local optimum (the "creep along, never crash" failure we hit).
        a_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(min=math.log(self.alpha_min))
        # soft-update targets
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.targ.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        self.updates += 1
        return float(q_loss), float(pi_loss), float(self.alpha.detach())

    def save(self, fp, extra=None):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "targ": self.targ.state_dict(), "log_alpha": self.log_alpha.detach(),
                    "obs_dim": self.obs_dim, "act_dim": self.act_dim, "nq": self.nq,
                    "updates": self.updates, **(extra or {})}, fp)

    def load(self, fp):
        d = torch.load(fp, map_location=self.device)
        self.actor.load_state_dict(d["actor"]); self.critic.load_state_dict(d["critic"])
        self.targ.load_state_dict(d["targ"])
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"].to(self.device))
        self.updates = d.get("updates", 0)
        return d
