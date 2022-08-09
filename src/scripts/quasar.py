import emcee
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from george import kernels, GP
from paths import figures

sns.set_context("notebook")
sns.set_style("ticks")

## Initialise random number generator
rng = np.random.default_rng(seed=1234)
np.random.seed(1231)

## Time stamps
T = 1
t_obs = rng.uniform(0, 1500, 100)
nobs = len(t_obs)
dt_true = 419.3
x_obs = np.concatenate([t_obs, t_obs + dt_true])
source_ids = np.concatenate([np.zeros_like(t_obs, "int"), np.ones_like(t_obs, "int")])
l1 = source_ids == 0
l2 = source_ids == 1

## Variability
gp_amp, gp_tau = 0.1, 5000
k = gp_amp**2 * kernels.Matern32Kernel(gp_tau)
gp = GP(k, mean=0)
p_gp_true = gp.get_parameter_vector()
rn = gp.sample(x_obs, 1).flatten()
rn1 = rn[l1]
rn2 = rn[l2]

## Observations
sig1 = 0.02
sig2 = 0.04
m1_true = 1.0
m2_true = 0.5
wn = rng.normal(0, 1, len(x_obs))
wn[l1] *= sig1
wn[l2] *= sig2
y1 = m1_true + rn1 + wn[l1]
y2 = m2_true + rn2 + wn[l2]


def lnprior_gp(p_gp):
    lnp = 0.0
    if (abs(p_gp - p_gp_true) > 5).any():
        return -np.inf
    return lnp


def lnprior_obs(p_obs, t1, t2):
    dt, m1, m2 = p_obs
    lnp = 0.0
    t2_shifted = t2 + dt
    if t2_shifted.min() > t1.max():
        return -np.inf
    if t2_shifted.max() < t1.min():
        return -np.inf
    if abs(m1 - m1_true) > 5:
        return -np.inf
    if abs(m2 - m2_true) > 5:
        return -np.inf
    return lnp


def lnprob(p, t, y, source_id, wn):
    p_gp = p[:2]
    res = lnprior_gp(p_gp)
    if not np.isfinite(res):
        return -np.inf
    p_obs = p[2:]
    l1 = source_id == 0
    t1 = t[l1]
    t2 = t[~l1]
    res += lnprior_obs(p_obs, t1, t2)
    if not np.isfinite(res):
        return -np.inf

    gp.set_parameter_vector(p[:2])
    x = np.concatenate([t1, t2 + p[2]])
    r = y.copy()
    r[l1] -= p_obs[1]
    r[~l1] -= p_obs[2]
    try:
        gp.compute(x, yerr=wn)
        ll = gp.log_likelihood(r)
        if not np.isfinite(ll):
            print("GP likelihood is not finite")
            print("Parameters:", p)
            print("Log prior:", res)
            print("Log likelihood:", ll)
        res += ll
    except:
        print("Could not evaluate GP likelihood")
        print("Parameters:", p)
        print("Log prior:", res)
    return res


p_true = np.concatenate([p_gp_true, [dt_true, m1_true, m2_true]])
ts = np.concatenate([t_obs, t_obs])
ys = np.concatenate([y1, y2])
print(p_true, lnprob(p_true, ts, ys, source_ids, wn))

ndim, nwalkers = len(p_true), 32
p0 = p_true + 1e-8 * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(ts, ys, source_ids, wn))
Nsteps = 3000
sampler.run_mcmc(p0, Nsteps, progress=True)


def eval_gp(p, t, y, source_id, wn, x_samp, sample=True):
    gp.set_parameter_vector(p[:2])
    dt, m1, m2 = p[2:]
    #    print(source_id)
    l1 = source_id == 0
    t1 = t[l1]
    t2 = t[~l1]
    #    print(t1.shape,t2.shape)
    x = np.concatenate([t1, t2 + dt])
    r = y.copy()
    r[l1] -= m1
    r[~l1] -= m2
    gp.compute(x, yerr=wn)
    if sample:
        s = gp.sample_conditional(r, x_samp).flatten()
        return s
    else:
        m, v = gp.predict(r, x_samp, return_var=True, return_cov=False)
        return m, np.sqrt(v)


tau = int(Nsteps / 50)

samples = sampler.get_chain(discard=5 * tau, thin=tau, flat=True)
t_grid = np.linspace(-500, 2500, 1000)

lnpr = sampler.get_log_prob(discard=5 * tau, thin=tau, flat=True)
i_best = np.argmax(lnpr)
p_best = samples[i_best]
# m, s = eval_gp(p_best, ts, ys, source_ids, wn, t_grid, sample=False)
# plt.plot(t_grid, m + p_best[3], 'C0--', alpha = 0.5)
# plt.fill_between(t_grid, m + p_best[3] + s, m + p_best[3] - s, color = 'C0', lw = 0, alpha = 0.2)
# plt.plot(t_grid - p_best[2], m + p_best[4], 'C1--', alpha=0.5)
# plt.fill_between(t_grid - p_best[2], m + p_best[4] + s, m + p_best[4] - s, color = 'C1', lw = 0, alpha = 0.2)
plt.errorbar(t_obs, y1, fmt="k.", yerr=sig1, capsize=0)
plt.errorbar(t_obs, y2, fmt="kx", yerr=sig2, capsize=0)

for s in samples[rng.integers(len(samples), size=24)]:
    v = eval_gp(s, ts, ys, source_ids, wn, t_grid)
    plt.plot(t_grid, v + s[3], "C0-", alpha=0.1)
    plt.plot(t_grid - s[2], v + s[4], "C3-", alpha=0.1)

plt.ylabel("flux")
plt.xlabel("time (days)")
plt.xlim(t_obs.min() - p_best[2] / 2, t_obs.max() + p_best[2] / 2)
# plt.tight_layout()
# plt.title("fit accounting for correlated noise");
plt.savefig(figures / "quasar.pdf")
