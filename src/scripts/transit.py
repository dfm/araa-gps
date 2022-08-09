import emcee
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ldtk import tess
from pytransit import LDTkLDModel, RoadRunnerModel

sns.set_style("ticks")

## Time stamps
T = 1
t_grid = np.linspace(-0.3, 0.3, 1000)
l = np.where(abs(t_grid) <= 0.2)[0][::10]
t_obs = t_grid[l]

# Transit
ldm = LDTkLDModel(
    teff=(5500, 150), logg=(4.5, 0.1), z=(0.0, 0.1), frozen=True, pbs=[tess]
)
tm = RoadRunnerModel(ldm)


def transit_func(p, x, per=1.0):
    ars = 3.0
    b = 0.1
    t0, rprs = p
    tm.set_data(x)
    return tm.evaluate(
        k=rprs, t0=t0, p=per, a=ars, i=np.arccos(b / ars), e=0, w=0, ldc=[None]
    )


rprs = 0.1
t0 = 0.0
p_tr_true = np.array([t0, rprs])
transit = transit_func(p_tr_true, t_obs)
# plt.figure(figsize=(6,5))
# plt.plot(t_obs, transit, 'k.');

## Add white noise
sig = 0.001
rng = np.random.default_rng(seed=1234)
wn = rng.normal(0, sig, len(t_obs))

## Add correlated noise
from george import GP, kernels

gp_amp, gp_tau = 0.002, 0.0005
k = gp_amp**2 * kernels.ExpSquaredKernel(gp_tau)
gp = GP(k, mean=0)
p_gp_true = gp.get_parameter_vector()
# seed = 1232 gives significant Rp/Rs offset
np.random.seed(1231)
rn = gp.sample(t_obs, 1).flatten()

y_obs = transit + wn + rn

## Fit with GP
def lnprior(p):
    lnp = 0
    p_gp, p_tr = p[:2], p[2:]
    if (abs(p_gp - p_gp_true) > 5).any():
        return -np.inf
    if (p_tr[1] < 0.0) or (p_tr[1] > 1):  # rprs between 0 and 1
        return -np.inf
    #    if p_tr[2] < 1: # a > Rs
    #        return -np.inf
    #    if (p_tr[-1] < 0.) or (p_tr[-1] > 1): # impact parameter between 0 and 1
    #        return -np.inf
    return lnp


def lnprob(p, x, y, wn):
    res = lnprior(p)
    if not np.isfinite(res):
        return -np.inf
    model = transit_func(p[2:], x)
    r = y - model
    gp.set_parameter_vector(p[:2])
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


p_true = np.concatenate([p_gp_true, p_tr_true])
print(p_true, lnprob(p_true, t_obs, y_obs, sig))

ndim, nwalkers = len(p_true), 32
p0 = p_true + 1e-8 * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t_obs, y_obs, sig))

sampler.run_mcmc(p0, 3000, progress=True)
tau = int(sampler.get_autocorr_time().max())


def eval_gp(p, x, y, wn, x_samp, sample=True):
    m = transit_func(p[2:], x_samp)
    r = y - transit_func(p[2:], x)
    gp.set_parameter_vector(p[:2])
    gp.compute(x, yerr=wn)
    if sample:
        v = gp.sample_conditional(r, x_samp).flatten()
    else:
        v = gp.predict(r, x_samp, return_var=False, return_cov=False)
    return m, v


samples = sampler.get_chain(discard=5 * tau, thin=tau, flat=True)

## White noise only fit
def lnprior_nogp(p_tr):
    lnp = 0
    if (p_tr[1] < 0.0) or (p_tr[1] > 1):  # rprs between 0 and 1
        return -np.inf
    #    if p_tr[2] < 1: # a > Rs
    #        return -np.inf
    #    if (p_tr[-1] < 0.) or (p_tr[-1] > 1): # impact parameter between 0 and 1
    #        return -np.inf
    return lnp


def lnprob_nogp(p, x, y, wn):
    res = lnprior_nogp(p)
    if not np.isfinite(res):
        return -np.inf
    model = transit_func(p, x)
    r = y - model
    gp.set_parameter_vector([-20, 20])
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


print(p_tr_true, lnprob_nogp(p_tr_true, t_obs, y_obs, sig))

ndim2, nwalkers = len(p_tr_true), 32
p0 = p_tr_true + 1e-8 * np.random.randn(nwalkers, ndim2)
sampler2 = emcee.EnsembleSampler(nwalkers, ndim2, lnprob_nogp, args=(t_obs, y_obs, sig))

sampler2.run_mcmc(p0, 3000, progress=True)
tau = int(sampler2.get_autocorr_time().max())
print(tau)
samples2 = sampler2.get_chain(discard=5 * tau, thin=tau, flat=True)

## Plot corner plot
fig1 = corner.corner(
    samples2,
    truths=p_tr_true,
    truth_color="k",
    color="C0",
    labels=labels[2:],
    quantiles=(0.16, 0.84),
)
corner.corner(samples[:, 2:], color="C3", fig=fig1, quantiles=(0.16, 0.84))
plt.savefig("transit_posteriors.pdf")

## Plot fits
plt.figure()
plt.errorbar(t_obs, y_obs, yerr=sig, fmt=".k", capsize=0)
for s in samples2[rng.integers(len(samples2), size=24)]:
    m = transit_func(s, t_grid)
    plt.plot(t_grid, m, color="C0", alpha=0.3)

for s in samples[rng.integers(len(samples), size=24)]:
    m, v = eval_gp(s, t_obs, y_obs, sig, t_grid)
    plt.plot(t_grid, m + v, color="C3", alpha=0.1)

plt.ylabel("normalised flux")
plt.xlabel("time (days)")
plt.xlim(-0.21, 0.21)
plt.savefig("transit.pdf")
