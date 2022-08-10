# %%
import numpyro

numpyro.set_host_device_count(2)

from astropy.table import Table
import matplotlib.pyplot as plt
from paths import figures, data

# %%
data = Table.read(data / "quasar.csv")

# %%
# plt.plot(data["jd"], data["a_mag"], ".")
# plt.plot(data["jd"], data["b_mag"] + 1, ".")
# plt.ylim(plt.ylim()[::-1])

# %%
from functools import partial
import jax
import jax.numpy as jnp
import tinygp
from tinygp import kernels, transforms, GaussianProcess


def time_delay_transform(lag, X):
    t, band = X
    return t - lag * band


def mean_func(means, X):
    t, band = X
    return means[band]


@tinygp.helpers.dataclass
class Multiband(kernels.quasisep.Wrapper):
    amplitudes: jnp.ndarray

    def coord_to_sortable(self, X):
        return X[0]

    def observation_model(self, X):
        return self.amplitudes[X[1]] * self.kernel.observation_model(X[0])


N = len(data)
X = jnp.concatenate((data["jd"].value, data["jd"].value)), jnp.concatenate(
    (jnp.zeros(N, dtype=int), jnp.ones(N, dtype=int))
)
y = jnp.concatenate((data["a_mag"].value, data["b_mag"].value))
diag = jnp.concatenate((data["a_mag_err"].value, data["b_mag_err"].value)) ** 2


def build_gp(params, X, diag):
    band = X[1]
    t = time_delay_transform(params["lag"], X)
    inds = jnp.argsort(t)
    kernel = Multiband(
        amplitudes=params["amps"],
        kernel=kernels.quasisep.Matern32(jnp.exp(params["log_ell"])),
    )
    mean = partial(mean_func, params["means"])
    return (
        GaussianProcess(kernel, (t[inds], band[inds]), diag=diag[inds], mean=mean),
        inds,
    )


@jax.jit
def loss(params):
    gp, inds = build_gp(params, X, diag)
    return -gp.log_probability(y[inds])


init = {
    "lag": 536.0,
    "log_ell": jnp.log(100.0),
    "amps": jnp.stack((jnp.std(data["a_mag"].value), jnp.std(data["b_mag"].value))),
    "means": jnp.stack(
        (jnp.median(data["a_mag"].value), jnp.median(data["b_mag"].value))
    ),
}
loss(init)

# %%
import jaxopt

opt = jaxopt.ScipyMinimize(fun=loss)

minimum = loss(init), init
lags = []
vals = []
for lag in jnp.linspace(0, 1000, 100):
    init["lag"] = lag
    soln = opt.run(init)
    lags.append(soln.params["lag"])
    vals.append(soln.state.fun_val)
    if soln.state.fun_val < minimum[0]:
        minimum = soln.state.fun_val, soln.params
# %%
# plt.plot(lags, vals, ".", alpha=0.1)

# %%
init = minimum[1]

# %%
init

# %%
t_lagged = X[0] - soln.params["lag"] * X[1]
# plt.scatter(
#     t_lagged,
#     (y - soln.params["means"][X[1]]) / soln.params["amps"][X[1]],
#     c=X[1],
#     s=10,
#     edgecolor="k",
#     linewidth=0.5,
# )
t_grid = jnp.linspace(t_lagged.min() - 200, t_lagged.max() + 200, 1000)

# %%
from numpyro import infer, distributions as dist


def model(X, diag, y):
    lag = numpyro.sample("lag", dist.Uniform(-1000.0, 1000.0))
    log_ell = numpyro.sample("log_ell", dist.Uniform(jnp.log(10), jnp.log(1000.0)))
    amps = numpyro.sample("amps", dist.Uniform(-5.0, 5.0), sample_shape=(2,))
    mean_a = numpyro.sample("mean_a", dist.Uniform(17.0, 18.0))
    delta_mean = numpyro.sample("delta_mean", dist.Uniform(-2.0, 2.0))
    means = jnp.stack((mean_a, mean_a + delta_mean))

    params = {
        "lag": lag,
        "log_ell": log_ell,
        "amps": amps,
        "means": means,
    }
    gp, inds = build_gp(params, X, diag)
    numpyro.sample("y", gp.numpyro_dist(), obs=y[inds])

    numpyro.deterministic(
        "pred_a",
        gp.condition(y[inds], (t_grid, jnp.zeros_like(t_grid, dtype=int))).gp.loc,
    )
    numpyro.deterministic(
        "pred_b",
        gp.condition(y[inds], (t_grid, jnp.ones_like(t_grid, dtype=int))).gp.loc,
    )


init_params = dict(soln.params)
init_params["mean_a"] = init_params["means"][0]
init_params["delta_mean"] = init_params["means"][1] - init_params["means"][0]
sampler = infer.MCMC(
    infer.NUTS(
        model,
        dense_mass=True,
        target_accept_prob=0.9,
        init_strategy=infer.init_to_value(values=init_params),
    ),
    num_warmup=1000,
    num_samples=1000,
    num_chains=2,
    progress_bar=True,
    chain_method="sequential",
)
sampler.run(jax.random.PRNGKey(34923), X, diag, y)


# %%
import corner
import arviz as az

# %%
inf_data = az.from_numpyro(sampler)

# %%
corner.corner(
    inf_data,
    var_names=["lag", "delta_mean"],
    labels=["time lag", "mean magnitude offset"],
)
plt.savefig(figures / "quasar2_posteriors.png", bbox_inches="tight")


# %%
samples = sampler.get_samples()
lag = jnp.median(samples["lag"])
pred_a = samples["pred_a"]
pred_b = samples["pred_b"]
inds = jax.random.randint(jax.random.PRNGKey(134), (12,), 0, len(pred_a))

offset = 0.3

plt.figure()
plt.plot(t_grid + lag, pred_a[inds, :].T, c="C0", alpha=0.3, lw=0.5)
plt.plot(t_grid + lag, pred_b[inds, :].T + offset, c="C1", alpha=0.3, lw=0.5)

plt.errorbar(
    data["jd"].value + lag,
    data["a_mag"].value,
    yerr=data["a_mag_err"].value,
    fmt="oC0",
    label="A",
    markersize=4,
    linewidth=1,
)
plt.errorbar(
    data["jd"].value,
    data["b_mag"].value + offset,
    yerr=data["a_mag_err"].value,
    fmt="oC1",
    label="B",
    markerfacecolor="white",
    markersize=4,
    linewidth=1,
)
plt.ylim(plt.ylim()[::-1])
plt.xlabel(f"time [days; A shifted +{lag:.0f} days]")
plt.ylabel(f"magnitude [B shifted +{offset}]")
plt.xlim(t_grid.min() + lag, t_grid.max() + lag)
plt.legend()
plt.savefig(figures / "quasar2.png", bbox_inches="tight")

# %%
