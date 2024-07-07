import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import norm
import plotly.graph_objects as go


def kernel_rbf(X1, X2, length_scale=1.0, sigma_f=1.0):
    """
    Radial Basis Function (RBF) kernel (also known as the Gaussian kernel).

    Args:
        X1 (array-like): First set of input points.
        X2 (array-like): Second set of input points.
        length_scale (float): Length scale parameter of the RBF kernel.
        sigma_f (float): Signal variance parameter of the RBF kernel.

    Returns:
        ndarray: Covariance matrix.
    """
    sqdist = cdist(X1, X2, "sqeuclidean")
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)


class GaussianProcess:
    """
    Gaussian Process Regression model.

    Args:
        kernel (callable): Covariance function.
        noise (float): Noise term for numerical stability.
    """

    def __init__(self, kernel, noise=1e-10):
        self.kernel = kernel
        self.noise = noise

    def fit(self, X, y):
        """
        Fit the Gaussian Process model.

        Args:
            X (array-like): Training input samples.
            y (array-like): Training target values.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.K = self.kernel(self.X_train, self.X_train) + self.noise**2 * np.eye(
            len(self.X_train)
        )
        self.L = np.linalg.cholesky(self.K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))

    def predict(self, X):
        """
        Predict using the Gaussian Process model.

        Args:
            X (array-like): Input samples for prediction.

        Returns:
            tuple: Mean and variance of the predictions.
        """
        K_trans = self.kernel(X, self.X_train)
        K_test = self.kernel(X, X)
        mu = K_trans @ self.alpha
        v = np.linalg.solve(self.L, K_trans.T)
        cov = K_test - v.T @ v
        return mu, np.diag(cov)


def expected_improvement(X, gp, y_max, xi=0.01):
    """
    Compute the Expected Improvement acquisition function.

    Args:
        X (array-like): Input samples.
        gp (GaussianProcess): Fitted Gaussian Process model.
        y_max (float): Maximum observed value of the objective function.
        xi (float): Exploration-exploitation trade-off parameter.

    Returns:
        ndarray: Expected Improvement values for the input samples.
    """
    mu, sigma = gp.predict(X)
    sigma = np.sqrt(sigma)
    with np.errstate(divide="warn"):
        imp = mu - y_max - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei


def bayesian_optimization(obj, bounds, n_iter=10, num_init_samples=2, seed=42):
    """
    Perform Bayesian Optimization on the objective function.

    Args:
        obj (callable): Objective function to be minimized.
        bounds (array-like): Bounds of the search space.
        n_iter (int): Number of optimization iterations.
        num_init_samples (int): Number of initial random samples.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Optimal input value and list of results for each iteration.
    """
    rng = np.random.default_rng(seed)
    X_sample = rng.uniform(bounds[0], bounds[1], size=(num_init_samples, 1))
    y_sample = obj(X_sample).reshape(-1, 1)
    results = []
    gp = GaussianProcess(kernel=kernel_rbf)
    results.append((gp, np.empty((0, 1)), np.empty((0, 1))))

    for i in range(n_iter):
        gp = GaussianProcess(kernel=kernel_rbf)
        gp.fit(X_sample, y_sample)
        results.append((gp, X_sample.copy(), y_sample.copy()))

        EI = lambda x: -expected_improvement(x.reshape(-1, 1), gp, np.max(y_sample))
        res = minimize(
            EI,
            rng.uniform(bounds[0], bounds[1], size=(1,)),
            bounds=[bounds],
            method="L-BFGS-B",
        )
        X_new = res.x.reshape(1, -1)
        y_new = obj(X_new).reshape(-1, 1)
        X_sample = np.vstack((X_sample, X_new))
        y_sample = np.vstack((y_sample, y_new))

    gp = GaussianProcess(kernel=kernel_rbf)
    gp.fit(X_sample, y_sample)
    results.append((gp, X_sample.copy(), y_sample.copy()))
    idx_min = np.argmin(y_sample)
    x_opt = X_sample[idx_min]
    return x_opt, results


def plot_gp_animation(results, bounds, objective):
    x = np.linspace(bounds[0], bounds[1], 1000).reshape(-1, 1)
    frames = []
    steps = []

    for i, (gp, X_sample, y_sample) in enumerate(results):
        if i == 0:
            mu = np.zeros_like(x).flatten()
            sigma = np.ones_like(x).flatten()
        else:
            mu, sigma = gp.predict(x)
            mu = mu.flatten()
            sigma = sigma.flatten()

        frame = go.Frame(
            data=[
                go.Scatter(
                    x=x.flatten(),
                    y=objective(x).flatten(),
                    mode="lines",
                    name="Objective Function",
                ),
                go.Scatter(
                    x=x.flatten(),
                    y=mu,
                    mode="lines",
                    name="GP (Mean)",
                    line=dict(dash="dash"),
                ),
                go.Scatter(
                    x=np.concatenate([x.flatten(), x.flatten()[::-1]]),
                    y=np.concatenate([mu - 1.96 * sigma, (mu + 1.96 * sigma)[::-1]]),
                    fill="toself",
                    fillcolor="rgba(255, 0, 0, 0.2)",
                    line=dict(color="rgba(255, 0, 0, 0)"),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                go.Scatter(
                    x=X_sample.flatten(),
                    y=y_sample.flatten(),
                    mode="markers",
                    name="Samples",
                    marker=dict(color="red", size=14),
                ),
                go.Scatter(
                    x=[X_sample[-1][0]] if len(X_sample) > 0 else [],
                    y=[y_sample[-1][0]] if len(y_sample) > 0 else [],
                    mode="markers",
                    marker=dict(color="orange", size=16),
                    name="Latest Sample",
                ),
                go.Scatter(
                    x=[X_sample[np.argmin(y_sample)][0]] if len(y_sample) > 0 else [],
                    y=[y_sample[np.argmin(y_sample)][0]] if len(y_sample) > 0 else [],
                    mode="markers+text",
                    marker=dict(color="green", size=18),
                    text=(
                        [
                            f"Optimum: x={X_sample[np.argmin(y_sample)][0]:.2f}, f(x)={y_sample[np.argmin(y_sample)][0]:.2f}"
                        ]
                        if len(y_sample) > 0
                        else []
                    ),
                    textposition="top center",
                    name="Current Optimum",
                ),
            ],
            name=f"Iteration {i}",
        )
        frames.append(frame)
        steps.append(
            dict(
                label=f"Iteration {i}",
                method="animate",
                args=[
                    [f"Iteration {i}"],
                    {"frame": {"duration": 1000, "redraw": True}, "mode": "immediate"},
                ],
            )
        )

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title=dict(
                text="Illustration of Bayesian Optimization with Gaussian Process",
                x=0.5,
                y=0.9,
                xanchor="center",
                yanchor="top",
            ),
            xaxis=dict(range=[bounds[0], bounds[1]], title="x"),
            yaxis=dict(title="f(x)"),
            width=1280,
            height=720,
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 1000, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "right",
                    "showactive": False,
                    "type": "buttons",
                    "x": 1.17,
                    "xanchor": "right",
                    "y": 0.78,
                    "yanchor": "top",
                },
                {
                    "buttons": steps,
                    "direction": "down",
                    "showactive": True,
                    "x": 1.17,
                    "xanchor": "right",
                    "y": 0.70,
                    "yanchor": "top",
                    "type": "dropdown",
                    "bgcolor": "rgba(255, 255, 255, 0.8)",
                },
            ],
        ),
        frames=frames,
    )

    fig.write_html(
        "bayesian_optimization_animation.html", auto_open=True, auto_play=False
    )

    fig.show()


# Example of objective function
def objective(X):
    return np.sin(-3 * X) + np.sin(X) + 0.2 * X**2 + 0.1 * X


# Perform Bayesian Optimization
bounds = np.array([-4, 4])
x_opt, results = bayesian_optimization(objective, bounds, n_iter=15, seed=111)

# Plot the optimization process
plot_gp_animation(results, bounds, objective)
