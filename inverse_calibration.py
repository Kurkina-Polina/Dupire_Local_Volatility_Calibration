import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

from crank_nicolson_dupire import solve_dupire_pde


def _laplacian_penalty(alpha_grid):
    """Discrete Laplacian-based smoothness penalty for variance nodes."""
    pen = 0.0
    m, n = alpha_grid.shape
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            center = alpha_grid[i, j]
            pen += (center - alpha_grid[i - 1, j]) ** 2
            pen += (center - alpha_grid[i + 1, j]) ** 2
            pen += (center - alpha_grid[i, j - 1]) ** 2
            pen += (center - alpha_grid[i, j + 1]) ** 2
    return pen / max((m - 2) * (n - 2), 1)


def _build_sigma_grid(alpha, K_nodes, T_nodes, K_full, T_full):
    """Bilinear interpolation from coarse variance nodes to PDE grid."""
    alpha_grid = alpha.reshape(len(T_nodes), len(K_nodes))
    interp = RegularGridInterpolator((T_nodes, K_nodes), alpha_grid, bounds_error=False, fill_value=None)
    TT, KK = np.meshgrid(T_full, K_full, indexing="ij")
    pts = np.stack([TT.ravel(), KK.ravel()], axis=1)
    sigma_grid = np.sqrt(np.clip(interp(pts).reshape(TT.shape), 1e-8, None))
    return sigma_grid


def calibrate_local_vol(market_prices, K_full, T_full, K_nodes, T_nodes, sigma_init, r, S0, lam=1e-2, maxiter=50, verbose=True):
    """
    Inverse problem: fit local volatility surface sigma(K,T) to market option prices.
    - market_prices: ndarray [M_full, N_full] target CALL prices on K_full x T_full grid
    - K_full, T_full: 1D arrays defining PDE grid used for pricing
    - K_nodes, T_nodes: coarse grids for variance parameters
    - sigma_init: initial guess (scalar) for volatility
    - lam: Tikhonov weight on surface smoothness (Laplacian of variance)
    - maxiter: optimizer iterations
    Returns calibrated sigma_grid over full PDE grid.
    """
    M_full, N_full = market_prices.shape
    alpha0 = np.full((len(T_nodes), len(K_nodes)), sigma_init**2)
    alpha0_vec = alpha0.ravel()
    
    iteration = [0]
    best_loss = [float('inf')]

    def loss(alpha_vec):
        iteration[0] += 1
        alpha_grid = alpha_vec.reshape(len(T_nodes), len(K_nodes))
        sigma_grid = _build_sigma_grid(alpha_vec, K_nodes, T_nodes, K_full, T_full)
        sigma_grid = np.clip(sigma_grid, 0.01, 2.0)
        
        _, _, C_model = solve_dupire_pde(
            S0=S0,
            r=r,
            initial_vol=sigma_init,
            K_min=K_full[0],
            K_max=K_full[-1],
            T_max=T_full[-1],
            N=len(K_full),
            M=len(T_full),
            sigma_grid=sigma_grid,
        )
        misfit = np.mean((C_model - market_prices) ** 2)
        reg = _laplacian_penalty(alpha_grid)
        total_loss = misfit + lam * reg
        
        if total_loss < best_loss[0]:
            best_loss[0] = total_loss
            if verbose and iteration[0] % 2 == 0:
                print(f"    Iter {iteration[0]:3d}: loss={total_loss:.6f}, misfit={misfit:.6f}, reg={reg:.6f}")
        
        return total_loss

    bounds = [(1e-6, 4.0)] * alpha0_vec.size
    res = minimize(loss, alpha0_vec, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "disp": False, "ftol": 1e-6, "gtol": 1e-5})
    sigma_calibrated = _build_sigma_grid(res.x, K_nodes, T_nodes, K_full, T_full)
    return sigma_calibrated, res


def demo_inverse_problem(S0, K_min=60.0, K_max=140.0, r=0.03, sigma_true = 0.20, sigma_init = 0.12, N=100, M=60):
    """Toy inverse problem on synthetic data (constant true vol)."""
    T_max = 1.0
 #FIXME: what K we need to use
    K_full = np.linspace(K_min, K_max, N)
    T_full = np.linspace(0.01, T_max, M)

    print("Inverse problem demo: synthetic constant vol")
    print(f"  PDE grid: {N} strikes × {M} times")
    print(f"  True sigma: {sigma_true}")

    # Forward prices with true sigma
    print("  Generating synthetic market prices...")
    sigma_true_grid = np.full((M, N), sigma_true)
    _, _, market_prices = solve_dupire_pde(
        S0=S0,
        r=r,
        initial_vol=sigma_true,
        K_min=K_min,
        K_max=K_max,
        T_max=T_max,
        N=N,
        M=M,
        sigma_grid=sigma_true_grid,
    )

    # Finer parameter grid for better recovery
    K_nodes = np.linspace(K_min, K_max, 10)
    T_nodes = np.linspace(0.01, T_max, 8)
    print(f"  Parameter grid: {len(K_nodes)} K-nodes × {len(T_nodes)} T-nodes")


    print(f"  Initial guess: sigma={sigma_init}")
    print("  Starting calibration with adaptive regularization...")
    
    sigma_calibrated, res = calibrate_local_vol(
        market_prices=market_prices,
        K_full=K_full,
        T_full=T_full,
        K_nodes=K_nodes,
        T_nodes=T_nodes,
        sigma_init=sigma_init,
        r=r,
        S0=S0,
        lam=2e-3,  # Smaller regularization for better fit
        maxiter=50,
        verbose=True,
    )

    err = np.mean((sigma_calibrated - sigma_true) ** 2) ** 0.5
    stats = {
        "min": float(np.min(sigma_calibrated)),
        "max": float(np.max(sigma_calibrated)),
        "mean": float(np.mean(sigma_calibrated)),
        "median": float(np.median(sigma_calibrated)),
        "rmse": float(err),
    }

    print(f"\n  Calibration complete!")
    print(f"  RMSE: {err:.6f} (target 0.0000)")
    print(f"  Recovered sigma stats:")
    print(f"    min={stats['min']:.4f}, max={stats['max']:.4f}")
    print(f"    mean={stats['mean']:.4f}, median={stats['median']:.4f}")
    print(f"  Optimizer: {res.message}")

    # Визуализация: сохраняем heatmap оцененной и истинной волатильности
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(sigma_true_grid, extent=[K_min, K_max, T_max, 0], aspect="auto", cmap="viridis")
    axes[0].set_title("True sigma")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("T")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(sigma_calibrated, extent=[K_min, K_max, T_max, 0], aspect="auto", cmap="viridis")
    axes[1].set_title("Calibrated sigma")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("T")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("inverse_demo_sigma.png", dpi=120)
    plt.close(fig)

    print("  plot saved to inverse_demo_sigma.png")


if __name__ == "__main__":
    demo_inverse_problem()
