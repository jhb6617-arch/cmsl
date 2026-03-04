import numpy as np
from tqdm import tqdm

np.random.seed(351)

# -------------- CONFIG --------------
target_count = 10000
max_attempts = 20000000

alpha_range = (5.0e8, 5.0e9)
beta_range  = (-5.0e11, -1.0e10)
gamma_range = (5.0e10, 9.0e12)
g_range     = (1.0e8, 9.0e9)
a0_range     = (5.0e8, 5.0e9)

max_alpha_minus_g_pos = 2.5e8
max_alpha_minus_g_neg = 2.0e8

E_max_dim = 3.0       # Fixed dimensional max field (MV/cm)
delta_E_dim = 0.15  # Fixed dimensional increment (MV/cm)
E_dim_array = np.arange(-E_max_dim, E_max_dim + delta_E_dim, delta_E_dim)

results = []

def compute_P0(alpha, beta, gamma, g):
    discriminant = beta**2 - 4 * gamma * (alpha - g)
    if discriminant < 0 or gamma == 0:
        return np.nan
    sqrt_term = np.sqrt(discriminant)
    numerator = 2 * abs(beta) + 2 * sqrt_term
    result = numerator / gamma
    if result < 0:
        return np.nan
    return np.sqrt(result)

for idx in tqdm(range(max_attempts), desc="Sampling Parameters"):
    a0 = np.random.uniform(*a0_range)
    alpha = np.random.uniform(*alpha_range)
    delta = np.random.uniform(-max_alpha_minus_g_neg, max_alpha_minus_g_pos)
    g = alpha - delta
    if not (g_range[0] <= g <= g_range[1]):
        continue

    beta  = np.random.uniform(*beta_range)
    gamma = np.random.uniform(*gamma_range)

    P0 = compute_P0(alpha, beta, gamma, g)
    if np.isnan(P0) or P0 < 0.37:
        continue

    E0 = (alpha * P0) / 4e8  # E0 in MV/cm
    if E0 > 0.8:
        continue

    E_max_prime = E_max_dim / E0

    delta_E_prime = delta_E_dim / E0

    a = (alpha + g) / 4
    d = (alpha - g) / 4
    b = beta / 32
    c = gamma / 192
    P = np.linspace(-2, 2, 1000)

    delta_check = 4 * gamma * (g - alpha) + beta**2 + 5 * gamma**2 * P**4 + 2 * beta * gamma * P**2
    if np.any(delta_check < 0):
        continue

    f1 = c * P**6 + b * P**4 + a * P**2
    f2 = c * P**6 + b * P**4 + d * P**2

    cond1 = (b**2 >= 1.8 * a * c) and (b**2 <= 3.5 * a * c)
    cond2 = np.all(f2 > -1.5e10)
    low_energy_mask = f1 < 1e10
    low_energy_P = P[low_energy_mask]
    cond3 = np.all((low_energy_P >= -1.5) & (low_energy_P <= 1.5))

    if not (cond1 and cond2 and cond3):
        continue

    def E_func(P): return 0.5 * (alpha + g) * P + (beta / 8) * P**3 + (gamma / 32) * P**5
    def dE_dP(P): return 0.5 * (alpha + g) + (3 * beta / 8) * P**2 + (5 * gamma / 32) * P**4

    dE = dE_dP(P)
    sign_changes = np.where(np.diff(np.sign(dE)))[0]
    if len(sign_changes) < 4:
        continue

    P_turning = (P[sign_changes] + P[sign_changes + 1]) / 2
    E_turning = E_func(P_turning)
    E1, E2, E3, E4 = E_turning[0], E_turning[1], E_turning[2], E_turning[3]

    E1_nonDim = E1 / (E0 * 1e8)
    E2_nonDim = E2 / (E0 * 1e8)
    E3_nonDim = E3 / (E0 * 1e8)
    E4_nonDim = E4 / (E0 * 1e8)

    cond4 = (E1_nonDim - E4_nonDim) < -0.05
    cond5 = (-3 <= E2_nonDim <= 3) and (-3 <= E3_nonDim <= 3)
    cond6 = (E1_nonDim - E2_nonDim) > 0.2
    cond7 = (E3_nonDim - E4_nonDim) < 0.85
    cond8 = E1 < E4

    if not cond8:
        continue

    result = {
        'alpha': alpha, 'beta': beta, 'gamma': gamma, 'g': g, 'a0': a0,
        'P0': P0, 'E0': E0,
        'E1': E1, 'E2': E2, 'E3': E3, 'E4': E4,
        'E1_nonDim': E1_nonDim, 'E2_nonDim': E2_nonDim,
        'E3_nonDim': E3_nonDim, 'E4_nonDim': E4_nonDim
    }

    results.append(result)

    if len(results) >= target_count:
        break

# Write summary TXT file
with open("fixed_field_data.txt", "w") as f:
    f.write("alpha\t\tbeta\t\tgamma\t\tg\t\ta0\t\tE1\t\tE2\t\tE3\t\tE4\t\tE1_ND\tE2_ND\tE3_ND\tE4_ND\tP0\tE0\n")
    for res in results:
        f.write(f"{res['alpha']:.6e}\t{res['beta']:.6e}\t{res['gamma']:.6e}\t{res['g']:.6e}\t"
                f"{res['a0']:.6e}\t"
                f"{res['E1']:.6e}\t{res['E2']:.6e}\t{res['E3']:.6e}\t{res['E4']:.6e}\t"
                f"{res['E1_nonDim']:.4f}\t{res['E2_nonDim']:.4f}\t{res['E3_nonDim']:.4f}\t{res['E4_nonDim']:.4f}\t"
                f"{res['P0']:.4f}\t{res['E0']:.4f}\n")

print(f"Total accepted: {len(results)}")
