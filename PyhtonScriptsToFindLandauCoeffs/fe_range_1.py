import numpy as np
import csv
from tqdm import tqdm

np.random.seed(351)  

# -------------- CONFIG -------------- # line number 8 and 9 says find out 10000 landau coeff from 30000000 attempts
target_count = 10000    
max_attempts = 30000000

# --- Parameter ranges --- #range
alpha_range = (-1e10, -1e8)
beta_range  = (-3.0e10, -3.0e8)
gamma_range = (1e10, 2e13)
a0_range     = (1.0e8, 1.0e9)
g_range = (-5e7, -5e8)

# --- Polarization grid (fixed) ---
P = np.linspace(-1.0, 1.0, 1000)   

# --- Store valid parameter sets here ---
results = []

E_max_dim = 3.0       # Fixed dimensional max field (MV/cm)
delta_E_dim = 0.15  # Fixed dimensional increment (MV/cm)
E_dim_array = np.arange(-E_max_dim, E_max_dim + delta_E_dim, delta_E_dim)

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

for attempt in tqdm(range(max_attempts), desc="Sampling parameters"):

    # Sample parameters
    alpha = np.random.uniform(*alpha_range)
    beta  = np.random.uniform(*beta_range)
    gamma = np.random.uniform(*gamma_range)
    #g = np.random.uniform(*g_range)
    log_g = np.random.uniform(np.log10(5e7), np.log10(5e8))
    g = -10**log_g


    P0 = compute_P0(alpha, beta, gamma, g)
    if np.isnan(P0) or P0 > 0.8: 
        continue

    E0 = (abs(alpha) * P0) / 4e8  # Characteristic electric field scale E0 in MV/cm

    E_max_prime = E_max_dim / E0
    if not (4.0 <= E_max_prime <= 22):  #Range for non-dimensional electric field
        continue

    if E_max_prime > 13:    
        a0 = np.random.uniform(1e8, 9e8)  
    else:
        a0 = np.random.uniform(*a0_range)  

    delta_E_prime = delta_E_dim / E0

    # --- Free Energy Curves ---
    a_afe = (alpha - g) / 4
    a_fe  = (alpha + g) / 4
    b = beta / 32
    c = gamma / 192

    f_afe = a_afe * P**2 + b * P**4 + c * P**6
    f_fe  = a_fe  * P**2 + b * P**4 + c * P**6

    # --- Critical points of f_fe ---
    quad_coeffs = [6*c, 4*b, 2*a_fe]
    Q_roots = np.roots(quad_coeffs)
    Q_roots = Q_roots[np.isreal(Q_roots) & (Q_roots.real > 0)].real

    # --- Define E and its derivative ---
    def E_func(P):
        return 0.5 * (alpha + g) * P + (beta / 8) * P**3 + (gamma / 32) * P**5

    def dE_dP(P):
        return 0.5 * (alpha + g) + (3 * beta / 8) * P**2 + (5 * gamma / 32) * P**4

    # --- Evaluate E and dE/dP ---
    E = E_func(P)
    dE = dE_dP(P)

    # --- Find turning points (sign changes in dE/dP) ---
    sign_changes = np.where(np.diff(np.sign(dE)) != 0)[0]
    if len(sign_changes) == 0:
        continue

    # --- Get Polarization values at turning points ---
    P1 = P[sign_changes[0]]
    P2 = P[sign_changes[1]]

    # --- Evaluate Electric Field at turning points ---
    E1 = E_func(P1)
    E2 = E_func(P2)

    if not E1>E2:
        continue

    if not (E1 > 1e8 and E1 < 3.0e8):
        continue

    # --- Non-dimensional versions (prevent divide by zero) ---
    E1_nonDim = E1 / (E0  * 1e8)if E0 != 0 else None
    E2_nonDim = E2 / (E0 * 1e8) if E0 != 0 else None

    if not (( (E1_nonDim - E2_nonDim) < 22) and (E1_nonDim - E2_nonDim) > 4):
        continue
    #print(f"E1_nonDim = {E1_nonDim:.4f}, E1 = {E1:.4e}, E0 = {E0:.4e}")

    result = {
        'alpha': alpha, 'beta': beta, 'gamma': gamma, 'g': g, 'a0': a0,
        'P0': P0, 'E0': E0,
        'E1': E1, 'E2': E2, 
        'E1_nonDim': E1_nonDim, 'E2_nonDim': E2_nonDim
    }

    results.append(result)

    if len(results) >= target_count:
        break

with open("Fe_field_data.txt", "w") as f:
    f.write("alpha\t\tbeta\t\tgamma\t\tg\t\ta0\t\tE1\t\tE2\t\tE1_ND\tE2_ND\tP0\tE0\n")
    for res in results:
        f.write(f"{res['alpha']:.6e}\t{res['beta']:.6e}\t{res['gamma']:.6e}\t{res['g']:.6e}\t"
                f"{res['a0']:.6e}\t"
                f"{res['E1']:.6e}\t{res['E2']:.6e}\t"
                f"{res['E1_nonDim']:.4f}\t{res['E2_nonDim']:.4f}\t"
                f"{res['P0']:.4f}\t{res['E0']:.4f}\n")

print(f"Saved {len(results)} valid parameter sets to 'Fe_field_data.txt'")
