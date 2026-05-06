import numpy as np
import matplotlib.pyplot as plt

from match_decomposition import ctqw_matching_step

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ===============================
# CONFIGURACIÓN SPH / CTQW
# ===============================
h = 1.2
dx = 0.5
dt = 1
nu = 1.0 / (h**2)
# Línea crítica c_crit = h^2/(2 dx), dt=1
c_crit = (h**2) / (2 * dx * dt)

# Rango de velocidades c (como en tu SPH)
c_values = np.logspace(-4, 0.4, 50)  # 10^-4 hasta ~2.51

# Condición inicial azul: u0(0)=1, u1(0)=0
u0_init, u1_init = 1.0, 0.0

# Tiempo de evolución CTQW
t = dt

# Simulador statevector
simulator = AerSimulator(method="statevector")

results_u0 = []
results_u1 = []

for c in c_values:
    # 1) Hamiltoniano H 2x2: La interacción entre 0 y 1
    # Multiplicamos por pi/4 para que en el punto crítico las partículas se crucen
    J = c * dx * nu 
    H = np.array([[0.0, J],
                  [J, 0.0]], dtype=float)

    # 2) Circuito CTQW via matching decomposition
    # Para 2 partículas, matching nos da una rotación Rx simple
    qc_evol = ctqw_matching_step(H, t, n_trotter_steps=1)

    # 3) Circuito completo
    qc = QuantumCircuit(1) # Solo necesitamos 1 qubit para 2 estados
    # Estado inicial |0> es u0=1, u1=0. No hace falta initialize si es |0>
    if u0_init < 1.0:
        # Si quisieras (0.8, 0.2), usarías amplitudes sqrt(0.8) y sqrt(0.2)
        initial_state = [np.sqrt(u0_init), np.sqrt(u1_init)]
        qc.initialize(initial_state, 0)
    
    qc.compose(qc_evol, inplace=True)
    qc.save_statevector()

    # 4) Ejecución
    state = simulator.run(qc).result().get_statevector()

    # 5) POST-PROCESO CORRECTO: Probabilidades [cite: 1591]
    # La advección cuántica mueve la PROBABILIDAD de un sitio a otro
    u0_t = np.abs(state[0])**2
    u1_t = np.abs(state[1])**2

    results_u0.append(u0_t)
    results_u1.append(u1_t)

# ===============================
# VISUALIZACIÓN (solo línea azul)
# ===============================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

ax1.plot(c_values, results_u0, 'o-', color='#1f77b4', label='(1.0, 0.0)')
ax2.plot(c_values, results_u1, 'o-', color='#1f77b4')

# Línea crítica c_crit = h^2/(2 dx), dt=1
c_crit = (h**2) / (2 * dx * dt)

for ax in (ax1, ax2):
    ax.set_xscale('log')
    ax.axvline(c_crit, color='gray', linestyle='--', alpha=0.7,
               label=r'$c = h^2/(2\Delta x)$' if ax is ax1 else "")
    ax.set_xlabel('Advection speed $c$', fontsize=12)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.set_xlim([1e-4, 2])

ax1.set_title('$u_0(1)$ solutions (CTQW + matching)', fontsize=13)
ax1.set_ylabel('Amplitude $u$', fontsize=12)
ax2.set_title('$u_1(1)$ solutions (CTQW + matching)', fontsize=13)

ax1.legend(title='Initial $(u_0(0), u_1(0))$')
plt.tight_layout()
plt.show()