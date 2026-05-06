import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- Configuración del Escenario SPH ---
h = 1.2          # Longitud de suavizado 
dx = 0.5         # Espaciado entre las dos partículas 
t_final = 1.0    # Tiempo final de la simulación
N = 1            # Número de vecinos (solo hay otra partícula)

# Gradiente del Kernel Triangular: gradW = 1/h^2 
gradW = 1.0 / (h**2)

def sph_advection_system(u, t, c):
    """
    Define el sistema de EDOs basado en la ecuación de advección SPH.
    du/dt = -c * sum( (u_j - u_i) * dx * gradW ) 
    """
    u0, u1 = u
    # Para la partícula 0, la vecina es la 1 (dirección positiva)
    du0_dt = -c * (u1 - u0) * dx * (-gradW)
    # Para la partícula 1, la vecina es la 0 (dirección negativa)
    du1_dt = -c * (u0 - u1) * dx * (gradW)
    return [du0_dt, du1_dt]

# --- Parámetros de la Simulación ---
c_values = np.logspace(-4, 0.4, 50)  # De 10^-4 hasta poco más de 10^0
initial_conditions = [
    (1.0, 0.0), (0.8, 0.2), (0.6, 0.4), 
    (0.4, 0.6), (0.2, 0.8), (0.0, 1.0)
]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# --- Ejecución ---
results_u0 = {cond: [] for cond in initial_conditions}
results_u1 = {cond: [] for cond in initial_conditions}

for c in c_values:
    for cond in initial_conditions:
        # Resolver el sistema desde t=0 hasta t=1
        t_span = np.linspace(0, t_final, 2)
        sol = odeint(sph_advection_system, cond, t_span, args=(c,))
        results_u0[cond].append(sol[-1, 0])
        results_u1[cond].append(sol[-1, 1])

# --- Visualización ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for i, cond in enumerate(initial_conditions):
    label = f"({cond[0]}, {cond[1]})"
    ax1.plot(c_values, results_u0[cond], 'x-', color=colors[i], label=label)
    ax2.plot(c_values, results_u1[cond], 'x-', color=colors[i])

# Estética y límites
c_instability = h**2 / (2 * dx) # Límite de inestabilidad 

for ax in [ax1, ax2]:
    ax.set_xscale('log')
    ax.axvline(c_instability, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Advection speed $c$')
    ax.grid(True, which="both", ls="-", alpha=0.2)

ax1.set_title('$u_0(1)$ solutions (Classical Corrected)')
ax1.set_ylabel('Amplitude $u$')
ax2.set_title('$u_1(1)$ solutions (Classical Corrected)')

ax1.legend(title='Initial $(u_0(0), u_1(0))$', bbox_to_anchor=(2.3, 1))
plt.tight_layout()
plt.show()