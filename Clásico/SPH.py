import numpy as np
import matplotlib.pyplot as plt


# CONFIGURACIÓN SPH

h = 1.2          # Longitud de suavizado (smoothing length)
dx = 0.5         # Espaciado entre partículas
dt = 1.0         # PASO DE TIEMPO (Δt = 1 en el paper, Sección III)
N_steps = 1    # Número de pasos temporales (1 para comparar con Fig. 7)

# Gradiente del kernel triangular (ecuación en página 5)
# dW/dr = -sgn(r)/h² para |r| < h
# Como dx = 0.5 < h = 1.2, las partículas están dentro del soporte
gradW01 = -1.0 / (h**2) 
gradW10 = 1.0 / (h**2)


# ALGORITMO SPH DISCRETO - ECUACIÓN (3) DEL PAPER
def sph_step_discreto(u0, u1, c):
    """
    Implementa la ecuación (3) del paper:
    u_j(t+Δt) = u_j(t) - c·Δt · Σ_k [(u_k - u_j) · Δx · ∇_j W_jk]
    
    Para 2 partículas, la suma solo tiene un término (k = vecina).
    """
    # --- Partícula 0: vecina es la 1 ---
    # Término: (u_1 - u_0) * dx * gradW_01
    suma_0 = (u1 - u0) * dx * gradW01
    u0_new = u0 - c * dt * suma_0
    
    # --- Partícula 1: vecina es la 0 ---
    # Término: (u_0 - u_1) * dx * gradW_10
    suma_1 = (u0 - u1) * (-dx) * gradW10
    u1_new = u1 - c * dt * suma_1
    
    return u0_new, u1_new


# PARÁMETROS DE SIMULACIÓN
c_values = np.logspace(-4, 0.4, 50)  # 10^-4 hasta 10^0 = 1

initial_conditions = [
    (1.0, 0.0),   # Azul
    (0.8, 0.2),   # Naranja  
    (0.6, 0.4),   # Verde
    (0.4, 0.6),   # Rojo
    (0.2, 0.8),   # Morado
    (0.0, 1.0)    # Marrón
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


# EJECUCIÓN DEL SPH
results_u0 = {cond: [] for cond in initial_conditions}
results_u1 = {cond: [] for cond in initial_conditions}

for c in c_values:
    for u0_init, u1_init in initial_conditions:
        # Condición inicial
        u0, u1 = u0_init, u1_init
        
        # Evolución temporal discreta (N_steps pasos)
        for step in range(N_steps):
            u0, u1 = sph_step_discreto(u0, u1, c)
        
        # Guardar resultado después de N_steps
        results_u0[(u0_init, u1_init)].append(u0)
        results_u1[(u0_init, u1_init)].append(u1)

# =============================================================================
# VISUALIZACIÓN (estilo Fig. 7 del paper)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for i, cond in enumerate(initial_conditions):
    label = f"({cond[0]}, {cond[1]})"
    ax1.plot(c_values, results_u0[cond], 'x-', color=colors[i], label=label, markersize=6)
    ax2.plot(c_values, results_u1[cond], 'x-', color=colors[i], markersize=6)

# Línea crítica de inestabilidad (ecuación 54 del paper)
# c_crit = h² / (2 * dx) cuando Δt = 1
c_crit = (h**2) / (2 * dx * dt)

for ax in [ax1, ax2]:
    ax.set_xscale('log')
    ax.axvline(c_crit, color='gray', linestyle='--', alpha=0.7, 
               label=f'$c = h^2/(2\\Delta x)$' if ax == ax1 else "")
    ax.set_xlabel('Advection speed $c$', fontsize=12)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.set_xlim([1e-4, 2])

ax1.set_title('$u_0(1)$ solutions (SPH Discrete)', fontsize=13)
ax1.set_ylabel('Amplitude $u$', fontsize=12)
ax2.set_title('$u_1(1)$ solutions (SPH Discrete)', fontsize=13)

ax1.legend(title='Initial $(u_0(0), u_1(0))$', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()