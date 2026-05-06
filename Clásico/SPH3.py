import numpy as np
import matplotlib.pyplot as plt


# CONFIGURACIÓN SPH

h = 1.2
dx = 0.5
dt = 0.05

gradW = -1.0 / (h**2)

c_crit = -1 / (2 * dx * dt * gradW)
print(f"Límite crítico de estabilidad: c_crit = {c_crit:.3f}")


# FUNCIONES

def sph_step_discreto(u0, u1, c):
    suma_0 = (u1 - u0) * dx * gradW
    u0_new = u0 - c * dt * suma_0
    suma_1 = (u0 - u1) * dx * gradW
    u1_new = u1 - c * dt * suma_1
    return u0_new, u1_new

def simular_temporal(u0_init, u1_init, c, N_steps):
    u0_history = [u0_init]
    u1_history = [u1_init]
    u0, u1 = u0_init, u1_init
    for step in range(N_steps):
        u0, u1 = sph_step_discreto(u0, u1, c)
        u0_history.append(u0)
        u1_history.append(u1)
    return np.array(u0_history), np.array(u1_history)


# PARÁMETROS

T_total = 40             
N_steps = int(T_total / dt)   

t_full = np.arange(N_steps + 1) * dt   
t      = t_full[1:]                     

u0_0, u1_0 = 1.0, 0.0

# c_values como múltiplos de c_crit
c_values_fijos = [0.1 * c_crit, 0.5 * c_crit, c_crit, 1.2 * c_crit, 1.5 * c_crit]
labels_c = [f'c = {c:.1f}' if c != c_crit else f'c = c_crit ({c:.1f})'
            for c in c_values_fijos]
colors_c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


# EJECUCIÓN

plt.figure(figsize=(14, 6))

# ---- Subplot 1: Evolución temporal ----
plt.subplot(1, 2, 1)
for c, label, color in zip(c_values_fijos, labels_c, colors_c):
    u0_hist, u1_hist = simular_temporal(u0_0, u1_0, c, N_steps)
    plt.plot(t, u0_hist[1:], '-',  color=color, lw=2, marker='o', markersize=4, label=f'$u_0$: {label}')
    plt.plot(t, u1_hist[1:], '--', color=color, lw=2, marker='x', markersize=4, label=f'$u_1$: {label}')

plt.axhline(0.5, color='black', linestyle=':', alpha=0.5, label='Equilibrio (0.5)')
plt.xscale('log')
plt.xlabel('Tiempo $t$ (escala log)')
plt.ylabel('Amplitud $u$')
plt.title(f'Evolución temporal — $(u_0, u_1)_0$ = ({u0_0}, {u1_0})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3, which='both')

# ---- Subplot 2: Conservación ----
plt.subplot(1, 2, 2)
plt.ylim(0.99, 1.01)
for c, label, color in zip(c_values_fijos, labels_c, colors_c):
    u0_hist, u1_hist = simular_temporal(u0_0, u1_0, c, N_steps)
    suma_total = (u0_hist + u1_hist)[1:]
    plt.plot(t, suma_total, '-', color=color, lw=1.5, label=label)

plt.axhline(u0_0 + u1_0, color='black', linestyle='--', alpha=0.5, label='Conservación ideal')
plt.xscale('log')
plt.xlabel('Tiempo $t$ (escala log)')
plt.ylabel('$u_0 + u_1$')
plt.title('Conservación de la cantidad total')
plt.legend()
plt.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()