import numpy as np
import matplotlib.pyplot as plt


# CONFIGURACIÓN SPH

h = 1.2          # Longitud de suavizado
dx = 0.5         # Espaciado entre partículas
dt = 1.0         # Paso de tiempo 

# Gradiente del kernel triangular
gradW = -1.0 / (h**2) 

# Límite crítico de estabilidad (ecuación 54 del paper)
c_crit = -1 / (2 * dx * dt * gradW)
print(f"Límite crítico de estabilidad: c_crit = {c_crit:.3f}")


# FUNCIÓN DE EVOLUCIÓN SPH

def sph_step_discreto(u0, u1, c):
    """
    Un paso temporal de la ecuación (3) del paper.
    """
    # Partícula 0
    suma_0 = (u1 - u0) * dx * gradW
    u0_new = u0 - c * dt * suma_0
    
    # Partícula 1  
    suma_1 = (u0 - u1) * dx * gradW
    u1_new = u1 - c * dt * suma_1
    
    return u0_new, u1_new

def simular_temporal(u0_init, u1_init, c, N_steps):
    """
    Simula N_steps pasos temporales y devuelve la historia completa.
    """
    u0_history = [u0_init]
    u1_history = [u1_init]
    
    u0, u1 = u0_init, u1_init
    
    for step in range(N_steps):
        u0, u1 = sph_step_discreto(u0, u1, c)
        u0_history.append(u0)
        u1_history.append(u1)

    
    return np.array(u0_history), np.array(u1_history)


# PARÁMETROS DE SIMULACIÓN TEMPORAL

N_steps = 10     # Número de pasos temporales a simular
t = np.arange(N_steps + 1) * dt  # Vector de tiempo

# Condición inicial
u0_0, u1_0 = 1.0, 0.0  # Partícula 0 con todo, partícula 1 con nada

# Valores de c a comparar (puedes añadir más)
# Estratégicamente elegimos: subcrítico, cercano a crítico, y supercrítico
c_values_fijos = [0.1, 0.5, c_crit, 1.5, 2.0]
labels_c = [f'c = {c:.2f}' if c != c_crit else f'c = c_crit ({c:.2f})' 
            for c in c_values_fijos]

colors_c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


# EJECUCIÓN

plt.figure(figsize=(14, 6))

# Subplot 1: Evolución temporal de ambas partículas
plt.subplot(1, 2, 1)
for c, label, color in zip(c_values_fijos, labels_c, colors_c):
    u0_hist, u1_hist = simular_temporal(u0_0, u1_0, c, N_steps)
    
    plt.plot(t[:len(u0_hist)], u0_hist, '-', color=color, lw=2, marker='o', markersize=4, label=f'$u_0$: {label}')
    plt.plot(t[:len(u1_hist)], u1_hist, '--', color=color, lw=2, marker='x', markersize=4, label=f'$u_1$: {label}')

plt.axhline((u0_0 + u1_0)/2, color='black', linestyle=':', alpha=0.5, label='Equilibrio (0.5)')
plt.xlabel('Tiempo $t$')
plt.ylabel('Amplitud $u$')
plt.title(f'Evolución temporal para $(u_0, u_1)_0$ = ({u0_0}, {u1_0})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Subplot 2: Conservación de la suma (debería ser constante = 1.0)
plt.subplot(1, 2, 2)
plt.ylim(0.99, 1.01) 
for c, label, color in zip(c_values_fijos, labels_c, colors_c):
    u0_hist, u1_hist = simular_temporal(u0_0, u1_0, c, N_steps)
    suma_total = u0_hist + u1_hist
    
    plt.plot(t[:len(suma_total)], suma_total, '-', color=color, label=label, marker='o', markersize=4)

plt.axhline(u0_0 + u1_0, color='black', linestyle='--', alpha=0.5, label='Conservación ideal')
plt.xlabel('Tiempo $t$')
plt.ylabel('$u_0 + u_1$')
plt.title('Conservación de la cantidad total')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()