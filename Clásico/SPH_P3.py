import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# CONFIGURACIÓN SPH (Valores modificables)

h = 1.2
c = 10**(-0.9)            # velocidad de advección fija
T_final = 30              # prueba larga pero todavía compacta
n_points = 500            # detalle fino en la línea

# Arrays de estado inicial (Puedes agregar más partículas aquí)
# La longitud de ambos arrays DEBE ser exactamente la misma
u_init = np.array([1.0, 0.0, 0.0, 0.0])              # Valores de advección U
x_pos  = np.array([0.0, 0.5, 1.0, 1.5])              # Posiciones X de cada partícula

# Validación de seguridad antes de arrancar
if len(u_init) != len(x_pos):
    print(f"ERROR: La longitud de u_init ({len(u_init)}) y x_pos ({len(x_pos)}) no coinciden.")
    sys.exit(1)

N_particles = len(u_init)

# Mallado temporal fino
t_values = np.linspace(0.0, T_final, n_points)
dt_local = t_values[1] - t_values[0]


# MATRIZ DE GRADIENTES DEL KERNEL

# Se precalcula la matriz de derivadas espaciales del kernel triangular.
# dW/dx_i = sign(x_i - x_j) / h^2   (solo si |x_i - x_j| < h)
gradW_matrix = np.zeros((N_particles, N_particles))

for i in range(N_particles):
    for j in range(N_particles):
        if i == j:
            continue
        dist = x_pos[i] - x_pos[j]
        # Condición del radio de influencia (compact support)
        if abs(dist) < h and dist != 0:
            gradW_matrix[i, j] = np.sign(dist) / (h**2)


# ECUACIÓN SPH DISCRETA PARA N PARTÍCULAS (Conservativa)

def sph_step_discreto_n(u, x, gradW, c, dt):
    """
    Realiza un paso de evolución temporal para N partículas usando la
    formulación "zeroth-order consistent" de Au-Yeung.
    """
    N = len(u)
    u_new = np.zeros(N)
    
    for i in range(N):
        suma_i = 0.0
        for j in range(N):
            if i == j:
                continue
            
            gW = gradW[i, j]
            # Si el gradiente es 0, significa que la partícula j está fuera del rango h
            if gW == 0.0:
                continue
                
            # Distancia x_ji = x_j - x_i
            x_ji = x[j] - x[i]
            
            # (u_j - u_i) * (x_j - x_i) * gradW_ij
            suma_i += (u[j] - u[i]) * x_ji * gW
            
        u_new[i] = u[i] - c * dt * suma_i
        
    return u_new


# EVOLUCIÓN TEMPORAL

# Matriz para almacenar el historial (filas = instantes de tiempo, columnas = partículas)
u_history = np.zeros((n_points, N_particles))
u_history[0, :] = u_init.copy()

u_current = u_init.copy()

# Bucle principal de simulación
for step in range(1, n_points):
    u_current = sph_step_discreto_n(u_current, x_pos, gradW_matrix, c, dt_local)
    u_history[step, :] = u_current.copy()


# VISUALIZACIÓN AUTOMATIZADA

fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

# Usamos un mapa de colores (colormap) para obtener N colores jerárquicos y distintos
colores = cm.get_cmap('tab10', N_particles)
marcadores = ['o', 's', 'd', '^', 'v', '<', '>', 'p', '*', 'h']

for i in range(N_particles):
    mk = marcadores[i % len(marcadores)] # Ciclar marcadores si hay más de 10 partículas
    ax1.plot(t_values, u_history[:, i], marker=mk, markevery=50, 
             linestyle='-', color=colores(i), lw=2, ms=4, label=f'$u_{i}(t)$ [x={x_pos[i]}]')

# Cálculo de la masa total y la línea de equilibrio
u_total = u_history.sum(axis=1)
masa_inicial_total = u_total[0]
equilibrio = masa_inicial_total / N_particles

ax1.plot(t_values, u_total, '--', color='purple', lw=2, label=r'$\Sigma u$ (Masa Total)')
ax1.axhline(equilibrio, color='gray', linestyle=':', lw=2, label=f'$u = {equilibrio:.3f}$ (Equilibrio)')

ax1.set_title(f'Evolución temporal SPH discreta - {N_particles} Partículas')
ax1.set_xlabel('Tiempo $t$')
ax1.set_ylabel('Amplitud $u$')
ax1.grid(True, alpha=0.3)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


# REPORTE FINAL EN CONSOLA

print("=== RESUMEN DE LA SIMULACIÓN ===")
print(f"Partículas simuladas: {N_particles}")
print(f"h (smoothing length): {h}")
print(f"dt_local = {dt_local:.6f}")
print("\n--- Valores finales en T_final ---")
for i in range(N_particles):
    print(f"u{i}(T) = {u_history[-1, i]:.6f}")
print(f"Suma Total = {u_total[-1]:.6f}")