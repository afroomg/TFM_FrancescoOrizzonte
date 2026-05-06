import numpy as np
import matplotlib.pyplot as plt


# CONFIGURACIÓN SPH

h = 1.2
dx = 0.5
c = 10**(-0.9)            # velocidad de advección fija
T_final = 30              # prueba larga pero todavía compacta
n_points = 500            # detalle fino en la línea

# Condición inicial: U0 = 1, U1 = 0, U2 = 0
u0_init, u1_init, u2_init = 1.0, 0.0, 0.0

# Mallado temporal fino
t_values = np.linspace(0.0, T_final, n_points)
dt_local = t_values[1] - t_values[0]

# Gradientes del kernel triangular W(r)
# Fórmula: gradW_ij = sign(x_i - x_j) / h^2
gradW01 = -1.0 / (h**2)   # x0 - x1 = -0.5 -> negativo
gradW02 = -1.0 / (h**2)   # x0 - x2 = -1.0 -> negativo

gradW10 =  1.0 / (h**2)   # x1 - x0 =  0.5 -> positivo
gradW12 = -1.0 / (h**2)   # x1 - x2 = -0.5 -> negativo

gradW20 =  1.0 / (h**2)   # x2 - x0 =  1.0 -> positivo
gradW21 =  1.0 / (h**2)   # x2 - x1 =  0.5 -> positivo


# ECUACIÓN SPH DISCRETA PARA 3 PARTÍCULAS (Conservativa)

def sph_step_discreto_3(u0, u1, u2, c, dt):
    
    # Partícula 0 (interactúa con la 1 a distancia dx y con la 2 a distancia 2dx)
    # suma_0 = sum_j (u_j - u_0) * (x_j - x_0) * gradW0j
    suma_0_1 = (u1 - u0) * (dx)   * gradW01
    suma_0_2 = (u2 - u0) * (2*dx) * gradW02
    u0_new = u0 - c * dt * (suma_0_1 + suma_0_2)

    # Partícula 1 (interactúa con la 0 a distancia -dx y con la 2 a distancia dx)
    # suma_1 = sum_j (u_j - u_1) * (x_j - x_1) * gradW1j
    suma_1_0 = (u0 - u1) * (-dx)  * gradW10
    suma_1_2 = (u2 - u1) * (dx)   * gradW12
    u1_new = u1 - c * dt * (suma_1_0 + suma_1_2)

    # Partícula 2 (interactúa con la 0 a distancia -2dx y con la 1 a distancia -dx)
    # suma_2 = sum_j (u_j - u_2) * (x_j - x_2) * gradW2j
    suma_2_0 = (u0 - u2) * (-2*dx) * gradW20
    suma_2_1 = (u1 - u2) * (-dx)   * gradW21
    u2_new = u2 - c * dt * (suma_2_0 + suma_2_1)

    return u0_new, u1_new, u2_new


# EVOLUCIÓN TEMPORAL

u0_values, u1_values, u2_values = [u0_init], [u1_init], [u2_init]

u0, u1, u2 = u0_init, u1_init, u2_init
for _ in range(n_points - 1):
    u0, u1, u2 = sph_step_discreto_3(u0, u1, u2, c, dt_local)
    u0_values.append(u0)
    u1_values.append(u1)
    u2_values.append(u2)

u0_values = np.array(u0_values)
u1_values = np.array(u1_values)
u2_values = np.array(u2_values)


# VISUALIZACIÓN

fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

ax1.plot(t_values, u0_values, 'o-', color='#1f77b4', lw=2, ms=3, label=r'$u_0(t)$')
ax1.plot(t_values, u1_values, 's-', color='#ff7f0e', lw=2, ms=3, label=r'$u_1(t)$')
ax1.plot(t_values, u2_values, 'd-', color='#2ca02c', lw=2, ms=3, label=r'$u_2(t)$')

# Masa total para comprobar conservación y línea asintótica
u_total = u0_values + u1_values + u2_values
ax1.plot(t_values, u_total, '--', color='purple', lw=2, label=r'$\Sigma u$ (Masa Total)')
ax1.axhline(1.0/3.0, color='gray', linestyle=':', lw=2, label=r'$u = 0.333$ (Equilibrio)')

ax1.set_title(r'Evolución temporal SPH discreta - 3 Partículas')
ax1.set_xlabel('Tiempo $t$')
ax1.set_ylabel('Amplitud $u$')
ax1.grid(True, alpha=0.3)
ax1.legend()

plt.tight_layout()
plt.show()

# Valores finales
print(f"dt_local = {dt_local:.6f}")
print(f"u0(T) = {u0_values[-1]:.6f}")
print(f"u1(T) = {u1_values[-1]:.6f}")
print(f"u2(T) = {u2_values[-1]:.6f}")
print(f"Suma Total = {u_total[-1]:.6f}")