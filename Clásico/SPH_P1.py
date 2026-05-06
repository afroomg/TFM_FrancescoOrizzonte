import numpy as np
import matplotlib.pyplot as plt


# CONFIGURACIÓN SPH

h = 1.2
dx = 0.5
c = 10**(-0.9)                 # velocidad de advección fija
T_final = 30           
n_points = 500            # detalle fino en la línea

# Condición inicial: Lína azul paper
u0_init, u1_init = 1.0, 0.0

t_values = np.linspace(0.0, T_final, n_points)
dt_local = t_values[1] - t_values[0]

# Gradiente del kernel triangular
gradW01 = -1.0 / (h**2)
gradW10 =  1.0 / (h**2)


def sph_step_discreto(u0, u1, c, dt):
    # Partícula 0
    suma_0 = (u1 - u0) * dx * gradW01
    u0_new = u0 - c * dt * suma_0

    # Partícula 1
    suma_1 = (u0 - u1) * (-dx) * gradW10
    u1_new = u1 - c * dt * suma_1

    return u0_new, u1_new


# EVOLUCIÓN TEMPORAL

u0_values = [u0_init]
u1_values = [u1_init]

u0, u1 = u0_init, u1_init
for _ in range(n_points - 1):
    u0, u1 = sph_step_discreto(u0, u1, c, dt_local)
    u0_values.append(u0)
    u1_values.append(u1)

u0_values = np.array(u0_values)
u1_values = np.array(u1_values)

u_avg = 0.5 * (u0_values + u1_values)
delta_u = u0_values - u1_values


# VISUALIZACIÓN

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel principal: amplitudes
ax1.plot(t_values, u0_values, 'o-', color='#1f77b4', lw=2, ms=4, label=r'$u_0(t)$')
ax1.plot(t_values, u1_values, 's-', color='#ff7f0e', lw=2, ms=4, label=r'$u_1(t)$')
ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.8, label=r'$u=0.5$')

ax1.set_title(r'Evolución temporal SPH discreta')
ax1.set_xlabel('Tiempo $t$')
ax1.set_ylabel('Amplitud $u$')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Panel auxiliar: diferencia entre partículas
ax2.plot(t_values, delta_u, 'd-', color='purple', lw=2, ms=4, label=r'$\Delta u(t)=u_0-u_1$')
ax2.axhline(0.0, color='gray', linestyle='--', alpha=0.8)

ax2.set_title(r'Relajación de la diferencia')
ax2.set_xlabel('Tiempo $t$')
ax2.set_ylabel(r'$\Delta u$')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

# Valores finales por comodidad
print(f"dt_local = {dt_local:.6f}")
print(f"u0({T_final}) = {u0_values[-1]:.6f}")
print(f"u1({T_final}) = {u1_values[-1]:.6f}")
print(f"u0 + u1 = {(u0_values[-1] + u1_values[-1]):.6f}")