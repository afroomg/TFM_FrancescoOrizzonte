import numpy as np
import matplotlib.pyplot as plt
from match_decomposition import ctqw_matching_step
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

#En este experimento utilizamos el simulador de AR Simulator con el método de matriz de densidad y
#vamos sacando el vector estado. Este no me gusta mucho, entonces no lo consideren para el estado final del TFM.


# 1. CONFIGURACIÓN FÍSICA SPH Y PARÁMETROS
h = 1.2
dx = 0.5
c = 10**(-0.5)  # Velocidad fija (~0.3162)
nu = 1.0 / (h**2)

# Tasa de transferencia clásica
J_classical = c * dx * nu 

# Parámetros de tiempo
T_final = 30.0
n_steps = 100  # Puntos en la gráfica y número de sub-pasos
dt = T_final / n_steps
t_values = np.linspace(0.0, T_final, n_steps)

# Condición inicial azul
u0_init, u1_init = 1.0, 0.0

print(f"Acoplamiento clásico J = {J_classical:.4f}, dt = {dt:.4f}")


# 2. COMPENSACIÓN DEL EFECTO ZENÓN

if J_classical * dt > 1.0:
    raise ValueError("El dt es muy grande para simular las probabilidades. Aumenta n_steps.")

J_eff = np.arcsin(np.sqrt(J_classical * dt)) / dt

# Creamos la matriz del Hamiltoniano Cuántico Efectivo
H_eff = np.array([[0.0, J_eff],
                  [J_eff, 0.0]], dtype=float)

# Simulador de Matriz de Densidad
simulator = AerSimulator(method="density_matrix")

results_u0 = [u0_init]
results_u1 = [u1_init]


# 3. CREACIÓN DEL PASO DE SIMULACIÓN (DECOHERENCIA PERFECTA)

qr = QuantumRegister(2, name="q")
qc_step = QuantumCircuit(qr, name="SPH_Step")

qc_evol = ctqw_matching_step(H_eff, dt, n_trotter_steps=1)
qc_step.compose(qc_evol, qubits=[0], inplace=True)

# Decoherencia
qc_step.cx(0, 1)
qc_step.reset(1)

print("\n--- Circuito de un Paso Temporal (CTQW -> Clásico) ---")
print(qc_step.draw(output='text'))


# 4. EJECUCIÓN DEL BUCLE TEMPORAL

# Partimos desde el step 1 porque el t=0 ya está guardado (condición inicial)
for step in range(1, n_steps):
    qc_main = QuantumCircuit(2)
    
    if u0_init < 1.0:
        qc_main.initialize([np.sqrt(u0_init), np.sqrt(u1_init)], 0)
        
    # Añadimos el paso de tiempo "step" veces
    for _ in range(step):
        qc_main.compose(qc_step, inplace=True)
        
    qc_main.save_density_matrix()
    
    # Ejecutar en el simulador
    rho = simulator.run(qc_main).result().data()['density_matrix']
    
    # Las probabilidades reales clásicas sobreviven en la diagonal
    results_u0.append(np.real(rho.data[0, 0]))
    results_u1.append(np.real(rho.data[1, 1]))

results_u0 = np.array(results_u0)
results_u1 = np.array(results_u1)
delta_u = results_u0 - results_u1


# 5. VISUALIZACIÓN CLÁSICA (IDÉNTICA A TU CAPTURA)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Amplitudes
ax1.plot(t_values, results_u0, '-', color='#1f77b4', lw=4, label='$u_0(t)$')
ax1.plot(t_values, results_u1, '-', color='#ff7f0e', lw=4, label='$u_1(t)$')
ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.8, label='$u=0.5$')

ax1.set_title('Evolución temporal Open CTQW (SPH Perfecto)', fontsize=13)
ax1.set_xlabel('Tiempo $t$')
ax1.set_ylabel('Amplitud $u$')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc='upper right')

# Panel 2: Diferencia
ax2.plot(t_values, delta_u, '-', color='purple', lw=4, label='$\\Delta u(t) = u_0 - u_1$')
ax2.axhline(0.0, color='gray', linestyle='--', alpha=0.8)

ax2.set_title('Relajación de la diferencia', fontsize=13)
ax2.set_xlabel('Tiempo $t$')
ax2.set_ylabel('$\\Delta u$')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([-1.05, 1.05])
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()