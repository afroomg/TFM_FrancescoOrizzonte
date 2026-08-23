import numpy as np
import matplotlib.pyplot as plt
from match_decomposition import ctqw_matching_step
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# CONFIGURACIÓN FÍSICA SPH / CTQW

h = 1.2
dx = 0.5
c = 10**(-0.5)  # Velocidad fija (~0.3162)
nu = 1.0 / (h**2) 

# Acoplamiento del Hamiltoniano
J = c * dx * nu 
H = np.array([[0.0, J],
              [J, 0.0]], dtype=float)

# Parámetros de tiempo
T_final = 30.0
n_points = 500
t_values = np.linspace(0.0, T_final, n_points)

# Condición inicial azul: u0(0)=1, u1(0)=0
u0_init, u1_init = 1.0, 0.0

# Simulador
simulator = AerSimulator(method="statevector")

results_u0 = []
results_u1 = []

print(f"Iniciando simulación cuántica CTQW...")
print(f"Acoplamiento J = {J:.4f}")


# BUCLE TEMPORAL CUÁNTICO

for t in t_values:
    # 1) Definir el número de pasos de Trotter dinámicamente
    # Mantenemos un tamaño de paso dt_step aprox de 0.5 para alta fidelidad
    # Si t es muy pequeño, con 1 paso basta.
    n_trotter = max(1, int(t / 0.5))
    
    # 2) Generar el circuito de evolución con tu función local
    qc_evol = ctqw_matching_step(H, t, n_trotter_steps=n_trotter)
    
    # 3) Construir circuito completo
    qc = QuantumCircuit(1) # 1 qubit = 2 estados (partículas 0 y 1)
    
    # Inicialización del estado (si fuera diferente de |0>)
    if u0_init < 1.0:
        initial_state = [np.sqrt(u0_init), np.sqrt(u1_init)]
        qc.initialize(initial_state, 0)
        
    # Añadir evolución y medir statevector
    qc.compose(qc_evol, inplace=True)
    qc.save_statevector()
    
    # 4) Ejecución
    state = simulator.run(qc).result().get_statevector()
    
    # 5) Extraer probabilidades |ψ|^2
    u0_t = np.abs(state[0])**2
    u1_t = np.abs(state[1])**2
    
    results_u0.append(u0_t)
    results_u1.append(u1_t)

results_u0 = np.array(results_u0)
results_u1 = np.array(results_u1)
delta_u = results_u0 - results_u1

print("Simulación completada. Generando gráficas...")


# VISUALIZACIÓN COMPARATIVA

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Amplitudes 
ax1.plot(t_values, results_u0, '-', color='#1f77b4', lw=2.5, label='$u_0(t)$')
ax1.plot(t_values, results_u1, '-', color='#ff7f0e', lw=2.5, label='$u_1(t)$')
ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.8, label='$u=0.5$')

ax1.set_title('Evolución temporal CTQW Unitario', fontsize=13)
ax1.set_xlabel('Tiempo $t$', fontsize=12)
ax1.set_ylabel('Amplitud de probabilidad $|\\psi|^2$', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc='upper right')

# Panel 2: Diferencia
ax2.plot(t_values, delta_u, '-', color='purple', lw=2.5, label='$\\Delta u(t) = u_0 - u_1$')
ax2.axhline(0.0, color='gray', linestyle='--', alpha=0.8)

ax2.set_title('Comportamiento de la diferencia', fontsize=13)
ax2.set_xlabel('Tiempo $t$', fontsize=12)
ax2.set_ylabel('$\\Delta u$', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([-1.05, 1.05])
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
