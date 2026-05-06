import numpy as np
import matplotlib.pyplot as plt

from match_decomposition import ctqw_matching_step
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from IPython.core.display_functions import display


# 1. PARÁMETROS FÍSICOS SPH

h  = 1.2
dx = 0.5
c  = 10**(-0.9)       # Velocidad fija (~0.3162)
nu = 1.0 / (h**2)

print(f"Parámetros físicos: h={h}, dx={dx}, c={c:.4f}, nu={nu:.4f}")


# 2. DEFINICIÓN DEL SISTEMA DE 4 PARTÍCULAS (2 Qubits)

N_particles = 4
x_pos = np.array([i * dx for i in range(N_particles)])

# Condición inicial (Partícula 0 tiene todo el fluido, estado |00>)
u_init = np.array([1.0, 0.0, 0.0, 0.0])

# Función para el cálculo del gradiente (Kernel triangular)
def get_gradW(xi, xj):
    dist = abs(xi - xj)
    if dist == 0 or dist >= h:
        return 0.0
    return np.sign(xi - xj) * nu

# Matriz Clásica J
J_cl_matrix = np.zeros((N_particles, N_particles))
for i in range(N_particles):
    for j in range(N_particles):
        if i != j:
            # Peso estricto absoluto como solicitaste
            peso = 0.5 * (abs(get_gradW(x_pos[i], x_pos[j])) + abs(get_gradW(x_pos[j], x_pos[i])))
            # Multiplicamos por c y por la distancia para obtener la tasa de transferencia
            J_cl_matrix[i, j] = c * abs(x_pos[i] - x_pos[j]) * peso

print("\nMatriz J Clásica (Tasa de transferencia):")
print(np.round(J_cl_matrix, 4))


# 3. PARÁMETROS DE SIMULACIÓN Y J_EFECTIVA

T_final  = 30.0
n_steps  = 100          # Número de pasos de tiempo (dt)
dt       = T_final / n_steps
shots    = 4000
backend  = AerSimulator()

# Compensación Efecto Zenón para toda la matriz
H_eff = np.zeros_like(J_cl_matrix)
for i in range(N_particles):
    for j in range(N_particles):
        if i != j:
            arg = J_cl_matrix[i, j] * dt
            if arg > 1.0:
                raise ValueError(f"J_classical*dt={arg:.4f} > 1.0. Aumenta n_steps.")
            # Aplicar la fórmula del efecto zenón adaptada a cada elemento
            H_eff[i, j] = np.arcsin(np.sqrt(arg)) / dt

print(f"\ndt={dt:.4f}")
print("Matriz Hamiltoniano H_eff (escalada por efecto Zenón):")
print(np.round(H_eff, 4))

# Sub-circuito de evolución unitaria CTQW usando Match Decomposition
# Nota: Pasamos la matriz H_eff de 4x4 a la función que ya tienes implementada
qc_evol = ctqw_matching_step(H_eff, dt, n_trotter_steps=1)


# 4. CONSTRUCCIÓN Y EJECUCIÓN DEL CIRCUITO ABIERTO (Qiskit 2.x API)

def build_and_run_markov_4p(n_pasos: int) -> list:
    """
    Simula la dinámica abierta con 4 partículas.
    Se añade 1 ancilla para inducir la disipación / medición intermedia.
    """
    if n_pasos == 0:
        return u_init.tolist()

    # 2 qubits de sistema + 1 ancilla
    qr_sys = QuantumRegister(2, name='q_sys')
    qr_anc = QuantumRegister(1, name='q_anc')
    
    cr_anc = ClassicalRegister(1, name='c_anc')  # Medición intermedia
    cr_sys = ClassicalRegister(2, name='c_sys')  # Medición final
    
    qc = QuantumCircuit(qr_sys, qr_anc, cr_anc, cr_sys)

    # Estado inicial: |00> (U0=1, resto=0). 
    
    for _ in range(n_pasos):
        # 1. Evolución Hamiltoniana (CTQW) en los 2 qubits del sistema
        qc.compose(qc_evol, qubits=[qr_sys[0], qr_sys[1]], inplace=True)
        
        # 2. Interacción con el entorno (Ancilla)
        # Entrelazamos ambos qubits del sistema con el ancilla para simular decoherencia global
        qc.cx(qr_sys[0], qr_anc[0])
        qc.cx(qr_sys[1], qr_anc[0])
        
        # 3. Disipación: Medición + Reset Clásico (Feedforward)
        qc.measure(qr_anc[0], cr_anc[0])
        
        with qc.if_test((cr_anc[0], 1)):
            qc.x(qr_anc[0])
            
        qc.barrier()

    # Medición final del estado del fluido (ambos qubits)
    qc.measure(qr_sys[0], cr_sys[0])
    qc.measure(qr_sys[1], cr_sys[1])

    # Ejecución
    tqc = transpile(qc, backend, optimization_level=1)
    job = backend.run(tqc, shots=shots)
    counts = job.result().get_counts()

    # Extraer las probabilidades de los 4 estados
    u_out = [0.0, 0.0, 0.0, 0.0]
    for bitstring, count in counts.items():
        bits = bitstring.split()
        # Qiskit formatea la salida como "c_sys c_anc" -> extraemos el primer bloque
        sys_bits = bits[0] if len(bits) > 1 else bitstring[:2]
        
        # El string es Big-Endian. Estado 0: '00', Estado 1: '01', Estado 2: '10', Estado 3: '11'
        idx = int(sys_bits, 2)
        u_out[idx] += count

    # Normalizar dividiendo por los shots
    return [c / shots for c in u_out]


# 5. BUCLE TEMPORAL Y RECOLECCIÓN DE DATOS

n_puntos = 30  
indices  = np.linspace(0, n_steps, n_puntos, dtype=int)
t_eval   = indices * dt

print(f"\nEjecutando {n_puntos} puntos temporales con {shots} shots...")
results = []

for i, n_pasos in enumerate(indices):
    u_out = build_and_run_markov_4p(n_pasos)
    results.append(u_out)
    print(f"  [{i+1:2d}/{n_puntos}] t={t_eval[i]:5.1f} | U = [{u_out[0]:.3f}, {u_out[1]:.3f}, {u_out[2]:.3f}, {u_out[3]:.3f}]")

results = np.array(results)


# 6. VISUALIZACIÓN

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for j in range(N_particles):
    ax.plot(t_eval, results[:, j], 'o-', color=colors[j], lw=2, ms=5, 
            label=rf'$u_{j}$ (x={x_pos[j]})')

# En un sistema de 4 partículas que conservan masa (1.0), el equilibrio es 1/4 = 0.25
ax.axhline(0.25, color='gray', linestyle=':', alpha=0.8, lw=2, label='Equilibrio (0.25)')

ax.set_title(f'CTQW Markoviano 4 Partículas (2 Qubits) | shots={shots}', fontsize=12)
ax.set_xlabel('Tiempo $t$')
ax.set_ylabel('Probabilidad $u$')
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()


# 7. DIBUJAR EL CIRCUITO VISUAL (MPL)

# Creamos un circuito con solo 2 pasos de Trotter para que se vea claro y 
# quepa bien en la pantalla sin saturarse.

# 2 qubits de sistema + 1 ancilla
qr_sys_draw = QuantumRegister(2, name='q_sys')
qr_anc_draw = QuantumRegister(1, name='q_anc')
cr_anc_draw = ClassicalRegister(1, name='c_anc')  
cr_sys_draw = ClassicalRegister(2, name='c_sys')  

qc_draw = QuantumCircuit(qr_sys_draw, qr_anc_draw, cr_anc_draw, cr_sys_draw)

# Añadimos solo 2 pasos para la visualización
for _ in range(2):
    # 1. CTQW
    qc_draw.compose(qc_evol, qubits=[qr_sys_draw[0], qr_sys_draw[1]], inplace=True)
    
    # 2. Decoherencia global 
    qc_draw.cx(qr_sys_draw[0], qr_anc_draw[0])
    qc_draw.cx(qr_sys_draw[1], qr_anc_draw[0])
    
    # 3. Medición + Reset
    qc_draw.measure(qr_anc_draw[0], cr_anc_draw[0])
    with qc_draw.if_test((cr_anc_draw[0], 1)):
        qc_draw.x(qr_anc_draw[0])
        
    qc_draw.barrier()

# Medición final
qc_draw.measure(qr_sys_draw[0], cr_sys_draw[0])
qc_draw.measure(qr_sys_draw[1], cr_sys_draw[1])

# Mostrar el diagrama visual. (En Jupyter se renderiza automáticamente debajo de la celda)
# Usamos fold=-1 para evitar que Qiskit corte la imagen en varias líneas si es muy larga.
fig2 = plt.figure(figsize=(12, 6))
qc_draw.draw(output='mpl', style='iqp', fold=-1)
plt.show()