import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from match_decomposition import ctqw_matching_step
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


# 1. PARÁMETROS FÍSICOS SPH Y PARÁMETROS
h  = 1.2
dx = 0.5
c  = 10**(-0.5)       # Velocidad fija (~0.3162)
nu = 1.0 / (h**2)
J_classical = c * dx * nu   # Tasa clásica de transferencia

print(f"Parámetros físicos: c={c:.4f}, J_classical={J_classical:.4f}")

T_final  = 30.0
n_steps  = 100          # Número de pasos de tiempo (dt)
dt       = T_final / n_steps

shots    = 4000
backend  = AerSimulator()

# 2. COMPENSACIÓN DEL EFECTO ZENÓN
arg = J_classical * dt
if arg > 1.0:
    raise ValueError(f"J_classical*dt={arg:.4f} > 1.0. Aumenta n_steps.")

J_eff = np.arcsin(np.sqrt(arg)) / dt

H_eff = np.array([[0.0, J_classical],
                  [J_classical, 0.0]], dtype=float)

print(f"dt={dt:.4f}, J_eff={J_eff:.6f}")
print(f"Verificación: sin²(J_eff·dt) = {np.sin(J_eff*dt)**2:.6f}  ←→  J_classical·dt = {arg:.6f}")

# Sub-circuito de evolución unitaria CTQW (Matching Decomposition)
qc_evol = ctqw_matching_step(H_eff, dt, n_trotter_steps=1)

print("\n Bloque de Matching")
print(qc_evol.decompose().draw(output='text'))


# 3. CONSTRUCCIÓN Y EJECUCIÓN (Qiskit 2.x API Moderna)

def build_and_run_markov(n_pasos: int, show_circuit: bool = False) -> tuple[float, float]:
    """
    Simula la dinámica abierta (Markoviana) mediante Mid-Circuit Measurement.
    """
    if n_pasos == 0:
        return 1.0, 0.0  # Condición inicial: todo el fluido en el nodo 0

    qr = QuantumRegister(2, name='q')
    cr_ancilla = ClassicalRegister(1, name='ancilla')  # Medición intermedia
    cr_sistema = ClassicalRegister(1, name='sistema')  # Medición final
    qc = QuantumCircuit(qr, cr_ancilla, cr_sistema)

    # Bucle de Trotterización + Decoherencia
    for _ in range(n_pasos):
        # 1. Evolución Hamiltoniana (CTQW)
        qc.compose(qc_evol, qubits=[qr[1]], inplace=True)
        
        # 2. Interacción con el entorno (Ancilla)
        qc.cx(qr[1], qr[0])
        
        # 3. Disipación: Medición + Reset Clásico (Feedforward)
        qc.measure(qr[0], cr_ancilla[0])
        
        with qc.if_test((cr_ancilla[0], 1)):
            qc.x(qr[0])
            
        qc.barrier()

    # Medición final del estado del fluido
    qc.measure(qr[1], cr_sistema[0])

    # Ejecución en AerSimulator
    tqc = transpile(qc, backend, optimization_level=1)
    job = backend.run(tqc, shots=shots)
    counts = job.result().get_counts()

    # Post-proceso: Los counts en Qiskit son 'cr_sistema cr_ancilla' (Big-Endian)
    # Queremos extraer solo el valor de cr_sistema, que es el caracter de la izquierda
    # ya que declaramos (qr, cr_ancilla, cr_sistema) en ese orden, o podemos buscar por key.
    count_0 = 0
    count_1 = 0
    for bitstring, count in counts.items():
        # Separamos el bitstring por espacios si los hay, 
        # la convención normal junta 'sistema ancilla'. Extraemos el bit de sistema (índice 0 string separado)
        bits = bitstring.split()
        sys_bit = bits[0] if len(bits) > 1 else bitstring[0]
        
        if sys_bit == '0':
            count_0 += count
        else:
            count_1 += count

    return count_0 / shots, count_1 / shots


# 4. BUCLE TEMPORAL Y RECOLECCIÓN DE DATOS

n_puntos = 40
indices  = np.linspace(0, n_steps, n_puntos, dtype=int)
t_eval   = indices * dt

print(f"\nEjecutando {n_puntos} puntos temporales con {shots} shots cada uno...")
print("="*65)

results_u0 = []
results_u1 = []

for i, n_pasos in enumerate(indices):
    u0, u1 = build_and_run_markov(n_pasos)
    results_u0.append(u0)
    results_u1.append(u1)
    print(f"  [{i+1:3d}/{n_puntos}] t={t_eval[i]:6.2f} | n_pasos={n_pasos:3d} | "
          f"u0={u0:.3f}, u1={u1:.3f}, Δu={u0-u1:+.3f}")

results_u0 = np.array(results_u0)
results_u1 = np.array(results_u1)
delta_u    = results_u0 - results_u1


# 5. VISUALIZACIÓN

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Amplitudes
ax1.plot(t_eval, results_u0, 'o-', color='#1f77b4', lw=2, ms=5, label='$u_0$ QSPH (shots)')
ax1.plot(t_eval, results_u1, 'o-', color='#ff7f0e', lw=2, ms=5, label='$u_1$ QSPH (shots)')
ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.7)

ax1.set_title(f'CTQW) | shots={shots}', fontsize=12)
ax1.set_xlabel('Tiempo $t$')
ax1.set_ylabel('Probabilidad $u$')
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Panel 2: Diferencia (Relaxation)
ax2.plot(t_eval, delta_u, 'o-', color='purple', lw=2, ms=5, label='$\\Delta u$ QSPH')
ax2.axhline(0.0, color='gray', linestyle=':', alpha=0.7)

ax2.set_title('Relajación $\\Delta u(t) = u_0 - u_1$', fontsize=12)
ax2.set_xlabel('Tiempo $t$')
ax2.set_ylabel('$\\Delta u$')
ax2.set_ylim(-0.1, 1.1)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.show(block=True)
