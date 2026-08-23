import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from match_decomposition import ctqw_matching_step
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ─────────────────────────────────────────────────────────
# 1. PARÁMETROS FÍSICOS SPH
# ─────────────────────────────────────────────────────────
h  = 1.2
dx = 0.5
c  = 10**(-0.5)
nu = 1.0 / (h**2)

display_step = 1      # ← pon None para silencio, o un int para ver ese paso

J_classical = c * dx * nu

print(f"Parámetros físicos: c={c:.4f}, J_classical={J_classical:.4f}")

T_final = 30.0
n_steps = 100
dt      = T_final / n_steps

shots   = 4000
backend = AerSimulator()

# ─────────────────────────────────────────────────────────
# 2. J efectivo (compensación Zenón)
# ─────────────────────────────────────────────────────────
arg = J_classical * dt
if arg > 1.0:
    raise ValueError(f"J_classical*dt={arg:.4f} > 1.0. Aumenta n_steps.")

J_eff = np.arcsin(np.sqrt(arg)) / dt

H_eff = np.array([[0.0, J_classical],
                  [J_classical, 0.0]], dtype=float)

print(f"dt={dt:.4f}, J_eff={J_eff:.6f}")
print(f"Verificación: sin²(J_eff·dt) = {np.sin(J_eff*dt)**2:.6f}  ←→  J_classical·dt = {arg:.6f}")

# ─────────────────────────────────────────────────────────
# 3. UN SOLO BLOQUE TROTTER (reutilizado n veces)
# ─────────────────────────────────────────────────────────
qc_step = ctqw_matching_step(H_eff, dt, n_trotter_steps=1)

print("\nBloque unitario de 1 paso Trotter (Matching Decomposition):")
print(qc_step.decompose().draw(output='text'))

# ─────────────────────────────────────────────────────────
# 4. FUNCIÓN: construye circuito con n_pasos bloques apilados
# ─────────────────────────────────────────────────────────
def display_circuit(qc: QuantumCircuit, n_pasos: int) -> None:
    fig = qc.draw(
        output="mpl",        # renderiza con matplotlib → imagen real, no texto
        style="iqp",         # estilo visual limpio de IBM
        fold=40,             # 40 puertas por fila antes de hacer salto
        scale=0.8,
        initial_state=True,  # muestra |0⟩ al inicio del hilo
        plot_barriers=True,  # separa visualmente cada bloque Trotter
    )
    #filename = f"circuit_step_{n_pasos:03d}.png"
    #fig.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show(block=True)
    plt.pause(0.5)
    plt.close(fig)

def build_and_run(n_pasos: int) -> tuple[float, float]:
    """
    Construye un circuito de 1 qubit con n_pasos bloques Trotter
    consecutivos y extrae las probabilidades P(|0>) y P(|1>).
    Estado inicial implícito: |0> (u0=1, u1=0).
    """
    if n_pasos == 0:
        return 1.0, 0.0   # condición inicial exacta

    qc = QuantumCircuit(1, 1)

    # Apilar n_pasos bloques del mismo circuito de evolución
    for _ in range(n_pasos):
        qc.compose(qc_step, qubits=[0], inplace=True)

    # Medición final
    qc.measure(0, 0)

     #if isinstance(display_step, int) and n_pasos == display_step:
       #  display_circuit(qc, n_pasos)

    tqc = transpile(qc, backend, optimization_level=1)
    job = backend.run(tqc, shots=shots)
    counts = job.result().get_counts()

    count_0 = counts.get('0', 0)
    count_1 = counts.get('1', 0)

    return count_0 / shots, count_1 / shots


qc_preview = QuantumCircuit(1, 1)
qc_preview.compose(qc_step, qubits=[0], inplace=True)
qc_preview.measure(0, 0)
#isplay_circuit(qc_preview, n_pasos=1)
# ─────────────────────────────────────────────────────────
# 5. BUCLE TEMPORAL
# ─────────────────────────────────────────────────────────
n_puntos = 30
indices  = np.linspace(0, n_steps, n_puntos, dtype=int)
t_eval   = indices * dt

print(f"\nEjecutando {n_puntos} puntos temporales ({shots} shots cada uno)...")
print("=" * 65)

results_u0 = []
results_u1 = []

for i, n_pasos in enumerate(indices):
    u0, u1 = build_and_run(n_pasos)
    results_u0.append(u0)
    results_u1.append(u1)
    print(f"  [{i+1:3d}/{n_puntos}] t={t_eval[i]:6.2f} | "
          f"n_pasos={n_pasos:3d} | u0={u0:.3f}, u1={u1:.3f}, Δu={u0-u1:+.3f}")

results_u0 = np.array(results_u0)
results_u1 = np.array(results_u1)
delta_u    = results_u0 - results_u1

# ─────────────────────────────────────────────────────────
# 6. VISUALIZACIÓN
# ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(t_eval, results_u0, 'o-', color='#1f77b4', lw=2, ms=5, label='$u_0$ QSPH')
ax1.plot(t_eval, results_u1, 'o-', color='#ff7f0e', lw=2, ms=5, label='$u_1$ QSPH')
ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.7)
ax1.set_title(f'CTQW Trotterizado | shots={shots}', fontsize=12)
ax1.set_xlabel('Tiempo $t$')
ax1.set_ylabel('Probabilidad $u$')
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

ax2.plot(t_eval, delta_u, 'o-', color='purple', lw=2, ms=5, label='$\\Delta u$ QSPH')
ax2.axhline(0.0, color='gray', linestyle=':', alpha=0.7)
ax2.set_title('Relajación $\\Delta u(t) = u_0 - u_1$', fontsize=12)
ax2.set_xlabel('Tiempo $t$')
ax2.set_ylabel('$\\Delta u$')
ax2.set_ylim(-1.2, 1.2)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.show(block=True)
