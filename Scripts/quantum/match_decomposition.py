import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate
from itertools import combinations


# 1. Utilidades de bitstrings


def int_to_bits(x, n):
    return format(x, f'0{n}b')

def bits_to_int(bits):
    return int(bits, 2)

def hamming_distance(a_bits, b_bits):
    return sum(aa != bb for aa, bb in zip(a_bits, b_bits))

def xor_mask(a_bits, b_bits):
    return ''.join('1' if aa != bb else '0' for aa, bb in zip(a_bits, b_bits))


# 2. De matriz de adyacencia a lista de aristas


def adjacency_to_edges(A, tol=1e-12):
    """
    A: np.ndarray real simétrica (2^n x 2^n)
    Devuelve lista de (u, v, weight) con u<v.
    """
    dim = A.shape[0]
    edges = []
    for u in range(dim):
        for v in range(u+1, dim):
            w = A[u, v]
            if abs(w) > tol:
                edges.append((u, v, float(w)))
    return edges


# 3. Matching greedy muy simple


def build_matchings_greedy(edges, num_nodes):
    """
    edges: lista de (u, v, w)
    Devuelve lista de matchings, cada matching es lista de aristas.
    Recorre aristas y va creando matchings donde no haya conflicto de vértices.
    """
    matchings = []
    for (u, v, w) in edges:
        placed = False
        for M in matchings:
            used_vertices = set()
            for (uu, vv, ww) in M:
                used_vertices.add(uu)
                used_vertices.add(vv)
            if u not in used_vertices and v not in used_vertices:
                M.append((u, v, w))
                placed = True
                break
        if not placed:
            matchings.append([(u, v, w)])
    return matchings


# 4. Circuito para una sola arista (sin compression)


def edge_to_circuit(qc, u, v, weight, t, n_qubits):
    """
    Implementa e^{-i H_edge t} para H_edge = weight (|u><v| + |v><u|)
    usando CX + multi-control-Rx.

    qc: QuantumCircuit
    u, v: int (índices de vértices)
    weight: float
    t: tiempo total (este edge usará ángulo 2*weight*t)
    n_qubits: número de qubits (log2(dim))
    """
    ubits = int_to_bits(u, n_qubits)
    vbits = int_to_bits(v, n_qubits)

    # Bits donde difieren
    diff_positions = [i for i, (a, b) in enumerate(zip(ubits, vbits)) if a != b]
    same_positions = [i for i in range(n_qubits) if i not in diff_positions]

    # Elegimos como target el primer bit que difiere (posición en [0..n_qubits-1])
    target = diff_positions[0]

    # Resto de bits que difieren serán arreglados por CX (técnica de Gonzales et al.)
    extra_diff = diff_positions[1:]

    # 4.1. Basis change: llevar (u,v) a Hamming distance 1 mediante CX
    #    Para cada bit d en extra_diff aplicamos CX(target -> d)
    for d in extra_diff:
        qc.cx(target, d)

    # 4.2. X gates para fijar controles según los bits comunes (same_positions)
    # Queremos que el estado |u> tenga todos estos qubits en |1> para un MCX limpio.
    # Si ubits[i] == '0', aplicamos X antes y después.
    x_positions = []
    for i in same_positions:
        if ubits[i] == '0':
            qc.x(i)
            x_positions.append(i)

    # 4.3. Multi-control Rx sobre el target
    theta = 2.0 * weight * t
    controls = same_positions  # controles: qubits con bits comunes

    if len(controls) == 0:
        # No hay controles: simple Rx
        qc.rx(theta, target)
    elif len(controls) == 1:
        # Un control: CRx se puede implementar como CX + Rz + ...,
        # pero aquí usamos un MCX + rotaciones; luego Qiskit transpila a CX+U3
        # Implementación básica: usar puente con ancillas si quieres.
        # Para demo, usamos un patrón con CNOT+Rx+CNOT para un control sobre target.
        c = controls[0]
        # CRx(theta) = (I⊗H) CNOT (I⊗Rz(theta)) CNOT (I⊗H)
        # Aquí usamos identidad conocida: CRx = (H target) CRz (H target),
        # pero Qiskit no tiene CRz nativo; usamos U3 y dejamos que transpile.
        qc.cu(theta, -np.pi/2, np.pi/2, 0, c, target)
    else:
        # Múltiples controles: construimos un MCX y lo "embeddeamos" en CRx
        # Estrategia sencilla:
        # 1) MCX(controls -> ancilla/target mapping)
        # 2) Rx en target
        # 3) MCX†
        # En Qiskit, MCXGate target=target, controls=controls
        mcx = MCXGate(len(controls))
        qc.append(mcx, controls + [target])
        qc.rx(theta, target)
        qc.append(mcx, controls + [target])

    # 4.4. Deshacer X en posiciones que cambiamos
    for i in x_positions:
        qc.x(i)

    # 4.5. Deshacer basis change (CX en extra_diff)
    for d in reversed(extra_diff):
        qc.cx(target, d)


# 5. Un Trotter step usando matching decomposition


def ctqw_matching_step(A, t, n_trotter_steps=1):
    """
    Construye un circuito que aproxima e^{-i A t} con n_trotter_steps
    usando matching decomposition.
    """
    dim = A.shape[0]
    n_qubits = int(np.log2(dim))
    assert 2**n_qubits == dim, "La dimensión de A debe ser potencia de 2"

    qr = QuantumRegister(n_qubits, 'q')
    qc = QuantumCircuit(qr, name='CTQW_matching')

    edges = adjacency_to_edges(A)
    matchings = build_matchings_greedy(edges, dim)

    dt = t / n_trotter_steps

    for _ in range(n_trotter_steps):
        for M in matchings:
            for (u, v, w) in M:
                edge_to_circuit(qc, u, v, w, dt, n_qubits)

    return qc