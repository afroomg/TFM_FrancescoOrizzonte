# Simulación SPH: Clásico y Quantum

Este repositorio recoge una serie de experimentos de simulación numérica basados en el método de **Smoothed Particle Hydrodynamics (SPH)**. El objetivo es explorar, por un lado, la formulación clásica del método y, por otro, una posible transcripción cuántica de la dinámica utilizando **Continuous-Time Quantum Walks (CTQW)** implementadas con Qiskit.

El proyecto está organizado en dos carpetas independientes: `Clásico` y `Quantum`.

## 📁 Estructura del Repositorio
```
.
├── Clásico/              # Simulaciones SPH clásicas
├── Quantum/              # Simulaciones SPH cuánticas (Qiskit)
├── Memoria/              # Versión más reciente de la memoria
├── requirements.txt      # Dependencias del proyecto
└── README.md
```
```
```

## 📂 Clásico/

Contiene los experimentos iniciales para comprender y validar la formulación SPH discreta. Se probaron distintas configuraciones con 2, 3 y 4 partículas.

### Archivos principales

- **`SPH_P1.py`** — Simulación SPH clásica con **2 partículas**. Es el caso más sencillo y sirve como referencia fundamental para comparar con la versión cuántica.
- **`SPH_P3.py`** — Simulación SPH clásica generalizada para **N partículas** (por defecto 4). Es la versión escalable del modelo clásico y el principal punto de comparación para el caso cuántico de 4 partículas.

### Archivos experimentales

- **`SPH.py`**, **`SPH2.py`**, **`SPH3.py`** — Primeros borradores y pruebas de concepto del método SPH.
- **`SPH_P2.py`** — Simulación con **3 partículas**. Fue útil para explorar interacciones más complejas, aunque el algoritmo cuántico posterior requiere un **número par de partículas** (por la codificación en qubits), por lo que este caso queda como referencia clásica pura.
- **`EDO_continuo.py`** — Implementación de la ecuación diferencial ordinaria (EDO) continua asociada, usada como referencia analítica.

> **Nota:** Todos los experimentos clásicos utilizan un kernel triangular con soporte compacto y una formulación "zeroth-order consistent" inspirada en Au-Yeung.

## 📂 Quantum/

Contiene la implementación cuántica de la dinámica SPH mediante **CTQW (Continuous-Time Quantum Walks)** con decoherencia inducida por medición (dinámica de Markov abierta). El hamiltoniano efectivo se construye a partir de la matriz clásica de tasas de transferencia y se compensa el **efecto Zenón** para recuperar la dinámica clásica en el límite de muchos shots.

### Archivos principales

- **`Q_SPH_P3.py`** — Simulación cuántica con **2 partículas (1 qubit)**. Es el caso cuántico que mejor resultado ha dado: la relajación de las amplitudes sigue fielmente la solución clásica analítica.
- **`Q_SPH_P4.py`** — Simulación cuántica con **4 partículas (2 qubits)**. Extiende el formalismo a un sistema mayor. Se observa que la dinámica de relajación hacia el equilibrio sigue la tendencia esperada, aunque aparece una **ligera oscilación o "rebote"** alrededor del valor de equilibrio, lo que sugiere que el acoplo con el entorno o la resolución temporal podría afinarse aún más.

### Archivos experimentales

- **`Q_SPH.py`**, **`Q_SPH_P2.py`** — Primeras iteraciones del algoritmo cuántico.
- **`Q_PSH_P1.py`** — Variante experimental del protocolo.

### Biblioteca auxiliar

- **`match_decomposition.py`** — Implementación del algoritmo de **Matching Decomposition** para descomponer el hamiltoniano en capas de puertas cuánticas aplicables. Permite simular $e^{-i H t}$ mediante Trotterización usando emparejamientos (matchings) de la matriz de adyacencia del grafo de interacción.

## ⚙️ Requisitos

Las dependencias están listadas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Dependencias principales:**
- `numpy`, `matplotlib`, `scipy` — computación numérica y visualización.
- `qiskit >= 1.2.0`, `qiskit-aer >= 0.15.0` — framework cuántico y simulador.
- `jupyterlab`, `pylatexenc` — entornos de desarrollo y renderizado de circuitos.

## 🚀 Uso Rápido

### Simulación clásica (4 partículas)
```bash
python Clásico/SPH_P3.py
```

### Simulación cuántica (2 partículas — mejor resultado)
```bash
python Quantum/Q_SPH_P3.py
```

### Simulación cuántica (4 partículas — con oscilación)
```bash
python Quantum/Q_SPH_P4.py
```

---

## 📌 Notas Importantes

1. **Número par de partículas:** El algoritmo cuántico codifica las partículas en qubits, por lo que el número de partículas debe ser potencia de 2 (2, 4, 8...). Los casos con 3 partículas (`SPH_P2.py`) son válidos clásicamente pero no tienen contrapartida cuántica directa.

2. **Efecto Zenón:** En los casos cuánticos se aplica una compensación no lineal $J_{\text{eff}} = \arcsin(\sqrt{J_{\text{cl}} \cdot dt}) / dt$ para que la probabilidad de transición por paso de Trotter coincida con la tasa clásica.

3. **Dinámica abierta:** La decoherencia se introduce mediante un qubit ancilla que se entrelaza con el sistema, se mide, y se reinicia (feedforward clásico) en cada paso temporal, forzando la evolución markoviana.

## 🎯 Estado Actual

- **Caso clásico (2 y 4 partículas):** Funcionamiento estable y resultados consistentes con la teoría.
- **Caso cuántico (2 partículas / 1 qubit):** Excelente concordancia con la solución clásica. El modelo funciona correctamente.
- **Caso cuántico (4 partículas / 2 qubits):** La tendencia de relajación es correcta, pero se observa una **pequeña oscilación** alrededor del equilibrio. Está pendiente de ajuste fino del acoplo con el entorno o la resolución temporal.
