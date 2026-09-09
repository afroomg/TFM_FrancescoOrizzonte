# TFM — SPH clásico vs. SPH cuántico

Trabajo de Fin de Máster que compara la formulación **clásica** del método
*Smoothed Particle Hydrodynamics* (SPH) con una transcripción **cuántica** de la
misma dinámica basada en *Continuous-Time Quantum Walks* (CTQW) implementadas con
Qiskit.

El repositorio alberga tanto el código de las simulaciones como la memoria del
TFM escrita en LaTeX.

## 📁 Estructura

```
.
├── Clásico/                # Simulación SPH clásica
│   └── SPH_P1.py           #   · 2 partículas (referencia)
├── Quantum/                # Simulación SPH cuántica (Qiskit)
│   ├── Q_SPH_P3.py         #   · 2 partículas / 1 qubit (mejor resultado)
│   └── match_decomposition.py  #   · Helper: Matching Decomposition
├── TFM_latex/              # Memoria del TFM en LaTeX
│   ├── main.tex            #   · Documento raíz
│   ├── cover.pdf           #   · Portada ya compilada
│   ├── referencias.bib     #   · Bibliografía
│   └── chapters/           #   · Capítulos (cap1…cap7) + imágenes
├── requirements.txt        # Dependencias de Python
└── README.md
```

## 📂 Clásico — `SPH_P1.py`

Simulación SPH discreta con **2 partículas** y kernel triangular (formulación inspirada en Au-Yeung). Es el caso más sencillo y sirve de referencia analítica para validar la versión cuántica. Únicamente depende de `numpy` y `matplotlib`.

```bash
python Clásico/SPH_P1.py
```

## 📂 Quantum — `Q_SPH_P3.py`

Simulación cuántica de la misma dinámica de 2 partículas mediante **CTQW** con decoherencia inducida por medición (dinámica de Markov abierta). El hamiltoniano eficiente se construye a partir de la matriz clásica de tasas de transferencia y se compensa el **efecto Zenón** para recuperar la dinámica clásica en el límite de muchos *shots*. La evolución unitaria $e^{-iHt}$ se implementa mediante **Matching Decomposition** (ver `match_decomposition.py`), que descompone el hamiltoniano en capas de puertas aplicables mediante Trotterización usando emparejamientos de la matriz de adyacencia.

```bash
python Quantum/Q_SPH_P3.py
```

### Biblioteca auxiliar — `match_decomposition.py`

Implementación del algoritmo de *Matching Decomposition* (Atalah *et al.*,2026). Expone `ctqw_matching_step(H, dt, n_trotter_steps)` y se apoya en `numpy`, `qiskit` e `itertools`. No es ejecutable por sí mismo; es importado por
`Q_SPH_P3.py`.

## ⚙️ Requisitos

```bash
pip install -r requirements.txt
```

Dependencias principales: `numpy`, `matplotlib`, `scipy`, `qiskit>=1.2.0`, `qiskit-aer>=0.15.0`, `pylatexenc`.

## 📝 Memoria (TFM_latex)

La memoria se redacta en LaTeX. El documento raíz es `TFM_latex/main.tex`, que incluye los capítulos de `chapters/` y la portada `cover.pdf`. Para compilar:

```bash
cd TFM_latex
pdflatex main && bibtex main && pdflatex main && pdflatex main
