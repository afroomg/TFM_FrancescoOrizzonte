import numpy as np
import matplotlib.pyplot as plt

# 1. Los datos (tu array 1D)

datos1 = [
    0.268, 0.213, 0.117, 0.110, 0.157, 0.137, 0.103, 0.186, 0.156, 0.103,
    0.162, 0.189, 0.146, 0.174, 0.186, 0.165, 0.113, 0.166, 0.194, 0.148,
    0.103, 0.151, 0.146, 0.155, 0.170, 0.194, 0.076, 0.109, 0.124, 0.119,
    0.126, 0.144, 0.152, 0.131, 0.103, 0.124, 0.117, 0.078, 0.117, 0.138,
    0.234, 0.090, 0.132, 0.126, 0.147, 0.151, 0.140, 0.138, 0.114, 0.178,
    0.130, 0.123, 0.103, 0.096, 0.111, 0.087, 0.111, 0.123, 0.165, 0.161,
    0.090, 0.194, 0.097, 0.194, 0.143, 0.123, 0.130, 0.233, 0.135, 0.183,
    0.182, 0.110, 0.139, 0.184, 0.111, 0.117, 0.167, 0.117, 0.103, 0.183,
    0.098, 0.287, 0.266, 0.192, 0.112, 0.104, 0.116, 0.095, 0.139, 0.158,
    0.082, 0.096, 0.110, 0.102, 0.091, 0.123, 0.173, 0.125, 0.119, 0.116,
    0.108, 0.262, 0.083, 0.126, 0.125, 0.091, 0.152, 0.125, 0.116, 0.110,
    0.155, 0.134, 0.129, 0.239, 0.125, 0.100, 0.164, 0.123, 0.108, 0.138,
    0.117, 0.113, 0.143, 0.095, 0.097, 0.110, 0.100, 0.102, 0.099, 0.086,
    0.110, 0.102, 0.144, 0.123, 0.125, 0.108, 0.089, 0.192, 0.125, 0.129,
    0.146, 0.116, 0.089, 0.137, 0.261, 0.171, 0.102, 0.091, 0.112, 0.137,
    0.091, 0.096, 0.095, 0.165, 0.113, 0.089, 0.106, 0.110, 0.138
]
datos = [x * 1000 for x in datos1]


num_datos = len(datos)
media = np.mean(datos)
varianza = np.var(datos, ddof=1)           # Varianza (variación estándar)
desv_tipica = np.std(datos, ddof=1)        # Desviación típica

# 3. Definir los rangos (desde 50 hasta 275 para asegurar que cubrimos hasta el 255+, de 25 en 25)
rangos = np.arange(75, 301, 25) 

# Extra: Comprobación por consola de los contadores en cada rango (como solicitaste)
contadores, limites = np.histogram(datos, bins=rangos)
print("--- CONTADORES POR RANGO ---")
for i in range(len(contadores)):
    print(f"Rango [{limites[i]:.0f} a {limites[i+1]:.0f}): {contadores[i]} datos")
print("----------------------------\n")

# 4. Configurar el gráfico
fig, ax = plt.subplots(figsize=(10, 7))

# Dibujar el histograma (matplotlib cuenta internamente y grafica)
ax.hist(datos, bins=rangos, color='#4C72B0', edgecolor='black', alpha=0.8)

# 5. Textos y etiquetas del gráfico
ax.set_xlabel('Diámetro (nm)', fontsize=20)
ax.set_ylabel('Número de partículas', fontsize=20)
ax.set_xticks(rangos) # Asegurar que el eje X muestre exactamente 50, 75, 100...

ax.tick_params(axis='x', labelsize=21, rotation=45)
ax.tick_params(axis='y', labelsize=21)

# 6. Crear la caja de texto para incluir las estadísticas en el gráfico
texto_estadisticas = (
    f"n: {num_datos}\n"
    f"Media: {media:.3f}\n".replace('.', ',') +
    f"Desviación Típica: {desv_tipica:.3f}".replace('.', ',')
)

# Configurar el estilo de la caja de texto
estilo_caja = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)

# Colocar el texto en la parte superior derecha del gráfico (coordenadas relativas 0.95, 0.95)
ax.text(0.95, 0.95, texto_estadisticas, transform=ax.transAxes, fontsize=19,
        verticalalignment='top', horizontalalignment='right', bbox=estilo_caja)

# Mejorar la estética añadiendo una cuadrícula sutil de fondo
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 7. Mostrar el resultado
plt.tight_layout()
plt.show()


