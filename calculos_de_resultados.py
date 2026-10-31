import numpy as np
import scipy.stats as stats

# ==========================================
# 1. DATOS DE ENTRADA (Promedios por sujeto)
# ==========================================

# Grupo 1 (Condición ciega) - N = 30
g1_tradicional = np.array([
    3.9, 3.4, 4.2, 4.5, 4.5, 3.2, 3.6, 2.5, 3.5, 3.8, 
    4.8, 4.9, 4.0, 4.5, 5.0, 3.3, 3.4, 4.9, 3.5, 4.8, 
    3.1, 4.6, 3.7, 3.5, 3.5, 4.7, 3.6, 5.0, 3.9, 4.7
])

g1_ia = np.array([
    2.9, 3.8, 3.9, 4.5, 4.0, 1.6, 3.5, 1.6, 4.8, 2.5, 
    4.6, 4.5, 4.0, 3.8, 5.0, 3.7, 4.3, 4.5, 2.9, 5.0, 
    3.3, 4.4, 4.3, 3.1, 1.9, 1.9, 1.9, 4.6, 4.1, 4.8
])

# Grupo 2 (Condición informada) - N = 30
g2_tradicional = np.array([
    2.8, 3.8, 3.7, 4.8, 3.6, 4.3, 3.3, 4.5, 4.2, 4.8, 
    4.4, 3.3, 4.7, 3.5, 4.1, 4.3, 3.2, 3.3, 3.8, 4.9, 
    4.6, 4.6, 5.0, 3.7, 3.5, 5.0, 4.8, 3.4, 4.8, 3.6
])

g2_ia = np.array([
    3.2, 1.8, 2.0, 3.2, 1.3, 3.5, 1.6, 3.0, 3.2, 2.7, 
    4.5, 2.3, 3.5, 3.5, 3.7, 3.5, 2.5, 2.0, 1.8, 4.5, 
    3.3, 3.5, 5.0, 2.5, 2.8, 3.1, 3.5, 2.1, 2.2, 1.6
])


# ==========================================
# 2. CÁLCULOS DE PRUEBAS T EMPAREJADAS
# ==========================================
print("--- PRUEBAS T EMPAREJADAS POR GRUPO ---")

# Medias globales
print(f"Grupo 1 - Tradicional (Media): {np.mean(g1_tradicional):.2f} | IA (Media): {np.mean(g1_ia):.2f}")
print(f"Grupo 2 - Tradicional (Media): {np.mean(g2_tradicional):.2f} | IA (Media): {np.mean(g2_ia):.2f}\n")

# Prueba t emparejada Grupo 1
t_g1, p_g1 = stats.ttest_rel(g1_tradicional, g1_ia)
print(f"Grupo 1 (Ciego): t(29) = {t_g1:.2f}, p = {p_g1:.4f}")

# Prueba t emparejada Grupo 2
t_g2, p_g2 = stats.ttest_rel(g2_tradicional, g2_ia)
print(f"Grupo 2 (Informado): t(29) = {t_g2:.2f}, p = {p_g2:.4e}\n")


# ==========================================
# 3. COMPARACIÓN DE CAÍDAS (Muestras Independientes)
# ==========================================
print("--- COMPARACIÓN DE CAÍDAS (G1 vs G2) ---")

# Cálculo de la brecha o caída por sujeto (Tradicional - IA)
caida_g1 = g1_tradicional - g1_ia
caida_g2 = g2_tradicional - g2_ia

print(f"Caída media Grupo 1: {np.mean(caida_g1):.2f}")
print(f"Caída media Grupo 2: {np.mean(caida_g2):.2f}")

# Prueba t de muestras independientes para las caídas
t_ind, p_ind = stats.ttest_ind(caida_g1, caida_g2)
print(f"Prueba t independiente: t(58) = {t_ind:.2f}, p = {p_ind:.4f}")