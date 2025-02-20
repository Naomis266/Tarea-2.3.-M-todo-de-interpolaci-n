import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

# Definir la función f(x) = sin(x) - x/2
def f(x):
    return np.sin(x) - x / 2

# Definir los puntos equidistantes en el intervalo [0,2]
x_points = np.linspace(0, 2, 3)  # Tres puntos equidistantes
y_points = f(x_points)  # Evaluar la función en estos puntos

# Construir el polinomio interpolante de Lagrange
polinomio = lagrange(x_points, y_points)

# Función del polinomio interpolante
def P(x):
    return polinomio(x)

# Método de Bisección para encontrar la raíz de P(x) = 0
def bisect(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:
        raise ValueError("El intervalo no contiene una raíz")
    
    iteraciones = []  # Lista para almacenar las iteraciones
    
    print("Iteración |    a     |    b     |    c     |     f(c)")
    print("----------------------------------------------------------")
    
    for i in range(max_iter):
        c = (a + b) / 2
        error = abs(b - a) / 2
        iteraciones.append((i + 1, a, b, c, func(c)))
        
        print(f"{i+1:9d} | {a:.6f} | {b:.6f} | {c:.6f} | {func(c):.6e}")
        
        if abs(func(c)) < tol or error < tol:
            break
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
    
    return c, iteraciones  # Retornar la raíz y la lista de iteraciones

# Aplicar el método de bisección sobre el polinomio de interpolación
x0, x2 = x_points[0], x_points[-1]  # Extremos del intervalo
root, iteraciones = bisect(P, x0, x2)

# Graficar el polinomio interpolante
x_vals = np.linspace(0, 2, 100)
y_vals = P(x_vals)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label="Polinomio de Lagrange", color="b")
plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.scatter(x_points, y_points, color="red", label="Puntos de interpolación")
plt.scatter(root, P(root), color="green", marker="x", s=100, label="Raíz encontrada")
plt.xlabel("x")
plt.ylabel("P(x)")
plt.title("Interpolación de Lagrange y raíz encontrada")
plt.legend()
plt.grid()
plt.show()

# Gráfica de convergencia y errores
iteraciones_x = [it[0] for it in iteraciones]  # Número de iteración
errores = [abs(it[4]) for it in iteraciones]  # Valor absoluto de f(c)

plt.figure(figsize=(8, 5))
plt.plot(iteraciones_x, errores, marker='o', linestyle='-', color='r', label="Error absoluto")
plt.xlabel("Iteración")
plt.ylabel("Error |f(c)|")
plt.title("Convergencia del Método de Bisección")
plt.yscale("log")  # Escala logarítmica para visualizar mejor la convergencia
plt.grid()
plt.legend()
plt.show()

# Modificaciónes realizado por: Luis Jorge Fuentes Tec