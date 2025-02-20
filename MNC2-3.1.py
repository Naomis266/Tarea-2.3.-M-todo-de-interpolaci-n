import numpy as np
import matplotlib.pyplot as plt

# Definir la función original
def f(x):
    return np.exp(-x) - x

# Interpolación de Lagrange
def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

# Método de Bisección con impresión de valores
def bisect(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:
        raise ValueError("El intervalo no contiene una raíz")
    
    iteraciones = []
    for i in range(max_iter):
        c = (a + b) / 2
        error = abs(b - a) / 2
        iteraciones.append((i + 1, a, b, c, func(c)))
        
        if abs(func(c)) < tol or error < tol:
            break
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
    
    # Imprimir tabla de iteraciones
    print("Iteración |     a    |     b    |     c    |     f(c)")
    print("----------------------------------------------------------")
    for it in iteraciones:
        print(f"{it[0]:9d} | {it[1]:.6f} | {it[2]:.6f} | {it[3]:.6f} | {it[4]:.6e}")
    
    return c  # Retorna la mejor estimación de la raíz

# Selección de tres puntos equidistantes
x0 = 0.0
x1 = 0.5
x2 = 1.0
x_points = np.array([x0, x1, x2])
y_points = f(x_points)

# Construcción del polinomio interpolante
x_vals = np.linspace(x0, x2, 100)
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]

# Encontrar raíz del polinomio interpolante usando bisección
root = bisect(lambda x: lagrange_interpolation(x, x_points, y_points), x0, x2)

# Cálculo de errores
error_absoluto = abs(f(root))
error_relativo = error_absoluto / abs(root)
error_cuadratico = error_absoluto**2

# Gráfico de la función y la interpolación en ventana separada
plt.figure(figsize=(8, 5))
plt.plot(x_vals, f(x_vals), label="f(x) = e^(-x) - x", color='blue')
plt.plot(x_vals, y_interp, label="Interpolación de Lagrange", linestyle='dashed', color='red')
plt.scatter(x_points, y_points, color='black', zorder=3, label="Puntos de interpolación")
plt.axhline(0, color='gray', linestyle='--')
plt.title("Función Original e Interpolación")
plt.legend()
plt.grid()
plt.show()

# Gráfico de errores en ventana separada
plt.figure(figsize=(8, 5))
iteraciones = list(range(1, len(x_vals) + 1))
errores = [abs(f(x)) for x in x_vals]
plt.plot(iteraciones, errores, label="Error absoluto", color='green')
plt.yscale("log")
plt.xlabel("Iteraciones")
plt.ylabel("Error")
plt.title("Convergencia del error")
plt.legend()
plt.grid()
plt.show()

# Imprimir la raíz y errores
print(f"\nLa raíz aproximada usando interpolación es: {root:.6f}")
print(f"Error absoluto: {error_absoluto:.6e}")
print(f"Error relativo: {error_relativo:.6e}")
print(f"Error cuadrático: {error_cuadratico:.6e}")

# Modificaciónes realizado por: Luis Jorge Fuentes Tec