import numpy as np
import matplotlib.pyplot as plt

# Función original
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

# Método de Bisección
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
    
    # Imprimir iteraciones
    print("Iteración |    a     |    b     |    c     |     f(c)")
    print("----------------------------------------------------------")
    for it in iteraciones:
        print(f"{it[0]:9d} | {it[1]:.6f} | {it[2]:.6f} | {it[3]:.6f} | {it[4]:.6e}")
    
    return c, iteraciones

# Selección de cuatro puntos equidistantes
x0, x1, x2, x3 = 0.0, 0.33, 0.66, 1.0
x_points = np.array([x0, x1, x2, x3])
y_points = f(x_points)

# Construcción del polinomio interpolante
x_vals = np.linspace(x0, x3, 100)
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]

# Encontrar raíz del polinomio interpolante usando bisección
root, iteraciones = bisect(lambda x: lagrange_interpolation(x, x_points, y_points), x0, x3)

# Gráfica de convergencia
iteraciones_x = [it[0] for it in iteraciones]
errores = [abs(it[4]) for it in iteraciones]
plt.figure(figsize=(8, 6))
plt.plot(iteraciones_x, errores, marker='o', linestyle='-', color='purple', label="Error absoluto")
plt.xlabel("Iteración")
plt.ylabel("Error absoluto")
plt.title("Convergencia del Método de Bisección")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.savefig("convergencia_ejercicio3.png")
plt.show()

# Cálculo de errores
error_absoluto = abs(f(root))
error_relativo = error_absoluto / abs(root)
error_cuadratico = error_absoluto**2

# Imprimir resultados
print(f"La raíz aproximada es: {root:.6f}")
print(f"Error absoluto: {error_absoluto:.6e}")
print(f"Error relativo: {error_relativo:.6e}")
print(f"Error cuadrático: {error_cuadratico:.6e}")

# Modificaciónes realizado por: Luis Jorge Fuentes Tec