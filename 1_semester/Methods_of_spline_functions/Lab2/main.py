import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import quad


def f(x):
    return 1.0 / np.sqrt((1 - x**2) * (1 - 0.25 * x**2))

def exact_integral():
    result, _ = quad(f, 0, 0.5)
    return result

def df(x):
    denom = (1 - x**2) * (1 - 0.25 * x**2)
    num_prime = (-2*x)*(1 - 0.25*x**2) + (1 - x**2)*(-0.5*x)
    return -0.5 * num_prime / (denom**(3/2))

def cubic_spline_integration(a, b, n, f, df_a=None, df_b=None):
    x_nodes = np.linspace(a, b, n+1)
    y_nodes = f(x_nodes)

    if df_a is None:
        h = x_nodes[1] - x_nodes[0]
        df_a = (-3*y_nodes[0] + 4*y_nodes[1] - y_nodes[2]) / (2*h)
    if df_b is None:
        h = x_nodes[-1] - x_nodes[-2]
        df_b = (3*y_nodes[-1] - 4*y_nodes[-2] + y_nodes[-3]) / (2*h)

    spline = CubicSpline(x_nodes, y_nodes, bc_type=((1, df_a), (1, df_b)))

    y_at_nodes = spline(x_nodes)
    if not np.allclose(y_at_nodes, y_nodes, atol=1e-10):
        print("⚠️ Сплайн не проходит через узлы!")
        print("Разница:", np.max(np.abs(y_at_nodes - y_nodes)))

    integral = spline.integrate(a, b)

    return integral, spline, x_nodes

def experiment_with_epsilon(a, b, epsilons, f, df_a=None, df_b=None):
    results = []
    exact_val = exact_integral()
    print(f"Точное значение интеграла: {exact_val:.10f}")
    print("-" * 60)
    print(f"{'Шаг ε':<10} {'Интеграл':<15} {'Ошибка':<15} {'Относ. ошибка (%)'}")
    print("-" * 60)

    for eps in epsilons:
        n = int((b - a) / eps)
        if n < 2:
            n = 2
        integral, _, _ = cubic_spline_integration(a, b, n, f, df_a, df_b)
        error = abs(integral - exact_val)
        rel_error_percent = (error / exact_val) * 100 if exact_val != 0 else 0
        results.append((eps, integral, error, rel_error_percent))
        print(f"{eps:<10.6f} {integral:<15.10f} {error:<15.10f} {rel_error_percent:<15.8f}")

    return results

if __name__ == "__main__":
    a, b = 0.0, 0.5

    df_a = df(a)
    df_b = df(b)

    print("Граничные условия 1-го рода:")
    print(f"f'(0) = {df_a:.10f}")
    print(f"f'(0.5) = {df_b:.10f}")
    print()

    epsilons = [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001]

    results = experiment_with_epsilon(a, b, epsilons, f, df_a, df_b)

    plt.figure(figsize=(10, 6))
    eps_vals = [r[0] for r in results]
    errors = [r[2] for r in results]
    plt.loglog(eps_vals, errors, 'o-', label='Ошибка')
    plt.xlabel('Шаг сетки ε')
    plt.ylabel('Абсолютная ошибка')
    plt.title('Сходимость метода кубического сплайна')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

    n_plot = 100
    x_fine = np.linspace(a, b, 1000)
    y_fine = f(x_fine)

    integral_approx, spline, x_nodes = cubic_spline_integration(a, b, n_plot, f, df_a, df_b)
    y_spline = spline(x_fine)

    plt.figure(figsize=(10, 6))
    plt.plot(x_fine, y_fine, 'b-', label='f(x)', linewidth=2)
    plt.plot(x_fine, y_spline, 'r--', label='Кубический сплайн', linewidth=2)
    plt.scatter(x_nodes, f(x_nodes), color='green', zorder=5, label='Узлы')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Функция и её кубический сплайн')
    plt.legend()
    plt.grid(True)
    plt.show()
