import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def cubic_spline_moments(x, y, y_prime_left=None, y_prime_right=None, epsilon=0.0):
    n = len(x) - 1
    h = np.diff(x)
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)

    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    if y_prime_left is not None and y_prime_right is not None:
        A[0, 0] = 2 * h[0]
        A[0, 1] = h[0]
        b[0] = 6 * ((y[1] - y[0]) / h[0] - (y_prime_left + epsilon))
        A[n, n - 1] = h[n - 1]
        A[n, n] = 2 * h[n - 1]
        b[n] = 6 * ((y_prime_right + epsilon) - (y[n] - y[n - 1]) / h[n - 1])
    else:

        A[0, 0] = 1
        A[n, n] = 1
        b[0] = 0
        b[n] = 0

    M = np.linalg.solve(A, b)
    return M


def evaluate_spline(x, y, M, x_eval):
    n = len(x) - 1
    h = np.diff(x)
    i = np.searchsorted(x, x_eval, side='right') - 1
    if i < 0:
        i = 0
    if i >= n:
        i = n - 1

    xi = x[i]
    xi1 = x[i + 1]
    hi = h[i]
    yi = y[i]
    yi1 = y[i + 1]
    Mi = M[i]
    Mi1 = M[i + 1]

    a = yi
    b = (yi1 - yi) / hi - hi * (Mi1 + 2 * Mi) / 6
    c = Mi / 2
    d = (Mi1 - Mi) / (6 * hi)

    t = x_eval - xi
    s = a + b * t + c * t ** 2 + d * t ** 3
    return s


def func1(x, a):
    return np.sin(a * x)


def func2(x, a):
    return a / (1 + 9 * x ** 2)


def func1_prime(x, a):
    return a * np.cos(a * x)


def func2_prime(x, a):
    denom = (1 + 9 * x ** 2) ** 2
    return -18 * a * x / denom


a_values = [1, 5, 10]
grid_sizes = [5, 10, 20, 40]
functions = [
    {"name": "sin(ax)", "func": func1, "prime": func1_prime, "x_range": (0, np.pi)},
    {"name": "a/(1+9x^2)", "func": func2, "prime": func2_prime, "x_range": (-1, 1)}
]

results_table = []

for func_info in functions:
    func_name = func_info["name"]
    f = func_info["func"]
    f_prime = func_info["prime"]
    x_min, x_max = func_info["x_range"]

    for a in a_values:
        print(f"\n=== Функция: {func_name}, a = {a} ===")

        for N in grid_sizes:
            print(f"  Сетка: {N} узлов")

            x_nodes = np.linspace(x_min, x_max, N)
            y_nodes = f(x_nodes, a)

            y_prime_left_exact = f_prime(x_min, a)
            y_prime_right_exact = f_prime(x_max, a)

            M_exact = cubic_spline_moments(x_nodes, y_nodes, y_prime_left_exact, y_prime_right_exact, epsilon=0.0)
            error_percent = 0.01
            y_prime_left_error = y_prime_left_exact * (1 + error_percent)
            y_prime_right_error = y_prime_right_exact * (1 + error_percent)
            M_error = cubic_spline_moments(x_nodes, y_nodes, y_prime_left_error, y_prime_right_error, epsilon=0.0)

            test_points = []
            for i in range(N - 1):
                mid = (x_nodes[i] + x_nodes[i + 1]) / 2
                test_points.append(mid)
                if len(test_points) >= 5:
                    break
            test_points = np.array(test_points[:5])

            true_vals_exact = f(test_points, a)
            approx_vals_exact = np.array([evaluate_spline(x_nodes, y_nodes, M_exact, xp) for xp in test_points])
            abs_error_exact = np.abs(true_vals_exact - approx_vals_exact)

            true_vals_error = f(test_points, a)
            approx_vals_error = np.array([evaluate_spline(x_nodes, y_nodes, M_error, xp) for xp in test_points])
            abs_error_error = np.abs(true_vals_error - approx_vals_error)

            for j, xp in enumerate(test_points):
                results_table.append({
                    "Функция": func_name,
                    "a": a,
                    "Сетка N": N,
                    "Тип граничных условий": "Точные",
                    "Значение x": xp,
                    "Точное значение": true_vals_exact[j],
                    "Приближенное значение": approx_vals_exact[j],
                    "Абсолютная погрешность": abs_error_exact[j]
                })

            for j, xp in enumerate(test_points):
                results_table.append({
                    "Функция": func_name,
                    "a": a,
                    "Сетка N": N,
                    "Тип граничных условий": "С ошибкой",
                    "Значение x": xp,
                    "Точное значение": true_vals_error[j],
                    "Приближенное значение": approx_vals_error[j],
                    "Абсолютная погрешность": abs_error_error[j]
                })

            print("  Контрольные точки (точные граничные условия):")
            for j, xp in enumerate(test_points):
                print(
                    f"    x={xp:.4f}: точное={true_vals_exact[j]:.6f}, аппрокс={approx_vals_exact[j]:.6f}, погрешность={abs_error_exact[j]:.2e}")

            print("  Контрольные точки (граничные условия с ошибкой):")
            for j, xp in enumerate(test_points):
                print(
                    f"    x={xp:.4f}: точное={true_vals_error[j]:.6f}, аппрокс={approx_vals_error[j]:.6f}, погрешность={abs_error_error[j]:.2e}")

print("\n" + "=" * 120)
print("ОБЩАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ:")
print("=" * 120)
print(
    f"{'Функция':<12} {'a':<3} {'Сетка N':<7} {'Тип ГУ':<12} {'x':<8} {'Точное':<12} {'Аппрокс':<12} {'Погрешность':<12}")
print("-" * 120)
for row in results_table:
    print(
        f"{row['Функция']:<12} {row['a']:<3} {row['Сетка N']:<7} {row['Тип граничных условий']:<12} {row['Значение x']:<8.4f} "
        f"{row['Точное значение']:<12.6f} {row['Приближенное значение']:<12.6f} {row['Абсолютная погрешность']:<12.2e}")

df = pd.DataFrame(results_table)
df.to_csv("spline_results.csv", index=False, float_format="%.8f")


agg_data = []
for (func, a, N, bc_type), group in df.groupby(['Функция', 'a', 'Сетка N', 'Тип граничных условий']):
    max_err = group['Абсолютная погрешность'].max()
    rms_err = np.sqrt((group['Абсолютная погрешность'] ** 2).mean())
    L = np.pi if 'sin' in func else 2.0
    agg_data.append({
        'Функция': func,
        'a': a,
        'Сетка N': N,
        'Тип ГУ': bc_type,
        'max_err': max_err,
        'rms_err': rms_err,
        'h': L / (N - 1)
    })

df_agg = pd.DataFrame(agg_data)
df_agg.to_csv("spline_summary_typeI.csv", index=False, float_format="%.6e")



print("\n" + "=" * 80)
print("СВОДНАЯ ТАБЛИЦА макс. погрешность  граничные условия типа I")
print("=" * 80)

example_rows = df_agg[
    (df_agg['Функция'] == 'sin(ax)') & (df_agg['a'] == 5)
    ].sort_values('Сетка N')

print(f"{'N':<4} {'h':<8} {'Точные ГУ':<12} {'ГУ с ошибкой':<15}")
print("-" * 50)
for N in sorted(example_rows['Сетка N'].unique()):
    row_exact = example_rows[(example_rows['Сетка N'] == N) & (example_rows['Тип ГУ'] == 'Точные')]
    row_error = example_rows[(example_rows['Сетка N'] == N) & (example_rows['Тип ГУ'] == 'С ошибкой')]

    h_val = L / (N - 1) if 'sin' in 'sin(ax)' else 2.0 / (N - 1)
    err_exact = f"{row_exact['max_err'].iloc[0]:.2e}" if not row_exact.empty else "–"
    err_error = f"{row_error['max_err'].iloc[0]:.2e}" if not row_error.empty else "–"

    print(f"{N:<4} {h_val:<8.3f} {err_exact:<12} {err_error:<15}")

plt.figure(figsize=(12, 5))
for idx, func_info in enumerate(functions):
    func_name = func_info["name"]
    plt.subplot(1, 2, idx + 1)
    for a in [1, 5, 10]:
        subset = df_agg[(df_agg['Функция'] == func_name) & (df_agg['a'] == a)]
        exact = subset[subset['Тип ГУ'] == 'Точные'].sort_values('h')
        error = subset[subset['Тип ГУ'] == 'С ошибкой'].sort_values('h')
        if not exact.empty:
            plt.loglog(exact['h'], exact['max_err'], 'o-', label=f'Точные, a={a}')
        if not error.empty:
            plt.loglog(error['h'], error['max_err'], 'x--', label=f'ГУ +1% ош., a={a}')
    plt.xlabel('Шаг сетки h')
    plt.ylabel('Макс. погрешность')
    plt.title(f'Сходимость: {func_name} (только тип I)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
plt.tight_layout()
plt.savefig("convergence_typeI.png", dpi=150)
plt.show()

func_name = "sin(ax)"
a = 5
N = 10
x_min, x_max = (0, np.pi)
x_nodes = np.linspace(x_min, x_max, N)
y_nodes = func1(x_nodes, a)
ypl = func1_prime(x_min, a)
ypr = func1_prime(x_max, a)
M_exact = cubic_spline_moments(x_nodes, y_nodes, ypl, ypr)
ypl_err = ypl * 1.01
ypr_err = ypr * 1.01
M_error = cubic_spline_moments(x_nodes, y_nodes, ypl_err, ypr_err)

x_plot = np.linspace(x_min, x_max, 500)
y_true = func1(x_plot, a)
y_spline_exact = np.array([evaluate_spline(x_nodes, y_nodes, M_exact, xp) for xp in x_plot])
y_spline_error = np.array([evaluate_spline(x_nodes, y_nodes, M_error, xp) for xp in x_plot])

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_true, 'k-', linewidth=2.5, label='Точная функция')
plt.plot(x_plot, y_spline_exact, '--', linewidth=2, label='Сплайн (точные ГУ)')
plt.plot(x_plot, y_spline_error, '-.', linewidth=2, label='Сплайн (ГУ +1% ошибка)')
plt.scatter(x_nodes, y_nodes, color='red', zorder=5, label='Узлы интерполяции')
plt.title(f'Сравнение: {func_name}, a={a}, N={N} (только тип I)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig("comparison_typeI.png", dpi=150)
plt.show()

N_fixed = 20
plt.figure(figsize=(10, 5))
for idx, func_info in enumerate(functions):
    func_name = func_info["name"]
    plt.subplot(1, 2, idx + 1)
    a_vals = []
    err_exact = []
    for a in a_values:
        row_e = df_agg[
            (df_agg['Функция'] == func_name) &
            (df_agg['a'] == a) &
            (df_agg['Сетка N'] == N_fixed) &
            (df_agg['Тип ГУ'] == 'Точные')
            ]
        if not row_e.empty:
            a_vals.append(a)
            err_exact.append(row_e['max_err'].iloc[0])
    plt.plot(a_vals, err_exact, 'o-', label='Точные ГУ')
    plt.xlabel('Параметр a')
    plt.ylabel('Макс. погрешность')
    plt.title(f'Влияние a (N={N_fixed}, тип I)')
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.savefig("error_vs_a_typeI.png", dpi=150)
plt.show()
