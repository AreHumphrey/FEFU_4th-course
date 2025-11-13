import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import math


class DescartesFolium:
    def __init__(self, a=1.0):
        self.a = a

    def parametric_equation(self, t):
        t = np.asarray(t)
        denominator = 1 + t ** 3
        mask = np.abs(denominator) > 1e-10

        x = np.full_like(t, np.nan)
        y = np.full_like(t, np.nan)

        x[mask] = 3 * self.a * t[mask] / denominator[mask]
        y[mask] = 3 * self.a * t[mask] ** 2 / denominator[mask]

        return x, y


class RationalSpline:
    def __init__(self, points, weights=None, degree=3):
        self.points = np.array(points)
        self.n_points = len(points)
        self.degree = degree

        if weights is None:
            self.weights = np.ones(self.n_points)
        else:
            self.weights = np.array(weights)

    def chord_length_parameterization(self):
        chord_lengths = np.zeros(self.n_points)
        for i in range(1, self.n_points):
            dx = self.points[i, 0] - self.points[i - 1, 0]
            dy = self.points[i, 1] - self.points[i - 1, 1]
            chord_lengths[i] = chord_lengths[i - 1] + math.sqrt(dx ** 2 + dy ** 2)

        if chord_lengths[-1] > 0:
            chord_lengths /= chord_lengths[-1]

        return chord_lengths

    def fit_spline(self, s=0):

        t = self.chord_length_parameterization()

        sort_idx = np.argsort(t)
        t_sorted = t[sort_idx]
        points_sorted = self.points[sort_idx]
        weights_sorted = self.weights[sort_idx]

        self.spline_x = interpolate.UnivariateSpline(t_sorted, points_sorted[:, 0],
                                                     w=weights_sorted, s=s, k=3)
        self.spline_y = interpolate.UnivariateSpline(t_sorted, points_sorted[:, 1],
                                                     w=weights_sorted, s=s, k=3)

        return self.spline_x, self.spline_y

    def evaluate(self, t_eval):
        if not hasattr(self, 'spline_x'):
            self.fit_spline()

        x_eval = self.spline_x(t_eval)
        y_eval = self.spline_y(t_eval)

        return x_eval, y_eval


def generate_descartes_points(a=1.0, n_points=300):
    descartes = DescartesFolium(a)

    t_left = np.linspace(-20, -1.01, n_points // 2) 
    t_right = np.linspace(-0.99, 20, n_points // 2) 

    t_all = np.concatenate([t_left, t_right])
    x_all, y_all = descartes.parametric_equation(t_all)

    valid_mask = ~np.isnan(x_all) & ~np.isnan(y_all) & (np.abs(x_all) < 10) & (np.abs(y_all) < 10)
    x_valid = x_all[valid_mask]
    y_valid = y_all[valid_mask]
    t_valid = t_all[valid_mask]

    sort_idx = np.argsort(t_valid)
    x_sorted = x_valid[sort_idx]
    y_sorted = y_valid[sort_idx]

    points = np.column_stack([x_sorted, y_sorted])

    return points, descartes


def calculate_proper_error(descartes, points, spline, n_eval=1000):
    
    t_dense = np.linspace(-10, 10, n_eval * 2)
    x_exact_dense, y_exact_dense = descartes.parametric_equation(t_dense)

    mask = ~np.isnan(x_exact_dense) & ~np.isnan(y_exact_dense) & \
           (np.abs(x_exact_dense) < 10) & (np.abs(y_exact_dense) < 10)
    x_exact = x_exact_dense[mask]
    y_exact = y_exact_dense[mask]

    t_eval = np.linspace(0, 1, n_eval)
    x_spline, y_spline = spline.evaluate(t_eval)

    spline_mask = ~np.isnan(x_spline) & ~np.isnan(y_spline) & \
                  (np.abs(x_spline) < 10) & (np.abs(y_spline) < 10)
    x_spline_clean = x_spline[spline_mask]
    y_spline_clean = y_spline[spline_mask]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(x_exact, y_exact, 'k-', alpha=0.7, linewidth=2, label='Точная кривая')
    plt.plot(points[:, 0], points[:, 1], 'ro', markersize=3, alpha=0.6, label='Точки аппроксимации')
    plt.plot(x_spline_clean, y_spline_clean, 'b-', alpha=0.7, linewidth=1, label='Сплайн')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Сравнение кривых')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    from scipy.spatial import cKDTree

    exact_points = np.column_stack([x_exact, y_exact])
    spline_points = np.column_stack([x_spline_clean, y_spline_clean])

    tree = cKDTree(exact_points)
    distances, indices = tree.query(spline_points)

    plt.subplot(1, 3, 2)
    plt.hist(distances, bins=50, alpha=0.7, color='red')
    plt.xlabel('Расстояние до точной кривой')
    plt.ylabel('Частота')
    plt.title('Распределение ошибок')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.scatter(x_spline_clean, y_spline_clean, c=distances, cmap='hot', s=10)
    plt.colorbar(label='Ошибка')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Пространственное распределение ошибок')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

    rmse = np.sqrt(np.mean(distances ** 2))
    max_error = np.max(distances)
    mean_error = np.mean(distances)

    print(f"Корректная среднеквадратичная ошибка: {rmse:.6f}")
    print(f"Корректная максимальная ошибка: {max_error:.6f}")
    print(f"Корректная средняя ошибка: {mean_error:.6f}")

    return rmse, max_error, mean_error


def main_corrected():
    print("корректные эксперименты с параметрическим рациональным сплайном")
    print("=" * 60)

    points, descartes = generate_descartes_points(a=1.0, n_points=200)

    print(f"Параметр декартова листа: a = {descartes.a}")
    print(f"Количество точек аппроксимации: {len(points)}")

    smoothing_params = [0, 0.0001, 0.001, 0.01]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, s in enumerate(smoothing_params):
        ax = axes[i]

        spline = RationalSpline(points)
        spline.fit_spline(s=s)

        t_eval = np.linspace(0, 1, 1000)
        x_spline, y_spline = spline.evaluate(t_eval)

        t_exact = np.linspace(-10, 10, 1000)
        x_exact, y_exact = descartes.parametric_equation(t_exact)
        mask = ~np.isnan(x_exact) & ~np.isnan(y_exact) & (np.abs(x_exact) < 10) & (np.abs(y_exact) < 10)

        ax.plot(x_exact[mask], y_exact[mask], 'k-', alpha=0.5, linewidth=1, label='Точная')
        ax.plot(points[:, 0], points[:, 1], 'ro', markersize=2, alpha=0.6, label='Точки')
        ax.plot(x_spline, y_spline, 'b-', linewidth=1.5, label=f'Сплайн (s={s})')

        ax.set_title(f'Параметр сглаживания s = {s}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("Анализ ошибочки:")
    print("=" * 60)

    spline_optimal = RationalSpline(points)
    spline_optimal.fit_spline(s=0.001)

    rmse, max_err, mean_err = calculate_proper_error(descartes, points, spline_optimal)

    print(f"\nИтоговые результаты:")
    print(f"- RMSE: {rmse:.6f} (должна быть < 0.1 для хорошей аппроксимации)")
    print(f"- Максимальная ошибка: {max_err:.6f}")
    print(f"- Средняя ошибка: {mean_err:.6f}")

    if rmse < 0.1:
        print("Аппроксимация нормальная")
    else:
        print("Аппроксимация плохо робит")


if __name__ == "__main__":
    main_corrected()
