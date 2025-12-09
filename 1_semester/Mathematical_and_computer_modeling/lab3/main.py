import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({
    "font.family": "DejaVU Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

def competition_model(t, z, r1, r2, K1, K2, alpha, beta):
    x, y = z
    dxdt = r1 * x * (1 - (x + alpha * y) / K1)
    dydt = r2 * y * (1 - (y + beta * x) / K2)
    return [dxdt, dydt]

# Параметры модели
r1 = 1.0      # скорость роста вида X
r2 = 0.8      # скорость роста вида Y
K1 = 50       # ёмкость среды для X
K2 = 40       # ёмкость среды для Y
alpha = 0.6   # влияние Y на X (на сколько особей Y "равны" одной особи X)
beta  = 1.2   # влияние X на Y

# Начальные условия
x0 = 10
y0 = 8
z0 = [x0, y0]

t_start = 0
t_end = 50
t_eval = np.linspace(t_start, t_end, 1000)

sol = solve_ivp(
    fun=lambda t, z: competition_model(t, z, r1, r2, K1, K2, alpha, beta),
    t_span=[t_start, t_end],
    y0=z0,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-8,
    atol=1e-10
)

x, y = sol.y


A = np.array([[1, alpha],
              [beta, 1]])
b = np.array([K1, K2])
try:
    x_eq, y_eq = np.linalg.solve(A, b)
    equilibrium_exists = (x_eq >= 0) and (y_eq >= 0)
except np.linalg.LinAlgError:
    x_eq, y_eq = np.nan, np.nan
    equilibrium_exists = False

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Модель конкуренции: "Вид X vs Вид Y"', fontsize=16, fontweight='bold', y=1.02)

axes[0].plot(sol.t, x, 'b-', linewidth=2.2, label='Вид X')
axes[0].plot(sol.t, y, 'r-', linewidth=2.2, label='Вид Y')
axes[0].set_xlabel('Время, условные единицы')
axes[0].set_ylabel('Численность')
axes[0].set_title('Динамика популяций во времени')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(x, y, 'm-', linewidth=2, label='Траектория')
if equilibrium_exists:
    axes[1].plot(x_eq, y_eq, 'ko', markersize=10, label='Равновесие (сосуществование)')

axes[1].plot(K1, 0, 'bo', markersize=8, label='Только X ($K_1, 0$)')
axes[1].plot(0, K2, 'ro', markersize=8, label='Только Y ($0, K_2$)')

axes[1].set_xlabel('Вид X')
axes[1].set_ylabel('Вид Y')
axes[1].set_title('Фазовый портрет')
axes[1].legend()
axes[1].grid(True)
axes[1].set_xlim(0, max(K1, max(x)) * 1.1)
axes[1].set_ylim(0, max(K2, max(y)) * 1.1)

info_text = (
    f"Параметры:\n"
    f"r₁={r1}, r₂={r2}\n"
    f"K₁={K1}, K₂={K2}\n"
    f"α={alpha}, β={beta}\n"
    f"Начало: x₀={x0}, y₀={y0}\n"
)
if equilibrium_exists:
    info_text += f"Равновесие: ({x_eq:.1f}, {y_eq:.1f})"
else:
    info_text += "Сосуществование невозможно"

axes[0].text(0.02, 0.98, info_text, transform=axes[0].transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

plt.tight_layout()
plt.savefig('competition_model_simulation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ МОДЕЛИ КОНКУРЕНЦИИ")
print("="*60)
print(f"• Начальные условия: x₀ = {x0}, y₀ = {y0}")
print(f"• Параметры: r1={r1}, r2={r2}, K1={K1}, K2={K2}, α={alpha}, β={beta}")

if equilibrium_exists:
    print(f"• Равновесие сосуществования: x* = {x_eq:.2f}, y* = {y_eq:.2f}")

    if x[-1] > K1 * 0.95 and y[-1] < 0.05 * K2:
        outcome = "Вид X вытеснил вид Y"
    elif y[-1] > K2 * 0.95 and x[-1] < 0.05 * K1:
        outcome = "Вид Y вытеснил вид X"
    elif abs(x[-1] - x_eq) < 1 and abs(y[-1] - y_eq) < 1:
        outcome = "Устойчивое сосуществование"
    else:
        outcome = "Переходный режим (наблюдается приближение к равновесию)"
else:
    outcome = "Сосуществование невозможно по параметрам"
    print("• Равновесие сосуществования — отрицательное или не существует")

print(f"\n• Вывод: {outcome}.")
print("  В модели конкуренции исход зависит от соотношения ёмкостей среды")
print("  и взаимного влияния видов: возможны вытеснение или совместное существование.")
