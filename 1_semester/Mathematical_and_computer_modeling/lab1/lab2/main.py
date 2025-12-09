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

def lotka_volterra(t, z, alpha, beta, gamma, delta):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

alpha = 1.0
beta  = 0.1
gamma = 1.5
delta = 0.075

x0 = 40
y0 = 9
z0 = [x0, y0]

t_start = 0
t_end = 30
t_eval = np.linspace(t_start, t_end, 1000)

sol = solve_ivp(
    fun=lambda t, z: lotka_volterra(t, z, alpha, beta, gamma, delta),
    t_span=[t_start, t_end],
    y0=z0,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-8,
    atol=1e-10
)

x, y = sol.y

x_eq = gamma / delta
y_eq = alpha / beta

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Модель Лотки–Вольтерры: "Зайцы – Рыси"', fontsize=16, fontweight='bold', y=1.02)

axes[0].plot(sol.t, x, 'b-', linewidth=2.2, label='Жертвы (зайцы)')
axes[0].plot(sol.t, y, 'r-', linewidth=2.2, label='Хищники (рыси)')
axes[0].set_xlabel('Время, условные единицы')
axes[0].set_ylabel('Численность')
axes[0].set_title('Динамика популяций во времени')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(x, y, 'm-', linewidth=2, label='Траектория')
axes[1].plot(x_eq, y_eq, 'ko', markersize=10, label='Равновесие')
axes[1].set_xlabel('Жертвы, x')
axes[1].set_ylabel('Хищники, y')
axes[1].set_title('Фазовый портрет')
axes[1].legend()
axes[1].grid(True)
axes[1].set_aspect('equal', adjustable='box')

info_text = (
    f"Параметры:\n"
    f"α = {alpha}, β = {beta}\n"
    f"γ = {gamma}, δ = {delta}\n"
    f"Начало: x₀ = {x0}, y₀ = {y0}\n"
    f"Равновесие: ({x_eq:.1f}, {y_eq:.1f})"
)
axes[0].text(0.02, 0.98, info_text, transform=axes[0].transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

plt.tight_layout()
plt.savefig('lotka_volterra_simulation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ МОДЕЛИ ЛОТКИ–ВОЛЬТЕРРЫ")
print("="*60)
print(f"• Начальные условия: x₀ = {x0}, y₀ = {y0}")
print(f"• Параметры: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
print(f"• Равновесная точка: x* = {x_eq:.2f}, y* = {y_eq:.2f}")
print(f"• Период колебаний (примерно): {t_end / (len(np.where(np.diff(np.sign(np.gradient(x))) != 0)[0]) // 2):.1f} ед. времени")
print("\n• Вывод: система демонстрирует устойчивые незатухающие колебания.")
print("  Это означает циклическое взаимодействие: рост жертв → рост хищников → падение жертв → падение хищников → повтор.")
