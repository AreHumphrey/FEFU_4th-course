import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})


def lorenz_system(t, xyz, sigma, r, b):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = r * x - y - x * z
    dzdt = x * y - b * z
    return [dxdt, dydt, dzdt]

experiments = [
    {
        'name': 'Устойчивый узел (r < 1)',
        'sigma': 10,
        'r': 0.5,
        'b': 8 / 3,
        't_end': 30,
        'initial': [1.0, 1.0, 1.0]
    },
    {
        'name': 'Устойчивые фокусы (1 < r < r_krit)',
        'sigma': 10,
        'r': 15,
        'b': 8 / 3,
        't_end': 50,
        'initial': [1.0, 1.0, 1.0]
    },
    {
        'name': 'Хаотический режим (классические параметры)',
        'sigma': 10,
        'r': 28,
        'b': 8 / 3,
        't_end': 50,
        'initial': [1.0, 1.0, 1.0]
    }
]

print("=" * 60)
print("МОДЕЛЬ ЛОРЕНЦА: ЧИСЛЕННОЕ ИССЛЕДОВАНИЕ ПОВЕДЕНИЯ")
print("=" * 60)

for exp in experiments:
    name = exp['name']
    sigma, r, b = exp['sigma'], exp['r'], exp['b']
    t_end = exp['t_end']
    z0 = exp['initial']

    print(f"\n• {name}")
    print(f"  Параметры: σ={sigma}, r={r}, b={b:.3f}")

    t_eval = np.linspace(0, t_end, 10000)

    sol = solve_ivp(
        fun=lambda t, xyz: lorenz_system(t, xyz, sigma, r, b),
        t_span=(0, t_end),
        y0=z0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )

    x, y, z = sol.y

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, lw=0.8, color='tab:blue')
    ax.set_title(f'Аттрактор Лоренца\n{name}', fontsize=14, pad=20)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.tight_layout()
    filename = f"lorenz_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.show()

print("\n" + "-" * 60)
print("ЧУВСТВИТЕЛЬНОСТЬ К НАЧАЛЬНЫМ УСЛОВИЯМ (ХАОС)")
print("-" * 60)

sigma, r, b = 10, 28, 8 / 3
t_end = 30
t_eval = np.linspace(0, t_end, 10000)

z0_1 = [1.0, 1.0, 1.0]
z0_2 = [1.0, 1.0, 1.0001] 

sol1 = solve_ivp(lorenz_system, (0, t_end), z0_1, t_eval=t_eval, args=(sigma, r, b), rtol=1e-8, atol=1e-10)
sol2 = solve_ivp(lorenz_system, (0, t_end), z0_2, t_eval=t_eval, args=(sigma, r, b), rtol=1e-8, atol=1e-10)

x1, y1, z1 = sol1.y
x2, y2, z2 = sol2.y

dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

plt.figure(figsize=(10, 4))
plt.plot(sol1.t, dist, color='red', linewidth=1.5)
plt.xlabel('Время')
plt.ylabel('Евклидово расстояние между траекториями')
plt.title('Экспоненциальное расхождение траекторий при хаосе\n(начальные условия отличаются на 0.0001 в z)')
plt.grid(True, alpha=0.4)
plt.yscale('log')  
plt.tight_layout()
plt.savefig('lorenz_sensitivity.png', dpi=200, bbox_inches='tight')
plt.show()

print(f"• Параметры: σ={sigma}, r={r}, b={b:.3f}")
print("• Начальные условия:")
print(f"    Траектория 1: {z0_1}")
print(f"    Траектория 2: {z0_2}")
print("• Наблюдается экспоненциальное расхождение → признак хаоса.")
print("• Это демонстрирует 'эффект бабочки': малое изменение → большие последствия.")

print("\n" + "=" * 60)
print("ВЫВОДЫ:")
print("• При r < 1 — единственная устойчивая точка (0,0,0).")
print("• При 1 < r < r_krit — две устойчивые точки, спиралевидное затухание.")
print("• При r = 28 — хаотический аттрактор, чувствительность к начальным условиям.")
print("=" * 60)
