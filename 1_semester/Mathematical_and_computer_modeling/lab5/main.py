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
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})


def brusselator(t, xy, A, B):
    x, y = xy
    dxdt = A - (B + 1) * x + x ** 2 * y
    dydt = B * x - x ** 2 * y
    return [dxdt, dydt]

experiments = [
    {
        'name': 'Устойчивое равновесие (B < 1 + A²)',
        'A': 1.0,
        'B': 2.0,
        't_end': 30,
        'initial': [1.2, 1.8]
    },
    {
        'name': 'Устойчивый предельный цикл (B > 1 + A²)',
        'A': 1.0,
        'B': 3.0,
        't_end': 50,
        'initial': [1.2, 1.8]
    },
    {
        'name': 'Цикл из другого начального условия',
        'A': 1.0,
        'B': 3.0,
        't_end': 50,
        'initial': [2.5, 0.5]
    }
]

print("=" * 60)
print("МОДЕЛЬ БРЮССЕЛЯТОРА: АВТОКОЛЕБАТЕЛЬНАЯ ХИМИЧЕСКАЯ СИСТЕМА")
print("=" * 60)

cyclic_trajectories = []

for exp in experiments:
    name = exp['name']
    A, B = exp['A'], exp['B']
    t_end = exp['t_end']
    z0 = exp['initial']

    t_eval = np.linspace(0, t_end, 5000)

    sol = solve_ivp(
        fun=lambda t, xy: brusselator(t, xy, A, B),
        t_span=(0, t_end),
        y0=z0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-9,
        atol=1e-12
    )

    x, y = sol.y
    x_eq, y_eq = A, B / A

    print(f"\n• {name}")
    print(f"  Параметры: A={A}, B={B} → 1 + A² = {1 + A ** 2:.1f}")
    print(f"  Равновесие: (x*, y*) = ({x_eq:.2f}, {y_eq:.2f})")

    if "цикл" in name or "Цикл" in name:
        cyclic_trajectories.append((x, y, z0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Брюсселятор\n{name}', fontsize=15, fontweight='bold', y=1.02)

    axes[0].plot(sol.t, x, 'b-', label='x(t)')
    axes[0].plot(sol.t, y, 'r-', label='y(t)')
    axes[0].axhline(y_eq, color='orange', linestyle='--', linewidth=1, label=f'x*, y*')
    axes[0].set_xlabel('Время')
    axes[0].set_ylabel('Концентрация')
    axes[0].set_title('Временная динамика')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(x, y, 'm-', linewidth=1.5)
    axes[1].plot(x_eq, y_eq, 'ko', markersize=8, label='Равновесие')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Фазовый портрет')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    filename = f"brusselator_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.show()

if len(cyclic_trajectories) == 2:
    plt.figure(figsize=(10, 6))
    for i, (x, y, z0) in enumerate(cyclic_trajectories):
        plt.plot(x, y, label=f'Начало: {z0}', linewidth=2)
    plt.plot(1.0, 3.0, 'ko', markersize=8, label='Равновесие (1, 3)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Автоколебания: разные начальные условия → один и тот же предельный цикл')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('brusselator_same_limit_cycle.png', dpi=200, bbox_inches='tight')
    plt.show()

print("\n" + "=" * 60)
print("ВЫВОДЫ:")
print("• При B < 1 + A² система стремится к устойчивому равновесию.")
print("• При B > 1 + A² — бифуркация Хопфа, возникает устойчивый предельный цикл.")
print("• Колебания — автоколебания: амплитуда и форма не зависят от начальных условий.")
print("• Это пример самоорганизации в химической системе.")
print("=" * 60)
