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

def malthus_model(t, N, r):
    return r * N

def logistic_model(t, N, r, K):
    return r * N * (1 - N / K)

r = 0.02
K = 20e9
N0 = 1e9
t_start = 0
t_end = 200
num_points = 500

t_eval = np.linspace(t_start, t_end, num_points)

sol_malthus = solve_ivp(
    fun=lambda t, N: malthus_model(t, N, r),
    t_span=[t_start, t_end],
    y0=[N0],
    t_eval=t_eval,
    method='RK45',
    rtol=1e-8,
    atol=1e-10
)

sol_logistic = solve_ivp(
    fun=lambda t, N: logistic_model(t, N, r, K),
    t_span=[t_start, t_end],
    y0=[N0],
    t_eval=t_eval,
    method='RK45',
    rtol=1e-8,
    atol=1e-10
)

N_analytic_malthus = N0 * np.exp(r * sol_malthus.t)
N_analytic_logistic = K / (1 + ((K - N0) / N0) * np.exp(-r * sol_logistic.t))

err_malthus = np.mean(np.abs(sol_malthus.y[0] - N_analytic_malthus)) / N0 * 100
err_logistic = np.mean(np.abs(sol_logistic.y[0] - N_analytic_logistic)) / N0 * 100

stabilization_threshold = 0.99 * K
stabilization_idx = np.where(sol_logistic.y[0] >= stabilization_threshold)[0]

if len(stabilization_idx) > 0:
    stabilization_year = sol_logistic.t[stabilization_idx[0]]
else:
    stabilization_year = None

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Сравнение моделей роста популяции: Мальтус vs Ферхюльст', 
             fontsize=16, fontweight='bold', y=1.02)

ax1 = axes[0]

ax1.plot(sol_malthus.t, sol_malthus.y[0], 'r-', linewidth=2.2, label='Мальтус (численное)')
ax1.plot(sol_logistic.t, sol_logistic.y[0], 'g-', linewidth=2.2, label='Логистика (численное)')

ax1.plot(sol_malthus.t, N_analytic_malthus, 'r--', linewidth=1.2, alpha=0.7, label='Мальтус (аналитическое)')
ax1.plot(sol_logistic.t, N_analytic_logistic, 'g--', linewidth=1.2, alpha=0.7, label='Логистика (аналитическое)')

ax1.axhline(y=K, color='orange', linestyle='--', linewidth=2, label=f'Ёмкость среды $K = {K/1e9:.0f}$ млрд')

if stabilization_year is not None:
    ax1.axvline(x=stabilization_year, color='purple', linestyle=':', linewidth=2,
                label=f'99% от $K$ (~{stabilization_year:.0f} г.)')
    ax1.scatter([stabilization_year], [0.99 * K], color='purple', s=60, zorder=5)

ax1.set_xlabel('Время, лет')
ax1.set_ylabel('Численность популяции, чел.')
ax1.set_title('Динамика роста популяции', fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True)
ax1.set_ylim(0, K * 1.1)

info_text = f"Параметры:\n$r = {r}$ год⁻¹\n$K = {K/1e9:.0f}$ млрд\n$N_0 = {N0/1e9:.0f}$ млрд"
ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

ax2 = axes[1]

N_range = np.linspace(0, 1.2 * K, 500)
dNdt_malthus = malthus_model(0, N_range, r)
dNdt_logistic = logistic_model(0, N_range, r, K)

ax2.plot(N_range, dNdt_malthus, 'r-', linewidth=2.2, label='Мальтус')
ax2.plot(N_range, dNdt_logistic, 'g-', linewidth=2.2, label='Логистика')

ax2.plot(0, 0, 'ko', markersize=8)
ax2.plot(K, 0, 'ko', markersize=8)

ax2.annotate('N=0\n(неустойч.)', xy=(0, 0), xytext=(K*0.05, -0.08*r*K),
             fontsize=11, ha='center',
             arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
ax2.annotate('N=K\n(устойч.)', xy=(K, 0), xytext=(K*0.8, 0.06*r*K),
             fontsize=11, ha='center',
             arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

ax2.axhline(0, color='black', linewidth=0.8)
ax2.axvline(K, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

ax2.set_xlabel('N, чел.')
ax2.set_ylabel('dN/dt, чел./год')
ax2.set_title('Фазовый портрет модели', fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1.2 * K)
ax2.set_ylim(-0.2 * r * K, 0.3 * r * K)

plt.tight_layout()

plt.savefig('population_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
print("="*60)

print(f"• Начальная численность: {N0/1e9:.1f} млрд")
print(f"• Скорость роста: r = {r} год⁻¹")
print(f"• Ёмкость среды: K = {K/1e9:.1f} млрд")

print(f"\n• Относительная ошибка численного решения:")
print(f"  — Мальтус:    {err_malthus:.2e} %")
print(f"  — Логистика:  {err_logistic:.2e} %")

if stabilization_year is not None:
    print(f"\n• Логистическая модель достигает 99% от K к {stabilization_year:.1f} году.")
else:
    print("\n• Стабилизация (99% от K) не достигнута в заданном временном интервале.")

print("\n• Аналитические выводы:")
print("  — Модель Мальтуса предсказывает неограниченный экспоненциальный рост.")
print("  — Логистическая модель учитывает ограниченность ресурсов и имеет")
print("    устойчивую равновесную точку при N = K.")
print("  — Точка N = 0 неустойчива в обеих моделях при r > 0.")
