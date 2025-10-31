import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

benchmarks = [
    'GPQA', 
    'MMLU',
    'MMLU-Pro',
    'MMMU'
    ]

# Load preprocessed CSVs from plot_scores_vs_t.py
files = sorted(glob.glob('data_with_dates_and_categories_*.csv'))
frames = []
for benchmark in benchmarks:
    path = f'data_with_dates_and_categories_{benchmark.lower()}.csv'
    df = pd.read_csv(path)
    # Ensure expected columns are present and typed
    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.dropna(subset=['release_date', 'score'])
    df['benchmark'] = benchmark
    frames.append(df[['model', 'score', 'open_vs_proprietary', 'release_date', 'benchmark']])

all_benchmarks = pd.concat(frames, ignore_index=True)
print(f"Total data points across all benchmarks: {len(all_benchmarks)}")


# Process Time-Series Data
results = []
for benchmark in benchmarks:
    bench_data = all_benchmarks[all_benchmarks['benchmark'] == benchmark].copy()
    bench_data = bench_data.sort_values('release_date')
    unique_dates = sorted(bench_data['release_date'].unique())
    for date in unique_dates:
        models_up_to_date = bench_data[bench_data['release_date'] <= date]
        proprietary_models = models_up_to_date[models_up_to_date['open_vs_proprietary'] == 'proprietary']
        best_proprietary = proprietary_models['score'].max() if len(proprietary_models) > 0 else 0
        opensource_models = models_up_to_date[models_up_to_date['open_vs_proprietary'] == 'open-source']
        best_opensource = opensource_models['score'].max() if len(opensource_models) > 0 else 0
        best_overall = max(best_proprietary, best_opensource)
        gap = best_proprietary - best_opensource
        if best_opensource > 0 and best_proprietary > 0:
            results.append({
                'benchmark': benchmark,
                'date': date,
                'best_overall': best_overall,
                'best_proprietary': best_proprietary,
                'best_opensource': best_opensource,
                'gap': gap
            })

df_results = pd.DataFrame(results)
df_results.to_csv('current_gaps_vs_current_best_scores.csv', index=False)

# Create Scatter Plot
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')
colors = {
    'GPQA': '#E74C3C',      # Red
    'MMLU': '#27AE60',      # Teal/Green
    'MMLU-Pro': '#3498DB',  # Blue
    'MMMU': '#E67E22'       # Orange
}
for benchmark in ['GPQA', 'MMLU', 'MMLU-Pro', 'MMMU']:
    bench_data = df_results[df_results['benchmark'] == benchmark]
    ax.scatter(
        bench_data['best_overall'] * 100,
        bench_data['gap'] * 100,
        c=colors[benchmark],
        marker='o',
        s=80,
        alpha=1.0,
        edgecolors=colors[benchmark],
        linewidths=0.5,
        label=benchmark
    )

# Add trend line
from scipy import stats
x_data = df_results['best_overall'].values * 100
y_data = df_results['gap'].values * 100
slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
line_x = np.array([x_data.min(), x_data.max()])
line_y = slope * line_x + intercept
ax.plot(line_x, line_y, color='gray', linestyle='--', linewidth=2, alpha=0.5, zorder=1)
print(f"Trend line: slope={slope:.3f}, intercept={intercept:.3f}, p-value={p_value:.3e}")

ax.set_xlabel('Current Best Score', fontsize=32, fontweight='bold')
ax.set_ylabel('Current Gap', fontsize=32, fontweight='bold')
ax.tick_params(axis='both', which='major', length=12, width=2, direction='out', pad=10)
ax.tick_params(axis='both', labelsize=20)
ax.set_xlim(50, 100)
ax.set_ylim(-10, 40)
ax.legend(loc='upper right', fontsize=24, framealpha=0.95)
plt.tight_layout()
plt.savefig('current_gaps_vs_current_best_scores.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
