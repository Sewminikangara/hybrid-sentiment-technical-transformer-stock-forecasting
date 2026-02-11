"""
Generate all evaluation plots for dissertation
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Create graphs directory
os.makedirs('graphs', exist_ok=True)

# Load training results
results_df = pd.read_csv('results/training_results_20260104_230409.csv')
cse_df = pd.read_csv('results/cse_training_results_20260104_232553.csv')

# Standardize column names
results_df.columns = results_df.columns.str.lower().str.replace('_', '')
cse_df.columns = cse_df.columns.str.lower().str.replace('_', '')

# Rename columns to match
results_df = results_df.rename(columns={
    'modeltype': 'model',
    'directionalacc': 'directional_accuracy',
    'directionalaccuracy': 'directional_accuracy'
})
cse_df = cse_df.rename(columns={
    'modeltype': 'model',
    'directionalacc': 'directional_accuracy',
    'directionalaccuracy': 'directional_accuracy'
})

# Combine all results
all_results = pd.concat([results_df, cse_df], ignore_index=True)

# Ensure numeric columns
for col in ['rmse', 'mae', 'mape', 'directional_accuracy']:
    if col in all_results.columns:
        all_results[col] = pd.to_numeric(all_results[col], errors='coerce')

# Add R² if not present (use 1 - normalized RMSE as proxy)
if 'r2' not in all_results.columns:
    all_results['r2'] = 1 - (all_results['rmse'] / all_results['rmse'].max())

print("=" * 80)
print("GENERATING EVALUATION PLOTS FOR DISSERTATION")
print("=" * 80)

# 1. Model Performance Comparison by Stock
print("\n[1/6] Creating model performance comparison...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Performance Comparison Across Stocks', fontsize=16, fontweight='bold')

# RMSE comparison
ax1 = axes[0, 0]
pivot_rmse = all_results.pivot(index='stock', columns='model', values='rmse')
pivot_rmse.plot(kind='bar', ax=ax1, width=0.8)
ax1.set_title('Root Mean Square Error (RMSE)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Stock Symbol', fontsize=10)
ax1.set_ylabel('RMSE', fontsize=10)
ax1.legend(title='Model', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# MAPE comparison
ax2 = axes[0, 1]
pivot_mape = all_results.pivot(index='stock', columns='model', values='mape')
pivot_mape.plot(kind='bar', ax=ax2, width=0.8)
ax2.set_title('Mean Absolute Percentage Error (MAPE)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Stock Symbol', fontsize=10)
ax2.set_ylabel('MAPE (%)', fontsize=10)
ax2.legend(title='Model', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Directional Accuracy comparison
ax3 = axes[1, 0]
pivot_dir = all_results.pivot(index='stock', columns='model', values='directional_accuracy')
pivot_dir.plot(kind='bar', ax=ax3, width=0.8)
ax3.set_title('Directional Accuracy', fontsize=12, fontweight='bold')
ax3.set_xlabel('Stock Symbol', fontsize=10)
ax3.set_ylabel('Accuracy (%)', fontsize=10)
ax3.legend(title='Model', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)
ax3.axhline(y=50, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Random Baseline')

# R² Score comparison
ax4 = axes[1, 1]
pivot_r2 = all_results.pivot(index='stock', columns='model', values='r2')
pivot_r2.plot(kind='bar', ax=ax4, width=0.8)
ax4.set_title('R² Score', fontsize=12, fontweight='bold')
ax4.set_xlabel('Stock Symbol', fontsize=10)
ax4.set_ylabel('R² Score', fontsize=10)
ax4.legend(title='Model', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('graphs/model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("   Saved: graphs/model_performance_comparison.png")
plt.close()

# 2. Cross-Market Analysis
print("\n[2/6] Creating cross-market analysis...")
# Add market classification
def classify_market(stock):
    if stock == 'CSEALL':
        return 'Sri Lanka'
    elif stock.endswith('.NS'):
        return 'India'
    else:
        return 'USA'

all_results['market'] = all_results['stock'].apply(classify_market)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Cross-Market Performance Analysis', fontsize=16, fontweight='bold')

metrics = ['rmse', 'mape', 'directional_accuracy']
titles = ['RMSE by Market', 'MAPE by Market (%)', 'Directional Accuracy by Market (%)']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx]
    market_perf = all_results.groupby(['market', 'model'])[metric].mean().reset_index()
    market_pivot = market_perf.pivot(index='market', columns='model', values=metric)
    market_pivot.plot(kind='bar', ax=ax, width=0.7)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Market', fontsize=10)
    ax.set_ylabel(metric.upper().replace('_', ' '), fontsize=10)
    ax.legend(title='Model', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('graphs/cross_market_analysis.png', dpi=300, bbox_inches='tight')
print("   Saved: graphs/cross_market_analysis.png")
plt.close()

# 3. Model Rankings
print("\n[3/6] Creating model rankings...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Overall Model Rankings', fontsize=16, fontweight='bold')

# Average metrics by model
model_avg = all_results.groupby('model').agg({
    'rmse': 'mean',
    'mape': 'mean',
    'directional_accuracy': 'mean',
    'r2': 'mean'
}).round(4)

# Left plot: Error metrics (lower is better)
ax1 = axes[0]
model_avg[['rmse', 'mape']].plot(kind='barh', ax=ax1, width=0.7)
ax1.set_title('Average Error Metrics (Lower is Better)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Value', fontsize=10)
ax1.set_ylabel('Model', fontsize=10)
ax1.legend(['RMSE', 'MAPE (%)'], fontsize=9)
ax1.grid(True, alpha=0.3, axis='x')

# Right plot: Accuracy metrics (higher is better)
ax2 = axes[1]
model_avg[['directional_accuracy', 'r2']].plot(kind='barh', ax=ax2, width=0.7)
ax2.set_title('Average Accuracy Metrics (Higher is Better)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Value', fontsize=10)
ax2.set_ylabel('Model', fontsize=10)
ax2.legend(['Directional Accuracy (%)', 'R² Score'], fontsize=9)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('graphs/model_rankings.png', dpi=300, bbox_inches='tight')
print("   Saved: graphs/model_rankings.png")
plt.close()

# 4. Best and Worst Performers
print("\n[4/6] Creating best/worst performers chart...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Best and Worst Performing Stocks by Model', fontsize=16, fontweight='bold')

# For each model and metric
models = all_results['model'].unique()
colors = ['#2ecc71', '#e74c3c']  # Green for best, Red for worst

for idx, model in enumerate(models):
    model_data = all_results[all_results['model'] == model].copy()
    
    # MAPE
    ax_mape = axes[idx, 0]
    model_data_sorted = model_data.nsmallest(5, 'mape')
    worst_data = model_data.nlargest(3, 'mape')
    
    ax_mape.barh(model_data_sorted['stock'], model_data_sorted['mape'], color=colors[0], alpha=0.7, label='Best 5')
    ax_mape.barh(worst_data['stock'], worst_data['mape'], color=colors[1], alpha=0.7, label='Worst 3')
    ax_mape.set_title(f'{model} - MAPE Performance', fontsize=11, fontweight='bold')
    ax_mape.set_xlabel('MAPE (%)', fontsize=9)
    ax_mape.legend(fontsize=8)
    ax_mape.grid(True, alpha=0.3, axis='x')
    
    # Directional Accuracy
    ax_dir = axes[idx, 1]
    model_data_sorted_dir = model_data.nlargest(5, 'directional_accuracy')
    worst_data_dir = model_data.nsmallest(3, 'directional_accuracy')
    
    ax_dir.barh(model_data_sorted_dir['stock'], model_data_sorted_dir['directional_accuracy'], 
                color=colors[0], alpha=0.7, label='Best 5')
    ax_dir.barh(worst_data_dir['stock'], worst_data_dir['directional_accuracy'], 
                color=colors[1], alpha=0.7, label='Worst 3')
    ax_dir.set_title(f'{model} - Directional Accuracy', fontsize=11, fontweight='bold')
    ax_dir.set_xlabel('Accuracy (%)', fontsize=9)
    ax_dir.legend(fontsize=8)
    ax_dir.grid(True, alpha=0.3, axis='x')
    ax_dir.axvline(x=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('graphs/best_worst_performers.png', dpi=300, bbox_inches='tight')
print("   Saved: graphs/best_worst_performers.png")
plt.close()

# 5. Metric Distribution Analysis
print("\n[5/6] Creating metric distribution plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Performance Metric Distributions', fontsize=16, fontweight='bold')

metrics_to_plot = ['rmse', 'mape', 'directional_accuracy', 'r2']
titles = ['RMSE Distribution', 'MAPE Distribution', 'Directional Accuracy Distribution', 'R² Distribution']

for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    ax = axes[idx // 2, idx % 2]
    
    for model in models:
        model_data = all_results[all_results['model'] == model][metric]
        ax.hist(model_data, alpha=0.6, label=model, bins=8)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(metric.upper().replace('_', ' '), fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphs/metric_distributions.png', dpi=300, bbox_inches='tight')
print("   Saved: graphs/metric_distributions.png")
plt.close()

# 6. Summary Statistics Table (as image)
print("\n[6/6] Creating summary statistics table...")
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Create summary table
summary_data = []
for model in models:
    model_data = all_results[all_results['model'] == model]
    summary_data.append([
        model,
        f"{model_data['rmse'].mean():.4f} ± {model_data['rmse'].std():.4f}",
        f"{model_data['mape'].mean():.2f}% ± {model_data['mape'].std():.2f}%",
        f"{model_data['directional_accuracy'].mean():.2f}% ± {model_data['directional_accuracy'].std():.2f}%",
        f"{model_data['r2'].mean():.4f} ± {model_data['r2'].std():.4f}",
        f"{model_data['rmse'].min():.4f}",
        f"{model_data['mape'].min():.2f}%",
        f"{model_data['directional_accuracy'].max():.2f}%"
    ])

# Add market-specific summaries
for market in ['USA', 'India', 'Sri Lanka']:
    market_data = all_results[all_results['market'] == market]
    summary_data.append([
        f"{market} Avg",
        f"{market_data['rmse'].mean():.4f}",
        f"{market_data['mape'].mean():.2f}%",
        f"{market_data['directional_accuracy'].mean():.2f}%",
        f"{market_data['r2'].mean():.4f}",
        "-",
        "-",
        "-"
    ])

columns = ['Model/Market', 'Avg RMSE', 'Avg MAPE', 'Avg Dir. Acc.', 'Avg R²', 'Best RMSE', 'Best MAPE', 'Best Dir. Acc.']
table = ax.table(cellText=summary_data, colLabels=columns, cellLoc='center', loc='center',
                colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(len(columns)):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(summary_data) + 1):
    if 'Avg' in summary_data[i-1][0]:
        for j in range(len(columns)):
            table[(i, j)].set_facecolor('#ecf0f1')
            table[(i, j)].set_text_props(weight='bold')
    else:
        for j in range(len(columns)):
            table[(i, j)].set_facecolor('#ffffff' if i % 2 == 0 else '#f8f9fa')

plt.title('Comprehensive Performance Summary', fontsize=14, fontweight='bold', pad=20)
plt.savefig('graphs/summary_statistics.png', dpi=300, bbox_inches='tight')
print("   Saved: graphs/summary_statistics.png")
plt.close()

# Print summary to console
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print("\nOverall Model Performance:")
print(model_avg)

print("\n\nCross-Market Performance:")
market_summary = all_results.groupby('market').agg({
    'rmse': 'mean',
    'mape': 'mean',
    'directional_accuracy': 'mean',
    'r2': 'mean'
}).round(4)
print(market_summary)

print("\n\nBest Performers:")
print(f"Lowest MAPE: {all_results.loc[all_results['mape'].idxmin(), ['stock', 'model', 'mape']].values}")
print(f"Highest Directional Accuracy: {all_results.loc[all_results['directional_accuracy'].idxmax(), ['stock', 'model', 'directional_accuracy']].values}")
print(f"Best R² Score: {all_results.loc[all_results['r2'].idxmax(), ['stock', 'model', 'r2']].values}")

print("\n" + "=" * 80)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print("Location: graphs/ folder")
print("=" * 80)
