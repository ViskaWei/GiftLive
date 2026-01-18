#!/usr/bin/env python3
"""
Improved EDA Visualization Script
Based on Section 7.5 prompts from exp_kuailive_eda_20260108.md

This script generates 4 improved figures that consolidate multiple observations:
1. Fig 1: Amount distribution (replaces Fig1+Fig2+Fig3)
2. Fig 2: User & Streamer concentration (replaces Fig4+Fig5)
3. Fig 3: Sparsity panorama (replaces Fig6+Fig7+Fig9)
4. Fig 4: Temporal patterns (upgrades Fig8)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import json
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# Set plot style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Paths
DATA_DIR = Path("/home/swei20/GiftLive/data/KuaiLive")
RESULTS_FILE = Path("/home/swei20/GiftLive/gift_allocation/results/eda_stats_20260108.json")
OUTPUT_DIR = Path("/home/swei20/GiftLive/KuaiLive/img")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load gift data for visualization."""
    print("Loading data...")
    gift_df = pd.read_csv(DATA_DIR / "gift.csv")
    click_df = pd.read_csv(DATA_DIR / "click.csv")
    streamer_df = pd.read_csv(DATA_DIR / "streamer.csv")
    return gift_df, click_df, streamer_df

def load_stats():
    """Load statistics from JSON file."""
    with open(RESULTS_FILE, 'r') as f:
        stats = json.load(f)
    return stats

def plot_fig1_amount_distribution_comprehensive(gift_df, stats, output_path):
    """
    Prompt 1: Replace Fig1+Fig2+Fig3
    One figure explaining: heavy-tail + log(1+x) value + key percentiles
    """
    amounts = gift_df['gift_price'].values
    log_amounts = np.log1p(amounts)
    
    amount_stats = stats['amount_stats']
    p50 = amount_stats['median']
    p90 = amount_stats['p90']
    p99 = amount_stats['p99']
    mean_val = amount_stats['mean']
    max_val = amount_stats['max']
    
    fig = plt.figure(figsize=(18, 6))
    
    # Panel A: Raw distribution (log scale on x-axis and y-axis)
    ax1 = plt.subplot(1, 3, 1)
    # Filter to reasonable range for visualization
    amounts_filtered = amounts[amounts <= p99 * 2]  # Show up to ~2x P99
    hist, bins, _ = ax1.hist(amounts_filtered, bins=100, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')  # Add log scale for y-axis
    ax1.set_xlabel('Gift Amount (log scale)', fontweight='bold')
    ax1.set_ylabel('Count (log scale)', fontweight='bold')
    ax1.set_title('Raw is extremely heavy-tailed', fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Add percentile lines
    ax1.axvline(p50, color='red', linestyle='--', linewidth=2, label=f'P50={p50:.0f}')
    ax1.axvline(p90, color='orange', linestyle='--', linewidth=2, label=f'P90={p90:.0f}')
    ax1.axvline(p99, color='purple', linestyle='--', linewidth=2, label=f'P99={p99:.0f}')
    ax1.legend(loc='upper left', fontsize=9)  # Move legend to upper left to avoid overlap
    
    # Add text annotations - repositioned to avoid overlap
    ax1.text(0.98, 0.15, f'Mean={mean_val:.1f}\nMax={max_val:.0f}', 
             transform=ax1.transAxes, fontsize=9, verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax1.text(0.02, 0.15, 'Most gifts are small,\nlong tail to the right', 
             transform=ax1.transAxes, fontsize=9, verticalalignment='bottom',
             horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Panel B: log1p distribution
    ax2 = plt.subplot(1, 3, 2)
    hist, bins, _ = ax2.hist(log_amounts, bins=80, alpha=0.7, color='forestgreen', edgecolor='black', linewidth=0.5, density=True)
    
    # Overlay normal fit
    mu, sigma = np.mean(log_amounts), np.std(log_amounts)
    x_norm = np.linspace(log_amounts.min(), log_amounts.max(), 200)
    y_norm = scipy_stats.norm.pdf(x_norm, mu, sigma)
    ax2.plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal fit', alpha=0.8)
    
    # Add percentile lines on log scale
    log_p50 = np.log1p(p50)
    log_p90 = np.log1p(p90)
    log_p99 = np.log1p(p99)
    ax2.axvline(log_p50, color='red', linestyle='--', linewidth=2, label=f'P50={p50:.0f}')
    ax2.axvline(log_p90, color='orange', linestyle='--', linewidth=2, label=f'P90={p90:.0f}')
    ax2.axvline(log_p99, color='purple', linestyle='--', linewidth=2, label=f'P99={p99:.0f}')
    
    ax2.set_xlabel('log(1 + Amount)', fontweight='bold')
    ax2.set_ylabel('Density (log scale)', fontweight='bold')
    ax2.set_yscale('log')  # Add log scale for y-axis
    ax2.set_title('log(1+x) is approximately normal → suitable for regression', 
                  fontweight='bold', fontsize=13)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Panel C: CCDF (log-log)
    ax3 = plt.subplot(1, 3, 3)
    sorted_amounts = np.sort(amounts)[::-1]  # Descending
    n = len(sorted_amounts)
    ccdf = np.arange(1, n + 1) / n  # P(X >= x)
    
    # Filter for better visualization
    mask = sorted_amounts > 0
    ax3.loglog(sorted_amounts[mask], ccdf[mask], 'b-', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Gift Amount (log scale)', fontweight='bold')
    ax3.set_ylabel('P(X ≥ x) (log scale)', fontweight='bold')
    ax3.set_title(f'Heavy-tail confirmed: P99/P50 = {p99/p50:.0f}x', 
                  fontweight='bold', fontsize=13)
    ax3.grid(True, alpha=0.3, which='both')
    
    # Add annotation
    ax3.text(0.98, 0.95, f'P99/P50 = {p99/p50:.0f}x\nPower-law like tail', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_fig2_concentration_comparison(gift_df, stats, output_path):
    """
    Prompt 2: Replace Fig4+Fig5
    One figure showing User & Streamer concentration (Gini + Top-share)
    """
    # Compute user and streamer aggregations
    user_gifts = gift_df.groupby('user_id')['gift_price'].sum().sort_values(ascending=True)
    streamer_gifts = gift_df.groupby('streamer_id')['gift_price'].sum().sort_values(ascending=True)
    
    # Build Lorenz curves
    def build_lorenz(values):
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals)
        total = cumsum[-1]
        p = np.arange(1, n + 1) / n  # Cumulative % of individuals
        L = cumsum / total  # Cumulative % of value
        return p, L
    
    p_user, L_user = build_lorenz(user_gifts.values)
    p_streamer, L_streamer = build_lorenz(streamer_gifts.values)
    
    # Get stats
    gini_user = stats['gini']['user']
    gini_streamer = stats['gini']['streamer']
    top_concentration = stats['top_concentration']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Perfect equality line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect equality', alpha=0.5)
    
    # User Lorenz curve
    ax.plot(p_user, L_user, 'b-', linewidth=3, label=f'Users (Gini={gini_user:.3f})', alpha=0.8)
    
    # Streamer Lorenz curve
    ax.plot(p_streamer, L_streamer, 'r-', linewidth=3, label=f'Streamers (Gini={gini_streamer:.3f})', alpha=0.8)
    
    # Annotate key points
    # Top 1% point
    idx_1pct_user = int(len(p_user) * 0.99)
    idx_1pct_streamer = int(len(p_streamer) * 0.99)
    ax.plot(p_user[idx_1pct_user], L_user[idx_1pct_user], 'bo', markersize=10, zorder=5)
    ax.plot(p_streamer[idx_1pct_streamer], L_streamer[idx_1pct_streamer], 'ro', markersize=10, zorder=5)
    
    # Top 10% point
    idx_10pct_user = int(len(p_user) * 0.90)
    idx_10pct_streamer = int(len(p_streamer) * 0.90)
    ax.plot(p_user[idx_10pct_user], L_user[idx_10pct_user], 'bs', markersize=8, zorder=5)
    ax.plot(p_streamer[idx_10pct_streamer], L_streamer[idx_10pct_streamer], 'rs', markersize=8, zorder=5)
    
    # Add annotations
    ax.annotate(f'Top 1% users:\n{top_concentration["top_1pct_users_share"]*100:.1f}% revenue',
                xy=(p_user[idx_1pct_user], L_user[idx_1pct_user]),
                xytext=(0.25, 0.75), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax.annotate(f'Top 1% streamers:\n{top_concentration["top_1pct_streamers_share"]*100:.1f}% revenue',
                xy=(p_streamer[idx_1pct_streamer], L_streamer[idx_1pct_streamer]),
                xytext=(0.25, 0.55), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    ax.set_xlabel('Cumulative % of Individuals (from low to high)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Cumulative % of Revenue', fontweight='bold', fontsize=12)
    ax.set_title('Revenue is dominated by a tiny fraction of users & streamers', 
                 fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Place legend first to get its position
    legend = ax.legend(loc='lower right', fontsize=11)
    
    # Add text for bottom 90% - place it away from legend (lower left)
    ax.text(0.02, 0.15, f'Bottom 90% users:\n{100-top_concentration["top_10pct_users_share"]*100:.1f}% revenue',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            verticalalignment='bottom')
    
    # Add summary text box - place it in upper left, away from annotations
    textstr = f'User Gini: {gini_user:.3f}\nStreamer Gini: {gini_streamer:.3f}\n\n'
    textstr += f'Top 10% users: {top_concentration["top_10pct_users_share"]*100:.1f}% revenue\n'
    textstr += f'Top 10% streamers: {top_concentration["top_10pct_streamers_share"]*100:.1f}% revenue'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_fig3_sparsity_panorama(gift_df, click_df, streamer_df, stats, output_path):
    """
    Prompt 3: Replace Fig6+Fig7+Fig9
    One figure explaining: sparsity + cold start + interaction structure
    """
    # Compute distributions
    user_gifts = gift_df.groupby('user_id')['gift_price'].count()
    streamer_gifts = gift_df.groupby('streamer_id')['gift_price'].count()
    
    fig = plt.figure(figsize=(16, 12))
    
    # (1) Top-left: User gift_count distribution (CCDF)
    ax1 = plt.subplot(2, 2, 1)
    sorted_counts = np.sort(user_gifts.values)[::-1]
    n = len(sorted_counts)
    ccdf = np.arange(1, n + 1) / n
    median_count = np.median(user_gifts.values)
    
    ax1.loglog(sorted_counts, ccdf, 'b-', linewidth=2, alpha=0.7)
    ax1.axvline(median_count, color='red', linestyle='--', linewidth=2, label=f'Median={median_count:.0f}')
    ax1.set_xlabel('Gifts per User (log scale)', fontweight='bold')
    ax1.set_ylabel('P(X ≥ x) (log scale)', fontweight='bold')
    ax1.set_title('Most users gift only 1-2 times', fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3, which='both')
    # Place legend first, then text box in a different corner
    ax1.legend(loc='upper left', fontsize=9)
    ax1.text(0.98, 0.15, f'Median = {median_count:.0f}\nMean = {user_gifts.mean():.2f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # (2) Top-right: Streamer received_gift_count distribution (CCDF)
    ax2 = plt.subplot(2, 2, 2)
    sorted_counts_s = np.sort(streamer_gifts.values)[::-1]
    n_s = len(sorted_counts_s)
    ccdf_s = np.arange(1, n_s + 1) / n_s
    median_count_s = np.median(streamer_gifts.values)
    
    ax2.loglog(sorted_counts_s, ccdf_s, 'r-', linewidth=2, alpha=0.7)
    ax2.axvline(median_count_s, color='red', linestyle='--', linewidth=2, label=f'Median={median_count_s:.0f}')
    ax2.set_xlabel('Gifts per Streamer (log scale)', fontweight='bold')
    ax2.set_ylabel('P(X ≥ x) (log scale)', fontweight='bold')
    ax2.set_title('Long-tail receiving pattern', fontweight='bold', fontsize=13)
    ax2.grid(True, alpha=0.3, which='both')
    # Place legend first, then text box in a different corner
    ax2.legend(loc='upper left', fontsize=9)
    ax2.text(0.98, 0.15, f'Median = {median_count_s:.0f}\nMean = {streamer_gifts.mean():.2f}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # (3) Bottom-left: Interaction matrix spy plot
    ax3 = plt.subplot(2, 2, 3)
    # Sample top N users and streamers
    top_n = 200
    top_users = gift_df.groupby('user_id')['gift_price'].sum().nlargest(top_n).index
    top_streamers = gift_df.groupby('streamer_id')['gift_price'].sum().nlargest(top_n).index
    
    # Build interaction matrix
    gift_sample = gift_df[
        gift_df['user_id'].isin(top_users) & 
        gift_df['streamer_id'].isin(top_streamers)
    ]
    
    # Create sparse matrix representation
    user_map = {uid: i for i, uid in enumerate(sorted(top_users))}
    streamer_map = {sid: i for i, sid in enumerate(sorted(top_streamers))}
    
    matrix = np.zeros((len(top_users), len(top_streamers)), dtype=bool)
    for _, row in gift_sample.iterrows():
        u_idx = user_map[row['user_id']]
        s_idx = streamer_map[row['streamer_id']]
        matrix[u_idx, s_idx] = True
    
    # Plot spy plot
    ax3.spy(matrix, markersize=0.5, aspect='auto', color='blue')
    density = stats['sparsity']['matrix_density']
    ax3.set_xlabel('Streamers (Top 200)', fontweight='bold')
    ax3.set_ylabel('Users (Top 200)', fontweight='bold')
    ax3.set_title('Interaction matrix is extremely sparse', fontweight='bold', fontsize=13)
    ax3.text(0.98, 0.02, f'Density = {density*100:.4f}%',
             transform=ax3.transAxes, fontsize=11, verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # (4) Bottom-right: Cold start bar chart
    ax4 = plt.subplot(2, 2, 4)
    total_streamers = len(streamer_df)
    streamers_with_gifts = stats['basic_counts']['unique_gift_streamers']
    streamers_cold_start = total_streamers - streamers_with_gifts
    cold_start_ratio = stats['sparsity']['cold_start_streamer_ratio']
    
    categories = ['Total\nStreamers', 'With Gifts', 'Cold Start']
    values = [total_streamers, streamers_with_gifts, streamers_cold_start]
    colors = ['gray', 'green', 'red']
    
    bars = ax4.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Count', fontweight='bold')
    ax4.set_title(f'Cold start problem: {cold_start_ratio*100:.1f}% streamers have no gifts', 
                  fontweight='bold', fontsize=13)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}\n({val/total_streamers*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add annotation
    ax4.text(0.5, 0.95, f'Cold start ratio = {cold_start_ratio*100:.1f}%',
             transform=ax4.transAxes, fontsize=11, verticalalignment='top',
             horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle('Extreme sparsity makes naive modeling fragile', 
                 fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_fig4_temporal_patterns(gift_df, stats, output_path):
    """
    Prompt 4: Upgrade Fig8
    Temporal pattern showing both gift count and total amount
    """
    # Extract time features
    gift_df = gift_df.copy()
    gift_df['timestamp'] = pd.to_datetime(gift_df['timestamp'], unit='ms')
    gift_df['hour'] = gift_df['timestamp'].dt.hour
    gift_df['day_of_week'] = gift_df['timestamp'].dt.dayofweek  # 0=Monday
    
    # Aggregate by day_of_week x hour
    temporal_agg = gift_df.groupby(['day_of_week', 'hour']).agg({
        'gift_price': ['count', 'sum']
    }).reset_index()
    temporal_agg.columns = ['day_of_week', 'hour', 'gift_count', 'total_amount']
    
    # Create heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Heatmap 1: Gift count
    pivot_count = temporal_agg.pivot(index='day_of_week', columns='hour', values='gift_count')
    pivot_count = pivot_count.fillna(0)
    
    im1 = ax1.imshow(pivot_count.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax1.set_xticks(range(24))
    ax1.set_xticklabels(range(24))
    ax1.set_yticks(range(7))
    ax1.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax1.set_xlabel('Hour of Day', fontweight='bold')
    ax1.set_ylabel('Day of Week', fontweight='bold')
    ax1.set_title('Gift Count by Day & Hour', fontweight='bold', fontsize=13)
    plt.colorbar(im1, ax=ax1, label='Gift Count')
    
    # Highlight peak hour
    peak_hour = stats['temporal']['peak_hour']
    ax1.axvline(peak_hour, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Peak: {peak_hour}:00')
    # Highlight 20-22 interval
    ax1.axvspan(20, 22, alpha=0.2, color='green', label='Evening peak (20-22)')
    ax1.legend(loc='upper right')
    
    # Heatmap 2: Total amount
    pivot_amount = temporal_agg.pivot(index='day_of_week', columns='hour', values='total_amount')
    pivot_amount = pivot_amount.fillna(0)
    
    im2 = ax2.imshow(pivot_amount.values, cmap='YlGnBu', aspect='auto', interpolation='nearest')
    ax2.set_xticks(range(24))
    ax2.set_xticklabels(range(24))
    ax2.set_yticks(range(7))
    ax2.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax2.set_xlabel('Hour of Day', fontweight='bold')
    ax2.set_ylabel('Day of Week', fontweight='bold')
    ax2.set_title('Total Gift Amount by Day & Hour', fontweight='bold', fontsize=13)
    plt.colorbar(im2, ax=ax2, label='Total Amount')
    
    # Highlight peak hour
    ax2.axvline(peak_hour, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Peak: {peak_hour}:00')
    ax2.axvspan(20, 22, alpha=0.2, color='green', label='Evening peak (20-22)')
    ax2.legend(loc='upper right')
    
    plt.suptitle('Temporal patterns: Peak at 12:00 and 20:00-22:00', 
                 fontweight='bold', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def main():
    print("=" * 60)
    print("Improved EDA Visualization - Based on Section 7.5 Prompts")
    print("=" * 60)
    
    # Load data and stats
    gift_df, click_df, streamer_df = load_data()
    stats = load_stats()
    
    print("\n=== Generating Improved Figures ===")
    
    # Generate 4 improved figures
    plot_fig1_amount_distribution_comprehensive(
        gift_df, stats, 
        OUTPUT_DIR / "fig1_amount_distribution_comprehensive.png"
    )
    
    plot_fig2_concentration_comparison(
        gift_df, stats,
        OUTPUT_DIR / "fig2_concentration_comparison.png"
    )
    
    plot_fig3_sparsity_panorama(
        gift_df, click_df, streamer_df, stats,
        OUTPUT_DIR / "fig3_sparsity_panorama.png"
    )
    
    plot_fig4_temporal_patterns(
        gift_df, stats,
        OUTPUT_DIR / "fig4_temporal_patterns.png"
    )
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
