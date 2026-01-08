#!/usr/bin/env python3
"""
KuaiLive EDA Script
Experiment ID: EXP-20260108-gift-allocation-01
MVP: MVP-0.1

This script performs exploratory data analysis on the KuaiLive dataset,
focusing on gift behavior patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Set plot style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
DATA_DIR = Path("/home/swei20/GiftLive/data/KuaiLive")
OUTPUT_DIR = Path("/home/swei20/GiftLive/experiments/gift_allocation")
IMG_DIR = OUTPUT_DIR / "img"
RESULTS_DIR = OUTPUT_DIR / "results"

def compute_gini(values):
    """Compute Gini coefficient from array of values."""
    values = np.array(values, dtype=float)
    values = values[values > 0]  # Remove zeros
    if len(values) == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    gini = (2 * np.sum((np.arange(1, n+1) * values)) - (n + 1) * np.sum(values)) / (n * np.sum(values))
    return gini

def load_data():
    """Load all necessary CSV files."""
    print("Loading data...")
    
    gift_df = pd.read_csv(DATA_DIR / "gift.csv")
    click_df = pd.read_csv(DATA_DIR / "click.csv")
    user_df = pd.read_csv(DATA_DIR / "user.csv")
    streamer_df = pd.read_csv(DATA_DIR / "streamer.csv")
    
    print(f"  gift.csv: {len(gift_df):,} rows")
    print(f"  click.csv: {len(click_df):,} rows")
    print(f"  user.csv: {len(user_df):,} rows")
    print(f"  streamer.csv: {len(streamer_df):,} rows")
    
    return gift_df, click_df, user_df, streamer_df

def analyze_gift_basics(gift_df, click_df, user_df):
    """Analyze basic gift statistics."""
    print("\n=== Gift Basics ===")
    
    stats = {
        "total_gift_records": len(gift_df),
        "unique_gift_users": gift_df['user_id'].nunique(),
        "unique_gift_streamers": gift_df['streamer_id'].nunique(),
        "unique_gift_rooms": gift_df['live_id'].nunique(),
        "total_users": user_df['user_id'].nunique(),
        "unique_click_users": click_df['user_id'].nunique(),
    }
    
    # Gift rate: users who gifted / total users who clicked
    stats["gift_rate"] = stats["unique_gift_users"] / stats["unique_click_users"]
    
    print(f"  Total gift records: {stats['total_gift_records']:,}")
    print(f"  Unique users who gifted: {stats['unique_gift_users']:,}")
    print(f"  Unique streamers received gifts: {stats['unique_gift_streamers']:,}")
    print(f"  Unique click users: {stats['unique_click_users']:,}")
    print(f"  Gift rate: {stats['gift_rate']*100:.2f}%")
    
    return stats

def analyze_amount_distribution(gift_df):
    """Analyze gift amount distribution."""
    print("\n=== Amount Distribution ===")
    
    amounts = gift_df['gift_price'].values
    log_amounts = np.log1p(amounts)
    
    stats = {
        "mean": float(np.mean(amounts)),
        "std": float(np.std(amounts)),
        "median": float(np.median(amounts)),
        "min": float(np.min(amounts)),
        "max": float(np.max(amounts)),
        "p10": float(np.percentile(amounts, 10)),
        "p25": float(np.percentile(amounts, 25)),
        "p50": float(np.percentile(amounts, 50)),
        "p75": float(np.percentile(amounts, 75)),
        "p90": float(np.percentile(amounts, 90)),
        "p95": float(np.percentile(amounts, 95)),
        "p99": float(np.percentile(amounts, 99)),
        "log_mean": float(np.mean(log_amounts)),
        "log_std": float(np.std(log_amounts)),
    }
    
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Std: {stats['std']:.2f}")
    print(f"  Median (P50): {stats['median']:.2f}")
    print(f"  P90: {stats['p90']:.2f}")
    print(f"  P95: {stats['p95']:.2f}")
    print(f"  P99: {stats['p99']:.2f}")
    print(f"  Max: {stats['max']:.2f}")
    
    return stats, amounts

def analyze_user_dimension(gift_df):
    """Analyze gift behavior by user."""
    print("\n=== User Dimension Analysis ===")
    
    user_gifts = gift_df.groupby('user_id').agg({
        'gift_price': ['count', 'sum']
    }).reset_index()
    user_gifts.columns = ['user_id', 'gift_count', 'total_amount']
    
    # Gini coefficient for user spending
    user_gini = compute_gini(user_gifts['total_amount'].values)
    
    # Top-K user contribution
    user_gifts_sorted = user_gifts.sort_values('total_amount', ascending=False)
    total_amount = user_gifts['total_amount'].sum()
    n_users = len(user_gifts)
    
    top_1pct_n = max(1, int(n_users * 0.01))
    top_5pct_n = max(1, int(n_users * 0.05))
    top_10pct_n = max(1, int(n_users * 0.10))
    
    top_1pct_share = user_gifts_sorted.head(top_1pct_n)['total_amount'].sum() / total_amount
    top_5pct_share = user_gifts_sorted.head(top_5pct_n)['total_amount'].sum() / total_amount
    top_10pct_share = user_gifts_sorted.head(top_10pct_n)['total_amount'].sum() / total_amount
    
    stats = {
        "unique_gifting_users": n_users,
        "avg_gifts_per_user": float(user_gifts['gift_count'].mean()),
        "avg_amount_per_user": float(user_gifts['total_amount'].mean()),
        "user_gini": float(user_gini),
        "top_1pct_share": float(top_1pct_share),
        "top_5pct_share": float(top_5pct_share),
        "top_10pct_share": float(top_10pct_share),
    }
    
    print(f"  Unique gifting users: {n_users:,}")
    print(f"  Avg gifts per user: {stats['avg_gifts_per_user']:.2f}")
    print(f"  User Gini coefficient: {user_gini:.4f}")
    print(f"  Top 1% users contribute: {top_1pct_share*100:.1f}%")
    print(f"  Top 5% users contribute: {top_5pct_share*100:.1f}%")
    print(f"  Top 10% users contribute: {top_10pct_share*100:.1f}%")
    
    return stats, user_gifts

def analyze_streamer_dimension(gift_df):
    """Analyze gift behavior by streamer."""
    print("\n=== Streamer Dimension Analysis ===")
    
    streamer_gifts = gift_df.groupby('streamer_id').agg({
        'gift_price': ['count', 'sum']
    }).reset_index()
    streamer_gifts.columns = ['streamer_id', 'gift_count', 'total_amount']
    
    # Gini coefficient for streamer revenue
    streamer_gini = compute_gini(streamer_gifts['total_amount'].values)
    
    # Top-K streamer share
    streamer_gifts_sorted = streamer_gifts.sort_values('total_amount', ascending=False)
    total_amount = streamer_gifts['total_amount'].sum()
    n_streamers = len(streamer_gifts)
    
    top_1pct_n = max(1, int(n_streamers * 0.01))
    top_5pct_n = max(1, int(n_streamers * 0.05))
    top_10pct_n = max(1, int(n_streamers * 0.10))
    
    top_1pct_share = streamer_gifts_sorted.head(top_1pct_n)['total_amount'].sum() / total_amount
    top_5pct_share = streamer_gifts_sorted.head(top_5pct_n)['total_amount'].sum() / total_amount
    top_10pct_share = streamer_gifts_sorted.head(top_10pct_n)['total_amount'].sum() / total_amount
    
    stats = {
        "unique_receiving_streamers": n_streamers,
        "avg_gifts_per_streamer": float(streamer_gifts['gift_count'].mean()),
        "avg_amount_per_streamer": float(streamer_gifts['total_amount'].mean()),
        "streamer_gini": float(streamer_gini),
        "top_1pct_share": float(top_1pct_share),
        "top_5pct_share": float(top_5pct_share),
        "top_10pct_share": float(top_10pct_share),
    }
    
    print(f"  Unique receiving streamers: {n_streamers:,}")
    print(f"  Avg gifts per streamer: {stats['avg_gifts_per_streamer']:.2f}")
    print(f"  Streamer Gini coefficient: {streamer_gini:.4f}")
    print(f"  Top 1% streamers receive: {top_1pct_share*100:.1f}%")
    print(f"  Top 5% streamers receive: {top_5pct_share*100:.1f}%")
    print(f"  Top 10% streamers receive: {top_10pct_share*100:.1f}%")
    
    return stats, streamer_gifts

def analyze_temporal(gift_df):
    """Analyze temporal patterns of gifts."""
    print("\n=== Temporal Analysis ===")
    
    # Convert timestamp to datetime
    gift_df = gift_df.copy()
    gift_df['datetime'] = pd.to_datetime(gift_df['timestamp'], unit='ms')
    gift_df['hour'] = gift_df['datetime'].dt.hour
    gift_df['day_of_week'] = gift_df['datetime'].dt.dayofweek
    
    # Hourly pattern
    hourly_stats = gift_df.groupby('hour').agg({
        'gift_price': ['count', 'sum', 'mean']
    }).reset_index()
    hourly_stats.columns = ['hour', 'count', 'total_amount', 'avg_amount']
    
    peak_hour = hourly_stats.loc[hourly_stats['count'].idxmax(), 'hour']
    
    print(f"  Peak hour: {peak_hour}:00")
    
    return {'peak_hour': int(peak_hour)}, hourly_stats, gift_df

def analyze_sparsity(gift_df, click_df, streamer_df):
    """Analyze sparsity of user-streamer interaction matrix."""
    print("\n=== Sparsity Analysis ===")
    
    n_gift_users = gift_df['user_id'].nunique()
    n_gift_streamers = gift_df['streamer_id'].nunique()
    n_interactions = len(gift_df.groupby(['user_id', 'streamer_id']).size())
    
    matrix_size = n_gift_users * n_gift_streamers
    density = n_interactions / matrix_size if matrix_size > 0 else 0
    
    # Cold start analysis
    click_users = set(click_df['user_id'].unique())
    gift_users = set(gift_df['user_id'].unique())
    cold_start_user_ratio = 1 - len(gift_users) / len(click_users) if len(click_users) > 0 else 0
    
    all_streamers = set(streamer_df['streamer_id'].unique())
    gift_streamers = set(gift_df['streamer_id'].unique())
    cold_start_streamer_ratio = 1 - len(gift_streamers) / len(all_streamers) if len(all_streamers) > 0 else 0
    
    stats = {
        "n_users": n_gift_users,
        "n_streamers": n_gift_streamers,
        "n_unique_pairs": n_interactions,
        "matrix_density": float(density),
        "cold_start_user_ratio": float(cold_start_user_ratio),
        "cold_start_streamer_ratio": float(cold_start_streamer_ratio),
    }
    
    print(f"  Matrix size: {n_gift_users:,} users x {n_gift_streamers:,} streamers")
    print(f"  Unique user-streamer pairs: {n_interactions:,}")
    print(f"  Matrix density: {density*100:.4f}%")
    print(f"  Cold start user ratio: {cold_start_user_ratio*100:.1f}%")
    print(f"  Cold start streamer ratio: {cold_start_streamer_ratio*100:.1f}%")
    
    return stats

# ==================== Plotting Functions ====================

def plot_fig1_amount_log_distribution(amounts, output_path):
    """Fig1: Gift amount distribution (log scale)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    log_amounts = np.log1p(amounts)
    ax.hist(log_amounts, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Gift Amount (log scale: log(1+x))')
    ax.set_ylabel('Count')
    ax.set_title('Gift Amount Distribution (Log Scale)')
    
    # Add statistics
    mean_val = np.mean(log_amounts)
    median_val = np.median(log_amounts)
    ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")

def plot_fig2_amount_raw_distribution(amounts, output_path):
    """Fig2: Gift amount distribution (raw)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Clip extreme values for visualization
    amounts_clipped = np.clip(amounts, 0, np.percentile(amounts, 99))
    ax.hist(amounts_clipped, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax.set_xlabel('Gift Amount')
    ax.set_ylabel('Count')
    ax.set_title('Gift Amount Distribution (Raw, clipped at P99)')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")

def plot_fig3_percentiles(amounts, output_path):
    """Fig3: Gift amount percentiles."""
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    values = [np.percentile(amounts, p) for p in percentiles]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([f'P{p}' for p in percentiles], values, color='teal', edgecolor='black')
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Gift Amount')
    ax.set_title('Gift Amount by Percentile')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")

def plot_fig4_user_lorenz(user_gifts, output_path):
    """Fig4: User Lorenz curve."""
    amounts = np.sort(user_gifts['total_amount'].values)
    cumsum = np.cumsum(amounts)
    cumsum_pct = cumsum / cumsum[-1] * 100
    x = np.arange(1, len(amounts) + 1) / len(amounts) * 100
    
    gini = compute_gini(amounts)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, cumsum_pct, 'b-', linewidth=2, label='User Gift Distribution')
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1, label='Perfect Equality')
    ax.fill_between(x, cumsum_pct, x, alpha=0.3)
    
    ax.set_xlabel('Cumulative % of Users (sorted by amount)')
    ax.set_ylabel('Cumulative % of Total Gift Amount')
    ax.set_title(f'User Lorenz Curve (Gini = {gini:.4f})')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Add annotation
    ax.annotate(f'Gini = {gini:.4f}', xy=(70, 30), fontsize=14, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")

def plot_fig5_streamer_lorenz(streamer_gifts, output_path):
    """Fig5: Streamer Lorenz curve."""
    amounts = np.sort(streamer_gifts['total_amount'].values)
    cumsum = np.cumsum(amounts)
    cumsum_pct = cumsum / cumsum[-1] * 100
    x = np.arange(1, len(amounts) + 1) / len(amounts) * 100
    
    gini = compute_gini(amounts)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, cumsum_pct, 'r-', linewidth=2, label='Streamer Revenue Distribution')
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1, label='Perfect Equality')
    ax.fill_between(x, cumsum_pct, x, alpha=0.3, color='red')
    
    ax.set_xlabel('Cumulative % of Streamers (sorted by revenue)')
    ax.set_ylabel('Cumulative % of Total Revenue')
    ax.set_title(f'Streamer Lorenz Curve (Gini = {gini:.4f})')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Add annotation
    ax.annotate(f'Gini = {gini:.4f}', xy=(70, 30), fontsize=14,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")

def plot_fig6_gifts_per_user(user_gifts, output_path):
    """Fig6: Distribution of gifts per user."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    counts = user_gifts['gift_count'].values
    counts_clipped = np.clip(counts, 0, np.percentile(counts, 99))
    
    ax.hist(counts_clipped, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax.set_xlabel('Number of Gifts per User')
    ax.set_ylabel('Count of Users')
    ax.set_title('Distribution of Gifts per User (clipped at P99)')
    
    # Add statistics
    ax.axvline(np.mean(counts), color='red', linestyle='--', 
               label=f'Mean: {np.mean(counts):.1f}')
    ax.axvline(np.median(counts), color='green', linestyle='--', 
               label=f'Median: {np.median(counts):.0f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")

def plot_fig7_gifts_per_streamer(streamer_gifts, output_path):
    """Fig7: Distribution of gifts per streamer."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    counts = streamer_gifts['gift_count'].values
    counts_clipped = np.clip(counts, 0, np.percentile(counts, 99))
    
    ax.hist(counts_clipped, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Number of Gifts per Streamer')
    ax.set_ylabel('Count of Streamers')
    ax.set_title('Distribution of Gifts per Streamer (clipped at P99)')
    
    # Add statistics
    ax.axvline(np.mean(counts), color='red', linestyle='--', 
               label=f'Mean: {np.mean(counts):.1f}')
    ax.axvline(np.median(counts), color='green', linestyle='--', 
               label=f'Median: {np.median(counts):.0f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")

def plot_fig8_hourly_pattern(hourly_stats, output_path):
    """Fig8: Hourly gift pattern."""
    fig, ax1 = plt.subplots(figsize=(6, 5))
    
    x = hourly_stats['hour'].values
    
    # Bar chart for count
    bars = ax1.bar(x, hourly_stats['count'], color='steelblue', alpha=0.7, label='Gift Count')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Gift Count', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # Line chart for total amount
    ax2 = ax1.twinx()
    ax2.plot(x, hourly_stats['total_amount'], 'r-o', linewidth=2, markersize=6, label='Total Amount')
    ax2.set_ylabel('Total Amount', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax1.set_title('Hourly Gift Pattern')
    ax1.set_xticks(range(24))
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")

def plot_fig9_interaction_matrix(gift_df, output_path, sample_size=100):
    """Fig9: User-Streamer interaction matrix (sampled)."""
    # Sample top users and streamers for visualization
    top_users = gift_df.groupby('user_id')['gift_price'].sum().nlargest(sample_size).index
    top_streamers = gift_df.groupby('streamer_id')['gift_price'].sum().nlargest(sample_size).index
    
    df_subset = gift_df[gift_df['user_id'].isin(top_users) & gift_df['streamer_id'].isin(top_streamers)]
    
    # Create pivot table
    pivot = df_subset.groupby(['user_id', 'streamer_id'])['gift_price'].sum().unstack(fill_value=0)
    
    # Apply log transform for better visualization
    pivot_log = np.log1p(pivot)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot_log, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'log(1 + Gift Amount)'})
    ax.set_xlabel('Streamer ID (top 100)')
    ax.set_ylabel('User ID (top 100)')
    ax.set_title(f'User-Streamer Interaction Matrix (Top {sample_size} x {sample_size})')
    
    # Remove tick labels for cleaner visualization
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")

def main():
    print("=" * 60)
    print("KuaiLive EDA - EXP-20260108-gift-allocation-01")
    print("=" * 60)
    
    # Load data
    gift_df, click_df, user_df, streamer_df = load_data()
    
    # Analysis
    basic_stats = analyze_gift_basics(gift_df, click_df, user_df)
    amount_stats, amounts = analyze_amount_distribution(gift_df)
    user_stats, user_gifts = analyze_user_dimension(gift_df)
    streamer_stats, streamer_gifts = analyze_streamer_dimension(gift_df)
    temporal_stats, hourly_stats, gift_df_with_time = analyze_temporal(gift_df)
    sparsity_stats = analyze_sparsity(gift_df, click_df, streamer_df)
    
    # Generate plots
    print("\n=== Generating Plots ===")
    plot_fig1_amount_log_distribution(amounts, IMG_DIR / "gift_amount_distribution.png")
    plot_fig2_amount_raw_distribution(amounts, IMG_DIR / "gift_amount_distribution_raw.png")
    plot_fig3_percentiles(amounts, IMG_DIR / "gift_amount_percentiles.png")
    plot_fig4_user_lorenz(user_gifts, IMG_DIR / "user_lorenz_curve.png")
    plot_fig5_streamer_lorenz(streamer_gifts, IMG_DIR / "streamer_lorenz_curve.png")
    plot_fig6_gifts_per_user(user_gifts, IMG_DIR / "gifts_per_user_distribution.png")
    plot_fig7_gifts_per_streamer(streamer_gifts, IMG_DIR / "gifts_per_streamer_distribution.png")
    plot_fig8_hourly_pattern(hourly_stats, IMG_DIR / "hourly_pattern.png")
    plot_fig9_interaction_matrix(gift_df, IMG_DIR / "user_streamer_interaction_matrix.png")
    
    # Compile results
    results = {
        "experiment_id": "EXP-20260108-gift-allocation-01",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gift_rate": basic_stats["gift_rate"],
        "amount_stats": {
            "mean": amount_stats["mean"],
            "std": amount_stats["std"],
            "median": amount_stats["median"],
            "p90": amount_stats["p90"],
            "p95": amount_stats["p95"],
            "p99": amount_stats["p99"],
            "max": amount_stats["max"],
        },
        "gini": {
            "user": user_stats["user_gini"],
            "streamer": streamer_stats["streamer_gini"],
        },
        "top_concentration": {
            "top_1pct_users_share": user_stats["top_1pct_share"],
            "top_5pct_users_share": user_stats["top_5pct_share"],
            "top_10pct_users_share": user_stats["top_10pct_share"],
            "top_1pct_streamers_share": streamer_stats["top_1pct_share"],
            "top_5pct_streamers_share": streamer_stats["top_5pct_share"],
            "top_10pct_streamers_share": streamer_stats["top_10pct_share"],
        },
        "sparsity": {
            "matrix_density": sparsity_stats["matrix_density"],
            "cold_start_user_ratio": sparsity_stats["cold_start_user_ratio"],
            "cold_start_streamer_ratio": sparsity_stats["cold_start_streamer_ratio"],
        },
        "basic_counts": {
            "total_gift_records": basic_stats["total_gift_records"],
            "unique_gift_users": basic_stats["unique_gift_users"],
            "unique_gift_streamers": basic_stats["unique_gift_streamers"],
            "unique_click_users": basic_stats["unique_click_users"],
        },
        "temporal": {
            "peak_hour": temporal_stats["peak_hour"],
        }
    }
    
    # Save results
    output_file = RESULTS_DIR / "eda_stats_20260108.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n=== Results saved to {output_file} ===")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Gift Rate: {results['gift_rate']*100:.2f}%")
    print(f"User Gini: {results['gini']['user']:.4f}")
    print(f"Streamer Gini: {results['gini']['streamer']:.4f}")
    print(f"Top 1% Users Share: {results['top_concentration']['top_1pct_users_share']*100:.1f}%")
    print(f"Top 1% Streamers Share: {results['top_concentration']['top_1pct_streamers_share']*100:.1f}%")
    print(f"Matrix Density: {results['sparsity']['matrix_density']*100:.4f}%")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    results = main()
