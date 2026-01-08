#!/usr/bin/env python3
"""
Multi-Task Learning for Gift Prediction
========================================

Experiment: EXP-20260108-gift-allocation-06 (MVP-1.3)

Goal: Verify if multi-task learning (using dense signals like watch/comment/like)
can improve sparse gift prediction performance.

Tasks:
- Task 1: watch_time (regression, MSE)
- Task 2: has_comment (binary, BCE)
- Task 3: has_gift (binary, Focal Loss)
- Task 4: gift_amount (regression, MSE, conditional on has_gift)

Comparison:
- Single-Task: Only train has_gift
- Multi-Task: Train all 4 tasks jointly

Decision Rule:
- If Multi-Task PR-AUC > Single-Task PR-AUC by ≥3% → Accept multi-task
- Else → Keep single-task

Author: Viska Wei
Date: 2026-01-08
"""

import os
import sys
import json
import random
import warnings
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============ Configuration ============
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Paths
BASE_DIR = Path("/home/swei20/GiftLive")
DATA_DIR = BASE_DIR / "data" / "KuaiLive"
OUTPUT_DIR = BASE_DIR / "gift_allocation"
IMG_DIR = OUTPUT_DIR / "img"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

for d in [IMG_DIR, RESULTS_DIR, MODELS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Plot settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def log_message(msg: str, level: str = "INFO"):
    """Print log message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "📝", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")


# ============ Data Loading and Preparation ============

def load_data():
    """Load all data files."""
    log_message("Loading data files...")
    
    gift = pd.read_csv(DATA_DIR / "gift.csv")
    user = pd.read_csv(DATA_DIR / "user.csv")
    streamer = pd.read_csv(DATA_DIR / "streamer.csv")
    room = pd.read_csv(DATA_DIR / "room.csv")
    click = pd.read_csv(DATA_DIR / "click.csv")
    comment = pd.read_csv(DATA_DIR / "comment.csv")
    like = pd.read_csv(DATA_DIR / "like.csv")
    
    log_message(f"Click: {len(click):,}, Gift: {len(gift):,}, Comment: {len(comment):,}, Like: {len(like):,}")
    
    return gift, user, streamer, room, click, comment, like


def create_user_features(gift: pd.DataFrame, click: pd.DataFrame, user: pd.DataFrame) -> pd.DataFrame:
    """Create user-level features."""
    log_message("Creating user features...")
    
    user_gift_stats = gift.groupby('user_id').agg({
        'gift_price': ['count', 'sum', 'mean', 'std', 'max'],
        'streamer_id': 'nunique',
        'live_id': 'nunique'
    }).reset_index()
    user_gift_stats.columns = [
        'user_id', 'user_gift_count', 'user_gift_sum', 'user_gift_mean', 
        'user_gift_std', 'user_gift_max', 'user_unique_streamers', 'user_unique_rooms'
    ]
    user_gift_stats['user_gift_std'] = user_gift_stats['user_gift_std'].fillna(0)
    
    user_click_stats = click.groupby('user_id').agg({
        'watch_live_time': ['count', 'sum', 'mean'],
        'streamer_id': 'nunique',
        'live_id': 'nunique'
    }).reset_index()
    user_click_stats.columns = [
        'user_id', 'user_watch_count', 'user_watch_time_sum', 'user_watch_time_mean',
        'user_watch_unique_streamers', 'user_watch_unique_rooms'
    ]
    
    user_features = user[['user_id', 'age', 'gender', 'device_brand', 'device_price',
                          'fans_num', 'follow_num', 'accu_watch_live_cnt', 
                          'accu_watch_live_duration', 'is_live_streamer', 'is_photo_author',
                          'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3',
                          'onehot_feat4', 'onehot_feat5', 'onehot_feat6']].copy()
    
    user_features = user_features.merge(user_gift_stats, on='user_id', how='left')
    user_features = user_features.merge(user_click_stats, on='user_id', how='left')
    
    numeric_cols = user_features.select_dtypes(include=[np.number]).columns
    user_features[numeric_cols] = user_features[numeric_cols].fillna(0)
    
    return user_features


def create_streamer_features(gift: pd.DataFrame, room: pd.DataFrame, streamer: pd.DataFrame) -> pd.DataFrame:
    """Create streamer-level features."""
    log_message("Creating streamer features...")
    
    streamer_gift_stats = gift.groupby('streamer_id').agg({
        'gift_price': ['count', 'sum', 'mean', 'std', 'max'],
        'user_id': 'nunique',
        'live_id': 'nunique'
    }).reset_index()
    streamer_gift_stats.columns = [
        'streamer_id', 'streamer_gift_count', 'streamer_gift_sum', 'streamer_gift_mean',
        'streamer_gift_std', 'streamer_gift_max', 'streamer_unique_givers', 'streamer_unique_rooms'
    ]
    streamer_gift_stats['streamer_gift_std'] = streamer_gift_stats['streamer_gift_std'].fillna(0)
    
    streamer_room_stats = room.groupby('streamer_id').agg({
        'live_id': 'count',
        'live_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0,
    }).reset_index()
    streamer_room_stats.columns = ['streamer_id', 'streamer_room_count', 'streamer_main_live_type']
    
    streamer_features = streamer[['streamer_id', 'gender', 'age', 'device_brand', 'device_price',
                                   'live_operation_tag', 'fans_user_num', 'fans_group_fans_num',
                                   'follow_user_num', 'accu_live_cnt', 'accu_live_duration',
                                   'accu_play_cnt', 'accu_play_duration',
                                   'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3',
                                   'onehot_feat4', 'onehot_feat5', 'onehot_feat6']].copy()
    
    streamer_features = streamer_features.rename(columns={
        'gender': 'streamer_gender', 'age': 'streamer_age',
        'device_brand': 'streamer_device_brand', 'device_price': 'streamer_device_price',
        'onehot_feat0': 'streamer_onehot_feat0', 'onehot_feat1': 'streamer_onehot_feat1',
        'onehot_feat2': 'streamer_onehot_feat2', 'onehot_feat3': 'streamer_onehot_feat3',
        'onehot_feat4': 'streamer_onehot_feat4', 'onehot_feat5': 'streamer_onehot_feat5',
        'onehot_feat6': 'streamer_onehot_feat6',
    })
    
    streamer_features = streamer_features.merge(streamer_gift_stats, on='streamer_id', how='left')
    streamer_features = streamer_features.merge(streamer_room_stats, on='streamer_id', how='left')
    
    numeric_cols = streamer_features.select_dtypes(include=[np.number]).columns
    streamer_features[numeric_cols] = streamer_features[numeric_cols].fillna(0)
    
    return streamer_features


def create_interaction_features(gift: pd.DataFrame, click: pd.DataFrame) -> pd.DataFrame:
    """Create user-streamer interaction features."""
    log_message("Creating interaction features...")
    
    user_streamer_click = click.groupby(['user_id', 'streamer_id']).agg({
        'watch_live_time': ['count', 'sum', 'mean']
    }).reset_index()
    user_streamer_click.columns = [
        'user_id', 'streamer_id', 
        'pair_watch_count', 'pair_watch_time_sum', 'pair_watch_time_mean'
    ]
    
    user_streamer_gift = gift.groupby(['user_id', 'streamer_id']).agg({
        'gift_price': ['count', 'sum', 'mean']
    }).reset_index()
    user_streamer_gift.columns = [
        'user_id', 'streamer_id',
        'pair_gift_count', 'pair_gift_sum', 'pair_gift_mean'
    ]
    
    interaction = user_streamer_click.merge(user_streamer_gift, on=['user_id', 'streamer_id'], how='outer')
    interaction = interaction.fillna(0)
    
    return interaction


def encode_categorical(df: pd.DataFrame, cat_columns: list) -> pd.DataFrame:
    """Label encode categorical columns."""
    df = df.copy()
    for col in cat_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')
            df[col] = df[col].astype('category').cat.codes
    return df


def prepare_multitask_data(gift, user, streamer, room, click, comment, like):
    """Prepare data for multi-task learning.
    
    Base: click data (all interactions)
    Labels:
    - watch_time: continuous (from click)
    - has_comment: binary (if user commented in this session)
    - has_gift: binary (if user gifted in this session)
    - gift_amount: continuous (sum of gift_price if gifted)
    """
    log_message("=" * 50)
    log_message("Preparing multi-task data...")
    log_message("=" * 50)
    
    # Create feature tables
    user_features = create_user_features(gift, click, user)
    streamer_features = create_streamer_features(gift, room, streamer)
    interaction = create_interaction_features(gift, click)
    
    # Base: click data
    click_base = click.copy()
    click_base['timestamp_dt'] = pd.to_datetime(click_base['timestamp'], unit='ms')
    click_base['hour'] = click_base['timestamp_dt'].dt.hour
    click_base['day_of_week'] = click_base['timestamp_dt'].dt.dayofweek
    click_base['is_weekend'] = (click_base['day_of_week'] >= 5).astype(int)
    click_base['date'] = click_base['timestamp_dt'].dt.date
    
    # Room info
    room_info = room[['live_id', 'live_type', 'live_content_category']].drop_duplicates('live_id')
    click_base = click_base.merge(room_info, on='live_id', how='left')
    
    # Label 1: watch_time (already in click)
    click_base['label_watch_time'] = np.log1p(click_base['watch_live_time'])
    
    # Label 2: has_comment
    comment_flag = comment.groupby(['user_id', 'streamer_id', 'live_id']).size().reset_index(name='comment_count')
    comment_flag['has_comment'] = 1
    click_base = click_base.merge(comment_flag[['user_id', 'streamer_id', 'live_id', 'has_comment']], 
                                   on=['user_id', 'streamer_id', 'live_id'], how='left')
    click_base['label_has_comment'] = click_base['has_comment'].fillna(0).astype(int)
    
    # Label 3 & 4: has_gift and gift_amount
    gift_agg = gift.groupby(['user_id', 'streamer_id', 'live_id']).agg({
        'gift_price': 'sum'
    }).reset_index().rename(columns={'gift_price': 'total_gift_price'})
    click_base = click_base.merge(gift_agg, on=['user_id', 'streamer_id', 'live_id'], how='left')
    click_base['total_gift_price'] = click_base['total_gift_price'].fillna(0)
    click_base['label_has_gift'] = (click_base['total_gift_price'] > 0).astype(int)
    click_base['label_gift_amount'] = np.log1p(click_base['total_gift_price'])
    
    log_message(f"Click base: {len(click_base):,} records")
    log_message(f"  has_comment rate: {click_base['label_has_comment'].mean()*100:.2f}%")
    log_message(f"  has_gift rate: {click_base['label_has_gift'].mean()*100:.2f}%")
    
    # Merge features
    df = click_base.merge(user_features, on='user_id', how='left')
    df = df.merge(streamer_features, on='streamer_id', how='left')
    df = df.merge(interaction, on=['user_id', 'streamer_id'], how='left')
    
    # Fill NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Encode categorical
    cat_columns = [
        'age', 'gender', 'device_brand', 'device_price', 'fans_num', 'follow_num',
        'accu_watch_live_cnt', 'accu_watch_live_duration',
        'streamer_gender', 'streamer_age', 'streamer_device_brand', 'streamer_device_price',
        'live_operation_tag', 'fans_user_num', 'fans_group_fans_num', 'follow_user_num',
        'accu_live_cnt', 'accu_live_duration', 'accu_play_cnt', 'accu_play_duration',
        'live_type', 'live_content_category'
    ]
    df = encode_categorical(df, cat_columns)
    
    log_message(f"Final dataset: {len(df):,} records, {df.shape[1]} columns")
    
    return df


def temporal_split(df: pd.DataFrame):
    """Split data by time."""
    log_message("Performing temporal split...")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    test_start = max_date - pd.Timedelta(days=6)
    val_start = test_start - pd.Timedelta(days=7)
    
    log_message(f"Train: {min_date} to {val_start - pd.Timedelta(days=1)}")
    log_message(f"Val: {val_start} to {test_start - pd.Timedelta(days=1)}")
    log_message(f"Test: {test_start} to {max_date}")
    
    train = df[df['date'] < val_start].copy()
    val = df[(df['date'] >= val_start) & (df['date'] < test_start)].copy()
    test = df[df['date'] >= test_start].copy()
    
    log_message(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
    log_message(f"Train gift rate: {train['label_has_gift'].mean()*100:.2f}%")
    
    return train, val, test


def get_feature_columns(df: pd.DataFrame):
    """Get feature columns."""
    exclude_cols = [
        'user_id', 'live_id', 'streamer_id', 'timestamp', 'timestamp_dt', 'date',
        'watch_live_time', 'has_comment', 'total_gift_price',
        'label_watch_time', 'label_has_comment', 'label_has_gift', 'label_gift_amount'
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


# ============ PyTorch Dataset and Models ============

class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning."""
    
    def __init__(self, df: pd.DataFrame, feature_cols: list):
        self.features = torch.FloatTensor(df[feature_cols].values)
        self.label_watch_time = torch.FloatTensor(df['label_watch_time'].values)
        self.label_has_comment = torch.FloatTensor(df['label_has_comment'].values)
        self.label_has_gift = torch.FloatTensor(df['label_has_gift'].values)
        self.label_gift_amount = torch.FloatTensor(df['label_gift_amount'].values)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'watch_time': self.label_watch_time[idx],
            'has_comment': self.label_has_comment[idx],
            'has_gift': self.label_has_gift[idx],
            'gift_amount': self.label_gift_amount[idx]
        }


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class MultiTaskMLP(nn.Module):
    """Multi-Task MLP with shared tower and task-specific heads."""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], head_dim=32, 
                 dropout=0.2, tasks=['watch_time', 'has_comment', 'has_gift', 'gift_amount']):
        super().__init__()
        self.tasks = tasks
        
        # Shared tower
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        self.shared_tower = nn.Sequential(*layers)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task in tasks:
            self.task_heads[task] = nn.Sequential(
                nn.Linear(hidden_dims[-1], head_dim),
                nn.ReLU(),
                nn.Linear(head_dim, 1)
            )
    
    def forward(self, x):
        shared = self.shared_tower(x)
        outputs = {}
        for task in self.tasks:
            outputs[task] = self.task_heads[task](shared).squeeze(-1)
        return outputs


def train_epoch(model, dataloader, optimizer, task_weights, device, tasks):
    """Train one epoch."""
    model.train()
    total_loss = 0
    task_losses = {t: 0 for t in tasks}
    
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    for batch in dataloader:
        features = batch['features'].to(device)
        optimizer.zero_grad()
        
        outputs = model(features)
        
        loss = 0
        for task in tasks:
            target = batch[task].to(device)
            
            if task == 'watch_time':
                t_loss = mse_loss(outputs[task], target)
            elif task == 'has_comment':
                t_loss = bce_loss(outputs[task], target)
            elif task == 'has_gift':
                t_loss = focal_loss(outputs[task], target)
            elif task == 'gift_amount':
                # Only compute loss for gift samples
                mask = batch['has_gift'].to(device) > 0.5
                if mask.sum() > 0:
                    t_loss = mse_loss(outputs[task][mask], target[mask])
                else:
                    t_loss = torch.tensor(0.0, device=device)
            else:
                t_loss = torch.tensor(0.0, device=device)
            
            weighted_loss = task_weights.get(task, 0.25) * t_loss
            loss += weighted_loss
            task_losses[task] += t_loss.item()
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    n_batches = len(dataloader)
    return total_loss / n_batches, {t: v / n_batches for t, v in task_losses.items()}


def evaluate(model, dataloader, device, tasks):
    """Evaluate model."""
    model.eval()
    all_outputs = {t: [] for t in tasks}
    all_targets = {t: [] for t in tasks}
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            outputs = model(features)
            
            for task in tasks:
                all_outputs[task].extend(outputs[task].cpu().numpy())
                all_targets[task].extend(batch[task].numpy())
    
    return {t: np.array(all_outputs[t]) for t in tasks}, {t: np.array(all_targets[t]) for t in tasks}


def compute_metrics(outputs, targets):
    """Compute metrics for has_gift task."""
    y_pred = outputs['has_gift']
    y_true = targets['has_gift']
    
    # Sigmoid for probabilities
    y_pred_prob = 1 / (1 + np.exp(-y_pred))
    
    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    
    # ECE
    ece = compute_ece(y_true, y_pred_prob)
    
    # Top-K capture (using gift_amount as ranking signal)
    if 'gift_amount' in outputs:
        # Combine has_gift probability with amount prediction for ranking
        y_pred_amount = outputs['gift_amount']
        y_pred_ev = y_pred_prob * np.expm1(np.maximum(y_pred_amount, 0))
    else:
        y_pred_ev = y_pred_prob
    
    y_true_amount = np.expm1(targets['gift_amount']) if 'gift_amount' in targets else y_true
    
    top_1pct = compute_capture_rate(y_true_amount, y_pred_ev, 0.01)
    top_5pct = compute_capture_rate(y_true_amount, y_pred_ev, 0.05)
    top_10pct = compute_capture_rate(y_true_amount, y_pred_ev, 0.10)
    
    # Spearman
    spearman, _ = stats.spearmanr(y_true_amount, y_pred_ev)
    
    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'ece': ece,
        'top_1pct_capture': top_1pct,
        'top_5pct_capture': top_5pct,
        'top_10pct_capture': top_10pct,
        'spearman': spearman
    }


def compute_ece(y_true, y_pred_prob, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    
    for i in range(n_bins):
        mask = (y_pred_prob >= bin_edges[i]) & (y_pred_prob < bin_edges[i+1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred_prob[mask].mean()
            ece += (mask.sum() / total) * np.abs(bin_acc - bin_conf)
    
    return ece


def compute_capture_rate(y_true, y_pred, top_pct):
    """Compute Top-K% capture rate."""
    n = len(y_true)
    k = max(1, int(n * top_pct))
    
    true_rank = np.argsort(np.argsort(-y_true))
    pred_rank = np.argsort(np.argsort(-y_pred))
    
    true_topk = set(np.where(true_rank < k)[0])
    pred_topk = set(np.where(pred_rank < k)[0])
    
    if len(true_topk) == 0:
        return 0.0
    return len(true_topk & pred_topk) / len(true_topk)


# ============ Training Pipeline ============

def train_model(train_loader, val_loader, input_dim, tasks, task_weights, 
                epochs=50, lr=1e-3, patience=10, model_name="multitask"):
    """Train a model with early stopping."""
    log_message(f"Training {model_name} model...")
    log_message(f"Tasks: {tasks}")
    log_message(f"Device: {DEVICE}")
    
    model = MultiTaskMLP(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        head_dim=32,
        dropout=0.2,
        tasks=tasks
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    best_pr_auc = 0
    best_epoch = 0
    no_improve = 0
    history = {'train_loss': [], 'val_pr_auc': [], 'task_losses': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss, task_losses = train_epoch(model, train_loader, optimizer, task_weights, DEVICE, tasks)
        
        # Evaluate on validation
        outputs, targets = evaluate(model, val_loader, DEVICE, tasks)
        metrics = compute_metrics(outputs, targets)
        
        history['train_loss'].append(train_loss)
        history['val_pr_auc'].append(metrics['pr_auc'])
        history['task_losses'].append(task_losses)
        
        scheduler.step(metrics['pr_auc'])
        
        if metrics['pr_auc'] > best_pr_auc:
            best_pr_auc = metrics['pr_auc']
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log_message(f"Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Val PR-AUC={metrics['pr_auc']:.4f}")
        
        if no_improve >= patience:
            log_message(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    log_message(f"Best epoch: {best_epoch+1}, Best PR-AUC: {best_pr_auc:.4f}", "SUCCESS")
    log_message(f"Training time: {training_time:.1f}s")
    
    return model, history, training_time, best_epoch


# ============ Plotting Functions ============

def plot_pr_auc_comparison(single_metrics, multi_metrics, save_path):
    """Fig1: PR-AUC comparison bar chart."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    models = ['Single-Task\n(has_gift only)', 'Multi-Task\n(4 tasks)']
    values = [single_metrics['pr_auc'], multi_metrics['pr_auc']]
    colors = ['steelblue', 'coral']
    
    bars = ax.bar(models, values, color=colors, alpha=0.8)
    
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    delta = multi_metrics['pr_auc'] - single_metrics['pr_auc']
    delta_pct = delta * 100
    ax.set_title(f'PR-AUC Comparison\n(Δ={delta_pct:+.2f} pp)', fontsize=12)
    ax.set_ylabel('PR-AUC', fontsize=11)
    ax.set_ylim(0, max(values) * 1.15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_training_curves(single_history, multi_history, save_path):
    """Fig2: Training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss
    axes[0].plot(single_history['train_loss'], 'steelblue', label='Single-Task', lw=2)
    axes[0].plot(multi_history['train_loss'], 'coral', label='Multi-Task', lw=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Training Loss', fontsize=11)
    axes[0].set_title('Training Loss', fontsize=12)
    axes[0].legend()
    
    # Validation PR-AUC
    axes[1].plot(single_history['val_pr_auc'], 'steelblue', label='Single-Task', lw=2)
    axes[1].plot(multi_history['val_pr_auc'], 'coral', label='Multi-Task', lw=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Validation PR-AUC', fontsize=11)
    axes[1].set_title('Validation PR-AUC', fontsize=12)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_topk_capture(single_metrics, multi_metrics, save_path):
    """Fig3: Top-K% capture comparison."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    metrics = ['Top-1%', 'Top-5%', 'Top-10%']
    single_vals = [
        single_metrics['top_1pct_capture'] * 100,
        single_metrics['top_5pct_capture'] * 100,
        single_metrics['top_10pct_capture'] * 100
    ]
    multi_vals = [
        multi_metrics['top_1pct_capture'] * 100,
        multi_metrics['top_5pct_capture'] * 100,
        multi_metrics['top_10pct_capture'] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, single_vals, width, label='Single-Task', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, multi_vals, width, label='Multi-Task', color='coral', alpha=0.8)
    
    for bar, val in zip(bars1, single_vals):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, multi_vals):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Capture Rate (%)', fontsize=11)
    ax.set_title('Top-K% Capture Rate Comparison', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_all_metrics(single_metrics, multi_metrics, save_path):
    """Fig4: All metrics comparison."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    metrics_names = ['PR-AUC', 'ROC-AUC', 'Spearman']
    single_vals = [
        single_metrics['pr_auc'] * 100,
        single_metrics['roc_auc'] * 100,
        single_metrics['spearman'] * 100
    ]
    multi_vals = [
        multi_metrics['pr_auc'] * 100,
        multi_metrics['roc_auc'] * 100,
        multi_metrics['spearman'] * 100
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, single_vals, width, label='Single-Task', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, multi_vals, width, label='Multi-Task', color='coral', alpha=0.8)
    
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title('All Metrics Comparison', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_task_loss_evolution(multi_history, save_path):
    """Fig5: Multi-task loss evolution."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    tasks = list(multi_history['task_losses'][0].keys())
    colors = {'watch_time': 'green', 'has_comment': 'blue', 'has_gift': 'red', 'gift_amount': 'purple'}
    
    for task in tasks:
        losses = [epoch_losses[task] for epoch_losses in multi_history['task_losses']]
        ax.plot(losses, label=task, color=colors.get(task, 'gray'), lw=2)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Multi-Task: Per-Task Loss Evolution', fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def save_results(single_metrics, multi_metrics, single_time, multi_time, 
                 single_best_epoch, multi_best_epoch, data_stats, save_path):
    """Save results to JSON."""
    delta_pr_auc = multi_metrics['pr_auc'] - single_metrics['pr_auc']
    delta_pct = delta_pr_auc * 100
    
    # Decision rule: PR-AUC improvement ≥ 3%
    conclusion = "accept" if delta_pct >= 3.0 else "reject"
    
    results = {
        "experiment_id": "EXP-20260108-gift-allocation-06",
        "mvp": "MVP-1.3",
        "timestamp": datetime.now().isoformat(),
        "data": data_stats,
        "single_task": {k: float(v) for k, v in single_metrics.items()},
        "multi_task": {k: float(v) for k, v in multi_metrics.items()},
        "delta": {
            "pr_auc": float(delta_pr_auc),
            "pr_auc_pct_points": float(delta_pct),
            "top_1pct": float(multi_metrics['top_1pct_capture'] - single_metrics['top_1pct_capture']),
            "top_5pct": float(multi_metrics['top_5pct_capture'] - single_metrics['top_5pct_capture']),
        },
        "training": {
            "single_task_time_seconds": float(single_time),
            "single_task_best_epoch": int(single_best_epoch),
            "multi_task_time_seconds": float(multi_time),
            "multi_task_best_epoch": int(multi_best_epoch),
            "device": str(DEVICE)
        },
        "decision": {
            "conclusion": conclusion,
            "rule": "If PR-AUC Δ ≥ 3 percentage points → accept multi-task",
            "dg5_status": "CLOSED" if conclusion == "reject" else "VALIDATED"
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log_message(f"Results saved to: {save_path}", "SUCCESS")
    return results


# ============ Main ============

def main():
    log_message("=" * 70)
    log_message("Multi-Task Learning Experiment - MVP-1.3")
    log_message("EXP-20260108-gift-allocation-06")
    log_message("=" * 70)
    
    # Load data
    gift, user, streamer, room, click, comment, like = load_data()
    
    # Prepare data
    df = prepare_multitask_data(gift, user, streamer, room, click, comment, like)
    
    # Temporal split
    train, val, test = temporal_split(df)
    
    # Feature columns
    feature_cols = get_feature_columns(df)
    log_message(f"Number of features: {len(feature_cols)}")
    
    # Data stats for results
    data_stats = {
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "n_features": len(feature_cols),
        "gift_rate": float(train['label_has_gift'].mean()),
        "comment_rate": float(train['label_has_comment'].mean())
    }
    
    # Create datasets
    train_dataset = MultiTaskDataset(train, feature_cols)
    val_dataset = MultiTaskDataset(val, feature_cols)
    test_dataset = MultiTaskDataset(test, feature_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False, num_workers=0)
    
    input_dim = len(feature_cols)
    
    # ============ Single-Task Model (has_gift only) ============
    log_message("=" * 50)
    log_message("TRAINING SINGLE-TASK MODEL (has_gift only)")
    log_message("=" * 50)
    
    single_tasks = ['has_gift', 'gift_amount']  # Need gift_amount for ranking
    single_weights = {'has_gift': 0.8, 'gift_amount': 0.2}
    
    single_model, single_history, single_time, single_best_epoch = train_model(
        train_loader, val_loader, input_dim,
        tasks=single_tasks, task_weights=single_weights,
        epochs=50, lr=1e-3, patience=10, model_name="single_task"
    )
    
    # Evaluate single-task on test
    single_outputs, single_targets = evaluate(single_model, test_loader, DEVICE, single_tasks)
    single_metrics = compute_metrics(single_outputs, single_targets)
    log_message(f"Single-Task Test PR-AUC: {single_metrics['pr_auc']:.4f}")
    log_message(f"Single-Task Test Top-1%: {single_metrics['top_1pct_capture']*100:.1f}%")
    
    # ============ Multi-Task Model (all 4 tasks) ============
    log_message("=" * 50)
    log_message("TRAINING MULTI-TASK MODEL (4 tasks)")
    log_message("=" * 50)
    
    multi_tasks = ['watch_time', 'has_comment', 'has_gift', 'gift_amount']
    multi_weights = {'watch_time': 0.3, 'has_comment': 0.2, 'has_gift': 0.3, 'gift_amount': 0.2}
    
    multi_model, multi_history, multi_time, multi_best_epoch = train_model(
        train_loader, val_loader, input_dim,
        tasks=multi_tasks, task_weights=multi_weights,
        epochs=50, lr=1e-3, patience=10, model_name="multi_task"
    )
    
    # Evaluate multi-task on test
    multi_outputs, multi_targets = evaluate(multi_model, test_loader, DEVICE, multi_tasks)
    multi_metrics = compute_metrics(multi_outputs, multi_targets)
    log_message(f"Multi-Task Test PR-AUC: {multi_metrics['pr_auc']:.4f}")
    log_message(f"Multi-Task Test Top-1%: {multi_metrics['top_1pct_capture']*100:.1f}%")
    
    # ============ Generate Figures ============
    log_message("Generating figures...")
    
    plot_pr_auc_comparison(single_metrics, multi_metrics, IMG_DIR / "mvp13_pr_auc_comparison.png")
    plot_training_curves(single_history, multi_history, IMG_DIR / "mvp13_training_curves.png")
    plot_topk_capture(single_metrics, multi_metrics, IMG_DIR / "mvp13_topk_capture.png")
    plot_all_metrics(single_metrics, multi_metrics, IMG_DIR / "mvp13_all_metrics.png")
    plot_task_loss_evolution(multi_history, IMG_DIR / "mvp13_task_loss_evolution.png")
    
    # ============ Save Results ============
    results = save_results(
        single_metrics, multi_metrics,
        single_time, multi_time,
        single_best_epoch, multi_best_epoch,
        data_stats,
        RESULTS_DIR / "multitask_results_20260108.json"
    )
    
    # ============ Print Summary ============
    print("\n" + "=" * 70)
    print("📊 EXPERIMENT SUMMARY - MVP-1.3: Multi-Task Learning")
    print("=" * 70)
    print(f"Data: Click-based with multi-task labels")
    print(f"Train/Val/Test: {data_stats['train_size']:,} / {data_stats['val_size']:,} / {data_stats['test_size']:,}")
    print(f"Gift rate: {data_stats['gift_rate']*100:.2f}%, Comment rate: {data_stats['comment_rate']*100:.2f}%")
    print("-" * 70)
    print("SINGLE-TASK MODEL (has_gift + gift_amount):")
    print(f"  PR-AUC:          {single_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:         {single_metrics['roc_auc']:.4f}")
    print(f"  ECE:             {single_metrics['ece']:.4f}")
    print(f"  Top-1% Capture:  {single_metrics['top_1pct_capture']*100:.1f}%")
    print(f"  Top-5% Capture:  {single_metrics['top_5pct_capture']*100:.1f}%")
    print(f"  Spearman:        {single_metrics['spearman']:.4f}")
    print(f"  Training time:   {single_time:.1f}s")
    print("-" * 70)
    print("MULTI-TASK MODEL (4 tasks):")
    print(f"  PR-AUC:          {multi_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:         {multi_metrics['roc_auc']:.4f}")
    print(f"  ECE:             {multi_metrics['ece']:.4f}")
    print(f"  Top-1% Capture:  {multi_metrics['top_1pct_capture']*100:.1f}%")
    print(f"  Top-5% Capture:  {multi_metrics['top_5pct_capture']*100:.1f}%")
    print(f"  Spearman:        {multi_metrics['spearman']:.4f}")
    print(f"  Training time:   {multi_time:.1f}s")
    print("-" * 70)
    print("COMPARISON (Multi - Single):")
    delta_pr = results['delta']['pr_auc_pct_points']
    delta_top1 = results['delta']['top_1pct'] * 100
    print(f"  Δ PR-AUC:        {delta_pr:+.2f} pp {'✅' if delta_pr >= 3 else '❌'} (threshold: ≥3pp)")
    print(f"  Δ Top-1%:        {delta_top1:+.1f} pp")
    print("-" * 70)
    print(f"🎯 DECISION: {results['decision']['conclusion'].upper()}")
    print(f"   DG5 Status: {results['decision']['dg5_status']}")
    if results['decision']['conclusion'] == 'accept':
        print("   → Multi-task learning shows significant improvement!")
    else:
        print("   → Multi-task improvement < 3pp. Keep single-task approach.")
    print("=" * 70)
    
    log_message("Experiment completed!", "SUCCESS")
    
    return results


if __name__ == "__main__":
    results = main()
