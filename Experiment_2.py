# ========================================
# GRAFT: Ablation Study
# Testing importance of Stochastic Gates and Residual MLP
# Variants: Full GRAFT, No STG, Linear Only
# Datasets: GBSG, METABRIC, SUPPORT, NWTCO, FLCHAIN, AIDS
# Noise levels: 0×, 3×, 5×, 7×, 10×
# 3-fold CV with 3 random seeds
# GPU-enabled
# ========================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from pycox.datasets import gbsg, metabric, support
import torch
import torch.nn as nn
import torch.optim as optim
import torchsort
import matplotlib.pyplot as plt
import warnings
import math
warnings.filterwarnings('ignore')

# GPU Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("GRAFT: Ablation Study")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("="*80)

# ========================================
# Helper Functions
# ========================================

def preprocess_df_with_noise(df, duration_col, event_col, drop_cols=None, noise_multiplier=0, seed=42):
    """Preprocess survival dataset with optional noise features"""
    df = df.copy()
    
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    
    df = df[df[duration_col] > 0].copy()
    
    durations = df[duration_col].astype(float).values
    events = df[event_col].astype(int).values
    X = df.drop(columns=[duration_col, event_col])
    
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    num_data = X[num_cols].copy().fillna(X[num_cols].median())
    
    if len(cat_cols) > 0:
        cat_data = pd.get_dummies(X[cat_cols].astype(str), drop_first=True)
        X_proc = pd.concat([num_data.reset_index(drop=True), cat_data.reset_index(drop=True)], axis=1)
    else:
        X_proc = num_data
    
    # Add noise features before standardization
    n_original_features = X_proc.shape[1]
    if noise_multiplier > 0:
        n_noise_features = int(n_original_features * noise_multiplier)
        np.random.seed(seed)
        noise_features = np.random.randn(len(X_proc), n_noise_features)
        noise_df = pd.DataFrame(noise_features, columns=[f'noise_{i}' for i in range(n_noise_features)])
        X_proc = pd.concat([X_proc.reset_index(drop=True), noise_df.reset_index(drop=True)], axis=1)
    
    # Standardize everything together
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_proc)
    
    all_cols = X_proc.columns.tolist()
    
    return X_scaled, durations, events, all_cols, n_original_features

def harrell_c_index(scores, durations, events):
    """Compute C-index"""
    return concordance_index(durations, scores, events)

def integrated_brier_ipcw(t_tr, e_tr, t_te, e_te, S_pred, times_grid):
    """Compute Integrated Brier Score with IPCW"""
    km_cens = KaplanMeierFitter()
    km_cens.fit(t_tr, 1 - e_tr)
    
    brier_scores = []
    for i, t in enumerate(times_grid):
        G_t = km_cens.survival_function_at_times(t).values[0] if t <= t_tr.max() else km_cens.survival_function_at_times(t_tr.max()).values[0]
        if G_t < 0.01:
            continue
        
        bs = 0.0
        for j in range(len(t_te)):
            S_pred_t = S_pred[j, i]
            if t_te[j] <= t and e_te[j] == 1:
                G_tj = km_cens.survival_function_at_times(t_te[j]).values[0]
                if G_tj > 0.01:
                    bs += (0 - S_pred_t) ** 2 / G_tj
            elif t_te[j] > t:
                bs += (1 - S_pred_t) ** 2 / G_t
        
        bs /= len(t_te)
        brier_scores.append(bs)
    
    if len(brier_scores) == 0:
        return np.nan
    
    ibs = np.trapz(brier_scores, times_grid[:len(brier_scores)])
    ibs /= (times_grid[len(brier_scores)-1] - times_grid[0])
    return ibs

# ========================================
# Local KM Imputation Functions
# ========================================

def find_adaptive_neighborhoods(
    X, times, events,
    k_min=30, min_events=10, max_k=200,
    metric="euclidean"
):
    """Pre-compute adaptive neighborhoods for all censored individuals"""
    n, d = X.shape
    censored_indices = np.where(events == 0)[0]

    nn_model = NearestNeighbors(
        n_neighbors=min(n, max_k),
        metric=metric
    )

    nn_model.fit(X)
    distances, indices = nn_model.kneighbors(X)

    neighborhoods = {}
    for i in censored_indices:
        neighbors = []
        n_events = 0

        for neighbor_idx in indices[i]:
            if neighbor_idx == i:
                continue

            neighbors.append(neighbor_idx)
            if events[neighbor_idx] == 1:
                n_events += 1

            if len(neighbors) >= k_min and n_events >= min_events:
                break

        neighborhoods[i] = neighbors

    return neighborhoods

def prefit_local_km_curves(times, events, neighborhoods):
    """Pre-fit all KM curves for speed optimization"""
    km_curves = {}
    
    for i, neighbor_idx in neighborhoods.items():
        t_neighbors = times[neighbor_idx]
        e_neighbors = events[neighbor_idx]
        
        km = KaplanMeierFitter()
        try:
            km.fit(t_neighbors, e_neighbors)
            km_curves[i] = km
        except:
            km_curves[i] = None
    
    return km_curves

def sample_from_conditional_km(km_fit, t_cens, n_samples=1):
    """Sample event times from conditional KM: S(t | T > t_cens)"""
    sf = km_fit.survival_function_
    km_times = sf.index.values
    km_surv = sf.values.flatten()
    
    S_at_cens = np.interp(t_cens, km_times, km_surv)
    
    if S_at_cens < 1e-6:
        return np.full(n_samples, t_cens * 1.1)
    
    mask = km_times > t_cens
    if not mask.any():
        return np.full(n_samples, t_cens * 1.1)
    
    conditional_times = km_times[mask]
    conditional_surv = km_surv[mask] / S_at_cens
    
    pdf = -np.diff(np.concatenate([[1.0], conditional_surv]))
    pdf = np.maximum(pdf, 0)
    pdf_sum = pdf.sum()
    
    if pdf_sum <= 0:
        return np.full(n_samples, t_cens * 1.1)
    
    pdf = pdf / pdf_sum
    sampled_times = np.random.choice(conditional_times, size=n_samples, p=pdf)
    
    return sampled_times

def impute_minibatch_fast(batch_indices, times_full, events_full, km_curves, M=5, device='cpu'):
    """Generate M imputed log-time arrays for minibatch using pre-fitted KM curves"""
    batch_size = len(batch_indices)
    Y_star_samples = []
    
    for m in range(M):
        Y_star_batch = np.zeros(batch_size)
        
        for local_idx, global_idx in enumerate(batch_indices):
            t_i = times_full[global_idx]
            e_i = events_full[global_idx]
            
            if e_i == 1:
                Y_star_batch[local_idx] = np.log(t_i + 1e-8)
            else:
                if global_idx in km_curves and km_curves[global_idx] is not None:
                    km = km_curves[global_idx]
                    try:
                        sampled_time = sample_from_conditional_km(km, t_i, n_samples=1)[0]
                        Y_star_batch[local_idx] = np.log(sampled_time + 1e-8)
                    except:
                        Y_star_batch[local_idx] = np.log(t_i * 1.1 + 1e-8)
                else:
                    Y_star_batch[local_idx] = np.log(t_i * 1.1 + 1e-8)
        
        Y_star_samples.append(torch.tensor(Y_star_batch, dtype=torch.float32, device=device))
    
    return Y_star_samples

# ========================================
# Stochastic Gates with Gaussian Distribution
# ========================================

class StochasticGate(nn.Module):
    """
    Gaussian-based Stochastic Gates
    Gates are global (shared across all samples in batch)
    """
    def __init__(self, n_features, sigma=0.5):
        super().__init__()
        assert sigma > 0, "sigma must be positive"
        self.n_features = n_features
        self.sigma = sigma
        self.mu = nn.Parameter(torch.full((n_features,), 0.5))
    
    def sample_z(self, batch_size, training=True):
        """Sample gate values (global across batch for feature selection)"""
        if training:
            epsilon = torch.randn(self.n_features, device=self.mu.device) * self.sigma
            z = torch.clamp(self.mu + epsilon, 0.0, 1.0)
            z = z.unsqueeze(0).expand(batch_size, -1)
        else:
            z = torch.clamp(self.mu, 0.0, 1.0)
            z = z.unsqueeze(0).expand(batch_size, -1)
        return z
    
    def regularization(self, reg_weight=1.0):
        """Compute sparsity regularization"""
        standardized_mu = self.mu / self.sigma
        prob_active = 0.5 * (1 + torch.erf(standardized_mu / math.sqrt(2)))
        l0_approx = prob_active.sum()
        return reg_weight * l0_approx

# ========================================
# GRAFT Model Variants
# ========================================

def soft_rank_loss_torchsort(scores, Y_star, regularization_strength=0.1):
    """Differentiable ranking loss using torchsort"""
    soft_ranks_pred = torchsort.soft_rank(
        scores.unsqueeze(0),
        regularization_strength=regularization_strength
    ).squeeze(0)
    
    Y_star_ranks = torch.argsort(torch.argsort(Y_star, descending=False)).float() + 1
    
    sr_c = soft_ranks_pred - soft_ranks_pred.mean()
    yr_c = Y_star_ranks - Y_star_ranks.mean()
    corr = torch.dot(sr_c, yr_c) / (torch.norm(sr_c) * torch.norm(yr_c) + 1e-8)
    
    return -corr

# Variant 1: Full GRAFT (with STG and MLP)
class GRAFT_Full(nn.Module):
    def __init__(self, n_features, hidden_dim=32, sigma=0.5, dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.stochastic_gates = StochasticGate(
            n_features=n_features,
            sigma=sigma
        )
        self.f_mlp = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features)
        )
        self.beta = nn.Parameter(torch.randn(n_features) * 0.1)
        self.mu = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, training=True):
        batch_size = x.shape[0]
        gates = self.stochastic_gates.sample_z(batch_size, training=training)
        gated_x = gates * x
        f_x = self.f_mlp(gated_x)
        gated_features = gated_x + f_x
        score = torch.sum(self.beta * gated_features, dim=1) + self.mu
        return score
    
    def regularization(self, reg_weight=0.01):
        return self.stochastic_gates.regularization(reg_weight=reg_weight)

# Variant 2: GRAFT without STG (no gates, but has MLP)
class GRAFT_NoSTG(nn.Module):
    def __init__(self, n_features, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.f_mlp = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features)
        )
        self.beta = nn.Parameter(torch.randn(n_features) * 0.1)
        self.mu = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, training=True):
        f_x = self.f_mlp(x)
        features = x + f_x
        score = torch.sum(self.beta * features, dim=1) + self.mu
        return score
    
    def regularization(self, reg_weight=0.01):
        return 0.0

# Variant 3: GRAFT without STG and MLP (just linear)
class GRAFT_Linear(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.beta = nn.Parameter(torch.randn(n_features) * 0.1)
        self.mu = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, training=True):
        score = torch.sum(self.beta * x, dim=1) + self.mu
        return score
    
    def regularization(self, reg_weight=0.01):
        return 0.0

def train_graft_variant(X_tr, t_tr, e_tr, model_type='full', hidden_dim=32, n_epochs=1000, 
                       batch_size=64, lr=1e-3, l2=1e-4, sigma=0.5, 
                       reg_weight=0.01, regularization_strength=0.1, dropout=0.2, 
                       M=5, k_min=30, min_events=10, patience=8, device=None):
    """Train GRAFT variant with Local KM imputation and early stopping"""
    
    if device is None:
        device = DEVICE
    
    n_samples = len(X_tr)
    
    # Pre-compute neighborhoods and KM curves
    neighborhoods = find_adaptive_neighborhoods(X_tr, t_tr, e_tr, k_min=k_min, 
                                               metric="euclidean", min_events=min_events)
    km_curves = prefit_local_km_curves(t_tr, e_tr, neighborhoods)
    
    # Create appropriate model
    if model_type == 'full':
        model = GRAFT_Full(
            n_features=X_tr.shape[1],
            hidden_dim=hidden_dim,
            sigma=sigma,
            dropout=dropout
        ).to(device)
    elif model_type == 'no_stg':
        model = GRAFT_NoSTG(
            n_features=X_tr.shape[1],
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(device)
    elif model_type == 'linear':
        model = GRAFT_Linear(
            n_features=X_tr.shape[1]
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    n_batches_per_epoch = max(1, n_samples // batch_size)
    
    # Early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        model.train()
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        
        for batch_idx in range(n_batches_per_epoch):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = perm[start_idx:end_idx]
            
            X_batch = X_tr[batch_indices]
            Y_star_samples = impute_minibatch_fast(batch_indices, t_tr, e_tr, km_curves, M=M, device=device)
            
            X_batch_t = torch.tensor(X_batch, dtype=torch.float32, device=device)
            scores = model(X_batch_t, training=True)
            
            ranking_losses = []
            for Y_star_t in Y_star_samples:
                loss_m = soft_rank_loss_torchsort(scores, Y_star_t, 
                                                 regularization_strength=regularization_strength)
                ranking_losses.append(loss_m)
            
            ranking_loss = torch.stack(ranking_losses).mean()
            reg_loss = model.regularization(reg_weight=reg_weight)
            total_loss = ranking_loss + reg_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        epoch_loss /= n_batches_per_epoch
        
        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    model.eval()
    return model

def predict_graft(X, model):
    """Predict with GRAFT"""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        scores = model(X_t, training=False).cpu().numpy()
    return scores

# ========================================
# Load Datasets
# ========================================

def load_all_datasets():
    """Load all 6 datasets"""
    datasets = {}
    
    print("\nLoading datasets...")
    
    # GBSG
    df_gbsg = gbsg.read_df()
    df_gbsg = df_gbsg[df_gbsg['duration'] > 0]
    datasets['GBSG'] = {
        'df': df_gbsg,
        'duration_col': 'duration',
        'event_col': 'event',
        'drop_cols': None
    }
    print(f"  GBSG: {len(df_gbsg)} samples")
    
    # METABRIC
    df_metabric = metabric.read_df()
    df_metabric = df_metabric[df_metabric['duration'] > 0]
    datasets['METABRIC'] = {
        'df': df_metabric,
        'duration_col': 'duration',
        'event_col': 'event',
        'drop_cols': None
    }
    print(f"  METABRIC: {len(df_metabric)} samples")
    
    # SUPPORT
    df_support = support.read_df()
    df_support = df_support[df_support['duration'] > 0]
    datasets['SUPPORT'] = {
        'df': df_support,
        'duration_col': 'duration',
        'event_col': 'event',
        'drop_cols': None
    }
    print(f"  SUPPORT: {len(df_support)} samples")
    
    # AIDS
    try:
        df_aids = pd.read_csv('aids.csv')
        df_aids = df_aids[df_aids['duration'] > 0]
        datasets['AIDS'] = {
            'df': df_aids,
            'duration_col': 'duration',
            'event_col': 'event',
            'drop_cols': ['Unnamed: 0']
        }
        print(f"  AIDS: {len(df_aids)} samples")
    except FileNotFoundError:
        print("  AIDS: File not found (aids.csv)")
    
    # FLCHAIN
    try:
        df_flchain = pd.read_csv('flchain_final.csv')
        df_flchain = df_flchain[df_flchain['duration'] > 0]
        datasets['FLCHAIN'] = {
            'df': df_flchain,
            'duration_col': 'duration',
            'event_col': 'event',
            'drop_cols': ['chapter', 'Unnamed: 0']
        }
        print(f"  FLCHAIN: {len(df_flchain)} samples")
    except FileNotFoundError:
        print("  FLCHAIN: File not found (flchain_final.csv)")
    
    # NWTCO
    try:
        df_nwtco = pd.read_csv('nwtco.csv')
        df_nwtco = df_nwtco[df_nwtco['edrel'] > 0]
        datasets['NWTCO'] = {
            'df': df_nwtco,
            'duration_col': 'edrel',
            'event_col': 'rel',
            'drop_cols': ['rownames', 'seqno', 'study', 'instit', 'in.subcohort']
        }
        print(f"  NWTCO: {len(df_nwtco)} samples")
    except FileNotFoundError:
        print("  NWTCO: File not found (nwtco.csv)")
    
    return datasets

# ========================================
# Display Functions
# ========================================

def display_single_dataset_results(dataset_name, results):
    """Display results for a single dataset"""
    print(f"\n{'='*100}")
    print(f"RESULTS: {dataset_name}")
    print(f"{'='*100}")
    
    variant_names = {
        'full': 'Full GRAFT',
        'no_stg': 'No STG',
        'linear': 'Linear Only'
    }
    
    noise_mults = results['noise_multipliers']
    variants = ['full', 'no_stg', 'linear']
    
    # C-Index table
    print("\nC-INDEX:")
    print("-" * 100)
    header = f"{'Noise':<10}"
    for variant in variants:
        header += f"{variant_names[variant]:>28}"
    print(header)
    print("-" * 100)
    
    for noise_mult in noise_mults:
        row = f"{noise_mult}×{'':<8}"
        for variant in variants:
            if noise_mult in results['variants'][variant]['cindex']:
                data = results['variants'][variant]['cindex'][noise_mult]
                row += f"{data['mean']:.4f} ± {data['std']:.4f}          "
            else:
                row += f"{'N/A':>28}"
        print(row)
    
    # IBS table
    print("\nINTEGRATED BRIER SCORE:")
    print("-" * 100)
    header = f"{'Noise':<10}"
    for variant in variants:
        header += f"{variant_names[variant]:>28}"
    print(header)
    print("-" * 100)
    
    for noise_mult in noise_mults:
        row = f"{noise_mult}×{'':<8}"
        for variant in variants:
            if noise_mult in results['variants'][variant]['ibs']:
                data = results['variants'][variant]['ibs'][noise_mult]
                row += f"{data['mean']:.4f} ± {data['std']:.4f}          "
            else:
                row += f"{'N/A':>28}"
        print(row)

def display_combined_results(all_results):
    """Display combined results for all datasets"""
    print("\n" + "="*120)
    print("COMBINED RESULTS: ALL DATASETS - ABLATION STUDY")
    print("="*120)
    
    datasets = list(all_results.keys())
    variants = ['full', 'no_stg', 'linear']
    variant_names = {
        'full': 'Full GRAFT',
        'no_stg': 'No STG',
        'linear': 'Linear Only'
    }
    
    noise_mults = all_results[datasets[0]]['noise_multipliers']
    
    for noise_mult in noise_mults:
        print(f"\n{'='*120}")
        print(f"NOISE LEVEL: {noise_mult}×")
        print(f"{'='*120}")
        
        # C-Index table
        print("\nC-INDEX:")
        print("-" * 120)
        header = "Dataset".ljust(15)
        for variant in variants:
            header += f"{variant_names[variant]}".center(25)
        print(header)
        print(" " * 15 + "Mean      Std      " * len(variants))
        print("-" * 120)
        
        for dataset in datasets:
            row = f"{dataset:15s}"
            for variant in variants:
                if noise_mult in all_results[dataset]['variants'][variant]['cindex']:
                    data = all_results[dataset]['variants'][variant]['cindex'][noise_mult]
                    row += f"{data['mean']:8.4f}  {data['std']:8.4f}     "
                else:
                    row += f"{'N/A':^25s}"
            print(row)
        
        # IBS table
        print("\nINTEGRATED BRIER SCORE:")
        print("-" * 120)
        header = "Dataset".ljust(15)
        for variant in variants:
            header += f"{variant_names[variant]}".center(25)
        print(header)
        print(" " * 15 + "Mean      Std      " * len(variants))
        print("-" * 120)
        
        for dataset in datasets:
            row = f"{dataset:15s}"
            for variant in variants:
                if noise_mult in all_results[dataset]['variants'][variant]['ibs']:
                    data = all_results[dataset]['variants'][variant]['ibs'][noise_mult]
                    row += f"{data['mean']:8.4f}  {data['std']:8.4f}     "
                else:
                    row += f"{'N/A':^25s}"
            print(row)

# ========================================
# Ablation Experiment
# ========================================

def run_ablation_experiment(datasets, noise_multipliers=[0, 3, 5, 7, 10], n_folds=3, seeds=[42, 43, 44]):
    """Run ablation experiment"""
    
    all_results = {}
    model_variants = ['full', 'no_stg', 'linear']
    
    for dataset_name in datasets.keys():
        print(f"\n{'='*80}")
        print(f"PROCESSING DATASET: {dataset_name}")
        print(f"{'='*80}")
        
        dataset_info = datasets[dataset_name]
        df = dataset_info['df']
        duration_col = dataset_info['duration_col']
        event_col = dataset_info['event_col']
        drop_cols = dataset_info['drop_cols']
        
        all_results[dataset_name] = {
            'noise_multipliers': noise_multipliers,
            'variants': {
                'full': {'cindex': {}, 'ibs': {}},
                'no_stg': {'cindex': {}, 'ibs': {}},
                'linear': {'cindex': {}, 'ibs': {}}
            }
        }
        
        for noise_mult in noise_multipliers:
            print(f"\n  Noise Multiplier: {noise_mult}×", end="")
            if noise_mult == 0:
                print(" (Baseline - No Noise)")
            else:
                print()
            
            seed_results = {
                'full': {'cindex': [], 'ibs': []},
                'no_stg': {'cindex': [], 'ibs': []},
                'linear': {'cindex': [], 'ibs': []}
            }
            
            for seed_idx, seed in enumerate(seeds):
                print(f"    Seed {seed}", end=" ")
                
                X, t, e, cols, n_orig = preprocess_df_with_noise(
                    df, duration_col, event_col,
                    drop_cols=drop_cols, 
                    noise_multiplier=noise_mult, 
                    seed=seed
                )
                
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
                
                fold_results = {
                    'full': {'cindex': [], 'ibs': []},
                    'no_stg': {'cindex': [], 'ibs': []},
                    'linear': {'cindex': [], 'ibs': []}
                }
                
                for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                    X_tr, X_te = X[train_idx], X[test_idx]
                    t_tr, t_te = t[train_idx], t[test_idx]
                    e_tr, e_te = e[train_idx], e[test_idx]
                    
                    for variant in model_variants:
                        try:
                            model = train_graft_variant(
                                X_tr, t_tr, e_tr, 
                                model_type=variant, 
                                dropout=0.2,
                                M=5, 
                                k_min=30, 
                                min_events=10,
                                batch_size=64,
                                sigma=0.5,
                                reg_weight=0.01,
                                n_epochs=1000,
                                patience=8,
                                device=DEVICE
                            )
                            scores = predict_graft(X_te, model)
                            cindex = harrell_c_index(scores, t_te, e_te)
                            
                            # Compute IBS
                            times_grid = np.linspace(t_te.min(), t_te.max(), 50)
                            scores_tr = predict_graft(X_tr, model)
                            temp_df = pd.DataFrame(X_tr, columns=cols)
                            temp_df['score'] = scores_tr
                            temp_df['duration'] = t_tr
                            temp_df['event'] = e_tr
                            cph_temp = CoxPHFitter()
                            cph_temp.fit(temp_df[['score', 'duration', 'event']], 
                                        duration_col='duration', event_col='event')
                            base_cumhaz = cph_temp.baseline_cumulative_hazard_
                            base_cumhaz_interp = np.interp(times_grid, base_cumhaz.index.values, 
                                                          base_cumhaz.values.flatten())
                            test_df_temp = pd.DataFrame(X_te, columns=cols)
                            test_df_temp['score'] = scores
                            linpred = cph_temp.predict_partial_hazard(test_df_temp[['score']]).values.flatten()
                            S_pred = np.exp(-np.outer(linpred, base_cumhaz_interp))
                            ibs = integrated_brier_ipcw(t_tr, e_tr, t_te, e_te, S_pred, times_grid)
                            
                            fold_results[variant]['cindex'].append(cindex)
                            fold_results[variant]['ibs'].append(ibs)
                        except Exception as ex:
                            print(f"[{variant} err]", end=" ")
                            pass
                
                # Average across folds for this seed
                for variant in model_variants:
                    if len(fold_results[variant]['cindex']) > 0:
                        seed_results[variant]['cindex'].append(np.mean(fold_results[variant]['cindex']))
                    if len(fold_results[variant]['ibs']) > 0:
                        valid_ibs = [x for x in fold_results[variant]['ibs'] if not np.isnan(x)]
                        if valid_ibs:
                            seed_results[variant]['ibs'].append(np.mean(valid_ibs))
                
                print("✓")
            
            # Store mean and std across seeds for this noise level
            for variant in model_variants:
                if len(seed_results[variant]['cindex']) > 0:
                    all_results[dataset_name]['variants'][variant]['cindex'][noise_mult] = {
                        'mean': np.mean(seed_results[variant]['cindex']),
                        'std': np.std(seed_results[variant]['cindex']),
                        'values': seed_results[variant]['cindex']
                    }
                if len(seed_results[variant]['ibs']) > 0:
                    all_results[dataset_name]['variants'][variant]['ibs'][noise_mult] = {
                        'mean': np.mean(seed_results[variant]['ibs']),
                        'std': np.std(seed_results[variant]['ibs']),
                        'values': seed_results[variant]['ibs']
                    }
        
        display_single_dataset_results(dataset_name, all_results[dataset_name])
    
    return all_results

# ========================================
# Visualization
# ========================================

def plot_ablation_results(all_results):
    """Create summary plots for ablation study"""
    
    datasets_list = list(all_results.keys())
    variants = ['full', 'no_stg', 'linear']
    variant_names = {
        'full': 'Full GRAFT',
        'no_stg': 'No STG',
        'linear': 'Linear Only'
    }
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    markers = ['o', 's', '^']
    
    # Combined plot - all datasets C-Index
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, dataset_name in enumerate(datasets_list):
        ax = axes[idx]
        noise_mults = all_results[dataset_name]['noise_multipliers']
        
        for variant_idx, variant in enumerate(variants):
            means = []
            stds = []
            for noise_mult in noise_mults:
                if noise_mult in all_results[dataset_name]['variants'][variant]['cindex']:
                    data = all_results[dataset_name]['variants'][variant]['cindex'][noise_mult]
                    means.append(data['mean'])
                    stds.append(data['std'])
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            
            ax.errorbar(noise_mults, means, yerr=stds, 
                       label=variant_names[variant], color=colors[variant_idx], marker=markers[variant_idx],
                       markersize=8, linewidth=2, capsize=4, capthick=1.5, alpha=0.8)
        
        ax.set_xlabel('Noise Multiplier', fontsize=11, fontweight='bold')
        ax.set_ylabel('C-Index', fontsize=11, fontweight='bold')
        ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(noise_mults)
        ax.set_xticklabels([f'{x}×' for x in noise_mults])
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ablation_cindex_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("  Saved: ablation_cindex_summary.png")
    
    # Combined plot - all datasets IBS
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, dataset_name in enumerate(datasets_list):
        ax = axes[idx]
        noise_mults = all_results[dataset_name]['noise_multipliers']
        
        for variant_idx, variant in enumerate(variants):
            means = []
            stds = []
            for noise_mult in noise_mults:
                if noise_mult in all_results[dataset_name]['variants'][variant]['ibs']:
                    data = all_results[dataset_name]['variants'][variant]['ibs'][noise_mult]
                    means.append(data['mean'])
                    stds.append(data['std'])
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            
            ax.errorbar(noise_mults, means, yerr=stds, 
                       label=variant_names[variant], color=colors[variant_idx], marker=markers[variant_idx],
                       markersize=8, linewidth=2, capsize=4, capthick=1.5, alpha=0.8)
        
        ax.set_xlabel('Noise Multiplier', fontsize=11, fontweight='bold')
        ax.set_ylabel('Integrated Brier Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(noise_mults)
        ax.set_xticklabels([f'{x}×' for x in noise_mults])
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ablation_ibs_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("  Saved: ablation_ibs_summary.png")

# ========================================
# Main Execution
# ========================================

if __name__ == "__main__":
    
    datasets = load_all_datasets()
    
    print("\n" + "="*80)
    print("Starting ablation experiment")
    print("- 6 datasets: GBSG, METABRIC, SUPPORT, AIDS, FLCHAIN, NWTCO")
    print("- 3 variants: Full GRAFT, No STG, Linear Only")
    print("- 3 seeds: 42, 43, 44")
    print("- Noise levels: 0×, 3×, 5×, 7×, 10×")
    print("- Early stopping: 1000 epochs, patience 8")
    print("- Local KM imputation with Euclidean distance")
    print("="*80)
    
    all_results = run_ablation_experiment(datasets, 
                                         noise_multipliers=[0, 3, 5, 7, 10], 
                                         n_folds=3, 
                                         seeds=[42, 43, 44])
    
    display_combined_results(all_results)
    
    print("\n" + "="*80)
    print("Creating summary plots...")
    print("="*80)
    plot_ablation_results(all_results)
    
    print("\n" + "="*80)
    print("ABLATION EXPERIMENT COMPLETE")
    print("="*80)
