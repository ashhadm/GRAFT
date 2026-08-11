# ========================================
# GRAFT: Comprehensive Baseline Comparison
# Models: GRAFT, CoxPH, Weibull, DeepHit, DeepSurv, RSF
# Datasets: GBSG, METABRIC, SUPPORT, AIDS, FLCHAIN, NWTCO
# 3-fold CV with 3 random seeds
# GPU-enabled
# ========================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from lifelines import CoxPHFitter, WeibullAFTFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import torch
import torch.nn as nn
import torch.optim as optim
import torchsort
import torchtuples as tt
from pycox.datasets import gbsg, metabric, support
from pycox.models import DeepHitSingle, CoxPH as PyCoxCoxPH
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import warnings
import math
warnings.filterwarnings('ignore')

# GPU Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("GRAFT: Comprehensive Baseline Comparison")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("="*80)

# ========================================
# Helper Functions
# ========================================

def preprocess_df(df, duration_col, event_col, drop_cols=None):
    """Preprocess survival dataset"""
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
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_proc)
    
    all_cols = X_proc.columns.tolist()
    return X_scaled, durations, events, all_cols

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
    metric="euclidian",
    shrinkage=0.1,
):
    """Pre-compute adaptive neighborhoods for all censored individuals"""
    n, d = X.shape
    censored_indices = np.where(events == 0)[0]

    if metric == "mahalanobis":
        cov = np.cov(X, rowvar=False)
        alpha = float(shrinkage)
        cov_reg = (1.0 - alpha) * cov + alpha * np.eye(d)
        VI = np.linalg.pinv(cov_reg)
        nn_model = NearestNeighbors(
            n_neighbors=min(n, max_k),
            metric="mahalanobis",
            metric_params={"VI": VI},
        )
    else:
        nn_model = NearestNeighbors(
            n_neighbors=min(n, max_k),
            metric="euclidean",
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
    Gaussian-based Stochastic Gates from Yamada et al. 2020
    Gates are global (shared across all samples in batch)
    During training: z_j = clamp(μ_j + ε_j, 0, 1) where ε_j ~ N(0, σ²)
    During inference: z_j = clamp(μ_j, 0, 1)
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
        """Compute sparsity regularization: λ * Σ P(z_j > 0)"""
        standardized_mu = self.mu / self.sigma
        prob_active = 0.5 * (1 + torch.erf(standardized_mu / math.sqrt(2)))
        l0_approx = prob_active.sum()
        return reg_weight * l0_approx

# ========================================
# GRAFT Model
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

class GRAFT(nn.Module):
    """Gated Residual AFT with Gaussian Stochastic Gates"""
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

def train_graft(X_tr, t_tr, e_tr, hidden_dim=32, n_epochs=1000, batch_size=64, lr=1e-3, l2=1e-4,
                sigma=0.5, reg_weight=0.01, regularization_strength=0.1, dropout=0.2, 
                M=5, k_min=30, min_events=10, patience=8, device=None):
    """Train GRAFT model with local KM imputation and early stopping"""
    
    if device is None:
        device = DEVICE
    
    n_samples = len(X_tr)
    
    # Pre-compute neighborhoods and KM curves
    neighborhoods = find_adaptive_neighborhoods(X_tr, t_tr, e_tr, k_min=k_min, 
                                                metric="euclidian", min_events=min_events)
    km_curves = prefit_local_km_curves(t_tr, e_tr, neighborhoods)
    
    # Create model and move to device
    model = GRAFT(
        n_features=X_tr.shape[1],
        hidden_dim=hidden_dim,
        sigma=sigma,
        dropout=dropout
    ).to(device)
    
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
            reg_loss = model.stochastic_gates.regularization(reg_weight=reg_weight)
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
# Baseline: Cox PH
# ========================================

def train_and_predict_cox(X_tr, t_tr, e_tr, X_te, t_te, e_te, cols):
    """Train Cox PH and return predictions"""
    train_df = pd.DataFrame(X_tr, columns=cols)
    train_df['duration'] = t_tr
    train_df['event'] = e_tr
    test_df = pd.DataFrame(X_te, columns=cols)
    test_df['duration'] = t_te
    test_df['event'] = e_te
    
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(train_df, duration_col='duration', event_col='event')
    scores = -cph.predict_partial_hazard(test_df).values.flatten()
    
    times_grid = np.linspace(t_te.min(), t_te.max(), 50)
    base_cumhaz = cph.baseline_cumulative_hazard_
    base_cumhaz_interp = np.interp(times_grid, base_cumhaz.index.values, base_cumhaz.values.flatten())
    linpred = cph.predict_partial_hazard(test_df).values.flatten()
    S_pred = np.exp(-np.outer(linpred, base_cumhaz_interp))
    ibs = integrated_brier_ipcw(t_tr, e_tr, t_te, e_te, S_pred, times_grid)
    
    return scores, ibs

# ========================================
# Baseline: Weibull AFT
# ========================================

def train_and_predict_weibull(X_tr, t_tr, e_tr, X_te, t_te, e_te, cols):
    """Train Weibull AFT and return predictions"""
    train_df = pd.DataFrame(X_tr, columns=cols)
    train_df['duration'] = t_tr
    train_df['event'] = e_tr
    test_df = pd.DataFrame(X_te, columns=cols)
    test_df['duration'] = t_te
    test_df['event'] = e_te
    
    wbf = WeibullAFTFitter(penalizer=0.01)
    wbf.fit(train_df, duration_col='duration', event_col='event')
    scores = wbf.predict_median(test_df).values.flatten()
    
    times_grid = np.linspace(t_te.min(), t_te.max(), 50)
    S_pred = np.zeros((len(test_df), len(times_grid)))
    for i in range(len(test_df)):
        S_pred[i,:] = wbf.predict_survival_function(test_df.iloc[[i]], times=times_grid).values.flatten()
    ibs = integrated_brier_ipcw(t_tr, e_tr, t_te, e_te, S_pred, times_grid)
    
    return scores, ibs

# ========================================
# Baseline: DeepHit
# ========================================

def train_and_predict_deephit(X_tr, t_tr, e_tr, X_te, t_te, e_te, epochs=1000, patience=8):
    """Train DeepHit and return predictions"""
    
    try:
        num_durations = 100
        labtrans = DeepHitSingle.label_transform(num_durations)
        y_train = labtrans.fit_transform(t_tr, e_tr)
        
        in_features = X_tr.shape[1]
        num_nodes = [32, 32]
        out_features = labtrans.out_features
        batch_norm = True
        dropout = 0.1
        
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
        
        model = DeepHitSingle(net, tt.optim.Adam(0.01), alpha=0.3, sigma=0.1, duration_index=labtrans.cuts)
        
        batch_size = 64
        callbacks = [tt.callbacks.EarlyStopping(patience=patience)]
        
        model.fit(X_tr.astype('float32'), y_train, batch_size, epochs, 
                 callbacks=callbacks, verbose=False)
        
        surv = model.predict_surv_df(X_te.astype('float32'))
        
        times = surv.index.values
        median_survival = []
        for i in range(len(X_te)):
            surv_probs = surv.iloc[:, i].values
            try:
                idx = np.where(surv_probs <= 0.5)[0]
                if len(idx) > 0:
                    median_survival.append(times[idx[0]])
                else:
                    median_survival.append(times[-1])
            except:
                median_survival.append(times[-1])
        
        scores = np.array(median_survival)
        
        times_grid = np.linspace(t_te.min(), t_te.max(), 50)
        surv_times = surv.index.values
        S_pred = np.zeros((len(X_te), len(times_grid)))
        for i in range(len(X_te)):
            S_pred[i, :] = np.interp(times_grid, surv_times, surv.iloc[:, i].values)
        
        ibs = integrated_brier_ipcw(t_tr, e_tr, t_te, e_te, S_pred, times_grid)
        
        return scores, ibs
        
    except Exception as e:
        print(f"[DH err]", end=" ")
        return np.full(len(X_te), np.nan), np.nan

# ========================================
# Baseline: DeepSurv
# ========================================

def train_and_predict_deepsurv(X_tr, t_tr, e_tr, X_te, t_te, e_te, epochs=1000, patience=8):
    """Train DeepSurv and return predictions"""
    
    try:
        in_features = X_tr.shape[1]
        num_nodes = [32, 32]
        out_features = 1
        batch_norm = True
        dropout = 0.2
        
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout, output_bias=False)
        
        model = PyCoxCoxPH(net, tt.optim.Adam(0.01))
        
        batch_size = 64
        callbacks = [tt.callbacks.EarlyStopping(patience=patience)]
        
        model.fit(X_tr.astype('float32'), (t_tr, e_tr), batch_size, epochs, 
                 callbacks=callbacks, verbose=False)
        
        _ = model.compute_baseline_hazards()
        surv = model.predict_surv_df(X_te.astype('float32'))
        scores = -model.predict(X_te.astype('float32'))
        
        times_grid = np.linspace(t_te.min(), t_te.max(), 50)
        surv_times = surv.index.values
        S_pred = np.zeros((len(X_te), len(times_grid)))
        for i in range(len(X_te)):
            S_pred[i, :] = np.interp(times_grid, surv_times, surv.iloc[:, i].values)
        
        ibs = integrated_brier_ipcw(t_tr, e_tr, t_te, e_te, S_pred, times_grid)
        
        return scores, ibs
        
    except Exception as e:
        print(f"[DS err]", end=" ")
        return np.full(len(X_te), np.nan), np.nan

# ========================================
# Baseline: RSF
# ========================================

def train_and_predict_rsf(X_tr, t_tr, e_tr, X_te, t_te, e_te):
    """Train RSF and return predictions"""
    
    try:
        y_train = Surv.from_arrays(event=e_tr.astype(bool), time=t_tr)
        
        rsf = RandomSurvivalForest(
            n_estimators=100,
            min_samples_split=10,
            min_samples_leaf=15,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        )
        
        rsf.fit(X_tr, y_train)
        scores = -rsf.predict(X_te)
        
        min_time = max(t_te.min(), t_tr.min())
        max_time = min(t_te.max(), t_tr.max())
        times_grid = np.linspace(min_time, max_time, 50)
        
        surv_funcs = rsf.predict_survival_function(X_te, return_array=False)
        
        S_pred = np.zeros((len(X_te), len(times_grid)))
        for i, surv_func in enumerate(surv_funcs):
            S_pred[i, :] = surv_func(times_grid)
        
        ibs = integrated_brier_ipcw(t_tr, e_tr, t_te, e_te, S_pred, times_grid)
        
        return scores, ibs
        
    except Exception as e:
        print(f"[RSF err: {e}]", end=" ")
        return np.full(len(X_te), np.nan), np.nan

# ========================================
# Load Datasets
# ========================================

def load_datasets():
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
    
    models = ['GRAFT', 'CoxPH', 'Weibull', 'DeepHit', 'DeepSurv', 'RSF']
    
    # C-Index table
    print("\nC-INDEX:")
    print("-" * 100)
    header = "Model".ljust(15) + "| Fold-Averaged".center(30) + "| Seed-Averaged".center(30) + "|"
    print(header)
    print(" " * 15 + "|" + " Mean      Std".center(30) + "|" + " Mean      Std".center(30) + "|")
    print("-" * 100)
    
    for model in models:
        if 'fold_avg_cindex' in results['models'][model] and len(results['models'][model]['fold_avg_cindex']) > 0:
            fold_avg = results['models'][model]['fold_avg_cindex']
            seed_avg = results['models'][model]['seed_avg_cindex']
            
            fold_mean = np.mean(fold_avg)
            fold_std = np.std(fold_avg)
            seed_mean = np.mean(seed_avg)
            seed_std = np.std(seed_avg)
            
            row = f"{model:15s}| {fold_mean:8.4f}  {fold_std:8.4f}      | {seed_mean:8.4f}  {seed_std:8.4f}      |"
            print(row)
        else:
            print(f"{model:15s}| {'N/A':^30s}| {'N/A':^30s}|")
    
    # IBS table
    print("\nINTEGRATED BRIER SCORE:")
    print("-" * 100)
    header = "Model".ljust(15) + "| Fold-Averaged".center(30) + "| Seed-Averaged".center(30) + "|"
    print(header)
    print(" " * 15 + "|" + " Mean      Std".center(30) + "|" + " Mean      Std".center(30) + "|")
    print("-" * 100)
    
    for model in models:
        if 'fold_avg_ibs' in results['models'][model] and len(results['models'][model]['fold_avg_ibs']) > 0:
            fold_avg = results['models'][model]['fold_avg_ibs']
            seed_avg = results['models'][model]['seed_avg_ibs']
            
            fold_mean = np.mean(fold_avg)
            fold_std = np.std(fold_avg)
            seed_mean = np.mean(seed_avg)
            seed_std = np.std(seed_avg)
            
            row = f"{model:15s}| {fold_mean:8.4f}  {fold_std:8.4f}      | {seed_mean:8.4f}  {seed_std:8.4f}      |"
            print(row)
        else:
            print(f"{model:15s}| {'N/A':^30s}| {'N/A':^30s}|")

def display_combined_results(all_results):
    """Display combined results for all datasets"""
    print("\n" + "="*120)
    print("COMBINED RESULTS: ALL DATASETS")
    print("="*120)
    
    datasets = list(all_results.keys())
    models = ['GRAFT', 'CoxPH', 'Weibull', 'DeepHit', 'DeepSurv', 'RSF']
    
    # C-Index Fold-Averaged table
    print("\nC-INDEX (Fold-Averaged):")
    print("-" * 120)
    header = "Dataset".ljust(15)
    for model in models:
        header += f"{model}".center(20)
    print(header)
    print(" " * 15 + "Mean    Std   " * len(models))
    print("-" * 120)
    
    for dataset in datasets:
        row = f"{dataset:15s}"
        for model in models:
            if 'fold_avg_cindex' in all_results[dataset]['models'][model] and len(all_results[dataset]['models'][model]['fold_avg_cindex']) > 0:
                values = all_results[dataset]['models'][model]['fold_avg_cindex']
                mean_val = np.mean(values)
                std_val = np.std(values)
                row += f"{mean_val:6.4f} {std_val:6.4f}  "
            else:
                row += f"{'N/A':^20s}"
        print(row)
    
    # C-Index Seed-Averaged table
    print("\nC-INDEX (Seed-Averaged):")
    print("-" * 120)
    header = "Dataset".ljust(15)
    for model in models:
        header += f"{model}".center(20)
    print(header)
    print(" " * 15 + "Mean    Std   " * len(models))
    print("-" * 120)
    
    for dataset in datasets:
        row = f"{dataset:15s}"
        for model in models:
            if 'seed_avg_cindex' in all_results[dataset]['models'][model] and len(all_results[dataset]['models'][model]['seed_avg_cindex']) > 0:
                values = all_results[dataset]['models'][model]['seed_avg_cindex']
                mean_val = np.mean(values)
                std_val = np.std(values)
                row += f"{mean_val:6.4f} {std_val:6.4f}  "
            else:
                row += f"{'N/A':^20s}"
        print(row)
    
    # IBS Fold-Averaged table
    print("\nINTEGRATED BRIER SCORE (Fold-Averaged):")
    print("-" * 120)
    header = "Dataset".ljust(15)
    for model in models:
        header += f"{model}".center(20)
    print(header)
    print(" " * 15 + "Mean    Std   " * len(models))
    print("-" * 120)
    
    for dataset in datasets:
        row = f"{dataset:15s}"
        for model in models:
            if 'fold_avg_ibs' in all_results[dataset]['models'][model] and len(all_results[dataset]['models'][model]['fold_avg_ibs']) > 0:
                values = all_results[dataset]['models'][model]['fold_avg_ibs']
                mean_val = np.mean(values)
                std_val = np.std(values)
                row += f"{mean_val:6.4f} {std_val:6.4f}  "
            else:
                row += f"{'N/A':^20s}"
        print(row)
    
    # IBS Seed-Averaged table
    print("\nINTEGRATED BRIER SCORE (Seed-Averaged):")
    print("-" * 120)
    header = "Dataset".ljust(15)
    for model in models:
        header += f"{model}".center(20)
    print(header)
    print(" " * 15 + "Mean    Std   " * len(models))
    print("-" * 120)
    
    for dataset in datasets:
        row = f"{dataset:15s}"
        for model in models:
            if 'seed_avg_ibs' in all_results[dataset]['models'][model] and len(all_results[dataset]['models'][model]['seed_avg_ibs']) > 0:
                values = all_results[dataset]['models'][model]['seed_avg_ibs']
                mean_val = np.mean(values)
                std_val = np.std(values)
                row += f"{mean_val:6.4f} {std_val:6.4f}  "
            else:
                row += f"{'N/A':^20s}"
        print(row)

# ========================================
# Main Experiment
# ========================================

def run_experiment(datasets, n_folds=3, seeds=[42, 43, 44]):
    """Run comprehensive baseline comparison experiment"""
    
    all_results = {}
    
    for dataset_name in datasets.keys():
        print(f"\n{'='*80}")
        print(f"PROCESSING DATASET: {dataset_name}")
        print(f"{'='*80}")
        
        dataset_info = datasets[dataset_name]
        df = dataset_info['df']
        duration_col = dataset_info['duration_col']
        event_col = dataset_info['event_col']
        drop_cols = dataset_info['drop_cols']
        
        # Initialize results storage
        all_results[dataset_name] = {
            'models': {
                'GRAFT': {},
                'CoxPH': {},
                'Weibull': {},
                'DeepHit': {},
                'DeepSurv': {},
                'RSF': {}
            }
        }
        
        seed_fold_results = {}
        for model in ['GRAFT', 'CoxPH', 'Weibull', 'DeepHit', 'DeepSurv', 'RSF']:
            seed_fold_results[model] = {'cindex': [], 'ibs': []}
        
        for seed_idx, seed in enumerate(seeds):
            print(f"\n  Seed {seed}")
            
            X, t, e, cols = preprocess_df(df, duration_col, event_col, drop_cols=drop_cols)
            print(f"    Features: {len(cols)}, Samples: {len(X)}, Events: {e.sum()}")
            
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
            for model in seed_fold_results.keys():
                seed_fold_results[model]['cindex'].append([])
                seed_fold_results[model]['ibs'].append([])
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                print(f"    Fold {fold+1}/{n_folds} ", end="")
                
                X_tr, X_te = X[train_idx], X[test_idx]
                t_tr, t_te = t[train_idx], t[test_idx]
                e_tr, e_te = e[train_idx], e[test_idx]
                
                # GRAFT
                try:
                    model_graft = train_graft(X_tr, t_tr, e_tr, dropout=0.2, M=5, k_min=30, 
                                             min_events=10, batch_size=64, sigma=0.5, 
                                             reg_weight=0.01, n_epochs=1000, patience=8, 
                                             device=DEVICE)
                    scores_graft = predict_graft(X_te, model_graft)
                    cindex_graft = harrell_c_index(scores_graft, t_te, e_te)
                    
                    # Calibrate for IBS
                    times_grid = np.linspace(t_te.min(), t_te.max(), 50)
                    scores_graft_tr = predict_graft(X_tr, model_graft)
                    temp_df = pd.DataFrame(X_tr, columns=cols)
                    temp_df['score'] = scores_graft_tr
                    temp_df['duration'] = t_tr
                    temp_df['event'] = e_tr
                    cph_temp = CoxPHFitter()
                    cph_temp.fit(temp_df[['score', 'duration', 'event']], 
                                duration_col='duration', event_col='event')
                    base_cumhaz = cph_temp.baseline_cumulative_hazard_
                    base_cumhaz_interp = np.interp(times_grid, base_cumhaz.index.values, 
                                                  base_cumhaz.values.flatten())
                    test_df_temp = pd.DataFrame(X_te, columns=cols)
                    test_df_temp['score'] = scores_graft
                    linpred = cph_temp.predict_partial_hazard(test_df_temp[['score']]).values.flatten()
                    S_pred_graft = np.exp(-np.outer(linpred, base_cumhaz_interp))
                    ibs_graft = integrated_brier_ipcw(t_tr, e_tr, t_te, e_te, S_pred_graft, times_grid)
                    
                    seed_fold_results['GRAFT']['cindex'][seed_idx].append(cindex_graft)
                    seed_fold_results['GRAFT']['ibs'][seed_idx].append(ibs_graft)
                except Exception as ex:
                    print(f"[GR err]", end=" ")
                    seed_fold_results['GRAFT']['cindex'][seed_idx].append(np.nan)
                    seed_fold_results['GRAFT']['ibs'][seed_idx].append(np.nan)
                
                # Cox PH
                try:
                    scores_cox, ibs_cox = train_and_predict_cox(X_tr, t_tr, e_tr, X_te, t_te, e_te, cols)
                    cindex_cox = harrell_c_index(scores_cox, t_te, e_te)
                    seed_fold_results['CoxPH']['cindex'][seed_idx].append(cindex_cox)
                    seed_fold_results['CoxPH']['ibs'][seed_idx].append(ibs_cox)
                except Exception as ex:
                    print(f"[Cox err]", end=" ")
                    seed_fold_results['CoxPH']['cindex'][seed_idx].append(np.nan)
                    seed_fold_results['CoxPH']['ibs'][seed_idx].append(np.nan)
                
                # Weibull
                try:
                    scores_wbf, ibs_wbf = train_and_predict_weibull(X_tr, t_tr, e_tr, X_te, t_te, e_te, cols)
                    cindex_wbf = harrell_c_index(scores_wbf, t_te, e_te)
                    seed_fold_results['Weibull']['cindex'][seed_idx].append(cindex_wbf)
                    seed_fold_results['Weibull']['ibs'][seed_idx].append(ibs_wbf)
                except Exception as ex:
                    print(f"[Wbl err]", end=" ")
                    seed_fold_results['Weibull']['cindex'][seed_idx].append(np.nan)
                    seed_fold_results['Weibull']['ibs'][seed_idx].append(np.nan)
                
                # DeepHit
                try:
                    scores_dh, ibs_dh = train_and_predict_deephit(X_tr, t_tr, e_tr, X_te, t_te, e_te,
                                                                  epochs=1000, patience=8)
                    cindex_dh = harrell_c_index(scores_dh, t_te, e_te)
                    seed_fold_results['DeepHit']['cindex'][seed_idx].append(cindex_dh)
                    seed_fold_results['DeepHit']['ibs'][seed_idx].append(ibs_dh)
                except Exception as ex:
                    print(f"[DH err]", end=" ")
                    seed_fold_results['DeepHit']['cindex'][seed_idx].append(np.nan)
                    seed_fold_results['DeepHit']['ibs'][seed_idx].append(np.nan)
                
                # DeepSurv
                try:
                    scores_ds, ibs_ds = train_and_predict_deepsurv(X_tr, t_tr, e_tr, X_te, t_te, e_te,
                                                                   epochs=1000, patience=8)
                    cindex_ds = harrell_c_index(scores_ds, t_te, e_te)
                    seed_fold_results['DeepSurv']['cindex'][seed_idx].append(cindex_ds)
                    seed_fold_results['DeepSurv']['ibs'][seed_idx].append(ibs_ds)
                except Exception as ex:
                    print(f"[DS err]", end=" ")
                    seed_fold_results['DeepSurv']['cindex'][seed_idx].append(np.nan)
                    seed_fold_results['DeepSurv']['ibs'][seed_idx].append(np.nan)
                
                # RSF
                try:
                    scores_rsf, ibs_rsf = train_and_predict_rsf(X_tr, t_tr, e_tr, X_te, t_te, e_te)
                    cindex_rsf = harrell_c_index(scores_rsf, t_te, e_te)
                    seed_fold_results['RSF']['cindex'][seed_idx].append(cindex_rsf)
                    seed_fold_results['RSF']['ibs'][seed_idx].append(ibs_rsf)
                except Exception as ex:
                    print(f"[RSF err]", end=" ")
                    seed_fold_results['RSF']['cindex'][seed_idx].append(np.nan)
                    seed_fold_results['RSF']['ibs'][seed_idx].append(np.nan)
                
                print("✓")
        
        # Compute both fold-averaged and seed-averaged results
        for model in seed_fold_results.keys():
            cindex_array = np.array(seed_fold_results[model]['cindex'])
            ibs_array = np.array(seed_fold_results[model]['ibs'])
            
            # Method 1: Fold-Averaged
            fold_avg_cindex = []
            fold_avg_ibs = []
            for seed_idx in range(len(seeds)):
                valid_cindex = [x for x in cindex_array[seed_idx] if not np.isnan(x)]
                valid_ibs = [x for x in ibs_array[seed_idx] if not np.isnan(x)]
                if len(valid_cindex) > 0:
                    fold_avg_cindex.append(np.mean(valid_cindex))
                if len(valid_ibs) > 0:
                    fold_avg_ibs.append(np.mean(valid_ibs))
            
            # Method 2: Seed-Averaged
            seed_avg_cindex = []
            seed_avg_ibs = []
            for fold_idx in range(n_folds):
                valid_cindex = [cindex_array[seed_idx, fold_idx] for seed_idx in range(len(seeds)) 
                               if not np.isnan(cindex_array[seed_idx, fold_idx])]
                valid_ibs = [ibs_array[seed_idx, fold_idx] for seed_idx in range(len(seeds)) 
                            if not np.isnan(ibs_array[seed_idx, fold_idx])]
                if len(valid_cindex) > 0:
                    seed_avg_cindex.append(np.mean(valid_cindex))
                if len(valid_ibs) > 0:
                    seed_avg_ibs.append(np.mean(valid_ibs))
            
            all_results[dataset_name]['models'][model]['fold_avg_cindex'] = fold_avg_cindex
            all_results[dataset_name]['models'][model]['fold_avg_ibs'] = fold_avg_ibs
            all_results[dataset_name]['models'][model]['seed_avg_cindex'] = seed_avg_cindex
            all_results[dataset_name]['models'][model]['seed_avg_ibs'] = seed_avg_ibs
        
        display_single_dataset_results(dataset_name, all_results[dataset_name])
    
    return all_results

# ========================================
# Main Execution
# ========================================

if __name__ == "__main__":
    
    datasets = load_datasets()
    
    print("\n" + "="*80)
    print("Starting comprehensive baseline comparison with GRAFT")
    print("- 6 datasets: GBSG, METABRIC, SUPPORT, AIDS, FLCHAIN, NWTCO")
    print("- 6 models: GRAFT, CoxPH, Weibull, DeepHit, DeepSurv, RSF")
    print("- 3 seeds: 42, 43, 44")
    print("- 3-fold CV")
    print("- Early stopping: 1000 epochs, patience 8")
    print("- Both fold-averaged and seed-averaged results")
    print("="*80)
    
    all_results = run_experiment(datasets, n_folds=3, seeds=[42, 43, 44])
    
    display_combined_results(all_results)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
