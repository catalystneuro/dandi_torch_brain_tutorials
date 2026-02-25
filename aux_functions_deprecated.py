from collections import defaultdict
from pathlib import Path
from typing import Literal
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from omegaconf import OmegaConf
from temporaldata import Data, IrregularTimeSeries
from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import RandomFixedWindowSampler, SequentialFixedWindowSampler
from torch_brain.data import Dataset, collate


def download_model(local_path: str = None):
    """
    Download a pre-trained model file and save it to `local_path`.
    """
    if local_path is None:
        local_path = "./poyo_1.ckpt"
    local_path = Path(local_path).resolve()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        print(f"File already exists at: {local_path}")
        return
    
    #url = "https://nyu1.osn.mghpcc.org/brainsets-public/model-zoo/poyo_mp.ckpt"
    url = "https://nyu1.osn.mghpcc.org/brainsets-public/model-zoo/poyo_1.ckpt"

    print("Downloading model...")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=16384):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded file to: {local_path}")


def move_to_device(data, device=None):
    if device is None:
        device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda:0") if torch.cuda.is_available()
            else torch.device("cpu")
        )

    if isinstance(data, torch.Tensor):
        # Safest path: specify dtype on the move for float tensors.
        if data.is_floating_point():
            return data.to(device=device, dtype=torch.float32)
        else:
            return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    else:
        return data


def r2_score(y_pred, y_true):
    # Compute total sum of squares (variance of the true values)
    y_true_mean = torch.mean(y_true, dim=0, keepdim=True)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)

    # Compute residual sum of squares
    ss_res = torch.sum((y_true - y_pred) ** 2)

    # Handle zero or near-zero variance
    epsilon = 1e-2
    if ss_total < epsilon:
        return torch.tensor(float('nan'))

    # Compute R^2
    r2 = 1 - ss_res / ss_total

    return r2


def compute_r2(dataloader, model):
    model.eval()  # turn off dropout, etc.
    total_target = []
    total_pred = []
    with torch.no_grad():  # <-- crucial: no graph, no huge memory
        for batch in dataloader:
            batch = move_to_device(batch)
            pred = model(**batch["model_inputs"])
            target = batch["target_values"]

            # If your model returns [B, T, 1], squeeze to [B, T]
            if pred.dim() == 3 and pred.size(-1) == 1:
                pred = pred.squeeze(-1)

            mask = torch.ones_like(target, dtype=torch.bool)
            if "output_mask" in batch["model_inputs"]:
                mask = batch["model_inputs"]["output_mask"]
                if mask.dim() == 3 and mask.size(-1) == 1:
                    mask = mask.squeeze(-1)

            total_target.append(target[mask])
            total_pred.append(pred[mask])

    total_target = torch.cat(total_target)
    total_pred = torch.cat(total_pred)

    r2 = r2_score(total_pred.flatten(), total_target.flatten())
    return r2.item(), total_target, total_pred


def training_step(batch, model, optimizer):
    # Step 0. Clear old gradients
    optimizer.zero_grad()

    inputs = batch["model_inputs"]
    target = batch["target_values"]

    # Step 1. Do forward pass
    pred = model(**inputs)

    # shapes: [B, T, 1] -> [B, T]
    if pred.dim() == 3 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)

    # Step 2. Compute loss
    loss = F.mse_loss(pred, target)

    # Step 3. Backward pass
    loss.backward()

    # Step 4. Update model params
    optimizer.step()
    return loss


def plot_training_curves(r2_log, loss_log):
    """
    Plots the training curves: training loss and validation R2 score.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(0, len(loss_log), len(loss_log)), loss_log)
    plt.title("Training Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("MSE Loss")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(r2_log[1:])  # skip initial validation before training
    plt.title("Validation R2")
    plt.xlabel("Epochs")
    plt.ylabel("R2 Score")
    plt.grid()
    plt.tight_layout()
    plt.show()


def finetune(model, optimizer, train_loader, val_loader, num_epochs=50, epoch_to_unfreeze=30):
    # Freeze the backbone
    backbone_params = [
        p for p in model.named_parameters()
        if (
            'unit_emb' not in p[0]
            and 'session_emb' not in p[0]
            and 'readout' not in p[0]
            and p[1].requires_grad
        )
    ]
    for _, param in backbone_params:
        param.requires_grad = False

    # Store intermediate outputs for visualization
    train_outputs = {
        'n_epochs': num_epochs,
        'epoch_to_unfreeze': epoch_to_unfreeze,
        'unit_emb': [],
        'session_emb': [],
        'output_pred': [],
        'output_gt': [],
    }

    r2_log = []
    loss_log = []

    # Main progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Finetuning Progress", leave=True)

    for epoch in epoch_pbar:
        # Unfreeze backbone
        if epoch == epoch_to_unfreeze:
            for _, param in backbone_params:
                param.requires_grad = True
            print("\n🔓 Unfreezing entire model")

        # Validation before training step
        with torch.no_grad():
            model.eval()  # make sure we're in eval mode during validation
            r2, target, pred = compute_r2(val_loader, model)
            r2_log.append(r2)

        # Switch back to training mode
        model.train()
        
        running_loss = 0.0

        # Inner progress bar for training batches
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in batch_pbar:
            batch = move_to_device(batch)
            loss = training_step(batch, model, optimizer)
            loss_log.append(loss.item())
            running_loss += loss.item()

            # Update inner bar postfix
            batch_pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Val R2": f"{r2:.3f}"
            })

        avg_loss = running_loss / len(train_loader)
        epoch_pbar.set_postfix({
            "Avg Loss": f"{avg_loss:.4f}",
            "Val R2": f"{r2:.3f}"
        })

        # Store intermediate outputs
        train_outputs['unit_emb'].append(model.unit_emb.weight[1:].detach().cpu().numpy())
        train_outputs['session_emb'].append(model.session_emb.weight[1:].detach().cpu().numpy())
        train_outputs['output_gt'].append(target.detach().cpu().numpy())
        train_outputs['output_pred'].append(pred.detach().cpu().numpy())

        del target, pred

    # Final validation
    r2, _, _ = compute_r2(val_loader, model)
    r2_log.append(r2)
    print(f"\n✅ Done! Final validation R² = {r2:.3f}")

    return r2_log, loss_log, train_outputs


def get_dataset_config(brainset, readout_id, session_ids=None):
    all_sessions = [f.resolve() for f in Path(brainset).glob("*.h5")]
    if session_ids is not None:
        all_sessions = [s for s in all_sessions if s.name in session_ids]
    values = np.array([])
    for session_path in all_sessions:
        with h5py.File(session_path, "r") as f:
            session_data = Data.from_hdf5(f, lazy=True)
            train = session_data.select_by_interval(session_data.train_domain)
            values = np.append(values, getattr(train, readout_id).values)
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)

    session_names = [s.name.split(".h5")[0] for s in all_sessions]
    sessions_yaml = '\n'.join([f'          - {name}' for name in session_names])

    config = f"""
    - selection:
      - brainset: {brainset}
        sessions:
{sessions_yaml}
      config:
        readout:
          readout_id: {readout_id}
          timestamp_key: {readout_id}.timestamps
          value_key: {readout_id}.values
          normalize_mean: {mean_val}
          normalize_std: {std_val}
          metrics:
            - metric:
                _target_: torchmetrics.R2Score
    """
    config = OmegaConf.create(config)
    return config


def get_loaders(
    dir_path: str = ".",
    recording_id=None,
    cfg=None,
    window_length=1.0,
    batch_size=16,
    seed=0,
    device=None,
):
    """Sets up train and validation Datasets, Samplers, and DataLoaders"""
    # sensible defaults
    use_multiproc = True
    use_pin_memory = True
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
    # On Apple MPS, avoid multiprocessing/pinned memory to prevent _share_filename_ errors
    if device.type == "mps":
        use_multiproc = False
        use_pin_memory = False

    # -- Train --
    train_dataset = Dataset(
        root=dir_path,
        recording_id=recording_id,
        config=cfg,
        split="train",
    )
    # We use a random sampler to improve generalization during training
    train_sampling_intervals = train_dataset.get_sampling_intervals()
    train_sampler = RandomFixedWindowSampler(
        sampling_intervals=train_sampling_intervals,
        window_length=window_length,
        generator=torch.Generator().manual_seed(seed),
    )
    # Finally combine them in a dataloader
    train_loader = DataLoader(
        dataset=train_dataset,      # dataset
        sampler=train_sampler,      # sampler
        batch_size=batch_size,      # num of samples per batch
        collate_fn=collate,         # the collator
        num_workers=0 if not use_multiproc else 4,    # data sample processing (slicing, transforms, tokenization) happens in parallel; this sets the amount of that parallelization
        pin_memory=use_pin_memory,
        persistent_workers=False,   # important on macOS
    )

    # -- Validation --
    val_dataset = Dataset(
        root=dir_path,
        recording_id=recording_id,
        config=cfg,
        split="valid",
    )
    # For validation we don't randomize samples for reproducibility
    val_sampling_intervals = val_dataset.get_sampling_intervals()
    val_sampler = SequentialFixedWindowSampler(
        sampling_intervals=val_sampling_intervals,
        window_length=window_length,
    )
    # Combine them in a dataloader
    val_loader = DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=0 if not use_multiproc else 4,
        pin_memory=use_pin_memory,
        persistent_workers=False,
    )

    # -- Test --
    test_dataset = Dataset(
        root=dir_path,
        recording_id=recording_id,
        config=cfg,
        split="test",
    )
    test_sampling_intervals = test_dataset.get_sampling_intervals()
    test_sampler = SequentialFixedWindowSampler(
        sampling_intervals=test_sampling_intervals,
        window_length=window_length,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=0 if not use_multiproc else 4,
        pin_memory=use_pin_memory,
        persistent_workers=False,
    )

    train_dataset.disable_data_leakage_check()
    val_dataset.disable_data_leakage_check()
    test_dataset.disable_data_leakage_check()

    return (
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
        test_dataset,
        test_loader,
    )


def get_unit_ids(
    dataset,
    filter_str=["motor"],
    quality_score: float = 0.6,
):
    """
    Get unit IDs filtered by location names and quality score.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset containing recording data.
    filter_str : list[str], optional
        List of strings to filter location names (default: ["motor"]).
    quality_score : float, optional
        Minimum IBL quality score threshold (default: 0.6).
    
    Returns
    -------
    list
        List of filtered unit IDs.
    """
    unit_ids_list = []
    for k in dataset.recording_dict.keys():
        data = dataset.get_recording_data(k)
        valid_ids = list()
        for i, ln, qs in zip(data.units.id, data.units.location_names, data.units.ibl_quality_score):
            # Filter by location name and quality score
            if any(fs in ln.lower() for fs in filter_str) and qs > quality_score:
                valid_ids.append(i)
        unit_ids_list.extend(valid_ids)
    return unit_ids_list


class Transform:
    def __init__(self, model):
        self.model = model
        
        # Precompute valid units per brainset/session
        out = defaultdict(list)
        for k in model.unit_emb.vocab:
            if k == 'NA':
                continue
            s = str(k)  # handle np.str_ safely
            prefix, unit = s.rsplit('/unit_', 1)
            out[prefix].append(int(unit))
        self.valid_units_per_recording = dict(out)
        
    def __call__(self, data):
        """Filters data to use only spikes from motor areas."""

        ## Deprecated code for filtering based on filter_str ---------------------
        # unit_ids = data.units.id
        # spike_unit_index = data.spikes.unit_index
        # spike_timestamps = data.spikes.timestamps

        # valid_idx = list()
        # for idx, ln in zip(data.units.id, data.units.location_names):
        #     if self.filter_str in ln.lower():
        #         valid_idx.append(idx.split("_")[-1])  # keep only the numeric part

        # valid_idx = np.array(valid_idx)
        # mask = np.isin(spike_unit_index, valid_idx)
        ##-----------------------------------------------------------------------

        valid_units = self.valid_units_per_recording.get(data.session.id, [])
        
        # Filter spikes
        spike_unit_index = data.spikes.unit_index
        spike_timestamps = data.spikes.timestamps
        mask_spikes = np.isin(spike_unit_index, valid_units)

        # Create mapping from old unit index to new (filtered) unit index
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_units)}
        
        # Remap spike unit indices to new positions
        remapped_indices = np.array([old_to_new[idx] for idx in spike_unit_index[mask_spikes]])
        
        data.spikes = IrregularTimeSeries(
            timestamps=spike_timestamps[mask_spikes],
            unit_index=remapped_indices,  # <-- Use remapped indices!
            domain="auto",
        )

        # Filter units metadata
        units_ids = [int(i.split("_")[-1]) for i in data.units.id]
        mask_units = np.isin(units_ids, valid_units)
        data.units = data.units.select_by_mask(mask_units)

        return self.model.tokenize(data)


def run_test(
    test_dataset,
    test_loader,
    model,
    device=None,
):
    # Connect tokenizers to Datasets
    test_dataset.transform = Transform(model=model)
    
    model.eval()
    targets, preds, r2_scores = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = move_to_device(batch, device)
            pred = model(**batch["model_inputs"])
            target = batch["target_values"]
    
            if pred.dim() == 3 and pred.size(-1) == 1:
                pred = pred.squeeze(-1)
    
            mask = torch.ones_like(target, dtype=torch.bool)
            if "output_mask" in batch["model_inputs"]:
                mask = batch["model_inputs"]["output_mask"]
                if mask.dim() == 3 and mask.size(-1) == 1:
                    mask = mask.squeeze(-1)
    
            # Process each interval in the batch separately
            batch_size = pred.shape[0]
            for i in range(batch_size):
                interval_pred = pred[i]      # Shape: [T]
                interval_target = target[i]  # Shape: [T]
                interval_mask = mask[i]      # Shape: [T]
                
                # Apply mask to get valid predictions and targets
                masked_pred = interval_pred[interval_mask]
                masked_target = interval_target[interval_mask]
                
                # Only calculate R² if we have valid data points
                if len(masked_target) > 0:
                    r2 = r2_score(masked_pred, masked_target)
                    
                    targets.append(masked_target)
                    preds.append(masked_pred)
                    r2_scores.append(r2.item())

    return dict(targets=targets, preds=preds, r2_scores=r2_scores)


def plot_test_intervals(test_results, n_intervals=5, order: Literal["top", "bottom"] = "top"):
    """
    Plots the top/bottom n_intervals based on R² scores.
    Supports both single-model and multi-model comparison.
    
    Parameters
    ----------
    test_results : dict
        Either a single model result dict with keys 'targets', 'preds', 'r2_scores',
        or a dict of model results: {model_name: {targets, preds, r2_scores}, ...}
    n_intervals : int, optional
        Number of intervals to plot (default=5).
    order : Literal["top", "bottom"], optional
        Which intervals to plot. "top" shows best performers (default),
        "bottom" shows worst performers.
    
    Returns
    -------
    top_indices : list[int]
        Indices of the intervals that were plotted.
    """
    # Detect if single model or multi-model
    is_multi_model = 'targets' not in test_results
    
    if is_multi_model:
        model_names = list(test_results.keys())
        
        # Calculate average R² scores across models for ranking
        all_r2_scores = [test_results[name]['r2_scores'] for name in model_names]
        avg_r2_scores = np.mean(all_r2_scores, axis=0)
        
        # Use first model's targets (should be same for all)
        targets = test_results[model_names[0]]['targets']
    else:
        model_names = ['model']
        test_results = {'model': test_results}
        avg_r2_scores = np.array(test_results['model']['r2_scores'])
        targets = test_results['model']['targets']
    
    # Filter out nan values before sorting
    r2_array = np.array(avg_r2_scores)
    valid_mask = ~np.isnan(r2_array)
    valid_indices = np.where(valid_mask)[0]
    valid_r2_scores = r2_array[valid_indices]
    
    # Sort valid indices by their R² scores based on order
    if order == "top":
        # Highest to lowest (best performers)
        sorted_order = np.argsort(valid_r2_scores)[::-1]
    else:  # bottom
        # Lowest to highest (worst performers)
        sorted_order = np.argsort(valid_r2_scores)
    
    sorted_indices = valid_indices[sorted_order]
    
    # Select top n_intervals from valid intervals only
    n_plot = min(n_intervals, len(sorted_indices))
    top_indices = sorted_indices[:n_plot]
    
    # Create subplots
    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 3 * n_plot))
    
    # Handle single subplot case
    if n_plot == 1:
        axes = [axes]
    
    for i, idx in enumerate(top_indices):
        ax = axes[i]
        
        # Plot ground truth (same for all models)
        y_true = targets[idx].detach().cpu().numpy().flatten()
        ax.plot(y_true, label="Ground Truth", linewidth=2, alpha=0.9, color='black', linestyle='--')
        
        # Plot predictions for each model
        r2_parts = []
        for model_name in model_names:
            y_pred = test_results[model_name]['preds'][idx].detach().cpu().numpy().flatten()
            r2 = test_results[model_name]['r2_scores'][idx]
            
            # Plot with label
            label = f"{model_name}" if is_multi_model else "Prediction"
            ax.plot(y_pred, label=label, linewidth=1.5, alpha=0.8)
            
            # Build R² string
            if is_multi_model:
                r2_parts.append(f"{model_name}: {r2:.3f}")
            else:
                r2_parts.append(f"{r2:.3f}")
        
        # Add average R² if multi-model
        if is_multi_model:
            avg_r2 = avg_r2_scores[idx]
            r2_parts.append(f"avg: {avg_r2:.3f}")
            title_text = f"R² - {' | '.join(r2_parts)}"
        else:
            title_text = f"R² = {r2_parts[0]}"
        
        # Formatting
        ax.set_title(title_text, fontsize=12, fontweight='bold')
        ax.set_xlabel("Sample index", fontsize=10)
        ax.set_ylabel("Value (normalized)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
