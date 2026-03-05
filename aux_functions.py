from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Callable, Literal, Optional
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import (
    matthews_corrcoef,
    balanced_accuracy_score,
    roc_auc_score,
)
from temporaldata import Data, IrregularTimeSeries
from torch_brain.dataset import Dataset, SpikingDatasetMixin
from torch_brain.data import collate
from torch_brain.data.sampler import RandomFixedWindowSampler, SequentialFixedWindowSampler


class IBLBrainWideMapDataset(SpikingDatasetMixin, Dataset):
    """Dataset for IBL Brain Wide Map data:
    - wraps the IBL processed HDF5 files
    - injects a readout config (normalization stats, readout_id) into each 
      recording via `get_recording_hook`.

    Args:
        root: Root directory containing the ``ibl_processed/`` subdirectory.
        readout_id: Name of the readout modality (e.g. ``"wheel_velocity"``).
        normalize_mean: Mean value used for output normalization.
        normalize_std: Std deviation used for output normalization.
        recording_ids: Optional list of recording IDs (h5 file stems) to include.
            If ``None``, all ``*.h5`` files in the dataset directory are used.
        transform: Optional transform applied.
        dirname: Subdirectory name under ``root`` where the h5 files live.
            Defaults to ``"ibl_processed"``.
    """

    def __init__(
        self,
        root: str,
        readout_id: str,
        normalize_mean: float,
        normalize_std: float,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        dirname: str = "ibl_processed",
        **kwargs,
    ):
        self._readout_config = {
            "readout": {
                "readout_id": readout_id,
                "timestamp_key": f"{readout_id}.timestamps",
                "value_key": f"{readout_id}.values",
                "normalize_mean": normalize_mean,
                "normalize_std": normalize_std,
            }
        }
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "units.id"],
            **kwargs,
        )
        # Prefix every unit ID with its session ID to ensure global uniqueness
        self.spiking_dataset_mixin_uniquify_unit_ids = True

    def get_recording_hook(self, data: Data):
        """Inject readout config into every loaded recording."""
        data.config = deepcopy(self._readout_config)
        super().get_recording_hook(data)

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
        task: Optional[str] = None,
    ):
        """Return per-recording sampling intervals, optionally filtered by split and task.

        If ``task`` is provided, the returned intervals are the intersection of
        the split domain with ``task_aligned_intervals.<task>`` for each
        recording.  The sampler will then only draw windows from within those
        task-aligned portions of the selected split.
        """
        domain_key = "domain" if split is None else f"{split}_domain"
        result = {}
        for rid in self.recording_ids:
            recording = self.get_recording(rid)
            domain = getattr(recording, domain_key)
            if task is not None:
                task_interval = getattr(recording.task_aligned_intervals, task)
                domain = domain & task_interval
            result[rid] = domain
        return result


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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
    
    url = "https://nyu1.osn.mghpcc.org/brainsets-public/model-zoo/poyo_1.ckpt"

    print("Downloading model...")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=16384):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded file to: {local_path}")


def compute_normalization_stats(
    dir_path,
    recording_ids: list[str],
    readout_id: str,
    dirname: str = "ibl_processed",
) -> tuple[float, float]:
    """Compute mean and std of a readout signal over training data.

    Scans the h5 files for each recording in ``recording_ids`` and computes
    descriptive stats from the ``train_domain`` portion of the data.

    Args:
        dir_path: Root directory containing the dataset subdirectory.
        recording_ids: List of recording IDs (h5 file stems) to scan.
        readout_id: Attribute path to the readout signal (e.g. ``"wheel_velocity"``).
        dirname: Subdirectory name under ``dir_path``.

    Returns:
        ``(mean, std)`` as floats.
    """
    values = np.array([])
    for rid in recording_ids:
        session_path = Path(dir_path) / dirname / f"{rid}.h5"
        with h5py.File(session_path, "r") as f:
            session_data = Data.from_hdf5(f, lazy=True)
            train = session_data.select_by_interval(session_data.train_domain)
            attr = train
            for part in readout_id.split("."):
                attr = getattr(attr, part)
            if hasattr(attr, "values"):
                attr = attr.values
            values = np.append(values, np.asarray(attr))
    return float(np.nanmean(values)), float(np.nanstd(values))


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


def pearson_r(y_pred, y_true):
    """Compute the Pearson correlation coefficient between two 1-D tensors.

    Returns ``NaN`` when either signal has near-zero variance (e.g. a quiet
    window where the mouse is stationary), rather than producing the large
    negative values that R² can produce in those cases.

    Args:
        y_pred: Predicted values, shape ``(N,)``.
        y_true: Ground-truth values, shape ``(N,)``.

    Returns:
        Scalar tensor in ``[-1, 1]``, or ``NaN`` if variance is too low.
    """
    pred_z   = y_pred - y_pred.mean()
    true_z   = y_true - y_true.mean()
    denom    = pred_z.norm() * true_z.norm()
    if denom < 1e-6:
        return torch.tensor(float("nan"))
    return (pred_z * true_z).sum() / denom


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


def compute_classification_metrics(dataloader, model):
    """Compute MCC, balanced accuracy and AUROC for a classification model.

    Analogous to :func:`compute_r2` for regression models.

    Args:
        dataloader: A :class:`torch.utils.data.DataLoader` whose batches have
            ``"model_inputs"`` and ``"target_values"`` (integer class indices).
        model: A trained :class:`~mlp.MLPNeuralClassifier` instance.

    Returns:
        dict with keys ``"mcc"``, ``"bal_acc"``, and ``"auroc"``.
    """

    model.eval()
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            batch  = move_to_device(batch)
            logits = model(**batch["model_inputs"])          # (B, num_classes)
            probs  = torch.softmax(logits, dim=-1)[:, 1]    # P(positive class)
            pred   = logits.argmax(dim=-1)
            all_preds.append(pred.cpu())
            all_probs.append(probs.cpu())
            all_targets.append(batch["target_values"].cpu())

    preds   = torch.cat(all_preds).numpy()
    probs   = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()

    return {
        "mcc":      float(matthews_corrcoef(targets, preds)),
        "bal_acc":  float(balanced_accuracy_score(targets, preds)),
        "auroc":    float(roc_auc_score(targets, probs)),
    }


def training_step(
    batch,
    model,
    optimizer,
    task_type: Optional[Literal["regression", "classification"]] = "regression",
):
    """Perform a single training step.

    Args:
        batch: Batch dict from the DataLoader (must have ``"model_inputs"`` and
            ``"target_values"`` keys).
        model: The model to train.
        optimizer: Optimizer instance.
        task_type: ``"regression"`` uses MSE loss (default); ``"classification"``
            uses cross-entropy loss (expects ``target_values`` to be integer class
            indices of dtype ``torch.long``).

    Returns:
        Scalar loss tensor.
    """
    # Step 0. Clear old gradients
    optimizer.zero_grad()

    inputs = batch["model_inputs"]
    target = batch["target_values"]

    # Step 1. Do forward pass
    pred = model(**inputs)

    # Step 2. Compute loss
    if task_type == "regression":
        # shapes: [B, T, 1] -> [B, T]
        if pred.dim() == 3 and pred.size(-1) == 1:
            pred = pred.squeeze(-1)
        loss = F.mse_loss(pred, target)
    else:  # classification
        # pred: (B, num_classes)  target: (B,) long
        loss = F.cross_entropy(pred, target.long())

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


def get_loaders(
    dir_path = ".",
    recording_ids: list[str] = None,
    readout_id: str = "wheel_velocity",
    task: Optional[str] = None,
    window_length: float = 1.0,
    batch_size: int = 16,
    seed: int = 0,
    device=None,
    dirname: str = "ibl_processed",
):
    """Set up a single Dataset and three DataLoaders (train / val / test).

    A *single* :class:`IBLBrainWideMapDataset` instance is created and shared
    across all three DataLoaders.  Each loader obtains its own sampler built
    from the appropriate split intervals (``"train"``, ``"valid"``, ``"test"``).

    The dataset's ``transform`` attribute can be swapped in-place at any time
    (e.g. ``dataset.transform = Transform(model)``); the change automatically
    affects all three loaders.

    Args:
        dir_path: Root directory containing the dataset subdirectory.
        recording_ids: List of h5 file stems to include. Must be provided.
        readout_id: Name of the readout modality (e.g. ``"wheel_velocity"``).
        task: Optional task name to filter sampling intervals (e.g. ``"choice"``).
        window_length: Sliding-window length in seconds.
        batch_size: Samples per batch.
        seed: Random seed for the training sampler.
        device: Target device.  Used only to decide multiprocessing settings.
        dirname: Subdirectory name under ``dir_path`` where h5 files live.

    Returns:
        ``(dataset, train_loader, val_loader, test_loader)``
    """
    if recording_ids is None:
        raise ValueError("recording_ids must be provided.")

    # Decide multiprocessing / pin-memory settings based on device
    use_multiproc = True
    use_pin_memory = True
    if device is None:
        device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda:0") if torch.cuda.is_available()
            else torch.device("cpu")
        )
    # On Apple MPS avoid multiprocessing/pinned memory to prevent _share_filename_ errors
    if device.type == "mps":
        use_multiproc = False
        use_pin_memory = False

    # Compute normalization statistics from the training portion of the data
    normalize_mean, normalize_std = compute_normalization_stats(
        dir_path=dir_path,
        recording_ids=recording_ids,
        readout_id=readout_id,
        dirname=dirname,
    )

    # A single dataset instance shared by all three loaders
    dataset = IBLBrainWideMapDataset(
        root=dir_path,
        readout_id=readout_id,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        recording_ids=recording_ids,
        dirname=dirname,
    )

    num_workers = 0 if not use_multiproc else 4

    train_sampler = RandomFixedWindowSampler(
        sampling_intervals=dataset.get_sampling_intervals("train", task=task),
        window_length=window_length,
        generator=torch.Generator().manual_seed(seed),
        drop_short=True,
    )
    train_loader = DataLoader(
        dataset=dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=False,
    )

    val_sampler = SequentialFixedWindowSampler(
        sampling_intervals=dataset.get_sampling_intervals("valid", task=task),
        window_length=window_length,
        drop_short=True,
    )
    val_loader = DataLoader(
        dataset=dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=False,
    )

    test_sampler = SequentialFixedWindowSampler(
        sampling_intervals=dataset.get_sampling_intervals("test", task=task),
        window_length=window_length,
        drop_short=True,
    )
    test_loader = DataLoader(
        dataset=dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=False,
    )

    return dataset, train_loader, val_loader, test_loader


def get_unit_ids(
    dataset: IBLBrainWideMapDataset,
    filter_str: list[str] = ["motor"],
    quality_score: float = 0.6,
) -> list:
    """Return unit IDs filtered by brain-area location name and quality score.

    Parameters
    ----------
    dataset:
        An :class:`IBLBrainWideMapDataset` instance.
    filter_str:
        List of strings matched (case-insensitive) against ``units.location_names``.
    quality_score:
        Minimum IBL quality score threshold.

    Returns
    -------
    list
        Filtered and sorted unit IDs (already prefixed with ``session_id/``
        by ``spiking_dataset_mixin_uniquify_unit_ids``).
    """
    unit_ids_list = []
    for rid in dataset.recording_ids:
        data = dataset.get_recording(rid)
        valid_ids = [
            i
            for i, ln, qs in zip(
                data.units.id,
                data.units.location_names,
                data.units.ibl_quality_score,
            )
            if any(fs in ln.lower() for fs in filter_str) and qs > quality_score
        ]
        unit_ids_list.extend(valid_ids)
    return unit_ids_list


class Transform:
    """Filter spikes to a pre-defined set of units and tokenize for the model.

    Parameters
    ----------
    model:
        A POYO model whose ``unit_emb.vocab`` defines the set of known units.
    """

    def __init__(self, model):
        self.model = model
        
        # Precompute valid unit indices (integer part) per session prefix
        out = defaultdict(list)
        for k in model.unit_emb.vocab:
            if k == 'NA':
                continue
            s = str(k)  # handle np.str_ safely
            prefix, unit = s.rsplit('/unit_', 1)
            out[prefix].append(int(unit))
        self.valid_units_per_recording = dict(out)
        
    def __call__(self, data):
        """Filter spikes to vocab units and tokenize."""
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
            unit_index=remapped_indices,
            domain="auto",
        )

        # Filter units metadata
        units_ids = [int(i.split("_")[-1]) for i in data.units.id]
        mask_units = np.isin(units_ids, valid_units)
        data.units = data.units.select_by_mask(mask_units)

        return self.model.tokenize(data)


def run_test(
    dataset: IBLBrainWideMapDataset,
    test_loader: DataLoader,
    model,
    device=None,
):
    """Run inference on the test set and collect per-interval R² scores.

    Args:
        dataset: The shared dataset instance.  Its ``transform`` will be
            replaced with a :class:`Transform` for ``model``.
        test_loader: DataLoader using the test split sampler.
        model: Finetuned POYO model.
        device: Torch device.

    Returns:
        dict with keys ``targets``, ``preds``, and ``r2_scores``.
    """
    # Attach the model's tokenizer/filter as the dataset transform
    dataset.transform = Transform(model=model)
    
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
                interval_pred   = pred[i]
                interval_target = target[i]
                interval_mask   = mask[i]
                
                masked_pred   = interval_pred[interval_mask]
                masked_target = interval_target[interval_mask]
                
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
        sorted_order = np.argsort(valid_r2_scores)[::-1]
    else:  # bottom
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
            
            label = f"{model_name}" if is_multi_model else "Prediction"
            ax.plot(y_pred, label=label, linewidth=1.5, alpha=0.8)
            
            if is_multi_model:
                r2_parts.append(f"{model_name}: {r2:.3f}")
            else:
                r2_parts.append(f"{r2:.3f}")
        
        if is_multi_model:
            title_text = f"r - {' | '.join(r2_parts)}"
        else:
            title_text = f"r = {r2_parts[0]}"
        
        # Formatting
        ax.set_title(title_text, fontsize=12, fontweight='bold')
        ax.set_xlabel("Sample index", fontsize=10)
        ax.set_ylabel("Value (normalized)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
