from collections import defaultdict
from pathlib import Path
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
    plt.plot(r2_log)
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


def get_unit_ids(dataset, filter_str="motor"):
    unit_ids_list = []
    for k in dataset.recording_dict.keys():
        data = dataset.get_recording_data(k)
        valid_ids = list()
        for i, ln in zip(data.units.id, data.units.location_names):
            if filter_str in ln.lower():
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
    targets, preds = [], []
    
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
    
            targets.append(target[mask])
            preds.append(pred[mask])

    return targets, preds


def plot_test_results_from_lists(
    targets,
    preds,
    n_segments=5,
    segment_len=1500,
    seed=0,
    title_prefix="Test",
):
    """
    Given lists of masked target/pred tensors from your test loop, this function:
      1) concatenates them
      2) computes overall R^2
      3) plots n_segments random contiguous segments of length segment_len

    Parameters
    ----------
    targets, preds : list[torch.Tensor]
        Each element is a 1D tensor produced by target[mask] and pred[mask].
    n_segments : int
        Number of random segments to plot.
    segment_len : int
        Number of samples per plotted segment (on the concatenated masked stream).
    seed : int
        RNG seed for reproducible segment selection.
    title_prefix : str
        Prefix for plot titles.

    Returns
    -------
    r2 : float
    y_true : torch.Tensor (1D, CPU)
    y_pred : torch.Tensor (1D, CPU)
    chosen_starts : list[int]
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    def _r2_score_1d(y_pred_1d, y_true_1d):
        y_true_mean = torch.mean(y_true_1d)
        ss_total = torch.sum((y_true_1d - y_true_mean) ** 2)
        ss_res = torch.sum((y_true_1d - y_pred_1d) ** 2)
        # avoid divide-by-zero if signal is constant
        if ss_total.item() == 0.0:
            return torch.tensor(float("nan"), device=y_true_1d.device)
        return 1 - ss_res / ss_total

    # Concatenate and move to CPU for plotting
    y_true = torch.cat([t.detach().flatten().cpu() for t in targets], dim=0)
    y_pred = torch.cat([p.detach().flatten().cpu() for p in preds], dim=0)

    r2 = _r2_score_1d(y_pred, y_true).item()

    N = y_true.numel()
    if N == 0:
        raise ValueError("No samples found after masking. targets/preds are empty when concatenated.")

    # Determine valid start positions
    seg_len = int(segment_len)
    if seg_len <= 10:
        seg_len = min(200, N)  # sensible fallback
    seg_len = min(seg_len, N)

    max_start = N - seg_len
    rng = np.random.default_rng(seed)

    if max_start <= 0:
        chosen_starts = [0]
        n_plot = 1
    else:
        n_plot = min(int(n_segments), 20, max_start + 1)
        chosen_starts = rng.choice(max_start + 1, size=n_plot, replace=False)
        chosen_starts = sorted([int(s) for s in chosen_starts])

    # Plot
    plt.figure(figsize=(14, 2.8 * len(chosen_starts)))
    for i, s in enumerate(chosen_starts, start=1):
        e = s + seg_len
        ax = plt.subplot(len(chosen_starts), 1, i)
        ax.plot(y_true[s:e].numpy(), label="GT", linewidth=1.5)
        ax.plot(y_pred[s:e].numpy(), label="Pred", linewidth=1.5)
        ax.set_title(f"{title_prefix} segment {i}/{len(chosen_starts)} | idx {s}:{e} | overall R²={r2:.3f}")
        ax.set_xlabel("Sample index (within segment)")
        ax.set_ylabel("Wheel velocity (normalized)")
        ax.grid(True)
        if i == 1:
            ax.legend()

    plt.tight_layout()
    plt.show()

    return r2, y_true, y_pred, chosen_starts
