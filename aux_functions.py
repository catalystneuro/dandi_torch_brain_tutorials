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
from sklearn.metrics import matthews_corrcoef
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
        valid_labels: Optional[set] = None,
    ):
        """Return per-recording sampling intervals, optionally filtered by split, task, and label.

        If ``task`` is provided, the returned intervals are the intersection of
        the split domain with ``task_aligned_intervals.<task>`` for each
        recording.  The sampler will then only draw windows from within those
        task-aligned portions of the selected split.

        If ``valid_labels`` is also provided, only trial intervals whose label
        (the per-trial attribute stored under the same name as ``task``) is a
        member of ``valid_labels`` are kept.  Labels stored as bytes are
        decoded to ``str`` before comparison.  This is the recommended way to
        exclude trials with unseen labels (e.g. ``"no_go"``) from the sampler
        so they never reach the model.
        """
        domain_key = "domain" if split is None else f"{split}_domain"
        result = {}
        for rid in self.recording_ids:
            recording = self.get_recording(rid)
            domain = getattr(recording, domain_key)
            if task is not None:
                task_interval = getattr(recording.task_aligned_intervals, task)
                if valid_labels is not None:
                    trial_labels = getattr(task_interval, task)
                    decoded = [
                        lbl.decode() if isinstance(lbl, (bytes, np.bytes_)) else str(lbl)
                        for lbl in trial_labels
                    ]
                    mask = np.array([lbl in valid_labels for lbl in decoded])
                    task_interval = task_interval.select_by_mask(mask)
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
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch  = move_to_device(batch)
            logits = model(**batch["model_inputs"])   # (B, num_classes)
            pred   = logits.argmax(dim=-1)
            all_preds.append(pred.cpu())
            all_targets.append(batch["target_values"].cpu())

    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    return {
        "mcc": float(matthews_corrcoef(targets, preds)),
    }


def plot_cls_training_curves(cls_mcc_logs, cls_loss_logs, task=""):
    """Plot validation MCC per epoch and training cross-entropy loss.

    Args:
        cls_mcc_logs: dict {label: list of MCC values (one per epoch + final)}.
        cls_loss_logs: dict {label: list of loss values (one per training step)}.
        task: Task name string used in the figure title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for label in cls_mcc_logs:
        axes[0].plot(cls_mcc_logs[label], label=label, marker="o", markersize=3)
        axes[1].plot(cls_loss_logs[label], label=label, linewidth=0.8, alpha=0.85)
    axes[0].set_title("Validation MCC per epoch", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MCC")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Training cross-entropy loss", fontsize=12)
    axes[1].set_xlabel("Training steps")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Training curves — task: {task}", fontsize=13)
    plt.tight_layout()
    plt.show()


def run_cls_test(dataset, test_loader, cls_models, cls_unit_filters,
                 device=None, transform_cls=None):
    """Run test-set inference for all classification models and return metrics + predictions.

    Args:
        dataset: The shared :class:`IBLBrainWideMapDataset` instance.
        test_loader: :class:`torch.utils.data.DataLoader` for the test split.
        cls_models: dict ``{label: trained MLPNeuralClassifier}``.
        cls_unit_filters: dict ``{label: UnitFilter}`` from the training loop.
        device: Torch device.  Auto-detected if ``None``.
        transform_cls: A callable that wraps ``[unit_filter, model.tokenize]``
            into a single transform (typically
            :class:`torch_brain.transforms.Compose`).

    Returns:
        ``(cls_test_metrics, cls_test_preds, cls_test_targets_arr)`` —
        each a dict keyed by ``label``.
    """
    if device is None:
        device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda:0") if torch.cuda.is_available()
            else torch.device("cpu")
        )

    cls_test_metrics, cls_test_preds, cls_test_targets_arr = {}, {}, {}

    for label, m in cls_models.items():
        if transform_cls is not None:
            dataset.transform = transform_cls([cls_unit_filters[label], m.tokenize])
        m.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch  = move_to_device(batch, device)
                logits = m(**batch["model_inputs"])
                all_preds.append(logits.argmax(dim=-1).cpu())
                all_targets.append(batch["target_values"].cpu())

        preds   = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()

        cls_test_metrics[label]     = {"mcc": float(matthews_corrcoef(targets, preds))}
        cls_test_preds[label]       = preds
        cls_test_targets_arr[label] = targets
        print(f"{label:15s}  MCC={cls_test_metrics[label]['mcc']:.3f}")

    return cls_test_metrics, cls_test_preds, cls_test_targets_arr


def plot_cls_mcc_bar(cls_test_metrics, task=""):
    """Bar chart of test MCC per experiment (dashed line at 0 = chance).

    Args:
        cls_test_metrics: dict ``{label: {"mcc": float}}``.
        task: Task name string used in the figure title.
    """
    labels_list = list(cls_test_metrics.keys())
    mcc_vals    = [cls_test_metrics[l]["mcc"] for l in labels_list]
    x = np.arange(len(labels_list))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x, mcc_vals, color="steelblue")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, fontsize=12)
    ax.set_ylabel("MCC")
    ax.set_title(f"Test MCC — task: {task}", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cls_confusion_matrices(cls_test_metrics, cls_test_preds, cls_test_targets_arr,
                                 task="", class_names=None):
    """One confusion matrix per experiment, each subplot titled with its MCC.

    Args:
        cls_test_metrics: dict ``{label: {"mcc": float}}``.
        cls_test_preds: dict ``{label: np.ndarray}`` of predicted class indices.
        cls_test_targets_arr: dict ``{label: np.ndarray}`` of ground-truth class indices.
        task: Task name string used in the suptitle.
        class_names: List of two class-name strings (default ``["class 0", "class 1"]``).
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    if class_names is None:
        class_names = ["class 0", "class 1"]

    labels_list = list(cls_test_metrics.keys())
    fig, axes = plt.subplots(1, len(labels_list), figsize=(4 * len(labels_list), 4))
    if len(labels_list) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels_list):
        cm = confusion_matrix(cls_test_targets_arr[label], cls_test_preds[label])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{label}\nMCC={cls_test_metrics[label]['mcc']:.3f}", fontsize=11)

    fig.suptitle(f"Confusion matrices — task: {task}", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_population_mcc(population_results, task=""):
    """Violin + swarm plot of test MCC distributions across sessions, one violin per region condition.

    Each dot is one session; thin lines connect dots from the same session across conditions.
    The dashed line at 0 marks chance level.

    Args:
        population_results: dict ``{session_id: {label: mcc_float}}``.
        task: Task name string used in the figure title.
    """
    preferred_order = ["motor", "caudoputamen", "both"]
    all_labels = {lbl for ses in population_results.values() for lbl in ses}
    labels_list = [l for l in preferred_order if l in all_labels] + sorted(all_labels - set(preferred_order))
    n_conditions = len(labels_list)

    session_ids = list(population_results.keys())
    data = np.full((len(session_ids), n_conditions), np.nan)
    for i, sid in enumerate(session_ids):
        for j, lbl in enumerate(labels_list):
            data[i, j] = population_results[sid].get(lbl, np.nan)

    fig, ax = plt.subplots(figsize=(4 + n_conditions, 5))
    x_pos = np.arange(n_conditions)

    # Violin
    valid_cols = [data[:, j][~np.isnan(data[:, j])] for j in range(n_conditions)]
    if any(len(c) > 1 for c in valid_cols):
        parts = ax.violinplot(
            [c if len(c) > 1 else np.array([c[0], c[0]]) for c in valid_cols],
            positions=x_pos,
            showmedians=True,
            showextrema=True,
        )
        for pc in parts["bodies"]:
            pc.set_alpha(0.35)

    # Jittered individual dots + connecting lines across conditions per session
    rng = np.random.default_rng(0)
    for i in range(len(session_ids)):
        jitter = rng.uniform(-0.06, 0.06, n_conditions)
        row = data[i]
        valid = ~np.isnan(row)
        ax.plot(x_pos[valid] + jitter[valid], row[valid],
                color="steelblue", alpha=0.45, linewidth=0.8, zorder=2)
        ax.scatter(x_pos[valid] + jitter[valid], row[valid],
                   color="steelblue", s=30, zorder=3, alpha=0.8)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7, label="Chance (MCC=0)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_list, fontsize=12)
    ax.set_ylabel("Test MCC", fontsize=12)
    ax.set_title(f"Population MCC — task: {task}\n(n={len(session_ids)} sessions)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


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
    valid_labels: Optional[set] = None,
    window_length: float = 1.0,
    batch_size: int = 16,
    seed: int = 0,
    device=None,
    dirname: str = "ibl_processed",
):
    """Set up a single Dataset and three DataLoaders (train / val / test).

    A single :class:`IBLBrainWideMapDataset` instance is created and shared
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
        valid_labels: Optional set of label strings to include when ``task`` is
            set.  Trial intervals whose label (stored under the same key as
            ``task``) is **not** in this set are excluded from the sampler,
            so they never reach the model.  Pass ``set(LABEL_MAP.keys())``
            to silently discard ``"no_go"`` and any other unlabelled trials.
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
        sampling_intervals=dataset.get_sampling_intervals("train", task=task, valid_labels=valid_labels),
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
        sampling_intervals=dataset.get_sampling_intervals("valid", task=task, valid_labels=valid_labels),
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
        sampling_intervals=dataset.get_sampling_intervals("test", task=task, valid_labels=valid_labels),
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
