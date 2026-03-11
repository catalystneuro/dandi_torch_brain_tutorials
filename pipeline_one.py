# /// brainset-pipeline
# python-version = "3.10"
# dependencies = [
#   "ray<2.42",
#   "numpy~=1.23.5",
#   "ONE-api==3.1.1",
#   "ibllib==3.3.1",
#   "scipy==1.15.3",
# ]
# ///

import os

os.environ["ONE_REVISION_LAST_BEFORE"] = "2026-02-23"

import argparse
import logging
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import pandas as pd
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import RecordingTech, Species
from iblatlas.regions import BrainRegions
from one.alf.exceptions import ALFObjectNotFound
from one.api import ONE, One
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries
from utils import decimate_signal, regularize_timeseries, resample_timeseries

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--reprocess", action="store_true")
parser.add_argument("--list-sessions", type=str, default=False)
parser.add_argument(
    "--download-first",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether to download data first. Use --no-download-first to skip.",
)

PROBES_WITHOUT_DATA = ["f936a701-5f8a-4aa1-b7a9-9f8b5b69bc7c"]
QC_NEURAL_REMAPPING = {
    "Session ID": "eid",
    "Probe ID": "probe_id",
    "Quality (Raw data)": "qc_neural_raw",
    "QC Status": "qc_neural",
}
QC_BEHAVIOR_REMAPPING = {
    "Session ID": "eid",
    "leftCamera.paws": "qc_behavior_paws",
    "licks": "qc_behavior_licks",
    "leftCamera.ME": "qc_behavior_motion_energy",
    "wheel": "qc_behavior_wheel",
    "behavior_test_candidate": "qc_behavior",
}

TRIAL_KEYS_REMAPPING = {
    "intervals_0": "start",  # start of trial
    "stimOn_times": "stim_on_time",  # visual stimulus appearance
    "goCueTrigger_times": "go_cue_trigger_time",  # audio command sent
    "goCue_times": "go_cue_time",  # audio sound plays
    "firstMovement_times": "movement_onset_time",  # reaction start
    "response_times": "response_time",  # wheel threshold crossing
    "feedback_times": "feedback_time",  # outcome delivery
    "stimOff_times": "stim_off_time",  # visual stimulus removal
    "intervals_1": "end",  # end of trial
    "probabilityLeft": "probability_left",
    "feedbackType": "feedback_type",
    "rewardVolume": "reward_volume",
    "contrastLeft": "contrast_left",
    "contrastRight": "contrast_right",
}
TRIAL_TIMEKEYS = [
    "start",
    "end",
    "stim_on_time",
    "feedback_time",
    "go_cue_trigger_time",
    "go_cue_time",
    "stim_off_time",
    "response_time",
    "movement_onset_time",
]

# Switch left and right
LEFT_PAW_KEYS = {
    "pos": ["paw_r_x", "paw_r_y"],
    "likelihood": "paw_r_likelihood",
}

RIGHT_PAW_KEYS = {
    "pos": ["paw_l_x", "paw_l_y"],
    "likelihood": "paw_l_likelihood",
}

PRE_STIM_WINDOW = 0.9  # 0.9s before stimulus onset
POST_STIM_WINDOW = 0.1  # 0.1s after stimulus onset
BLOCK_PRIOR_WINDOW = 1.0  # 1s before stimulus onset
MOVEMENT_WINDOW = 1.0  # 1s after movement starts
FEEDBACK_WINDOW = 1.0  # 1s after reward/penalty

WHISKER_NOMINAL_FS = [60, 150]
POSE_NOMINAL_FS = [60, 150]

WHEEL_INIT_FS = 1000.0  # 1kHZ
WHEEL_DOWNSAMPLE_FACTORS = [10, 2]  # 1kHZ -> 100Hz -> 50Hz

GAP_TOLERANCE_SEC = 0.01  # 10HZ
TOLERANCE_HZ = 1.0  # 1HZ
TARGET_FS = 50  # 50kHZ

CONTRAST_MAP = {0.0: 0, 0.0625: 1, 0.125: 2, 0.25: 3, 1.0: 4}
REWARD_MAP = {-1: 0, 1: 1}
# In general 1 encode left and 0 right
STIMULUS_SIDE_MAP = {"Left": 1, "Right": 0}
CHOICE_MAP = {-1: 0, 0: 2, 1: 1}
BLOCK_MAP = {0.2: 0, 0.8: 1, 0.5: -1}  # the block with no prior (0.5) will be ignore

# filter trials
MIN_RT = 0.0
MAX_RT = 10.0
EXCLUDE_NO_CHOICE = True
KEYS_WITH_NAN_TO_EXCLUDE = [
    "stimOn_times",
    "choice",
    "feedback_times",
    "probabilityLeft",
    "firstMovement_times",
    "feedbackType",
]
QUERY = (
    f" | (firstMovement_times - stimOn_times < {MIN_RT})"
    f" | (firstMovement_times - stimOn_times > {MAX_RT})"
)
for event in KEYS_WITH_NAN_TO_EXCLUDE:
    QUERY += f" | {event}.isnull()"
if EXCLUDE_NO_CHOICE:
    QUERY += " | (choice == 0)"
SUCCESSFUL_TRIALS_QUERY = QUERY.lstrip(" |")
ONE_DETAILS_MAX_RETRIES = 3

NEURAL_QC_FILENAME = "bwm_ephys_qc.csv"
BEHAVIOR_QC_FILENAME = "bwm_behavior_qc.csv"

WHISKER_CAM = "leftCamera"
LIGHTNING_POSE_DSET = "_ibl_leftCamera.lightningPose.pqt"
LICK_TIMES_DSET = "licks.times.npy"
WHEEL_OBJ = "wheel"

PRETRAIN_TRAIN_RATIO = 0.9
PRETRAIN_VAL_RATIO = 0.1

EVAL_TRAIN_RATIO = 0.4
EVAL_VAL_RATIO = 0.2
EVAL_TEST_RATIO = 0.4


class Pipeline(BrainsetPipeline):
    brainset_id = "ibl_brain_wide_map_2023"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir: Path, args):
        # use pre-determined list of eids for train and test
        pipeline_dir = Path(__file__).resolve().parent
        with open(pipeline_dir / "pretrain_eids.txt") as fh:
            manifest_list = [
                {"id": line.strip(), "eid": line.strip(), "is_pretrain": True}
                for line in fh
            ]
        with open(pipeline_dir / "eval_eids.txt") as fh:
            manifest_list.extend(
                [
                    {"id": line.strip(), "eid": line.strip(), "is_pretrain": False}
                    for line in fh
                ]
            )

        manifest = pd.DataFrame(manifest_list).set_index("id")

        # load and store neural QC in manifest
        neural_qc_df = _get_neural_qc_df(pipeline_dir)
        manifest = manifest.merge(
            neural_qc_df,
            on="eid",
            how="left",
        ).set_index(manifest.index)

        # load and store behavior QC in manifest
        behavior_qc_df = _get_behavior_qc_df(pipeline_dir)
        manifest = manifest.merge(
            behavior_qc_df,
            on="eid",
            how="left",
        ).set_index(manifest.index)

        # connect to the ONE sdk
        # do this once only, and copy the handle for all manifest items
        # otherwise, an error will be raised from token file being overwritten
        one = ONE(
            base_url="https://openalyx.internationalbrainlab.org",
            username="intbrainlab",
            password="international",
            cache_dir=raw_dir,
        )
        manifest["one"] = one

        return manifest

    def download(self, manifest_item):
        if not self.args.download_first:
            self.update_status("Skipped Downloading (--no-download-first flag used)")
            # We still MUST return manifest_item so the process() method
            # has the eid and ONE instance to work with from the local cache!
            return manifest_item

        self.update_status(f"Starting Download")

        eid = manifest_item.eid
        one: One = manifest_item.one

        retries = 0
        while retries < ONE_DETAILS_MAX_RETRIES:
            try:
                self._download_from_one(one, eid)
                break
            except Exception as e:
                retries += 1
                if retries == ONE_DETAILS_MAX_RETRIES:
                    raise Exception(
                        f"Failed to download from ONE after {ONE_DETAILS_MAX_RETRIES} attempts"
                    )
                self.update_status(
                    f"Failed to download from ONE on attempt {retries}. Waiting before retrying..."
                )
                logging.warning(
                    f"Failed to download from ONE on attempt {retries}: {e}"
                )
                time.sleep(5)  # wait 5 seconds before retrying

        self.update_status(f"Ending Download")

        return manifest_item

    def _download_from_one(self, one: One, eid: str):
        one_details = one.get_details(eid)
        local_path = one_details["local_path"]
        session_loader = SessionLoader(one, session_path=local_path, eid=eid)

        self.update_status(f"Downloading Spikes")
        pids, probe_names = one.eid2pid(eid)
        for pid, probe_name in zip(pids, probe_names):
            if str(pid) in PROBES_WITHOUT_DATA:
                logging.warning(f"Probe {pid} has no data, skipping")
                continue
            spike_loader = SpikeSortingLoader(
                pid=pid, one=one, eid=eid, pname=probe_name
            )
            spike_loader.download_spike_sorting()

        self.update_status(f"Downloading Whisker")
        _one_load_object(
            one,
            eid,
            obj=WHISKER_CAM,
            collection="alf",
            attribute=["ROIMotionEnergy", "times"],
            download_only=True,
        )

        self.update_status(f"Downloading Pose")
        _one_load_dataset(
            one, eid, dataset=LIGHTNING_POSE_DSET, collection="alf", download_only=True
        )

        self.update_status(f"Downloading Wheel")
        _one_load_object(one, eid, obj=WHEEL_OBJ, collection="alf", download_only=True)

        self.update_status(f"Downloading Lick")
        _one_load_dataset(
            one, eid, dataset=LICK_TIMES_DSET, collection="alf", download_only=True
        )

        self.update_status(f"Downloading Trial")
        session_loader.load_trials()

    def process(self, manifest_item):
        eid = manifest_item.eid
        is_pretrain = manifest_item.is_pretrain
        one: One = manifest_item.one

        self.processed_dir.mkdir(exist_ok=True, parents=True)

        store_path = self.processed_dir / f"{eid}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status(f"Skipped Processing")
            return

        self.update_status(f"Processing session")

        brainset_description = BrainsetDescription(
            id="ibl_brain_wide_map_2023",
            origin_version="",
            derived_version="1.0.0",
            source="one-api",
            description=(
                "A key challenge in neuroscience is understanding how neurons in hundreds of interconnected brain regions integrate sensory inputs with "
                "previous expectations to initiate movements and make decisions. It is difficult to meet this challenge if different laboratories apply "
                "different analyses to different recordings in different regions during different behaviours. Here we report a comprehensive set of recordings "
                "from 621,733 neurons recorded with 699 Neuropixels probes across 139 mice in 12 laboratories. The data were obtained from mice performing "
                "a decision-making task with sensory, motor and cognitive components. The probes covered 279 brain areas in the left forebrain and midbrain "
                "and the right hindbrain and cerebellum. We provide an initial appraisal of this brain-wide map and assess how neural activity encodes key "
                "task variables. Representations of visual stimuli transiently appeared in classical visual areas after stimulus onset and then spread to "
                "ramp-like activity in a collection of midbrain and hindbrain regions that also encoded choices. Neural responses correlated with impending "
                "motor action almost everywhere in the brain. Responses to reward delivery and consumption were also widespread. This publicly available "
                "dataset represents a resource for understanding how computations distributed across and within brain areas drive behaviour."
            ),
        )

        # get subject metadata
        one_details = one.get_details(eid)
        subject_id = one_details["subject"]

        subject = SubjectDescription(
            id=subject_id,
            species=Species.MUS_MUSCULUS,
        )

        # extract experiment metadata
        recording_date = datetime.fromisoformat(one_details["start_time"]).strftime(
            "%Y%m%d"
        )
        lab_id = one_details["lab"]

        device_id = f"{lab_id}_{subject.id}_{recording_date}"

        # register session
        session_description = SessionDescription(
            id=eid,
            recording_date=datetime.strptime(recording_date, "%Y%m%d"),
        )

        # register device
        device_description = DeviceDescription(
            id=device_id,
            recording_tech=RecordingTech.NEUROPIXELS_ARRAY,
        )

        # extract spiking activity
        self.update_status(f"Extracting Spikes")
        spikes, units = extract_spikes(one, eid)

        # register neural QC into units table
        self.update_status(f"Registering Neural QC")
        units = register_neural_qc(units, manifest_item)

        # extract behavioral data
        self.update_status(f"Extracting Behavioral Data")

        session_loader = SessionLoader(
            one, session_path=one_details["local_path"], eid=eid
        )

        behavior_dict = {}

        if manifest_item.qc_behavior != "FAIL":
            # whisker motion energy
            # whisker data is always available and all other behavior timestamps will be aligned to it
            self.update_status(f"Extracting Whisker Data")
            whisker, whisker_raw_timestamps = extract_whisker(session_loader)
            whisker_resampled_timestamps = whisker.timestamps

            validate_whisker(whisker, ref_fs=TARGET_FS)
            behavior_dict["whisker"] = whisker

            # pose
            self.update_status(f"Extracting Pose Data")
            pose = extract_pose(one, eid, ref_raw_timestamps=whisker_raw_timestamps)

            if pose is not None:
                validate_pose(
                    pose,
                    ref_fs=TARGET_FS,
                    ref_timestamps=whisker_resampled_timestamps,
                )
                behavior_dict["pose"] = pose
            else:
                logging.warning(f"Pose data is not available or skipped because of QC")

            # lick
            # note: this file isn't always present (typically because the data from the right camera is not usable)
            self.update_status(f"Extracting Lick Data")
            licks = extract_licks(
                one, eid, ref_resampled_timestamps=whisker_resampled_timestamps
            )
            if licks is not None:
                validate_licks(
                    licks,
                    ref_fs=TARGET_FS,
                    ref_timestamps=whisker_resampled_timestamps,
                )
                behavior_dict["licks"] = licks
            else:
                logging.warning(f"Lick data is not available or skipped because of QC")

            # wheel
            self.update_status(f"Extracting Wheel Data")
            wheel = extract_wheel(
                session_loader, ref_resampled_timestamps=whisker_resampled_timestamps
            )
            if wheel is not None:
                validate_wheel(
                    wheel,
                    ref_fs=TARGET_FS,
                    ref_timestamps=whisker_resampled_timestamps,
                )
                behavior_dict["wheel"] = wheel
            else:
                logging.warning(f"Wheel data is not available or skipped because of QC")

        # register behavior QC
        self.update_status(f"Registering Behavior QC")
        behavior_qc = get_beh_qc(manifest_item)

        # extract trial data
        self.update_status(f"Extracting Trial and Task Intervals Data")
        trials = load_trials(session_loader)

        task_aligned_intervals = extract_task_aligned_intervals(trials)

        # register session
        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            lab=lab_id,
            # neural activity
            spikes=spikes,
            units=units,
            # stimuli and behavior
            trials=trials,
            task_aligned_intervals=task_aligned_intervals,
            **behavior_dict,
            behavior_qc=behavior_qc,
            domain=spikes.domain,
        )

        # define splits trial data
        self.update_status(f"Extracting Splits")

        if is_pretrain:
            train_ratio = PRETRAIN_TRAIN_RATIO
            valid_ratio = PRETRAIN_VAL_RATIO
            test_ratio = 0
        else:
            train_ratio = EVAL_TRAIN_RATIO
            valid_ratio = EVAL_VAL_RATIO
            test_ratio = EVAL_TEST_RATIO

        train_domain, valid_domain, test_domain = make_causal_splits(
            data.domain,
            task_aligned_intervals.domain,
            train_ratio,
            valid_ratio,
            test_ratio,
        )
        data.set_train_domain(train_domain)
        data.set_valid_domain(valid_domain)
        data.set_test_domain(test_domain)

        data.normalize = _get_beh_normalize(data, train_domain)

        # save data to disk
        self.update_status(f"Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        try:
            one.save_cache()  # explicitly save before shutdown
        except Exception as e:
            logging.warning(f"Error saving cache: {e}")


def _get_beh_normalize(data: Data, train_domain: Interval):
    """Compute dataset-level behavior normalization stats for present modalities.

    Returns nested stats under:
    normalize.<modality>.<signal>.mean/std
    """

    eps = 1e-8
    normalize = Data()
    train_slice = data.slice(
        train_domain.start[0],
        train_domain.end[0],
        reset_origin=False,
    )

    specs = {
        "whisker": ["motion_energy"],
        "wheel": ["speed"],
        "pose": ["right_paw_v_xy", "left_paw_v_xy"],
        "licks": ["licking_rate"],
    }

    for modality_name, signal_names in specs.items():
        modality = getattr(train_slice, modality_name, None)
        modality_norm = Data()

        if modality is not None:
            for signal_name in signal_names:
                if not hasattr(modality, signal_name):
                    continue
                values = np.asarray(getattr(modality, signal_name))
                setattr(
                    modality_norm,
                    signal_name,
                    Data(
                        mean=values.mean(axis=0),
                        std=np.maximum(values.std(axis=0), eps),
                    ),
                )

        setattr(normalize, modality_name, modality_norm)

    return normalize


def _get_neural_qc_df(pipeline_dir):
    neural_qc_df = pd.read_csv(pipeline_dir / NEURAL_QC_FILENAME)
    neural_qc_df = neural_qc_df[QC_NEURAL_REMAPPING.keys()].rename(
        columns=QC_NEURAL_REMAPPING
    )

    assert len(neural_qc_df.probe_id.unique()) == len(
        neural_qc_df
    ), "There are duplicate probe IDs in the neural QC dataframe"

    neural_qc_df["probe_num"] = neural_qc_df.groupby("eid").cumcount() + 1
    nc_pivot = neural_qc_df.pivot(index="eid", columns="probe_num")
    nc_pivot.columns = [f"{col}{num}" for (col, num) in nc_pivot.columns]
    neural_qc_df = nc_pivot.reset_index()

    assert len(neural_qc_df.eid.unique()) == len(
        neural_qc_df
    ), "There are duplicate eids in the neural QC dataframe"

    return neural_qc_df


def _get_behavior_qc_df(pipeline_dir):
    behavior_qc_df = pd.read_csv(pipeline_dir / BEHAVIOR_QC_FILENAME)
    behavior_qc_df = behavior_qc_df[QC_BEHAVIOR_REMAPPING.keys()].rename(
        columns=QC_BEHAVIOR_REMAPPING
    )
    assert len(behavior_qc_df.eid.unique()) == len(
        behavior_qc_df
    ), "There are duplicate eids in the behavior QC dataframe"

    return behavior_qc_df


def _one_load_object(one: One, eid: str, obj: str, **kwargs):
    try:
        return one.load_object(eid, obj=obj, **kwargs)
    except ALFObjectNotFound as e:
        logging.warning(f"Unable to download {obj} object, skipping")
        logging.warning(e)
        return None


def _one_load_dataset(one: One, eid: str, dataset: str, **kwargs):
    try:
        return one.load_dataset(eid, dataset=dataset, **kwargs)
    except ALFObjectNotFound as e:
        logging.warning(f"Unable to download {dataset} dataset, skipping")
        logging.warning(e)
        return None


def extract_spikes(one: One, eid: str):
    """Load and reindex spike-sorting outputs across probes for one session."""

    # Resolve probe insertion IDs and names for this experiment session.
    pids, probe_names = one.eid2pid(eid)

    spikes_list, units_list = [], []
    unit_ptr = 0

    # Load per-probe spikes and clusters, then concatenate them into
    # a session-level spike stream with globally unique unit indices.
    for pid, probe_name in zip(pids, probe_names):
        spike_loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=probe_name)
        spikes, clusters, channels = spike_loader.load_spike_sorting()

        if len(clusters) == 0:
            # A small number of known probe insertions are expected to have no cluster output;
            # this guards against silently dropping unexpected cases.
            assert str(pid) in PROBES_WITHOUT_DATA, f"Pid {pid} has no clusters"
            continue

        # Merge cluster metadata for unit-level annotations.
        clusters = SpikeSortingLoader.merge_clusters(
            spikes, clusters, channels, compute_metrics=False
        )

        # Offset probe-local cluster IDs so unit indices are unique across probes.
        spikes["clusters"] += unit_ptr
        spikes_list.append(spikes)

        # Build unit table and remap anatomical acronyms to Beryl labels.
        clusters["pid"] = str(pid)
        units = pd.DataFrame(clusters).rename(columns={"uuids": "id"})
        units["region"] = BrainRegions().acronym2acronym(
            clusters["acronym"], mapping="Beryl"
        )
        units_list.append(units)

        # Advance index offset for the next probe.
        num_units = len(clusters["cluster_id"])
        unit_ptr += num_units

        assert num_units == len(
            np.unique(clusters["cluster_id"])
        ), f"There are duplicate units"

        assert num_units == len(
            np.unique(spikes["clusters"])
        ), f"There are units that have no spikes"

    # Store spikes and sort the timestamps.
    spikes = IrregularTimeSeries(
        timestamps=np.concatenate([s["times"] for s in spikes_list]),
        unit_index=np.concatenate([s["clusters"] for s in spikes_list]),
        amplitude=np.concatenate([s["amps"] for s in spikes_list]),
        depth=np.concatenate([s["depths"] for s in spikes_list]),
        domain="auto",
    )
    spikes.sort()

    # Store units.
    units = ArrayDict.from_dataframe(
        pd.concat(units_list, ignore_index=True),
        unsigned_to_long=True,
    )
    units.location = np.stack([units.x, units.y, units.z], axis=1)

    # Unit UUIDs should be globally unique after concatenation.
    assert len(units.id) == len(np.unique(units.id)), f"uuids is not unique"

    return spikes, units


def register_neural_qc(units: ArrayDict, manifest_item: NamedTuple):
    """Register neural QC into units table"""

    qc_fields = [n for n in QC_NEURAL_REMAPPING.values() if n != "eid"]
    num_probes = len(np.unique(units.pid))

    for qc_field in qc_fields:
        pid_to_val = {
            getattr(manifest_item, f"probe_id{probe_num}"): getattr(
                manifest_item, f"{qc_field}{probe_num}"
            )
            for probe_num in range(1, num_probes + 1)
        }
        qc_arr = np.array([pid_to_val[pid] for pid in units.pid], dtype=object)
        setattr(units, qc_field, qc_arr)

    return units


def get_beh_qc(manifest_item: NamedTuple):
    """Construct an object to store behavior QC from manifest item"""

    qc_fields = [n for n in QC_BEHAVIOR_REMAPPING.values() if n != "eid"]
    beh_qc = Data(
        **{qc_field: getattr(manifest_item, qc_field) for qc_field in qc_fields}
    )
    return beh_qc


def extract_whisker(session_loader: SessionLoader, return_full: bool = False):
    """Load whisker motion energy, regularize timestamps, and resample to the target fs."""

    session_loader.load_motion_energy(views=["left"])

    df_motion_energy = session_loader.motion_energy["leftCamera"]
    raw_timestamps = df_motion_energy.times.values
    motion_energy = df_motion_energy.whiskerMotionEnergy.values

    assert len(raw_timestamps) == len(
        motion_energy
    ), f"Number of timestamps ({len(raw_timestamps)}) doesn't match number of motion energy values ({len(motion_energy)})"

    whisker = IrregularTimeSeries(
        timestamps=raw_timestamps, motion_energy=motion_energy, domain="auto"
    )

    # First, regularize the timestamp grid.
    regularized_whisker = regularize_timeseries(whisker, gap_tol=GAP_TOLERANCE_SEC)
    whisker_fs = 1 / np.median(np.diff(regularized_whisker.timestamps))

    # The source video is expected at 60 Hz or 150 Hz.
    assert any(
        np.abs(whisker_fs - fs) < TOLERANCE_HZ for fs in WHISKER_NOMINAL_FS
    ), f"Whisker sampling fs is not one of {WHISKER_NOMINAL_FS}Hz (or within {TOLERANCE_HZ}Hz), it is {whisker_fs}"

    # Then resample to the shared target fs (50 Hz).
    resampled_whisker = resample_timeseries(regularized_whisker, target_fs=TARGET_FS)

    # Return the resampled trace and raw timestamps for cross-modal alignment.
    if not return_full:
        return resampled_whisker, raw_timestamps

    return {
        "raw": whisker,
        "regularized": regularized_whisker,
        "resampled": resampled_whisker,
        "regularized_fs": whisker_fs,
        "resampled_fs": TARGET_FS,
    }


def extract_pose(
    one: One,
    eid: str,
    ref_raw_timestamps: np.ndarray,  # reference timestamps in the raw fs
    return_full: bool = False,  # return all data, not just the final resampled pose data
):
    """Load pose keypoints, regularize timestamps, and resample to the target fs."""

    pose_df = _one_load_dataset(one, eid, dataset=LIGHTNING_POSE_DSET, collection="alf")
    if pose_df is None:
        return None

    # The pose file does not include timestamps. Because whisker and pose data
    # are derived from the same video, reuse the video timestamps.
    t, len_val = len(ref_raw_timestamps), len(pose_df)
    assert (
        t == len_val
    ), f"Number of timestamps ({t}) doesn't match number of frames ({len_val})"

    # Extract paw keypoints (exclude nose, pupil, tube and tongue keypoints).
    # The tube is static, and tongue dynamics are represented by lick events.
    # Build pose time series on the video timestamp axis.
    pose = IrregularTimeSeries(
        timestamps=ref_raw_timestamps,
        # paws
        left_paw=pose_df[LEFT_PAW_KEYS["pos"]].to_numpy(),
        left_paw_likelihood=pose_df[LEFT_PAW_KEYS["likelihood"]].to_numpy(),
        right_paw=pose_df[RIGHT_PAW_KEYS["pos"]].to_numpy(),
        right_paw_likelihood=pose_df[RIGHT_PAW_KEYS["likelihood"]].to_numpy(),
        domain="auto",
    )

    # First, regularize both streams to a nearly uniform timestamp grid.
    regularized_pose = regularize_timeseries(pose, gap_tol=GAP_TOLERANCE_SEC)
    pose_fs = 1.0 / np.median(np.diff(regularized_pose.timestamps))

    # Validate that observed fss match expected nominal frame fss.
    assert any(
        np.abs(pose_fs - fs) < TOLERANCE_HZ for fs in POSE_NOMINAL_FS
    ), f"Pose sampling fs is not one of {POSE_NOMINAL_FS}Hz (or within {TOLERANCE_HZ}Hz), it is {pose_fs}"

    # Derive per-axis velocity for key tracked keypoints.
    dt = 1 / pose_fs
    regularized_pose.left_paw_v_xy = np.gradient(regularized_pose.left_paw, dt, axis=0)
    regularized_pose.right_paw_v_xy = np.gradient(
        regularized_pose.right_paw, dt, axis=0
    )

    # Resample both streams to the common 50 Hz target fs.
    resampled_pose = resample_timeseries(regularized_pose, target_fs=TARGET_FS)

    if not return_full:
        return resampled_pose

    return {
        "raw": pose,
        "regularized": regularized_pose,
        "resampled": resampled_pose,
        "regularized_fs": pose_fs,
        "resampled_fs": TARGET_FS,
    }


def extract_licks(
    one: One,
    eid: str,
    ref_resampled_timestamps: np.ndarray,
    return_full: bool = False,
):
    """Load lick timestamps, bin them on the pose grid, and return lick fs."""

    lick_timestamps = _one_load_dataset(
        one, eid, dataset=LICK_TIMES_DSET, collection="alf"
    )
    if lick_timestamps is None:
        return None

    binned_licking = np.zeros_like(ref_resampled_timestamps, dtype=np.int32)
    bin_index = (lick_timestamps - ref_resampled_timestamps[0]) * TARGET_FS
    bin_index = np.floor(bin_index).astype(int)
    np.add.at(binned_licking, bin_index, 1)

    # Convert binned counts to an approximate per-second fs.
    rate = binned_licking * float(TARGET_FS)

    licks = IrregularTimeSeries(
        timestamps=ref_resampled_timestamps, licking_rate=rate, domain="auto"
    )

    if not return_full:
        return licks

    return {
        "timestamps": lick_timestamps,
        "binned": binned_licking,
        "licks": licks,
        "target_fs": TARGET_FS,
    }


def extract_wheel(
    session_loader: SessionLoader,
    ref_resampled_timestamps: np.ndarray,
    return_full: bool = False,
):
    """Load wheel data, regularize timestamps, and resample to the target fs."""

    session_loader.load_wheel()

    timestamps = session_loader.wheel["times"].to_numpy().astype(np.float64)
    pos = session_loader.wheel["position"].to_numpy()
    vel = session_loader.wheel["velocity"].to_numpy()

    wheel = IrregularTimeSeries(timestamps=timestamps, pos=pos, vel=vel, domain="auto")

    # We will resample the wheel data to 50Hz and align it with the target timestamps
    # check that target_timestamps are within the range of the raw timestamps
    start_timestamp = timestamps[0] + np.remainder(
        np.abs(timestamps[0] - ref_resampled_timestamps[0]), 1 / TARGET_FS
    )
    end_timestamp = timestamps[-1] - np.remainder(
        np.abs(timestamps[-1] - ref_resampled_timestamps[0]), 1 / TARGET_FS
    )

    regularized_wheel = regularize_timeseries(
        wheel,
        gap_tol=GAP_TOLERANCE_SEC,
        target_fs=WHEEL_INIT_FS,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )

    # Move from 1kHz to 50Hz
    stride = int(WHEEL_INIT_FS // TARGET_FS)
    timestamps = regularized_wheel.timestamps[::stride]

    # step 2: Decimate from 1kHz to 100Hz, then 100Hz to 50Hz
    pos = decimate_signal(
        regularized_wheel.pos, WHEEL_DOWNSAMPLE_FACTORS, WHEEL_INIT_FS
    )
    vel = decimate_signal(
        regularized_wheel.vel, WHEEL_DOWNSAMPLE_FACTORS, WHEEL_INIT_FS
    )

    resampled_wheel = IrregularTimeSeries(
        timestamps=timestamps,
        pos=pos,
        vel=vel,
        speed=np.abs(vel),
        domain="auto",
    )

    if not return_full:
        return resampled_wheel

    return {
        "raw": wheel,
        "regularized": regularized_wheel,
        "resampled": resampled_wheel,
        "regularized_fs": WHEEL_INIT_FS,
        "resampled_fs": TARGET_FS,
    }


def load_trials(session_loader: SessionLoader):
    """Load trials data, remap columns, convert to Interval objects, and filter trials."""

    session_loader.load_trials()
    trials = session_loader.trials

    trials = trials.rename(columns=TRIAL_KEYS_REMAPPING)

    # relevant columns: choice, probability_left, ...
    trials = Interval.from_dataframe(trials, timekeys=TRIAL_TIMEKEYS)

    # choice is -1, 0 or 1, we will map it to 0, 2 and 1 respectively
    trials.choice = pd.Series(trials.choice).map(CHOICE_MAP).to_numpy()

    # feedback -1 is 0 and feedback 1 is 1
    trials.reward = pd.Series(trials.feedback_type).map(REWARD_MAP).to_numpy()

    # block is 0.2, 0.5 or 0.8, we will map it to 0, 1 and 2 respectively
    trials.block = pd.Series(trials.probability_left).map(BLOCK_MAP).to_numpy()

    # 0 if right, 1 if left
    side_labels = np.where(np.isnan(trials.contrast_left), "Right", "Left")
    trials.stimulus_side = pd.Series(side_labels).map(STIMULUS_SIDE_MAP).to_numpy()

    # sanity check: the encoded task variables are compatible
    assert np.all(
        # rewarded iff choice is not no choice (0) and choice is the same as stimulus side
        ((trials.choice != CHOICE_MAP[0]) & (trials.choice == trials.stimulus_side))
        == trials.reward
    ), "Stimulus side and contrast are not compatible"

    # left and right contrasts are 0., 0.0625, 0.125, 0.25, 1.0
    contrast = np.nan_to_num(trials.contrast_left) + np.nan_to_num(
        trials.contrast_right
    )
    trials.stimulus_contrast = pd.Series(contrast).map(CONTRAST_MAP).to_numpy()

    # filter trials
    trials.successful = ~session_loader.trials.eval(SUCCESSFUL_TRIALS_QUERY).values

    assert trials.is_disjoint(), f"Trials are not disjoint"
    trials.sort()

    return trials


def extract_task_aligned_intervals(trials: Interval):
    """Extract task-aligned intervals from trials data."""

    trials_idx = np.arange(len(trials))
    trials_idx = trials_idx[trials.successful]
    trials = trials.select_by_mask(trials.successful)

    # stimulus side
    stimulus_side = Interval(
        start=trials.stim_on_time - PRE_STIM_WINDOW,
        end=trials.stim_on_time + POST_STIM_WINDOW,
        stimulus_side=trials.stimulus_side,
        trials_idx=trials_idx,
    )
    # stimulus contrast
    stimulus_contrast = Interval(
        start=trials.stim_on_time - PRE_STIM_WINDOW,
        end=trials.stim_on_time + POST_STIM_WINDOW,
        stimulus_contrast=trials.stimulus_contrast,
        trials_idx=trials_idx,
    )

    # choice
    choice = Interval(
        start=trials.movement_onset_time,
        end=trials.movement_onset_time + MOVEMENT_WINDOW,
        choice=trials.choice,
        trials_idx=trials_idx,
    )

    # movement_intervals
    movement_intervals = Interval(
        start=trials.movement_onset_time,
        end=trials.movement_onset_time + MOVEMENT_WINDOW,
        trials_idx=trials_idx,
    )

    # reward
    reward = Interval(
        start=trials.feedback_time,
        end=trials.feedback_time + FEEDBACK_WINDOW,
        reward=trials.reward,
        trials_idx=trials_idx,
    )

    # block prior (we ignore the first block unbiased)
    block_prior = Interval(
        start=trials.stim_on_time - BLOCK_PRIOR_WINDOW,
        end=trials.stim_on_time,
        block=trials.block,
        trials_idx=trials_idx,
    )
    block_prior_mask = block_prior.block != BLOCK_MAP[0.5]
    block_prior = block_prior.select_by_mask(block_prior_mask)

    # sanity check: all the block with no prior are ignore (0.5 block prior)
    assert np.all(
        block_prior.block != BLOCK_MAP[0.5]
    ), "Some blocks have a prior of 0.5"

    tasks = [
        stimulus_side,
        stimulus_contrast,
        choice,
        movement_intervals,
        reward,
    ]

    # Get the length from the first valid attribute
    num_indices = len(tasks[0])
    assert all(len(t) == num_indices for t in tasks), "Tasks have mismatched lengths"
    assert (
        len(block_prior_mask) == num_indices
    ), "Block prior mask has mismatched length"

    domain_starts = np.full(num_indices, np.inf)
    domain_ends = np.full(num_indices, -np.inf)

    for task in tasks:
        # Using [:] to force loading from HDF5 into memory
        domain_starts = np.minimum(domain_starts, task.start[:])
        domain_ends = np.maximum(domain_ends, task.end[:])

    domain_starts[block_prior_mask] = np.minimum(
        domain_starts[block_prior_mask], block_prior.start[:]
    )
    domain_ends[block_prior_mask] = np.maximum(
        domain_ends[block_prior_mask], block_prior.end[:]
    )

    if np.any(domain_starts == np.inf) or np.any(domain_ends == -np.inf):
        raise AssertionError(f"Found trials with NO task-aligned data")

    domain = Interval(start=domain_starts, end=domain_ends)

    assert domain.is_disjoint(), "task_aligned_intervals domain is not disjoint"

    task_aligned_intervals = Data(
        block_prior=block_prior,
        stimulus_side=stimulus_side,
        stimulus_contrast=stimulus_contrast,
        choice=choice,
        reward=reward,
        movement_intervals=movement_intervals,
        domain=domain,
    )

    return task_aligned_intervals


def _snap_border_to_task_edge(border: float, task_domain: Interval) -> float:
    # Assumes task_domain is sorted + disjoint
    # border cuts this task
    mask = (task_domain.start < border) & (task_domain.end > border)
    if not np.any(mask):
        return border

    i = np.flatnonzero(mask)[0]  # disjoint => at most one
    left_edge = task_domain.start[i]
    right_edge = task_domain.end[i]

    # pick closest edge to preserve target ratios as much as possible
    if abs(border - left_edge) <= abs(right_edge - border):
        return left_edge
    else:
        return right_edge


def make_causal_splits(
    domain: Interval,
    task_aligned_domain: Interval,
    train_ratio: float = 0.4,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.4,
):
    assert np.isclose(
        train_ratio + valid_ratio + test_ratio, 1.0
    ), "split are not summing to one"

    start, end = domain.start[0], domain.end[-1]
    size_domain = end - start

    b1 = start + size_domain * train_ratio
    b2 = b1 + size_domain * valid_ratio

    b1 = _snap_border_to_task_edge(b1, task_aligned_domain)
    b2 = _snap_border_to_task_edge(b2, task_aligned_domain)

    b2 = max(b2, b1)

    train_domain = Interval(start, b1)
    valid_domain = Interval(b1, b2)
    if test_ratio > 0:
        test_domain = Interval(b2, end)
    else:
        test_domain = Interval(np.array([]), np.array([]))

    assert len(train_domain & valid_domain) == 0, "Leakage between train and val"
    assert len(valid_domain & test_domain) == 0, "Leakage between val and test"
    assert len(train_domain & test_domain) == 0, "Leakage between train and test"

    return train_domain, valid_domain, test_domain


def validate_whisker(whisker, ref_fs):
    assert np.allclose(
        np.diff(whisker.timestamps), 1 / ref_fs
    ), f"Whisker timestamps are not at the expected fs"


def validate_pose(pose, ref_fs, ref_timestamps):
    assert np.allclose(
        np.diff(pose.timestamps), 1 / ref_fs
    ), f"Pose timestamps are not at the expected fs"
    assert np.allclose(
        pose.timestamps, ref_timestamps
    ), f"Pose timestamps do not match whisker timestamps"


def validate_licks(licks, ref_fs, ref_timestamps):
    assert np.allclose(
        np.diff(licks.timestamps), 1 / ref_fs
    ), f"Lick timestamps are not at the expected fs"
    assert np.allclose(
        licks.timestamps, ref_timestamps
    ), f"Lick timestamps do not match whisker timestamps"


def validate_wheel(wheel, ref_fs, ref_timestamps):
    assert np.allclose(
        np.diff(wheel.timestamps), 1 / ref_fs
    ), f"Wheel timestamps are not at the expected fs"
    # TODO this check fails, double check whether timestamps are aligned
    # # wheel and whisker timestamps might start or end at different times, so we'll check whether the intersecting domains are aligned
    # # note: most of the time, whisker timestamps will be a subset of wheel timestamps
    # min_timestamp = max(whisker.timestamps.min(), wheel.timestamps.min())
    # max_timestamp = min(whisker.timestamps.max(), wheel.timestamps.max())
    # breakpoint()
    # assert np.allclose(
    #     wheel.timestamps[(wheel.timestamps >= min_timestamp) & (wheel.timestamps <= max_timestamp)],
    #     whisker.timestamps[(whisker.timestamps >= min_timestamp) & (whisker.timestamps <= max_timestamp)]
    # ), f"Wheel and whisker domains do not intersect at the same times"