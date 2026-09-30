import os
import numpy as np
import mne
from tqdm import tqdm
from autoreject import AutoReject

def preprocess_bdf(fname):
    """Apply highpass filter and autoreject to .bdf; save output as .npy"""
    tmin = 30  # seconds
    l_freq, h_freq = 0.5, None  # Hz
    notch_filter = 50  # Hz
    epoch_duration = 5  # seconds
    epoch_overlap = 0  # seconds
    seed = 123
    num_splits = 10  # autoreject splits

    # Load in raw EEG data
    raw = mne.io.read_raw_bdf(fname).load_data()
    raw.drop_channels(['VPVA', 'VNVB', 'HPHL', 'HNHR', 'Erbs', 'Mass', 'Status'])

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn')

    # Crop
    raw.crop(tmin=tmin)

    # High-pass filtering
    raw.filter(l_freq=l_freq, h_freq=h_freq)

    # Notch filter
    raw.notch_filter(notch_filter)

    # Epoch the data
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=True, overlap=epoch_overlap)

    # Run autoreject
    reject = AutoReject(random_state=seed, cv=num_splits, verbose=False, n_jobs=-1)
    epochs, autoreject_log = reject.fit_transform(epochs.copy(), return_log=True)

    # Re-reference to average
    epochs.set_eeg_reference(ref_channels="average", verbose=False)

    # Convert to numpy arrays
    eeg_data = epochs.get_data()

    # autoreject_log = None
    return eeg_data, autoreject_log


if __name__ == "__main__":
    mne.set_log_level('WARNING')

    # Collect all resting state .bdf files
    data_dir = '/users/ntolley/data/shared/TDBRAIN_v3/TDBRAIN_Dataset_V3_1/'
    eeg_paths, eeg_files = [], []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            if ('task-restE' in filename) and ('.bdf' in filename):
                eeg_paths.append(dirpath)
                eeg_files.append(filename)

    # Apply preprocessing
    for eeg_dir, eeg_file in tqdm(list(zip(eeg_paths, eeg_files))):
        fname = f'{eeg_dir}/{eeg_file}'
        eeg_data, autoreject_log = preprocess_bdf(fname)

        # Prepare file paths
        save_file = eeg_file.removesuffix('.bdf')  # remove .bdf extension
        save_path = f'{eeg_dir}/preprocessed'
        os.makedirs(save_path, exist_ok=True)

        # Save EEG data and autoreject logs
        np.save(f'{save_path}/{save_file}.npy', eeg_data)
        autoreject_log.save(f'{save_path}/{save_file}_autoreject_log.npz', overwrite=True)

