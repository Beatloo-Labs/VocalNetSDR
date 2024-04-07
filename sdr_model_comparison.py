import argparse
import os
import numpy as np
import soundfile as sf
from scipy import signal
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def parse_args():
    """
    Parses command-line arguments for the script.

    Returns a namespace object containing the arguments to the command. The function defines three arguments:
    --tracks_folder: The directory containing the tracks to process (default: "tracks").
    --models: A list of model names to be applied to the tracks.
    --threads: The number of parallel threads to use for processing (default: 5).

    Note: --models is a required argument.
    """
    parser = argparse.ArgumentParser(description='Process tracks with various neural network models.')
    parser.add_argument('--tracks_folder', type=str, default="songs", help='Folder containing the track folders to be processed')
    parser.add_argument('--models', nargs='+', required=True, help='List of neural network model names to process the tracks with')
    parser.add_argument('--threads', type=int, default=5, help='Number of threads for simultaneous track processing, default: 5. 5 workers use 8.5 GB of RAM on average')
    return parser.parse_args()


def sdr(references, estimates):
    """
    Calculates the Signal-to-Distortion Ratio (SDR) for given reference and estimated signals.

    Args:
        references: The original, unprocessed audio signals.
        estimates: The processed audio signals to compare against the references.

    Returns:
        The calculated SDR values for each pair of reference and estimate signals.
    """
    delta = 1e-7
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += delta
    den += delta
    return 10 * np.log10(num / den)


def get_sdr(original_signal, model_signal):
    """
    Trims and aligns original and model-processed signals before computing their SDR.

    Args:
        original_signal: The original audio signal array.
        model_signal: The signal processed by the neural network model.

    Returns:
        The SDR value comparing the original and model-processed signals.
    """
    min_length = min(original_signal.shape[1], model_signal.shape[1])
    original_signal_trimmed = original_signal[:, :min_length]
    model_signal_trimmed = model_signal[:, :min_length]
    sdr_value = sdr(np.array([original_signal_trimmed]), np.array([model_signal_trimmed]))
    return sdr_value


def lr_filter(audio, cutoff, filter_type, order=6, sr=44100):
    """
    Applies a low-pass or high-pass filter to an audio signal.

    Args:
        audio: The input audio signal to filter.
        cutoff: The cutoff frequency for the filter.
        filter_type: 'lowpass' or 'highpass', defining the type of filter to apply.
        order: The order of the filter (default: 6).
        sr: The sample rate of the audio signal (default: 44100).

    Returns:
        The filtered audio signal.
    """
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order//2, normal_cutoff, btype=filter_type, analog=False)
    sos = signal.tf2sos(b, a)
    filtered_audio = signal.sosfiltfilt(sos, audio)
    return filtered_audio


def process_track(root, track, weight_combinations, models):
    """
    Processes a single track with various weight combinations of neural network models.

    Args:
        root: The root directory containing the track.
        track: The specific track to process.
        weight_combinations: A list of tuples representing different weight combinations to apply to the models.

    Returns:
        A list of tuples containing the weight combinations and their corresponding SDR values for the track.
    """
    track_sdr_values = []
    original_vocals, sr1 = sf.read(os.path.join(root, track, "original_vocals.wav"))
    original_other, sr2 = sf.read(os.path.join(root, track, "original_other.wav"))
    original_mix = original_vocals.T + original_other.T

    model_signals = []
    srs = [sr1, sr2]  # Collect sample rates to check for consistency.
    for model in models:
        model_signal, sr = sf.read(os.path.join(root, track, f"vocals_{model}.wav"))
        model_signals.append(model_signal.T)
        srs.append(sr)

    # Check that all sampling frequencies match
    if len(set(srs)) > 1:
        raise ValueError(f"Sample rates are not the same across all files.\nFound sample rates: {', '.join(map(str, sorted(set(srs))))}.\nPlease ensure all files have the same sample rate, such as 44100 or 48000.")

    for first_val, second_val in weight_combinations:
        weights = np.array([first_val, second_val])
        vocals_low = lr_filter((weights[0] * model_signals[0] + weights[1] * model_signals[1]) / weights.sum(), 10000, 'lowpass', sr=sr1) * 1.01055
        vocals_high = lr_filter(model_signals[1], 10000, 'highpass', sr=sr1)
        summ_vocals = vocals_low + vocals_high

        sdr_vocals = get_sdr(original_vocals.T, summ_vocals)
        track_sdr_values.append((first_val, second_val, sdr_vocals[0]))

    return track_sdr_values


def main():
    """
    Main function to orchestrate the processing of tracks with neural network models based on command-line inputs.

    Processes each track in parallel using a ThreadPoolExecutor, evaluates the Signal-to-Distortion Ratio (SDR) for various model weight combinations, and identifies the combination yielding the highest average SDR.
    """
    args = parse_args()

    weight_step = 1.0
    weight_combinations = [(first_val, 10 - first_val) for first_val in np.arange(0, 10 + weight_step, weight_step)]
    all_sdr_values = []

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        for root, dirs, files in os.walk(args.tracks_folder):
            for track in dirs:
                futures.append(executor.submit(process_track, root, track, weight_combinations, args.models))

        for future in tqdm(futures, desc="Processing tracks"):
            all_sdr_values.extend(future.result())

    # Determine the best weight combination based on average SDR
    best_sdr = float('-inf')
    best_weights = (0, 0)
    avg_sdr_per_weight = {}

    # Accumulate SDRs for each weight combination
    for first_val, second_val, sdr_value in all_sdr_values:
        if (first_val, second_val) not in avg_sdr_per_weight:
            avg_sdr_per_weight[(first_val, second_val)] = []
        avg_sdr_per_weight[(first_val, second_val)].append(sdr_value)

    # Calculate average SDR for each weight combination
    for weights, sdr_values in avg_sdr_per_weight.items():
        avg_sdr = sum(sdr_values) / len(sdr_values)
        if avg_sdr > best_sdr:
            best_sdr = avg_sdr
            best_weights = weights

    print(f"Best AVG SDR: {round(best_sdr, 2)} with weights {args.models[0]} / {args.models[1]} {best_weights}")


if __name__ == "__main__":
    main()