"""
This script contains functions for converting midi csv files or midi text files into numpy matrices that represent notes per timestep.
When used directly, it uses these matrices to save them in a visual form (a.k.a. images).

Some of the same functions used in other scripts are, again, defined and used in here.
"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os
import os.path as osp
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _parse_args():
    """
    Parses arguments for the main script.
    """
    example_text = '''examples:

    python visualize.py --csv .../csv_dir/csvfile.csv --ticks 30
        --result-path .../results --prefix midi_image

    takes "csvfile" as an input, divides notes into timesteps with 30 ticks per each step, and
     stores the resulting images in the "results" directory with prefix "midi_image"
    resulting images will have names like these: midi_image0.jpg, midi_image1.jpg, ...

    python visualize.py --text .../txt_dir/txtfile.txt -r .../results -p
    '''

    parser = ArgumentParser(description='utility script for visualization of a midi file in either text or csv form.',
                            prog='python visualize.py',
                            epilog=example_text,
                            formatter_class=RawDescriptionHelpFormatter)

    in_group = parser.add_argument_group(title='input arguments',
                                         description='one of these arguments must be specified as the input')

    group = in_group.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-t',
        '--text',
        metavar='TEXT_PATH',
        help='File path for a midi text file to visualize.',
        type=str
        )
    group.add_argument(
        '-c',
        '--csv',
        metavar='CSV_PATH',
        help='File path for a midi csv file to visualize.',
        type=str
        )

    csv_group = parser.add_argument_group(title='csv optional argument',
                                          description="when using a midi csv file, you can "
                                                      "specify the number of ticks per each time step")

    csv_group.add_argument(
        '--ticks',
        help="Ticks per each time step (default: 25). "
             "Only used when visualizing midi csv files. "
             "Midi text files are already divided into time steps.",
        type=int,
        default=25
        )

    out_group = parser.add_argument_group(title='output arguments')

    out_group.add_argument(
        '-r',
        '--result-dir',
        required=True,
        help='directory for resulting images to be saved',
        type=str
        )

    out_group.add_argument(
        '-p',
        '--prefix',
        help="prefix for resulting images. by default, it will use "
             "the input filename as the prefix. using ' ' as the prefix "
             "will give images with no prefix; just numbers",
        type=str
        )

    parser.add_argument(
        '-v',
        '--verbose',
        help="make the process more verbose.",
        action='store_true',
        default=False
        )

    parser.add_argument(
        '-z',
        '--zero-pad',
        help="add zero paddings to the image number so that the number of digits match",
        action='store_true',
        default=False
        )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if not args.ticks > 0:
        parser.error("The value for ticks per time step must be at least 1.")

    if args.text and not osp.isfile(args.text):
        parser.error("The input text file does not exist. Please, check the file path and try again.")

    if args.csv and not osp.isfile(args.csv):
        parser.error("The input csv file does not exist. Please, check the file path and try again.")

    if not osp.isdir(args.result_dir):
        parser.error("The result directory does not exist. Please, make the directory first and try again.")

    reserved_chrs = '/ \\ ? % * : | \" < >'.split()
    if args.prefix:
        for reserved_chr in reserved_chrs:
            if reserved_chr in args.prefix:
                parser.error("The prefix cannot contain the following characters: / \\ ? % * : | \" < >")

    return args

def read_midi_csv(midi_file_path):
    """
    Given the file path of a converted midi csv file, reads and returns the data as a pandas DataFrame.
    Also, during the process, removes double quotes in the 5th column (e.g. "major")
     so that it can be processed later without much problem.
    """
    midi = pd.read_csv(midi_file_path, names=["Track", "Time", "Type", "Val1", "Val2", "Val3", "Val4"])
    midi.iloc[:, 4] = midi.iloc[:, 4].apply(lambda val: val.replace('"', '') if isinstance(val, str) else val)
    return midi

def drop_nonessentials(midi_dataframe):
    """
    Drops all items except for those that are essential for the music.
    The resulting dataframe is returned.
    """
    non_essential_list = ['header', 'end_of_file', 'start_track', 'end_track', 'tempo', 'note_on_c', 'note_off_c']
    non_essentials = midi_dataframe.iloc[:, 2].apply(lambda str: str.strip().lower() not in non_essential_list)
    return midi_dataframe.drop(index=midi_dataframe.loc[non_essentials].index).reset_index(drop=True)

def time_adjustment(midi_dataframe):
    """
    Drops 'Tempo' items and changes the time values accordingly.
    The resulting dataframe is returned.
    """
    base_midi = midi_dataframe.copy()
    base_midi.loc[:, 'index'] = base_midi.index
    modified_midi = midi_dataframe.copy()
    DEFAULT_TEMPO = 500000

    tempos = base_midi[base_midi.iloc[:, 2].apply(lambda str: str.strip().lower() == 'tempo')].sort_values([base_midi.columns[1], 'index'])

    tempo_change_time_points = tempos.iloc[:, 1].values.tolist()
    tempo_change_time_points.insert(0, 0)
    tempo_change_time_points.append(base_midi.iloc[:, 1].max() + 1)

    interval_multipliers = (tempos.iloc[:, 3] / DEFAULT_TEMPO).values.tolist()
    interval_multipliers.insert(0, 1.)

    last_time_point = tempo_change_time_points[0]

    for tempo_idx in range(len(tempo_change_time_points) - 1):

        selecting_condition = (base_midi.iloc[:, 1] > tempo_change_time_points[tempo_idx]) & (base_midi.iloc[:, 1] <= tempo_change_time_points[tempo_idx + 1])

        if selecting_condition.sum() > 0:

            multiplier = interval_multipliers[tempo_idx]

            times_since_tempo = base_midi.loc[selecting_condition, base_midi.columns[1]] - tempo_change_time_points[tempo_idx]

            scaled_times = times_since_tempo * multiplier

            adjusted_times = (scaled_times + last_time_point).values

            modified_midi.loc[selecting_condition, base_midi.columns[1]] = adjusted_times

            last_time_point = adjusted_times.max()

    modified_midi.iloc[:, 1] = modified_midi.iloc[:, 1].round()

    # Remove 'Tempo' lines
    modified_midi.drop(index=modified_midi.loc[modified_midi.iloc[:, 2].apply(lambda str: str.strip().lower() == 'tempo')].index, inplace=True)

    modified_midi.reset_index(drop=True, inplace=True)

    return modified_midi

def merge_tracks(midi_dataframe):
    """
    Combines multiple tracks into one track (track 1), then sorts the items by the time values.
    Returns the resulting dataframe.
    """
    midi = midi_dataframe.copy()

    # Change the number of tracks indicated by the header
    midi.loc[midi.iloc[:, 2].apply(lambda str: str.strip().lower() == 'header'), midi.columns[4]] = 1

    # Remove extra 'Start_track' and 'End_track'
    start_indices = midi.loc[midi.iloc[:, 2].apply(lambda str: str.strip().lower() == 'start_track')].index
    end_indices = midi.loc[midi.iloc[:, 2].apply(lambda str: str.strip().lower() == 'end_track')].index
    min_start_idx = midi.loc[start_indices, midi.columns[1]].idxmin()
    max_end_idx = midi.loc[end_indices, midi.columns[1]].idxmax()
    midi.drop(index=start_indices[start_indices != min_start_idx], inplace=True)
    midi.drop(index=end_indices[end_indices != max_end_idx], inplace=True)

    # Change all track numbers to 1 (other meta-data have 0 as their track number, so ignore any item with track number 0)
    midi.loc[midi.iloc[:, 0] > 1, midi.columns[0]] = 1

    # Sort items with track number 1 in ascending order by their time value
    midi.loc[midi.iloc[:, 0] == 1] = midi.loc[midi.iloc[:, 0] == 1].sort_values(midi.columns[1], axis=0, ascending=True).values

    midi.reset_index(drop=True, inplace=True)

    return midi

def midicsv_to_notematrix(midi_dataframe, ticks_per_step=25):
    """
    Converts a midi csv dataframe into a numpy matrix of shape [128, total_timesteps].
    Returns the matrix.
    """
    NUMBER_OF_PITCH = 128
    TRACK_COL = 0
    TIME_COL = 1
    TYPE_COL = 2
    CH_COL = 3
    PITCH_COL = 4
    VEL_COL = 5

    base_midi = midi_dataframe.copy()

    col_names = base_midi.columns

    # Simplify the data by having lower time resolution and only notes (no other meta-data or something like that)
    # Also, instead of the given 'End_track', make a new end point with some gap between it and the last note
    note_midi = base_midi.loc[base_midi.iloc[:, TYPE_COL].apply(lambda str: str.strip().lower() in ['note_on_c', 'note_off_c'])].copy()
    note_midi.sort_values(col_names[TIME_COL], inplace=True)
    note_midi.reset_index(drop=True, inplace=True)
    note_midi = note_midi.append({col_names[TRACK_COL]: 1, col_names[TIME_COL]: note_midi.iloc[:, 1].max() + 1920, col_names[TYPE_COL]: 'END',
                                  col_names[CH_COL]: 0, col_names[PITCH_COL]: 0, col_names[VEL_COL]: 0}, ignore_index=True)
    note_midi.iloc[:, TIME_COL] = (note_midi.iloc[:, TIME_COL] / ticks_per_step).round()

    # Total number of time steps
    n_steps = int(note_midi.iloc[:, TIME_COL].max())

    note_time_matrix = np.zeros((NUMBER_OF_PITCH, n_steps), dtype=np.uint8)

    for pitch in range(128):
        note_pitch_match = note_midi.loc[note_midi.iloc[:, PITCH_COL].astype(int) == pitch]
        if len(note_pitch_match) > 0:
            for note_idx in range(len(note_pitch_match) - 1):
                if note_pitch_match.iloc[note_idx, VEL_COL] > 0:
                    start_from = int(note_pitch_match.iloc[note_idx, TIME_COL])
                    end_before = int(note_pitch_match.iloc[note_idx + 1, TIME_COL])
                    note_time_matrix[pitch, start_from:end_before] = 1
                    # Notes aren't guaranteed to be off before being pressed again.
                    # A subsequent Note_on after at least two of the same kind should be treated as a separate note
                    # Checking 2 steps back minimizes note loss when there are more than 2 subsequent notes
                    if note_time_matrix[pitch, (start_from - 2):start_from].sum() == 2:
                        note_time_matrix[pitch, start_from - 1] = 0

    return note_time_matrix

def text_to_notematrix(midi_text):
    """
    Given a string from a midi text, converts it into a numpy matrix of shape [128, total_timesteps].
    Returns the matrix.
    """
    NUMBER_OF_PITCH = 128

    text_list = (' ' + midi_text).split(' 0')

    # Total number of time steps
    n_steps = len(text_list)

    note_time_matrix = np.zeros((NUMBER_OF_PITCH, n_steps), dtype=np.uint8)

    for time_step in range(n_steps):
        note_str = text_list[time_step].strip()
        if note_str != '':
            for note_num in note_str.split(' '):
                note_time_matrix[int(note_num) - 1, time_step] = 1

    return note_time_matrix

def notematrix_to_image(note_matrix):
    """
    Scales and fits the given matrix to multiple matrices with each having a size of (1080, 1920).
    (1080, 1920) -> 1080 rows, 1920 columns -> 1920 x 1080
    Returns the list of image matrices.
    """
    idx = []
    for pitch in range(128):
        for _ in range(8):
            idx.append(pitch)
    idx = idx[::-1]

    # Stack with zero paddings at the top and the bottom
    top_bottom_paddings = np.zeros((28, note_matrix.shape[1]), dtype=np.uint8)
    img = np.vstack(
        (
            top_bottom_paddings,
            note_matrix[idx, :],
            top_bottom_paddings
        )
    )

    # Add zero paddings to the right as necessary
    right_paddings = np.zeros((img.shape[0], 1920 - (img.shape[1] % 1920)), dtype=np.uint8)
    img = np.hstack((img, right_paddings))

    # img itself is just one very long image.
    # It needs to be broken down into multiple images with appropriate sizes (1080, 1920)
    imgs = [img[:, (1920 * n) : (1920 * (n + 1))] for n in range(img.shape[1] // 1920)]

    return imgs

def save_images(imgs, dir_path='.', prefix='', verbose=False, zero_pad=False):
    """
    Takes a list of numpy matrices and saves them as image files.
    No value is returned.

    If the directory path is given, images will be saved in that directory.
    If the prefix string is given, image files will have that prefix followed by their numbers.

    If the verbose is True, prints the progress for every image.

    If the zero_pad is True, pad zeros to the image numbers in order to match the number of digits.

    By default, the images are saved in the current directory with no prefix.
    """
    n_imgs = len(imgs)
    n_digits = len(str(n_imgs))
    if verbose:
        print(f"Saving {n_imgs} images:")
    for i, img in enumerate(imgs):
        image_num_str = f"{i:0{n_digits}d}" if zero_pad else str(i)
        filename = prefix + image_num_str + '.jpg'
        filepath = osp.join(dir_path, filename).replace('\\', '/')
        plt.imsave(filepath, img, cmap='gray', format='jpeg')
        if verbose:
            print(filepath, f"\t({i + 1} / {n_imgs})")

if __name__ == '__main__':

    args = _parse_args()

    csv_path = args.csv
    txt_path = args.text

    result_dir = args.result_dir

    if args.prefix:
        prefix = args.prefix.strip()
    elif args.csv:
        prefix = osp.basename(args.csv).rsplit('.', 1)[0]
    else:
        prefix = osp.basename(args.text).rsplit('.', 1)[0]

    ticks_per_step = args.ticks

    verbose = args.verbose
    zero_pad = args.zero_pad

    if csv_path:

        if verbose:
            print("\nReading the midi csv file ...")

        midi = read_midi_csv(csv_path)

        if verbose:
            print("\nDropping non-essential items ...")

        midi = drop_nonessentials(midi)

        if verbose:
            print("\nAdjusting time and dropping tempo changes ...")

        midi = time_adjustment(midi)

        if verbose:
            print("\nMerging tracks ...")

        midi = merge_tracks(midi)

        if verbose:
            print("\nConverting the midi csv into a note matrix ...")

        note_matrix = midicsv_to_notematrix(midi, ticks_per_step=ticks_per_step)

    else:

        if verbose:
            print("\nReading the midi text file ...")

        with open(txt_path, 'r', encoding='utf-8') as txt_f:
            midi_text = txt_f.read()

        if verbose:
            print("\nConverting the midi text into a note matrix ...")

        note_matrix = text_to_notematrix(midi_text)

    if verbose:
        print("\nScaling and fitting the note matrix into image matrices ...")

    images = notematrix_to_image(note_matrix)

    if verbose:
        print("\nSaving the images ...\n")

    save_images(images, dir_path=result_dir, prefix=prefix, verbose=verbose, zero_pad=zero_pad)

    if verbose:
        print("\n... Done!\n")
