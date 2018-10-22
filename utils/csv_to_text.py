"""
This script contains functions for simplifying and converting midi csv files.
"""
from argparse import ArgumentParser
import os.path as osp
import sys
import pandas as pd
import numpy as np

def _parse_args():
    """
    Parses arguments for the main script.
    """
    parser = ArgumentParser(description='utility script for converting a midi file in csv format into a text',
                            prog='python csv_to_text.py')

    parser.add_argument(
        '-t',
        '--ticks',
        help="Ticks per each time step (default: 25). "
             "Make sure to remember its value when converting the text back to a csv.",
        type=int,
        default=25
        )

    parser.add_argument(
        '-v',
        '--verbose',
        help="make the process more verbose.",
        action='store_true'
        )

    parser.add_argument(
        'csv',
        type=str,
        help="File path for the csv file to convert."
        )

    parser.add_argument(
        'text',
        nargs='?',
        type=str,
        help="File path for the resulting text file (Optional). By default, the text file"
             " will be generated in the same directory as the source csv file."
        )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if not args.ticks > 0:
        parser.error("The value for ticks per time step must be at least 1.")

    if not osp.isfile(args.csv):
        parser.error("The input csv file does not exist. Please, check the file path and try again.")

    if args.txt and not osp.isdir(osp.dirname(args.txt)):
        parser.error("The result path does not exist. Please, use an existing directory.")

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

def constantize_velocities(midi_dataframe, velocity=80):
    """
    Fixes all Note_on velocity values to a constant.
    Returns the resulting dataframe.
    Though it is not necessary, it can give you some sence as to how the machine generated musics would be.
    """
    midi = midi_dataframe.copy()

    note_on = midi.loc[midi.iloc[:, 2].apply(lambda str: str.strip().lower() in ['note_on_c', 'note_off_c'])]

    nonzero_vel_idx = note_on.loc[note_on.iloc[:, 5].apply(lambda vel: vel > 0)].index

    midi.loc[nonzero_vel_idx, midi.columns[5]] = velocity

    return midi

def midicsv_to_text(midi_dataframe, ticks_per_step=25):
    """
    Converts the given midi dataframe into a string form, where each line represents
     the notes that are pressed / being hold at that time step.
    Ticks per time step determines the length of each time step.
    The smaller the value is, the longer and more accurate the result will be.
    However, when putting it into a machine learning algorithm, consider how
     much the algorithm should take. Longer string will give the algorithm a hard time
     maintaining its memory.

    Returns the resulting string text.
    """
    NUMBER_OF_PITCH = 128
    TRACK_COL = 0
    TIME_COL = 1
    TYPE_COL = 2
    CH_COL = 3
    PITCH_COL = 4
    VEL_COL = 5

    next_line_num = ord('\n')

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

    text_list = []

    for time_step in range(n_steps):
        text = ''
        for pitch in note_time_matrix[:, time_step].nonzero()[0]:
            # To make the text look nice and organized, I'll use nextline to indicate time steps
            # All other pitches will come after the code for nextline
            text += chr(pitch + next_line_num + 1)
        text_list.append(text)

    # Separator between timesteps will be chr(128 + 1)
    midi_text = chr(next_line_num).join(text_list)

    return midi_text


if __name__ == '__main__':

    args = _parse_args()

    csv_path = args.csv
    txt_path = args.txt if args.txt else (args.csv.rsplit('.', 1)[0] + '.txt')
    ticks_per_step = args.ticks
    verbose = args.verbose

    if verbose:
        print("\nReading CSV file ...")

    csv = read_midi_csv(csv_path)

    if verbose:
        print("\nDropping non-essencial items ...")

    csv = drop_nonessentials(csv)

    if verbose:
        print("\nAdjusting time and dropping tempo changes ...")

    csv = time_adjustment(csv)

    if verbose:
        print("\nMerging tracks ...")

    csv = merge_tracks(csv)

    if verbose:
        print("\nConverting CSV into text ...")

    # Not necessary for the process
    # csv = constantize_velocities(csv)

    text = midicsv_to_text(csv, ticks_per_step=ticks_per_step)

    if verbose:
        print("\nWriting to text file ...")

    with open(txt_path, 'w', encoding='utf-8') as txt_f:
        txt_f.write(text)

    if verbose:
        print("\n... Done!\n")
