"""
This script contains a function for converting a text file into a csv file.
Unlike the conversion from csv to text, text to csv does not require as many functionalities.
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
    parser = ArgumentParser(description='utility script for converting a midi text file into a csv file',
                            prog='python text_to_csv.py')

    parser.add_argument(
        '-t',
        '--ticks',
        help="Ticks per each time step (default: 25). "
             "You should use the same value that you used to convert the original csv into the text.",
        type=int,
        default=25
        )

    parser.add_argument(
        '-v',
        '--velocity',
        help="Velocity value for notes (default: 80).",
        type=int,
        default=80
        )

    parser.add_argument(
        '-e',
        '--end-track',
        help="Add an 'End_track' item after the last note. "
             "Use it if you are not using the original midi text "
             "(e.g. if you're using a machine-generated text).",
        action='store_true',
        default=False
        )

    parser.add_argument(
        '--verbose',
        help="make the process more verbose.",
        action='store_true'
        )

    parser.add_argument(
        'text',
        type=str,
        help="File path for the text file to convert."
        )

    parser.add_argument(
        'csv',
        nargs='?',
        type=str,
        help="File path for the resulting csv file (Optional). By default, the csv file will be "
             "generated in the same directory as the source text file."
        )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if not args.ticks > 0:
        parser.error("The value for ticks per time step must be at least 1.")

    if not args.velocity >= 0:
        parser.error("The value for ticks per time step must be non-negative.")

    if not osp.isfile(args.txt):
        parser.error("The input text file does not exist. Please, check the file path and try again.")

    if args.csv and not osp.isdir(osp.dirname(args.csv)):
        parser.error("The result path does not exist. Please, use an existing directory.")

    return args

def text_to_midicsv(midi_text, ticks_per_step=25, vel=80, add_end_track=False):
    """
    Converts the given midi text file into a pandas dataframe.
    Then, the resulting dataframe is returned.

    When constructing the dataframe, it uses ticks_per_step to determine the time tick value of each note.
    All 'Note_on's will have the same velocity specified as a parameter.

    If add_end_track is True, it will add an 'End_track' 1920 ticks after the last item from the text.
    Otherwise, it will consider the last item in the text as the 'End_track'.
    """
    NUMBER_OF_PITCH = 128
    COL_NAMES = ['Track', 'Time', 'Type', 'Val1', 'Val2', 'Val3']
    HEADER = [0, 0, 'Header', 1, 1, 480]
    START_TRACK = [1, 0, 'Start_track', np.nan, np.nan, np.nan]

    next_line_num = ord('\n')

    text_list = midi_text.split(chr(next_line_num))

    # Total number of time steps
    n_steps = len(text_list)

    note_time_matrix = np.zeros((NUMBER_OF_PITCH, n_steps), dtype=np.uint8)

    for time_step in range(n_steps):
        note_str = text_list[time_step]
        if note_str != '':
            for note_chr in note_str:
                note_time_matrix[ord(note_chr) - next_line_num - 1, time_step] = 1

    data_lists = [HEADER, START_TRACK]

    if note_time_matrix[:, 0].nonzero()[0].any():
        for pitch in note_time_matrix[:, 0].nonzero()[0]:
            data_lists.append([1, 0, 'Note_on_c', 0, pitch, vel])
    for time_step in range(1, n_steps - 1):
        change_occured = note_time_matrix[:, time_step - 1] != note_time_matrix[:, time_step]
        for pitch in change_occured.nonzero()[0]:
            if note_time_matrix[pitch, time_step] == 1:
                velocity = vel
            else:
                velocity = 0
            data_lists.append([1, (time_step * ticks_per_step), 'Note_on_c', 0, pitch, velocity])
    if not add_end_track:
        data_lists.append([1, ((n_steps - 1) * ticks_per_step), 'End_track'])
    else:
        change_occured = note_time_matrix[:, n_steps - 2] != note_time_matrix[:, n_steps - 1]
        for pitch in change_occured.nonzero()[0]:
            if note_time_matrix[pitch, time_step] == 1:
                velocity = vel
            else:
                velocity = 0
            data_lists.append([1, ((n_steps - 1) * ticks_per_step), 'Note_on_c', 0, pitch, velocity])
        data_lists.append([1, ((n_steps - 1) * ticks_per_step) + 1920, 'End_track'])

    data_lists.append([0, 0, 'End_of_file'])

    midi_csv = pd.DataFrame(data=data_lists, columns=COL_NAMES)

    return midi_csv

if __name__ == '__main__':

    args = _parse_args()

    txt_path = args.txt
    csv_path = args.csv if args.csv else (args.txt.rsplit('.', 1)[0] + '.csv')
    ticks_per_step = args.ticks
    vel = args.velocity
    add_end = args.end_track
    verbose = args.verbose

    if verbose:
        print("\nReading the text file ...")

    with open(txt_path, 'r', encoding='utf-8') as txt_f:
        text = txt_f.read()

    if verbose:
        print("\nConverting the text into a csv dataframe ...")

    csv = text_to_midicsv(text, ticks_per_step=ticks_per_step, vel=vel, add_end_track=add_end)

    if verbose:
        print("\nSaving the csv file ...")

    csv.to_csv(csv_path, header=False, index=False)

    if verbose:
        print("\n... Done!\n")
