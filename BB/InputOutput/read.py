import csv
import datetime
import logging
import numpy as np
from BB.Number import number


def read_to_numpy_array(input_path, data_skip_lines):
    """
    Reads the input file, skipping over the first several lines if needed.
    """
    reader = csv.reader(open(input_path.strip()))

    num_cols = 0
    rows = []
    start = datetime.datetime.utcnow()
    for index, row in enumerate(reader):
        if index < data_skip_lines:
            continue

        if 0 == len(row):
            continue

        if 0 == num_cols:
            num_cols = len(row)

        if num_cols != len(row):
            raise Exception(
                "input file has line with incorrect length, line: #{}, "
                "contents: {}".format(index + 1, row))

        row_data = [[0] * num_cols]
        for index, column in enumerate(row):
            row_data[index] = number.to_float(column)

        rows.append(row_data)

    end = datetime.datetime.utcnow()
    logging.info('read {} in: {}'.format(input_path, (end - start)))

    numpy_array = np.array(rows)
    return numpy_array
