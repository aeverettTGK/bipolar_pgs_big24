#! /usr/bin/env python3

import glob
import os
import pandas as pd
import sys

root_dir = sys.argv[1]

dfs = []

for subdir, _, files in os.walk(root_dir):
    pattern = os.path.join(subdir, "important_var*.csv")
    for file_path in glob.glob(pattern):
        df = pd.read_csv(file_path)
        dfs.append(df)

concatenate = pd.concat(dfs, ignore_index = True, sort = False)

output_path = sys.argv[2]

concatenate.to_csv(output_path, index = False)
