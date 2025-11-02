import pandas as pd
import os

df = pd.read_csv('/mnt/data/omkumar/foundation_phase1/data_utils/classes_in_patients.csv')
sorted_df = df.sort_values(by = "num_classes")
sorted_df.to_csv('/mnt/data/omkumar/foundation_phase1/data_utils/classes_in_patients.csv')
