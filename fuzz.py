# Author: Aditya S. Mansharamani, adityam5@illinois.edu
# This code, along with any accompanying source code files ONLY,
# are released under the license specified in LICENSE.md.

import pandas as pd
import string
import random
import numpy as np

# Load in original data files
metadata = pd.read_csv("./metadata.csv").set_index("Key")
met_base = pd.read_csv("./metabolomics_baseline_2021_2.csv").set_index("Key")
met_end = pd.read_csv("./metabolomics_end_2021_2.csv").set_index("Key")

# Anonymize subject IDs
def anonymize_subject_ids(metadata, met_base, met_end):
    def anonymize_id(sid):
        sid = sid.split(".")
        h = hex(abs(hash(tuple(sid[0:-2]))))[2:]
        tag = ".".join(sid[-2:])
        return f"S{h}.{tag}"

    metadata.index = metadata.index.map(anonymize_id)
    met_base.index = met_base.index.map(anonymize_id)
    met_end.index = met_end.index.map(anonymize_id)

    return metadata, met_base, met_end


# Subset metadata to only include the fields that are absoloutely necessary.
def subset_metadata(metadata):
    metadata = metadata[["Study", "Treatment", "Treatment2"]]
    return metadata


# Fuzz metabolite values and names
def fuzz_metabolites(met_base, met_end):
    # Create new dataframes with random values
    met_base_rand = pd.DataFrame(np.random.randint(0, 100000, size=met_base.shape))
    met_base_rand.index = met_base.index
    met_base_rand.columns = met_base.columns

    met_end_rand = pd.DataFrame(np.random.randint(0, 100000, size=met_end.shape))
    met_end_rand.index = met_end.index
    met_end_rand.columns = met_end.columns

    # Change metabolite names
    def fuzz_metabolite(metabolite):
        h = abs(hash(metabolite))
        h = hex(h)[2:]
        return "M" + h[:5]

    met_base_rand.columns = met_base_rand.columns.map(fuzz_metabolite)
    met_end_rand.columns = met_end_rand.columns.map(fuzz_metabolite)

    # Introduce 15% missing values
    def introduce_missing(df):
        for col in df:
            for i, row_value in df[col].iteritems():
                if random.random() <= 0.15:
                    df[col][i] = np.nan

    introduce_missing(met_base_rand)
    introduce_missing(met_end_rand)

    return met_base_rand, met_end_rand


# Save all data
def save_data(metadata, met_base, met_end):
    metadata.to_csv("metadata_fuzzed.csv")
    met_base.to_csv("metabolomics_baseline_fuzzed.csv")
    met_end.to_csv("metabolomics_end_fuzzed.csv")


# Apply pipeline
metadata, met_base, met_end = anonymize_subject_ids(metadata, met_base, met_end)
metadata = subset_metadata(metadata)
met_base, met_end = fuzz_metabolites(met_base, met_end)
save_data(metadata, met_base, met_end)
