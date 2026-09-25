"""Output writing and validation functions."""


def write_tsv(data, path):
    data.to_csv(path, sep="\t", index=False)
