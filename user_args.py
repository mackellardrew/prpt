import argparse
import os
from pathlib import Path

# Handling inputs as a dir OR file of paths is too much trouble;
# keep it dir-only for now.  Also, mount Docker volumes as
# their native (local) paths; no more "/data" mappings.
parser = argparse.ArgumentParser()
parser.add_argument(
    help=("Path to dir containing all input FastQ read files"),
    type=Path,
    dest="indir",
)
parser.add_argument(
    help=("Path to dir containing the cloned SNVPhyl CLI repository"),
    type=Path,
    dest="snvdir",
)
parser.add_argument(
    "-e",
    "--email",
    help="User email for authenticating to NCBI FTP site",
    type=str,
    dest="email",
)
parser.add_argument(
    "-o",
    "--outdir",
    help="Directory to which to write the outputs",
    type=Path,
    dest="outdir",
)
parser.add_argument(
    "-t",
    "--threads",
    help="Number of threads to use",
    type=int,
    default=os.cpu_count(),
    dest="nthreads",
)
parser.add_argument(
    "-p",
    "--phylo",
    help=(
        "Phylogenetic tree-building program to use: "
        "one of either 'lyveset' or 'snvphyl'"
    ),
    type=str,
    default="SNVPHYL",
    dest="phylo",
)
parser.add_argument(
    "-r",
    "--reads",
    help=(
        "Reads to use in phylogenetic tree building: "
        "one of either 'raw' or 'trimmed'"
    ),
    type=str,
    default="TRIMMED",
    dest="phylo_read_type",
)
parser.add_argument(
    "--assemblies_dir",
    help=(
        "Path to dir containg existing genome assemblies; "
        "if supplied, skips SPAdes genome assembling "
        "(the slowest step in this pipeline)"
    ),
    type=Path,
    default=None,
    dest="assemblies_dir",
)

user_args = parser.parse_args()