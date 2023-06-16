#/!/usr/bin/env/python

"""A more nuanced second-pass to summarize approaches used in Pacific Rim 
Proficiency Training exercise to characterize HAI/AR samples in Summer 2020. 
All functions assume availability of Docker engine, Python>=3.5, etc. To be made more coherent and concise at a later date."""

import datetime
import docker
import gzip
import logging
import json
import os
import pandas as pd
import re
import shlex
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path, PurePath
from zipfile import ZipFile

from user_args import user_args

INDIR = user_args.indir.resolve()
SNVDIR = user_args.snvdir.resolve()

if user_args.assemblies_dir:
    ASSEMBLIESDIR = user_args.assemblies_dir.resolve(strict=True)
else:
    ASSEMBLIESDIR = None

if user_args.outdir:
    OUTDIR = user_args.outdir.resolve()
else:
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    OUTDIR = Path(f"{now}_PRPT").resolve()

EMAIL = user_args.email
NTHREADS = user_args.nthreads
PHYLO = user_args.phylo.lower()
PHYLO_READ_TYPE = user_args.phylo_read_type.lower()

CLIENT = docker.from_env()
USER = str(os.getuid())
GID = str(os.getgid())


def setup_logger() -> logging.Logger:
    """Returns a logger object writing to 'prpt_logs.txt'."""
    log_filepath = os.path.join(OUTDIR, "prpt_logs.txt")
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("prpt_logger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_filepath)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())

    return logger


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical(
        "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
    )

    return None


def recognize_sample_names(file_list: list, logger: logging.Logger):
    illumina_pattern = re.compile(r"^(.*)_S[0-9]+_L[0-9]{3}_R[12]_001\..*$")
    simple_pattern = re.compile(r"^(.*)[rR][12].*$")
    sample_names = set()
    for file in file_list:
        if ("fastq" in file) or ("fq" in file):
            _, filename = os.path.split(file)
            illumina_match = re.match(illumina_pattern, filename)
            simple_match = re.match(simple_pattern, filename)
            if illumina_match:
                sample_name = illumina_match.groups()[0]
            elif simple_match:
                sample_name = simple_match.groups()[0]
            sample_names.update([sample_name])
    if len(sample_names) == 0:
        file_list_str = '\n'.join(file_list)
        file_list_msg = (
            "Error: could not identify sample names from the following "
            f"files in {INDIR}: {file_list_str}"
            "\nPlease check dir path and ensure .fastq or .fq "
            "files are present."
        )
        logger.error(file_list_msg)
        sys.exit()

    return list(sample_names)


def get_read_pairs(
    filepath_list: list, samples: set, raw_or_trimmed: str, 
    logger: logging.Logger, require_pairs: bool = True, 
    ):
    """Output dict of sample: list_of_paths for input
    samples list and in_dir."""
    read_pairs = {sample: [] for sample in samples}
    for filepath in filepath_list:
        for sample in samples:
            if ("fastq" in filepath or "fq" in filepath) and (sample in filepath):
                if raw_or_trimmed == "raw":
                    read_pairs[sample].append(filepath)
                elif raw_or_trimmed == "trimmed":
                    if filepath[-7:] == "P.fastq":
                        read_pairs[sample].append(filepath)
    if require_pairs:
        read_pairs = {
            sample: path_list
            for sample, path_list in read_pairs.items()
            if len(path_list) == 2
        }
    for sample, pairs in read_pairs.items():
        read_pairs[sample] = sorted(pairs)

    return read_pairs


def map_read_pairs(read_pairs: dict, remapped: dict, raw_or_trimmed: str):
    """Converts paths of input reads from their local dir tree into 
    that expected by the container, given volume mount locations."""
    mapped_read_pairs = {sample: [] for sample in read_pairs}
    for sample, path_list in read_pairs.items():
        for filepath in path_list:
            _, filename = os.path.split(filepath)
            if raw_or_trimmed.lower() == "raw":
                mapped_dir = remapped.get(INDIR).get("bind")
            elif raw_or_trimmed.lower() == "trimmed":
                mapped_dir = os.path.join(
                    remapped.get(OUTDIR).get("bind"), "trimmed_reads"
                )
            new_path = os.path.join(mapped_dir, filename)
            mapped_read_pairs[sample].append(new_path)

    return mapped_read_pairs


def map_docker_dirs(indir: Path, outdir: Path):
    """Returns a dict mapping local dirs to their mounted 
    container locations, to be passed to Docker SDK."""
    volumes = {
        indir: {"bind": "/inputs", "mode": "ro"},
        outdir: {"bind": "/data", "mode": "rw"},
    }
    return volumes


def docker_create(
    image_name: str, volumes: dict, logger: logging.Logger, 
    name=None, tty=True, overwrite=True, **kwargs
    ):
    """Launch a new container image with Docker SDK.
    If overwrite=True and a container with the given name
    already exists; stop and remove it."""
    # Check that image is present
    images = CLIENT.images.list()
    local_img = CLIENT.images.get(image_name)
    if not local_img:
        repo, version = image_name.split(":")
        image_missing_msg = (
            f"Could not find local installation of image: '{repo}'; "
            f"attempting to download version: '{version}' now."
        )
        logger.info(image_missing_msg)
        CLIENT.images.pull(repo, tag=version)

    create_func = partial(
        CLIENT.containers.create,
        group_add=GID,
        image=image_name,
        name=name,
        tty=tty,
        volumes=volumes,
        user=USER,
        **kwargs,
    )
    creating_container_msg = (
        f"Attempting to start container {image_name}"
    )
    logger.info(creating_container_msg)
    try:
        container = create_func()
    except docker.errors.APIError as err:
        # container_err_msg = (
        #     f"Encountered error {err}"
        # )
        # logger.critical(container_err_msg)
        old_container = CLIENT.containers.get(name)
        old_container_msg = (
            f"Found running container: {old_container}"
        )
        if overwrite:
            old_msg_suffix = (
                "; attempting to stop and remove existing container."
            )
            old_container_msg += old_msg_suffix
            logger.warning(old_container_msg)
            old_container.stop()
            old_container.remove(v=True, force=True)
            container = create_func()
        else:
            container = old_container
            old_msg_suffix_2 = (
                "; attempting to use previously-existing container."
            )
            old_container_msg += old_msg_suffix_2
            logger.warning(old_container_msg)



    return container


def run_docker(
    instructions, logger: logging.Logger, redirect=False, stop_remove=True
    ):
    """Run a container based on input dict of cmds, then stop and remove container."""
    container = docker_create(
        instructions.get("image"),
        volumes=instructions.get("volumes"),
        name=instructions.get("name"),
        logger=logger,
        # Hacky workaround; fix later
        # healthcheck={'test':['NONE']}
        healthcheck={
            'test': [],
            'interval': int(1e9),
            'timeout': int(2.34e12),
            'retries': 3,
            'start_period': 0,
        }
    )
    container.start()
    outputs = {}
    cmds = instructions.get("cmds")
    if redirect:
        for dict_ in cmds:
            for outpath, cmd in dict_.items():
                output = container.exec_run(cmd)
                with open(outpath, "wb") as f:
                    f.write(output.output)
    else:
        for cmd in cmds:
            outputs[" ".join(cmd)] = container.exec_run(cmd).output.decode("utf-8")
    if stop_remove:
        container.stop()
        container.remove()

    if redirect == False:
        return outputs


def prep_fastqc(mapped_read_pairs, volumes, raw_or_trimmed):
    outdir = os.path.join(OUTDIR, f"{raw_or_trimmed}_read_qc")
    Path(outdir).mkdir(exist_ok=True)
    cmds = [
        [
            "fastqc",
            "-t",
            str(NTHREADS),
            *[read for readpair in mapped_read_pairs.values() for read in readpair],
            "-o",
            os.path.join("/data", f"{raw_or_trimmed}_read_qc"),
        ]
    ]
    instructions = {
        "fastqc_inputs": {
            "image": "staphb/fastqc:latest",
            "volumes": volumes,
            "name": "fastqc_container",
            "cmds": cmds,
        }
    }
    return instructions


def prep_multiqc(input_dir, volumes):
    outdir = os.path.join(input_dir, "multiqc")
    cmds = [["multiqc", "-o", outdir, input_dir]]
    instructions = {
        "multiqc_inputs": {
            "image": "staphb/multiqc:latest",
            "volumes": volumes,
            "name": "multiqc_container",
            "cmds": cmds,
        }
    }
    return instructions


def mash_pair_reads(paired_reads_dict, linux=True):
    """For input reads_dir path and sample: list_of_paths
    pairs, concatenate pairs of reads to a new fastq.gz
    file in new 'paired' dir within reads_dir."""
    paired_dir = PurePath(OUTDIR).joinpath("paired_reads")
    Path(paired_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(paired_dir)

    if linux:
        for sample, paths in paired_reads_dict.items():
            cmd = ["cat", *paths]
            outpath = os.path.join(paired_dir, f"{sample}.fastq.gz")
            with open(outpath, "w") as f:
                proc = subprocess.Popen(cmd, stdout=f)
                proc.communicate()
    # Trying to add an option to execute in memory, if
    # user not on linux... unsure if sound
    else:
        for sample, paths in paired_reads_dict.items():
            outpath = os.path.join("paired_reads", f"{sample}.fastq.gz")
            with open(outpath, "w") as f:
                with open(paths[0], "r") as f_1:
                    f.write(f_1.read())
                    f.write("\n")
                with open(paths[1], "r") as f_2:
                    f.write(f_2.read())
    # Maybe consider checks or returning exit codes


def prep_mash_sketch(samples, volumes):
    outdir = os.path.join(OUTDIR, "mash_sketches")
    Path(outdir).mkdir(exist_ok=True)
    cmds = []
    for sample in samples:
        paired_sample_path = os.path.join("/data", "paired_reads", f"{sample}.fastq.gz")
        sketch_path = os.path.join("/data", "mash_sketches", f"{sample}.msh")
        cmd = [
            "mash",
            "sketch",
            "-r",
            "-c",
            "30",
            "-m",
            "2",
            "-o",
            f"{sketch_path}",
            f"{paired_sample_path}",
        ]
        cmds.append(cmd)
    instructions = {
        "mash_sketch_inputs": {
            "image": "staphb/mash:latest",
            "volumes": volumes,
            "name": "mash_sketch_container",
            "cmds": cmds,
        }
    }
    return instructions


def prep_mash_dist(samples, volumes):
    outdir = os.path.join(OUTDIR, "mash_distances")
    Path(outdir).mkdir(exist_ok=True)
    cmds = []
    for sample in samples:
        sketch_path = os.path.join("/data", "mash_sketches", f"{sample}.msh")
        dist_path = os.path.join(OUTDIR, "mash_distances", f"{sample}_dist.tab")
        cmd = [
            "mash",
            "dist",
            "-p",
            f"{NTHREADS}",
            "/db/RefSeqSketchesDefaults.msh",
            sketch_path,
        ]
        cmds.append({dist_path: cmd})
    instructions = {
        "mash_dist_inputs": {
            "image": "staphb/mash:latest",
            "volumes": volumes,
            "name": "mash_dist_container",
            "cmds": cmds,
        }
    }
    return instructions


def prep_trimmomatic(mapped_read_pairs, volumes):

    trim_opts = (
        "ILLUMINACLIP:/Trimmomatic-0.39/adapters/"
        "TruSeq3-PE.fa:2:30:10 AVGQUAL:20 "
        "LEADING:20 SLIDINGWINDOW:4:25 MINLEN:75"
    )
    cmds = list()
    for sample, pairs in mapped_read_pairs.items():
        baseout = os.path.join("/data", "trimmed_reads", f"{sample}.fastq")
        Path(os.path.join(OUTDIR, "trimmed_reads")).mkdir(exist_ok=True)
        cmd = shlex.split(
            (
                f"trimmomatic PE -threads {NTHREADS} "
                f"{pairs[0]} {pairs[1]} -baseout {baseout} "
                f"{trim_opts}"
            )
        )
        cmds.append(cmd)
    instructions = {
        "trimmomatic_inputs": {
            "image": "staphb/trimmomatic:0.39",
            "volumes": volumes,
            "name": "trimmomatic_container",
            "cmds": cmds,
        }
    }
    return instructions


def prep_spades(indir, samples, volumes):
    chr_assembly_dir = os.path.join(volumes.get(OUTDIR).get("bind"), "chr_assemblies")
    plas_assembly_dir = os.path.join(
        volumes.get(OUTDIR).get("bind"), "plasmid_assemblies"
    )
    cmds = list()
    for sample in samples:
        pairs = [os.path.join(indir, f"{sample}_{n}P.fastq") for n in (1, 2)]
        container_chr_assembly_dir = os.path.join(chr_assembly_dir, sample)
        container_plas_assembly_dir = os.path.join(plas_assembly_dir, sample)
        chr_cmd = shlex.split(
            f"spades.py -t {NTHREADS} -1 {pairs[0]} -2 {pairs[1]} --isolate -o {container_chr_assembly_dir}"
        )
        plas_cmd = shlex.split(
            f"spades.py -t {NTHREADS} -1 {pairs[0]} -2 {pairs[1]} --plasmid -o {container_plas_assembly_dir}"
        )
        cmds.extend([chr_cmd, plas_cmd])
    instructions = {
        "spades_inputs": {
            "image": "staphb/spades:latest",
            "volumes": volumes,
            "name": "spades_container",
            "cmds": cmds,
        }
    }
    return instructions


def prep_quast(sample_build_dict, proj_dir, volumes):
    """Takes a dict mapping species keys to dicts containing
    'sample': [sample_list] and 'build': GCF string pairs,
    and outputs commands to run quast with those references
    on assemblies in the dir indicated."""

    cmds = list()
    mounted_proj_dir = volumes.get(proj_dir).get("bind")
    # assembly_dir = os.path.join(proj_dir, "chr_assemblies")
    mounted_ref_dir, mounted_assembly_dir = (
        os.path.join(mounted_proj_dir, dir_)
        for dir_ in ("reference_genomes", "chr_assemblies")
    )
    for _, d in sample_build_dict.items():
        try:
            ref_fasta = os.path.join(mounted_ref_dir, f"{d['build']}_genomic.fna")
            ref_gff = os.path.join(mounted_ref_dir, f"{d['build']}_genomic.gff")
        except KeyError:
            continue
        for sample in d["samples"]:
            in_dir = os.path.join(mounted_assembly_dir, sample)
            # qc_dir = os.path.join(assembly_dir, "qc")
            # out_dir = os.path.join(qc_dir, sample)
            # Path(out_dir).mkdir(parents=True, exist_ok=True)
            mounted_out_dir = os.path.join(mounted_assembly_dir, "qc", sample)
            scaff_file = os.path.join(in_dir, "scaffolds.fasta")
            cmd = shlex.split(
                (
                    f"quast.py -t {NTHREADS} {scaff_file} -r {ref_fasta} "
                    f"-g {ref_gff} -o {mounted_out_dir} -l {sample} "
                    "--fragmented"
                )
            )
            cmds.append(cmd)
    instructions = {
        "quast_inputs": {
            "image": "staphb/quast:latest",
            "volumes": volumes,
            "name": "quast_container",
            "cmds": cmds,
        }
    }
    return instructions


def prep_abricate(samples, volumes):
    outdir = os.path.join(OUTDIR, "abricate")
    Path(outdir).mkdir(exist_ok=True)
    cmds = list()
    for sample in samples:
        chr_assembly_path, plas_assembly_path = (
            os.path.join("/data", subdir, sample, "scaffolds.fasta")
            for subdir in ("chr_assemblies", "plasmid_assemblies")
        )
        chr_output_path, plas_output_path = (
            os.path.join(OUTDIR, "abricate", filename)
            for filename in (f"{sample}.tsv", f"{sample}_plasmid.tsv")
        )
        for input_path, output_path in zip(
            (chr_assembly_path, plas_assembly_path), (chr_output_path, plas_output_path)
        ):
            cmds.append({output_path: ["abricate", input_path]})
    instructions = {
        "image": "staphb/abricate:latest",
        "volumes": volumes,
        "name": "abricate_container",
        "cmds": cmds,
    }
    return instructions


def get_mash_dfs(dist_dir, logger:logging.Logger):
    mash_dfs = dict()
    columns = [
        "Reference-ID",
        "Query-ID",
        "Mash-distance",
        "P-value",
        "Matching-hashes",
    ]
    get_mash_msg = (
        f"Collating Mash dist ('.tab') files in {dist_dir}"
    )
    logger.info(get_mash_msg)
    for file in os.listdir(dist_dir):
        if file[-4:] == ".tab":
            filepath = os.path.join(dist_dir, file)
            # get rid of the trailing "_dist" in filename
            sample = file.split(".")[0][:-5]
            df = pd.read_csv(filepath, sep="\t", header=None, names=columns)
            df.sort_values("Mash-distance", inplace=True)
            df["species"] = df["Reference-ID"].apply(lambda x: x.split("-")[-1])
            df.reset_index(drop=True, inplace=True)
            mash_dfs[sample] = df
    found_dist_files_msg = (
        f"Found {len(mash_dfs)} Mash dist files."
    )
    logger.info(found_dist_files_msg)

    return mash_dfs


def write_sample_species(mash_dfs):
    sample_species_dict = dict()
    for sample, df in mash_dfs.items():
        sample_dict = df.loc[0, ["species", "Reference-ID"]].to_dict()
        for key in sample_dict:
            val = sample_dict[key]
            if val[-4:] == ".fna":
                sample_dict[key] = val[:-4]
        sample_species_dict[sample] = sample_dict
    outpath = os.path.join(OUTDIR, "sample_species.json")
    with open(outpath, "w") as f:
        f.write(json.dumps(sample_species_dict, indent=4))


def get_species(df, logger:logging.Logger):
    """Return species of top hit from input DataFrame
    of Mash results."""
    try:
        species = df.sort_values("Mash-distance").loc[0, "species"]
        return species
    except KeyError:
        logger.warning("Can't find the species in the input mash_dict!")


def format_species(species_str:str):
    """Recover simple 'Species_genus' string from various
    formats of input species_str"""
    import re

    # double-check species formatting
    # search_str = r"^\s?([a-zA-Z]+)[\s_]([a-zA-Z]+).*$"
    search_str = r"^\s?([a-zA-Z]+)[\s_]([a-zA-Z]+).*$"
    groups = re.search(search_str, species_str).groups()
    species = "_".join(groups)
    return species


def get_ftp_file(
    domain, credentials_dict, target_dir, filename, output_path, 
    logger:logging.Logger
):
    """Download a file and save to filename at output_path."""
    from ftplib import FTP

    ftp = FTP(domain)
    user = credentials_dict.get("user", None)
    passwd = credentials_dict.get("password", None)
    ftp.login(user, passwd)
    ftp.cwd(target_dir)
    download_msg = (
        f"Attempting to download {filename} in {target_dir} from {domain}."
    )
    logger.info(download_msg)
    with open(output_path, "wb") as f:
        ftp.retrbinary("RETR " + filename, f.write, 1024)
    ftp.quit()


def get_assembly_summary(species, email, logger, outdir=None):
    """Download 'assembly_summary.txt' file from
    NCBI FTP server, summarizing available genome
    assemblies for a given species."""
    if outdir is None:
        outdir = os.getcwd()
    ftp_url_base = "ftp.ncbi.nlm.nih.gov"
    credentials_dict = dict(user="anonymous", passwd=email)
    bacteria_prefix = "/genomes/refseq/bacteria/"
    # Hacky worharound; fix later
    if species == "Klebsiella_sp":
        species = "Klebsiella_pneumoniae"
    species_dir = f"{bacteria_prefix}{species}/"
    target_file = "assembly_summary.txt"
    outpath = os.path.join(outdir, f"{species}_assembly_summary.txt")
    get_ftp_file(
        ftp_url_base, credentials_dict, species_dir, 
        target_file, outpath, logger=logger
    )


def combine_mash_species(
    mash_dfs, logger:logging.Logger, drop_extraneous=True, 
):
    """Take a dict of sample: DataFrame pairs, detect
    the species of the samples, combine those with same
    species, and output species: concatenated DataFrame
    pairs."""
    species_dict = dict()
    for sample, df in mash_dfs.items():
        species = format_species(get_species(df, logger))
        sample_list = species_dict.setdefault(species, list())
        sample_list.append(sample)
        # try:
        #     species_dict[species].append(sample)
        # except KeyError:
        #     species_dict[species] = [sample]

    species_dfs = dict()
    for species, samples in species_dict.items():
        df = pd.concat(
            [mash_dfs[sample] for sample in samples],
            keys=[sample for sample in samples],
            names=["sample", "index"],
        )
        df["rank"] = df.groupby("Query-ID")["Mash-distance"].rank()
        if drop_extraneous:
            df = df[df["species"].str.contains(species)]
        species_dfs[species] = df
    return species_dfs


def get_best_hits(species_dfs, max_hits=None):
    """Take in dict of species: DataFrame pairs, find highest-ranking
    reference genome across all samples, and return a dict of
    species: Series pairs with number of hits indicated."""
    hits_dict = dict()
    for species, df in species_dfs.items():
        pivoted = df.reset_index(level=0).pivot(
            index="Reference-ID",
            columns="sample",
            values="rank"
        )
        averaged = pivoted.mean(axis=1).sort_values()
        averaged.name = "avg_rank"
        if max_hits:
            averaged = averaged.iloc[:max_hits]
        hits_dict[species] = averaged
    return hits_dict


def load_assembly_summaries(
    indir, species_list, email, logger:logging.Logger
):
    assembly_dfs = dict()

    def load_summary(filepath):
        df = pd.read_csv(filepath, sep="\t", header=1)
        return df

    for species in species_list:
        # species = format_species(species)
        filename = f"{species}_assembly_summary.txt"
        filepath = os.path.join(indir, filename)
        if not os.path.isfile(filepath):
            get_assembly_summary(
                species, email, logger=logger, outdir=indir
            )
        assembly_dfs[species] = load_summary(filepath)
    return assembly_dfs


def assemb_ids_from_mash(species_dfs, assembly_dfs, max_hits=None):
    """Takes in dicts of species: DataFrame pairs summarizing the Mash
    distance results for all species and the assemblies available from
    NCBI FTP site.  Outputs a dict of species: DataFrame pairs sorted
    by lowest average assembly distance for all samples in species.
    FTP path can then be used to download assemblies."""
    assemb_dict = dict()
    pattern = re.compile(".*(SAM[EDN][A-Z]?[0-9]+).*")
    all_species = sorted(
        list(set(species_dfs.keys()).intersection(set(assembly_dfs.keys())))
    )
    hits_dict = get_best_hits(species_dfs, max_hits=max_hits)
    for species in all_species:
        hits_df = hits_dict[species].to_frame().reset_index()
        hits_df["biosample"] = hits_df["Reference-ID"].str.extract(pattern)
        df = pd.merge(
            hits_df, assembly_dfs[species].copy(), on="biosample", how="inner"
        ).sort_values("avg_rank")
        assemb_dict[species] = df[
            ["bioproject", "biosample", "# assembly_accession", "ftp_path"]
        ]
    return assemb_dict


def get_abricate_dfs(sample_list, input_dir):
    """For a list of sample names and input_dir,
    return sample: DataFrame pairs in a dict."""

    columns = [
        "#FILE",
        "SEQUENCE",
        "START",
        "END",
        "STRAND",
        "GENE",
        "COVERAGE",
        "COVERAGE_MAP",
        "GAPS",
        "%COVERAGE",
        "%IDENTITY",
        "DATABASE",
        "ACCESSION",
        "PRODUCT",
        "RESISTANCE",
    ]
    df_dict = dict()
    for obj in os.listdir(input_dir):
        for sample in sample_list:
            if (sample in obj) and (obj[-4:] == ".tsv"):
                filepath = os.path.join(input_dir, obj)
                df = pd.read_csv(filepath, sep="\t", names=columns, skiprows=6)
                df.columns = [col.lower() for col in df.columns]
                df_dict[obj[:-4]] = df
    return df_dict


def combined_abr_df(df_dict, combine_plasmid=True):
    """Convert an input dict of sample: DataFrame pairs
    to a single concatenated DataFrame.  If combine_plasmid
    remove '_plasmid' from sample names, and add a 'plasmid'
    column."""
    import pandas as pd
    import re

    abr_df = pd.concat(
        df_dict.values(),
        keys=df_dict.keys(),
        names=["sample", "feature"],
        axis=0,
        sort=True,
    )

    def collapse_sample_names(sample):
        match = re.match(r"([a-zA-Z0-9]+)_[pP].*", sample)
        match_s = match.groups()[0] if match else sample
        return match_s

    if combine_plasmid:
        abr_df = abr_df.reset_index(level=0)
        abr_df["plasmid"] = (
            abr_df["sample"].str.contains("plas", flags=re.IGNORECASE).astype(int)
        )
        abr_df["sample"] = abr_df["sample"].apply(collapse_sample_names)
        abr_df.set_index(["sample", abr_df.index], inplace=True)
    abr_df["coverage_map"] = abr_df["coverage_map"].astype(str)

    return abr_df


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    import logging
    import boto3
    from botocore.exceptions import ClientError

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        _ = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def calc_cov(path_to_gen_stats_file, sample_to_gen_size_dict, sample_name_regex=None):
    """Calculate approximate genome coverage from
    MultiQC results in pat_to_gen_stats_file, with
    dict mapping genome size in Mbp to species names."""
    import numpy as np
    import pandas as pd

    df = pd.read_csv(path_to_gen_stats_file, sep="\t")
    df.columns = [
        col.replace("FastQC_mqc-generalstats-fastqc-", "") for col in df.columns
    ]
    df["total_bases"] = df["total_sequences"] * df["avg_sequence_length"]
    if sample_name_regex:
        repl = lambda x: x.group(1)
        df["real_sample"] = df["Sample"].str.replace(
            sample_name_regex, repl, regex=True
        )
    else:
        df["real_sample"] = df["Sample"]
    total_bases_S = df.groupby("real_sample")["total_bases"].sum()
    size_S = pd.Series(sample_to_gen_size_dict)
    if size_S.mean() < 1e6:
        size_S = size_S.apply(lambda x: x * 1e6)

    covs_S = total_bases_S / size_S

    return covs_S.sort_index()


def get_ftp_files(
    url, ext, logger:logging.Logger, credentials_dict={}, 
    outdir=os.getcwd(),
):
    """For FTP URL recovered from output of
    assemb_ids_from_mash, get the file specified
    in 'ext' like 'fna', 'gbff', or 'gff'.
    Download to outdir or else cwd."""
    import gzip
    import os
    from urllib.parse import urlparse

    parsed = urlparse(url)
    domain = parsed.netloc
    target_dir = parsed.path
    assembly_id = target_dir.split("/")[-1]

    translator = {
        "fna": "_genomic.fna.gz",
        "gbff": "_genomic.gbff.gz",
        "gff": "_genomic.gff.gz",
    }

    filename = assembly_id + translator[ext]
    filepath = os.path.join(outdir, filename)
    get_ftp_file(
        domain=domain,
        credentials_dict=credentials_dict,
        target_dir=target_dir,
        filename=filename,
        output_path=filepath,
        logger=logger
    )
    decompressed = os.path.join(outdir, filename[:-3])
    with gzip.open(filepath, "rb") as f_in:
        with open(decompressed, "wb") as f_out:
            f_out.write(f_in.read())
    # rename ".gbff" to ".gb"
    if ext == "gbff":
        os.rename(decompressed, decompressed[:-2])


def download_assemblies(
    assembly_ids_dict, species_dfs, ext, 
    logger:logging.Logger, n_assemblies=1
):
    """Downloads n_assemblies from assembly_ids_dict['species']' 'ftp_path' field."""
    ref_dir = os.path.join(OUTDIR, "reference_genomes")
    Path(ref_dir).mkdir(exist_ok=True)
    sample_build_dict = {species: dict() for species in assembly_ids_dict}
    for species, df in assembly_ids_dict.items():
        # if species_dfs[species].index.levels[0].shape[0] > 1:
        # Weird; getting some empty dataframes here
        if df.shape[0] == 0:
            continue
        ftp_paths = df.loc[: n_assemblies - 1, "ftp_path"]
        ftp_paths.apply(
            get_ftp_files, ext=ext, logger=logger, outdir=ref_dir
        )
        build_name = ftp_paths.loc[0].split("/")[-1]
        sample_build = sample_build_dict.get(species)
        sample_build["build"] = build_name
        sample_build["samples"] = species_dfs[species].index.levels[0]
    return sample_build_dict


def group_reads_by_species(
    sample_read_dict, logger: logging.Logger, read_pairs_dict=None
):
    """Takes in sample_read_dict mapping species to reference files and samples
    and copies either trimmed or raw reads for samples into new, species-specific
    dir, to prepare them for use in SNP calling with SNVPhyl."""

    master_dict = {
        sample: d
        for sample, d in sample_read_dict.items()
        if d.get("samples").shape[0] > 0
    }
    if PHYLO_READ_TYPE == "raw":
        for species, d in master_dict.items():
            species_reads_dir = os.path.join(OUTDIR, "untrimmed_reads", species)
            Path(species_reads_dir).mkdir(parents=True, exist_ok=True)
            old_paths = {
                sample: read_pairs_dict.get(sample) for sample in d.get("samples")
            }
            for sample, old_path_list in old_paths.items():
                for i, old_path in enumerate(old_path_list):
                    # filename = os.path.basename(old_path)
                    new_path = os.path.join(
                        species_reads_dir, f"{sample}_{i+1}.fastq.gz"
                    )
                    decompressed = new_path[:-3]
                    shutil.copyfile(old_path, new_path)
                    with gzip.open(new_path, "rb") as f_in:
                        with open(decompressed, "wb") as f_out:
                            f_out.write(f_in.read())
                    os.remove(new_path)
    if PHYLO_READ_TYPE == "trimmed":
        for species, d in master_dict.items():
            species_reads_dir = os.path.join(OUTDIR, "trimmed_reads", species)
            Path(species_reads_dir).mkdir(parents=True, exist_ok=True)
            for sample in d.get("samples"):
                old_paths = [
                    os.path.join(OUTDIR, "trimmed_reads", f"{sample}_{n}P.fastq")
                    for n in range(1, 3)
                ]
                new_paths = [
                    os.path.join(species_reads_dir, f"{sample}_{n}.fastq")
                    for n in range(1, 3)
                ]
                for old_path, new_path in zip(old_paths, new_paths):
                    shutil.copyfile(old_path, new_path)


def rename_reads_lyveset(read_pairs, rename_dir="renamed"):
    """Generates a rename_dir subdir of first parent dir
    in read_pairs dict, and copies raw reads to simplified
    sample named files.  Pursuant to interleaving reads
    prior to running LyveSET."""
    import os
    import shutil

    for sample, paths in read_pairs.items():
        os.chdir(os.path.dirname(paths[0]))
        if not os.path.isdir(rename_dir):
            Path(rename_dir).mkdir()
        for path in paths:
            newname = "_".join([sample, *os.path.basename(path).split(sep="_")[-2:]])
            shutil.copy(path, os.path.join(rename_dir, newname))


# def interleave_reads(input_dir, subset_samples=None):
#     """Uses containerized LyveSET to first interleave paired reads
#     as required for main LyveSET analysis.  If subset_samples
#     provided, will look for files with those names; otherwise,
#     will interleave all read files in input_dir."""
#     files = [
#         os.path.join("/data/renamed", file)
#         for file in os.listdir(f"{input_dir}/renamed")
#     ]
#     valid_extensions = [".fastq.gz", ".fastq", ".fq"]
#     read_files = [
#         file for file in files for ext in valid_extensions if file.endswith(ext)
#     ]
#     if subset_samples:
#         read_files = [
#             file for file in read_files for sub in subset_samples if sub in file
#         ]
#     cmd = f"shuffleSplitReads.pl --numcpus=8 -o /data/interleaved " + " ".join(
#         read_files
#     )
#     outputs = docker_exec(
#         [cmd], "staphb/lyveset:1.1.4f", "interleave_reads", f"-v {input_dir}:/data"
#     )
#     return outputs


# def run_lyveset(
#     interleaved_reads_dir,
#     reference_genome,
#     output_dir,
#     leaveout_reads=None,
#     other_opts=None,
# ):
#     """Run LyveSET using interleaved reads, a reference_genome,
#     and a given output_dir path."""
#     if os.path.isdir(output_dir):
#         from datetime import datetime

#         now = datetime.now()
#         formatted = datetime.strftime(now, "%Y%m%d_%H%M")
#         formatted_name = f"lyveset_{formatted}"
#         output_dir = os.path.join(output_dir, formatted_name)
#     mount_str, remapped = map_docker_dirs(
#         [interleaved_reads_dir, reference_genome, output_dir]
#     )
#     cmds = [f"set_manage.pl --create {remapped[output_dir]}"]
#     for file in os.listdir(interleaved_reads_dir):
#         # Don't have to check multiple extensions; should all be .fastq.gz
#         if file[-9:] == ".fastq.gz" and file not in leaveout_reads:
#             newpath = os.path.join(remapped[interleaved_reads_dir], file)
#             cmd = f"set_manage.pl {remapped[output_dir]} --add-reads {newpath}"
#             cmds.append(cmd)
#     launch_cmd = (
#         f"launch_set.pl --numcpus 8 {remapped[output_dir]} "
#         f"-ref {remapped[reference_genome]} "
#         f"--qsubxopts '-n 8'"
#     )
#     if other_opts:
#         launch_cmd = " ".join(launch_cmd, **other_opts)
#     cmds.append(launch_cmd)

#     outputs = docker_exec(
#         cmds, "staphb/lyveset:1.1.4f", formatted_name, f"-v {mount_str}"
#     )

#     return outputs


def run_snvphyl(snvphyl_dir, sample_build_dict, logger=logging.Logger):
    pipes = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    Path(os.path.join(OUTDIR, "snvphyl")).mkdir(exist_ok=True)
    if PHYLO_READ_TYPE == "raw":
        # reads_dir = os.path.join(OUTDIR, "untrimmed_reads")
        reads_dir = INDIR
    elif PHYLO_READ_TYPE == "trimmed":
        reads_dir = os.path.join(OUTDIR, "trimmed_reads")
    snvphyl_ex = os.path.join(snvphyl_dir, "bin", "snvphyl.py")
    species_list = list()
    for obj in os.listdir(reads_dir):
        obj_path = os.path.join(reads_dir, obj)
        if os.path.isdir(obj_path) and (obj in sample_build_dict):
            species_list.append(obj)
    results_dict = dict()
    for species in species_list:
        species_reads_dir = os.path.join(reads_dir, species)
        output_dir = os.path.join(OUTDIR, "snvphyl", species)
        ref_assembly = sample_build_dict[species].get("build")
        ref_gen_path = os.path.join(
            OUTDIR, "reference_genomes", f"{ref_assembly}_genomic.fna"
        )
        cmd = shlex.split(
            (
                f"python {snvphyl_ex} --deploy-docker --fastq-dir "
                f"{species_reads_dir} --reference-file {ref_gen_path} "
                f"--min-coverage 5 --output-dir {output_dir}"
            )
        )
        logger.info(cmd)
        proc = subprocess.Popen(cmd, **pipes)
        stdout, stderr = proc.communicate()
        results_dict[species] = [std.decode("utf-8") for std in (stdout, stderr)]
    return results_dict


def draw_tree(newick_file_path, yield_or_show, figsize=(4, 5)):
    """Render an input newick file to stdout."""
    from Bio import Phylo
    import matplotlib.pyplot as plt

    tree = next(Phylo.parse(newick_file_path, "newick"))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    Phylo.draw(tree, axes=ax, do_show=False)
    ax.axis("off")
    if yield_or_show == "show":
        plt.show()
        return None
    elif yield_or_show == "yield":
        return fig


def sorted_d(in_dict):
    """Sort dicts in alphabetical order; aids in
    summarizing results from different tools prior
    to reporting to external agency."""
    sortednames = sorted(in_dict.keys(), key=lambda x: x.lower())
    out_dict = {k: in_dict[k] for k in sortednames}
    return out_dict


def unzip_fastqc():
    raw_reads_qc_dir = os.path.join(OUTDIR, "raw_read_qc")
    os.chdir(raw_reads_qc_dir)
    for obj in os.listdir():
        if obj[-4:] == ".zip":
            with ZipFile(obj, "r") as zip_obj:
                zip_obj.extractall()


def ar_report_stats(in_dir):
    """Looks for `fastqc_data.txt` file in in_dir,
    uses to summarize stats needed for reporting
    results in PRPT exercise."""
    import os
    import pandas as pd

    qc_dfs = dict()

    for root, _, files in os.walk(in_dir):
        for file in files:
            if file == "fastqc_data.txt":
                filepath = os.path.join(root, file)
                start_line = 0
                end_line = None
                with open(filepath, "r") as f:
                    for i, line in enumerate(f.readlines()):
                        if ">>Per sequence quality scores" in line:
                            start_line = i + 1
                        if (start_line > 0) and (end_line is None):
                            if "END_MODULE" in line:
                                end_line = i - 1
                subdir = os.path.split(root)[-1]
                #                 sample, read = subdir.split('_')[1:3]
                # parsing the sample name from subdirs may require
                # tuning for different runs
                parts = subdir.split("_")
                sample, read = parts[0].split("-")[0], parts[3]
                S = pd.read_csv(
                    filepath,
                    sep="\t",
                    skiprows=start_line,
                    nrows=end_line - start_line,
                    index_col=0,
                )
                if sample in qc_dfs:
                    qc_dfs[sample][read] = S
                else:
                    qc_dfs[sample] = dict()
    sample_dfs = {sample: pd.concat(reads, axis=1) for sample, reads in qc_dfs.items()}
    q30_d = dict()
    avg_q = dict()
    total_reads_d = dict()

    for sample, df in sample_dfs.items():
        both = df.sum(axis=1)
        totals = both.sum()
        q30_proportion = both.iloc[30:].sum() / totals
        q30_d[sample] = int(q30_proportion * 100)
        avg_q[sample] = round((sum(both.values * both.index) / both.sum()), 1)
        total_reads_d[sample] = totals

    return sorted_d(avg_q), sorted_d(q30_d), sorted_d(total_reads_d)


def mock_assembly(assembly_dir_dict):
    """A function to skip the costly assembly step
    when troubleshooting this script."""
    for dir_name, dir_path in assembly_dir_dict.items():
        shutil.copytree(dir_path, os.path.join(OUTDIR, dir_name))


def main():
    """Runs the various functions of this script to process the data."""
    """1. Read QC: get_read_pairs(), map_docker_dirs(), docker_exec() 
        for FastQC and MultiQC
    2. Species identification: mash_pair_reads(), get_top_hits(), 
        get_species(), format_species(), get_assembly_summary(), 
        get_ftp_files()
    3. Trimming and repeat QC: map_docker_dirs(), docker_exec()
        for Trimmomatic, FastQC, and MultiQC
    4. (Check Quality): calc_cov(), review MultiQC results to see 
        if it makes sense to proceed to assembly
    5. Assembly: map_docker_dirs(), docker_exec() for SPAdes 
        and PlasmidSPAdes
    6. (Check Assembly): run_quast() (expects local installation) 
        using reference genomes downloaded under "Species identification" 
        step
    7. SNP Analysis: rename_reads_lyveset(), interleave_reads(), 
        run_lyveset()  -- OR --  run_snv_phyl()
    8. (Check SNP results): draw_tree(), and check distance 
        matrix manually
    9. Detect AMR & Virulence Alleles: docker_exec() on 
        'staphb/abricate:latest', abricate_dfs(), combined_abr_df()
    10. Summarize & Report: ar_report_stats() to get Q30 & other metrics"""
    OUTDIR.mkdir(exist_ok=True, parents=True)
    os.chdir(OUTDIR)
    prpt_logger = setup_logger()
    uncaught_exception_logger = partial(
        handle_exception, logger=prpt_logger
    )
    sys.excepthook = uncaught_exception_logger
    indir_filepaths = [os.path.join(INDIR, file) for file in os.listdir(INDIR)]
    sample_names = recognize_sample_names(indir_filepaths, prpt_logger)
    read_pairs = get_read_pairs(
        indir_filepaths, sample_names, "raw", prpt_logger
    )

    volumes = map_docker_dirs(INDIR, OUTDIR)
    mapped_read_pairs = map_read_pairs(read_pairs, volumes, "raw")

    raw_fastqc_cmds = prep_fastqc(mapped_read_pairs, volumes, "raw")
    raw_multiqc_cmds = prep_multiqc(os.path.join("/data", "raw_read_qc"), volumes)
    mash_pair_reads(read_pairs, linux=True)
    mash_sketch_cmds = prep_mash_sketch(sample_names, volumes)
    mash_dist_cmds = prep_mash_dist(sample_names, volumes)

    # Part 1: perform preliminary QC and Mash steps
    part_1_msg = (
        f"Initiating first phase of pipeline: preliminary QC and Mash dist."
    )
    prpt_logger.info(part_1_msg)
    results = {}
    for prepped_dict in [
        raw_fastqc_cmds,
        raw_multiqc_cmds,
        mash_sketch_cmds,
        mash_dist_cmds,
    ]:
        for step, instructions in prepped_dict.items():
            if step == "mash_dist_inputs":
                run_docker(instructions, prpt_logger, redirect=True)
            else:
                result = run_docker(instructions, prpt_logger)
                results[step] = result

    # Part 2: determine best reference genomes and download
    part_2_msg = (
        "Initiating second phase of pipeline: "
        "determination of species, and downloading reference genomes."
    )
    prpt_logger.info(part_2_msg)
    mash_dfs = get_mash_dfs(
        os.path.join(OUTDIR, "mash_distances"), prpt_logger
    )
    write_sample_species(mash_dfs)
    species_dfs = combine_mash_species(mash_dfs, prpt_logger)
    best_hits = get_best_hits(species_dfs)
    for species in best_hits:
        get_assembly_summary(
            species, EMAIL, outdir=OUTDIR, logger=prpt_logger
        )
    assembly_dfs = load_assembly_summaries(OUTDIR, list(best_hits.keys()), EMAIL, prpt_logger)
    assembly_dict = assemb_ids_from_mash(species_dfs, assembly_dfs,)
    exts = ["fna", "gff"]
    if PHYLO == "lyveset":
        exts.append("gbff")
    for ext in exts:
        sample_build_dict = download_assemblies(
            assembly_dict, species_dfs, ext=ext,
            logger=prpt_logger
        )

    # Part 3: trim, repeat QC, assemble, and check assembly
    part_3_msg = (
        "Initiating third phase of pipeline: "
        "read trimming, QC of trimmed reads, assembly, and QC of assembly."
    )
    prpt_logger.info(part_3_msg)
    trimmed_dir = os.path.join(OUTDIR, "trimmed_reads")
    container_trimmed_dir = os.path.join("/data", "trimmed_reads")
    trimmomatic_cmds = prep_trimmomatic(mapped_read_pairs, volumes)
    results["trimmomatic_inputs"] = run_docker(
        trimmomatic_cmds["trimmomatic_inputs"], prpt_logger
    )
    trimmed_read_pairs = get_read_pairs(
        os.listdir(trimmed_dir), sample_names, "trimmed", prpt_logger
    )
    mapped_trimmed_read_pairs = map_read_pairs(trimmed_read_pairs, volumes, "trimmed")
    trim_fastqc_cmds = prep_fastqc(mapped_trimmed_read_pairs, volumes, "trimmed")
    trim_multiqc_cmds = prep_multiqc(os.path.join("/data", "trimmed_read_qc"), volumes)
    spades_cmds = prep_spades(container_trimmed_dir, sample_names, volumes)
    quast_cmds = prep_quast(sample_build_dict, OUTDIR, volumes)
    genome_qc_dir = os.path.join("/data", "chr_assemblies", "qc")
    quast_multiqc_cmds = prep_multiqc(genome_qc_dir, volumes)
    prepped_dicts = [
        trim_fastqc_cmds,
        trim_multiqc_cmds,
        quast_cmds,
        quast_multiqc_cmds,
    ]
    if not ASSEMBLIESDIR:
        prepped_dicts.insert(2, spades_cmds)
    for prepped_dict in prepped_dicts:
        # Alternate for testing purposes; omit spades_cmds
        # test_assemblies_dir = "/dcm/data/prpt/tests/assemblies"
        # assemblies_dict = {
        #     dir_name: os.path.join(test_assemblies_dir, dir_name)
        #     for dir_name in ("chr_assemblies", "plasmid_assemblies")
        # }
        # mock_assembly(assemblies_dict)
        # quast_cmds = prep_quast(sample_build_dict, OUTDIR, volumes)
        # genome_qc_dir = os.path.join("/data", "chr_assemblies", "qc")
        # quast_multiqc_cmds = prep_multiqc(genome_qc_dir, volumes)
        # for prepped_dict in [
        #     trim_fastqc_cmds,
        #     trim_multiqc_cmds,
        #     quast_cmds,
        #     quast_multiqc_cmds,
        # ]:
        for step, instructions in prepped_dict.items():
            result = run_docker(instructions, logger=prpt_logger)
            results[step] = result

    # Part 4: SNP analysis, and Abricate
    part_4_msg = (
        "Initiating fourth phase of pipeline: "
        "SNP analysis, and geneome annotation from Abricate."
    )
    prpt_logger.info(part_4_msg)
    group_reads_by_species(
        sample_build_dict, prpt_logger, read_pairs_dict=read_pairs
    )
    # snvphyl_msg = (
    #     "Beginning SNVPhyl with options: "
    #     " ".join()
    # )
    results["snvphyl"] = run_snvphyl(
        SNVDIR, sample_build_dict, logger=prpt_logger
    )
    abricate_cmds = prep_abricate(sample_names, volumes)
    run_docker(abricate_cmds, logger=prpt_logger, redirect=True)
    abricate_dir = os.path.join(OUTDIR, "abricate")
    abricate_dfs_dict = get_abricate_dfs(sample_names, abricate_dir)
    abricate_dfs = combined_abr_df(abricate_dfs_dict)
    dfs_excel_path = os.path.join(OUTDIR, "abricate_dfs.xlsx")
    abricate_dfs.to_excel(dfs_excel_path)
    unzip_fastqc()
    avg_q, q30, total_reads = ar_report_stats(os.path.join(OUTDIR, "raw_read_qc"))
    read_qc_stats = {"avg_q": avg_q, "q30": q30, "total_reads": total_reads}
    with open(os.path.join(OUTDIR, "read_qc_stats.json"), "w") as f:
        f.write(json.dumps(read_qc_stats, indent=4))

    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        f.write(json.dumps(results, indent=4))
    pipeline_complete_msg = (
        "This pipeline ran to completion successfully."
    )
    prpt_logger.info(pipeline_complete_msg)


if __name__ == "__main__":
    main()
    # pass
