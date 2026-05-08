#!/usr/bin/env python3
"""
Genome database management CLI.

Commands:
  list                     Show all cached genomes
  download <species>       Download + index a genome (e.g. mouse, mm10, hg38)
  register <species> <fasta> <gtf>  Register locally provided FASTA + GTF
  show-supported           Print all known species/builds

Examples:
  python scripts/manage_genome_db.py list
  python scripts/manage_genome_db.py download mouse
  python scripts/manage_genome_db.py download hg38
  python scripts/manage_genome_db.py register mm10 /data/mm10.fa /data/mm10.gtf
  python scripts/manage_genome_db.py show-supported
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grn_agent.acquisition.genome_db import (
    GenomeDB,
    _BUILD_INDEX,
    _REGISTRY,
    ensure_genome,
)


def cmd_list(args: argparse.Namespace) -> None:
    db = GenomeDB(args.db_root or None)
    cached = db.list_cached()
    if not cached:
        print("No genomes cached yet.")
        print(f"DB location: {db._root}")
        return
    print(f"Cached genomes ({len(cached)}) in {db._root}:")
    print(f"{'Key':<14} {'Indexed':<10} {'Downloaded':<22} {'FASTA size'}")
    print("-" * 75)
    for g in cached:
        fasta_p = Path(g["fasta"])
        size = f"{fasta_p.stat().st_size / 1e9:.1f} GB" if fasta_p.is_file() else "missing"
        print(f"{g['key']:<14} {str(g['indexed']):<10} {g['downloaded_at'][:19]:<22} {size}")


def cmd_download(args: argparse.Namespace) -> None:
    db = GenomeDB(args.db_root or None)
    species = args.species
    print(f"[genome_db] Ensuring genome: {species}")
    fasta, gtf = db.ensure(species)
    entry = db.get(species)
    print(f"\nDone!")
    print(f"  FASTA : {fasta}")
    print(f"  GTF   : {gtf}")
    print(f"  Indexed: {entry.fasta_indexed if entry else '?'}")


def cmd_register(args: argparse.Namespace) -> None:
    db = GenomeDB(args.db_root or None)
    entry = db.register_existing(args.species, args.fasta, args.gtf)
    print(f"Registered '{args.species}' in genome DB.")
    print(f"  FASTA  : {entry.fasta_path}")
    print(f"  GTF    : {entry.gtf_path}")
    print(f"  Indexed: {entry.fasta_indexed}")


def cmd_show_supported(_: argparse.Namespace) -> None:
    print(f"{'Key':<14} {'Common name':<14} {'Latin name':<32} {'Assembly':<14} {'Ensembl release'}")
    print("-" * 90)
    seen = set()
    for b in _REGISTRY:
        if b.key not in seen:
            print(f"{b.key:<14} {b.species_common:<14} {b.species_latin:<32} {b.assembly:<14} {b.ensembl_release}")
            seen.add(b.key)
    print(f"\nAliases: {', '.join(sorted(set(_BUILD_INDEX.keys()) - seen))}")
    print(
        "\nTo add a new species, register the FASTA + GTF:\n"
        "  python scripts/manage_genome_db.py register <name> <fasta.fa> <annotation.gtf>"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Genome DB management")
    ap.add_argument("--db-root", default="", help="Override default genome DB root (~/.cache/grn_agent/genomes/)")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List cached genomes")

    dl = sub.add_parser("download", help="Download + index a genome")
    dl.add_argument("species", help="Species or build: mouse, mm10, hg38, rat, zebrafish, …")

    reg = sub.add_parser("register", help="Register existing local FASTA + GTF")
    reg.add_argument("species", help="Species or build key")
    reg.add_argument("fasta", help="Path to FASTA file (will be indexed if .fai missing)")
    reg.add_argument("gtf", help="Path to GTF file")

    sub.add_parser("show-supported", help="Show all supported species/builds")

    args = ap.parse_args()
    if args.db_root:
        args.db_root = args.db_root.strip() or None
    else:
        args.db_root = None

    dispatch = {
        "list": cmd_list,
        "download": cmd_download,
        "register": cmd_register,
        "show-supported": cmd_show_supported,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
