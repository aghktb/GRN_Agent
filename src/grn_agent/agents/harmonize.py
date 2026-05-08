from __future__ import annotations

from grn_agent.schemas import Dataset, GeneMeta


def harmonize_genes(dataset: Dataset, species: str) -> Dataset:
    """Ensure symbols stable; placeholder for Ensembl mapping."""
    out_genes: list[GeneMeta] = []
    for g in dataset.genes:
        sym = (g.symbol or g.gene_id).upper() if g.symbol or g.gene_id else g.gene_id
        out_genes.append(
            GeneMeta(
                gene_id=g.gene_id,
                symbol=sym,
                ensembl_id=g.ensembl_id,
            )
        )
    return dataset.model_copy(
        update={
            "genes": out_genes,
            "species": species,
            "metadata": {**dataset.metadata, "harmonized": True},
        }
    )
