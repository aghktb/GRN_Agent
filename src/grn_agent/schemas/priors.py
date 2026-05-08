from pydantic import BaseModel, Field


class PriorBundle(BaseModel):
    """Data-driven priors per candidate or per context (SDD §3.5)."""

    p_grnboost: float | None = None
    p_genie3: float | None = None
    p_pidc: float | None = None
    scenic_regulon_support: float | None = None
    bootstrap_stability: float | None = None
    ensemble_prior: float = Field(ge=0.0, le=1.0, default=0.0)
