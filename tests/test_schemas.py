import json

from grn_agent.schemas import Dataset, RunManifest, ScoredEdge


def test_dataset_roundtrip():
    d = Dataset(dataset_id="D", species="mouse", modalities=["scrna"], genes=[], samples=[])
    raw = d.model_dump()
    d2 = Dataset.model_validate(raw)
    assert d2.species == "mouse"


def test_scored_edge_json():
    e = ScoredEdge(
        source_tf="A",
        target_gene="B",
        p_present=0.9,
        logit=2.2,
        confidence_score=0.9,
    )
    s = json.dumps(e.model_dump(mode="json"))
    e2 = ScoredEdge.model_validate(json.loads(s))
    assert e2.p_present == 0.9


def test_run_manifest():
    m = RunManifest(run_id="r1", dataset_id="d1")
    assert m.model_version == "grnagent_v1"
