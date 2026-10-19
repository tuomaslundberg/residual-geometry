"""The thesis-sample labelling scheme (scripts/thesis_sample.py)."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "thesis_sample", Path(__file__).parents[1] / "scripts" / "thesis_sample.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)


def test_label_scheme():
    t = 0.4
    assert ts.label({"NA": 0.9}, t) == ("NA", False)                      # pure main
    assert ts.label({"NA": 0.9, "ne": 0.8}, t) == ("NA", False)           # sub coalesced
    assert ts.label({"NA": 0.9, "ne": 0.8, "sr": 0.5}, t) == ("NA", False)  # two subs, same main
    assert ts.label({"NA": 0.9, "OP": 0.5}, t)[0] is None                 # crosses main boundary
    assert ts.label({"NA": 0.9, "ob": 0.5}, t)[0] is None                 # sub of another main
    assert ts.label({"ne": 0.8}, t)[0] is None                            # sub without its main
    assert ts.label({"NA": 0.1}, t)[0] is None                            # no label
    assert ts.label({"NA": 0.9, "MT": 0.7}, t) == ("NA", True)            # MT is a flag
    assert ts.label({"MT": 0.7}, t) == ("MT", True)                       # MT alone
    assert ts.label({"MT": 0.7, "ne": 0.6}, t)[0] is None                 # orphan sub
    assert ts.label({"NA": 0.44}, ts.THRESHOLD["fin"])[0] is None         # Finnish threshold 0.45


def test_draw_is_uniform_seeded_and_excludes(tmp_path):
    import json
    from types import SimpleNamespace

    def write(name, labels):
        p = tmp_path / name
        p.write_text("".join(json.dumps({"id": f"{name}{i}", "label": l}) + "\n"
                             for i, l in enumerate(labels)))
        return p

    fi = write("fi", ["NA"] * 50 + ["SP"] * 9 + ["LY"] * 30)
    en = write("en", ["NA"] * 20 + ["SP"] * 40)
    out = tmp_path / "out"
    args = SimpleNamespace(fi=[fi], en=[en], per_class=10, seed=1, exclude=["LY"], out=str(out))
    ts.cmd_draw(args)
    rows = [json.loads(l) for l in open(out / "fi.jsonl")]
    assert sorted(r["label"] for r in rows) == ["NA"] * 10 + ["SP"] * 9
    first = (out / "fi.jsonl").read_text()
    ts.cmd_draw(args)
    assert (out / "fi.jsonl").read_text() == first
