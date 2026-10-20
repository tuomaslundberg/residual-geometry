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


def test_draw_caps_sites_drops_mt_hybrids_and_is_seeded(tmp_path):
    import json
    from collections import Counter
    from types import SimpleNamespace

    def write(name, rows):
        p = tmp_path / name
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return p

    fi_rows = ([{"id": f"a{i}", "label": "NA", "mt": False, "url": "http://big.fi/x"} for i in range(40)]
               + [{"id": f"b{i}", "label": "NA", "mt": False, "url": f"http://s{i}.fi/"} for i in range(6)]
               + [{"id": f"c{i}", "label": "NA", "mt": True, "url": f"http://m{i}.fi/"} for i in range(5)]
               + [{"id": f"d{i}", "label": "MT", "mt": True, "url": f"http://t{i}.fi/"} for i in range(3)]
               + [{"id": f"e{i}", "label": "LY", "mt": False, "url": "http://l.fi/"} for i in range(5)])
    fi = write("fi", fi_rows)
    en = write("en", [{"id": f"x{i}", "label": "NA", "mt": False, "url": f"http://e{i}.com/"} for i in range(20)])
    out = tmp_path / "out"
    args = SimpleNamespace(fi=[fi], en=[en], per_class=10, seed=1, exclude=["LY"], out=str(out),
                           host_cap=3, no_mt_hybrids=True)
    ts.cmd_draw(args)
    rows = [json.loads(l) for l in open(out / "fi.jsonl")]
    na = [r for r in rows if r["label"] == "NA"]
    assert len(na) == 9                                     # 3 from big.fi + 6 singletons
    assert Counter(r["url"] for r in na)["http://big.fi/x"] == 3
    assert not any(r["mt"] for r in na)                     # MT + NA dropped
    assert sum(r["label"] == "MT" for r in rows) == 3       # MT-only kept
    assert not any(r["label"] == "LY" for r in rows)
    first = (out / "fi.jsonl").read_text()
    ts.cmd_draw(args)
    assert (out / "fi.jsonl").read_text() == first
