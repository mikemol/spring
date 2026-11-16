import glob
from collections.abc import Iterable
from pathlib import Path

from penman import Graph

from spring_amr.penman import load as pm_load


def read_raw_amr_data(
    paths: Iterable[str | Path] | Path,
    use_recategorization: bool = False,
    dereify: bool = True,
    remove_wiki: bool = False,
) -> list[Graph]:
    assert paths

    # A lone Path is the only non-iterable input; wrap it as the original did.
    if isinstance(paths, Path):
        paths = [paths]

    graphs: list[Graph] = []
    for path_ in paths:
        for found in glob.glob(str(path_)):
            path = Path(found)
            graphs.extend(pm_load(path, dereify=dereify, remove_wiki=remove_wiki))

    assert graphs

    if use_recategorization:
        for g in graphs:
            metadata = g.metadata
            metadata["snt_orig"] = metadata["snt"]
            tokens = eval(metadata["tokens"])
            metadata["snt"] = " ".join(
                [
                    t
                    for t in tokens
                    if not (t.startswith(("-L", "-R")) and t.endswith("-"))
                ]
            )

    return graphs
