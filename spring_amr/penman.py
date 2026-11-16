from penman import Graph, Triple
from penman import encode as encode_
from penman import load as load_
from penman import loads as loads_
from penman.codec import FileOrFilename
from penman.model import Model
from penman.models import amr
from penman.models.noop import NoOpModel

op_model = Model()
noop_model = NoOpModel()
amr_model = amr.model
DEFAULT = op_model


def _get_model(dereify: bool | None) -> Model:
    if dereify is None:
        return DEFAULT
    if dereify:
        return op_model
    return noop_model


def _remove_wiki(graph: Graph) -> Graph:
    metadata = graph.metadata
    triples: list[Triple] = []
    for t in graph.triples:
        v1, rel, _v2 = t
        if rel == ":wiki":
            t = Triple(v1, rel, "+")
        triples.append(t)
    graph = Graph(triples)
    graph.metadata = metadata
    return graph


def load(
    source: FileOrFilename, dereify: bool | None = None, remove_wiki: bool = False
) -> list[Graph]:
    model = _get_model(dereify)
    out = load_(source=source, model=model)
    if remove_wiki:
        for i in range(len(out)):
            out[i] = _remove_wiki(out[i])
    return out


def loads(
    string: str, dereify: bool | None = None, remove_wiki: bool = False
) -> list[Graph]:
    model = _get_model(dereify)
    out = loads_(string=string, model=model)
    if remove_wiki:
        for i in range(len(out)):
            out[i] = _remove_wiki(out[i])
    return out


def encode(
    g: Graph, top: str | None = None, indent: int = -1, compact: bool = False
) -> str:
    model = amr_model
    return encode_(g=g, top=top, indent=indent, compact=compact, model=model)
