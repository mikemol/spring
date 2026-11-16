from collections import defaultdict
from collections.abc import Iterable, Iterator

from penman import Graph, Triple
from penman.graph import Constant

EntityRow = tuple[
    int,
    dict[str, tuple[int, Constant]],
    dict[Constant, str],
    defaultdict[str, list[Triple]],
]


def read_entities(
    sentences: Iterable[str], graphs: Iterable[Graph], just_tagged: bool = True
) -> Iterator[EntityRow]:
    for i, (_s, g) in enumerate(zip(sentences, graphs)):
        with_wikis: dict[str, tuple[int, Constant]] = {}
        name_to_entity: dict[Constant, str] = {}
        name_to_ops: defaultdict[str, list[Triple]] = defaultdict(list)

        for nt, t in enumerate(g.triples):
            n1, rel, n2 = t

            if n2 == "-" and just_tagged:
                continue

            if rel == ":wiki":
                with_wikis[n1] = (nt, n2)

        for t in g.triples:
            n1, rel, n2 = t
            if (n1 in with_wikis) and (rel == ":name"):
                name_to_entity[n2] = n1

        for nt, t in enumerate(g.triples):
            n1, rel, n2 = t
            if (n1 in name_to_entity) and rel.startswith(":op"):
                name_to_ops[n1].append(t)

        yield (i, with_wikis, name_to_entity, name_to_ops)
