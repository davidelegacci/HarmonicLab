from itertools import combinations
import networkx as nx


# https://iq.opengenus.org/algorithm-to-find-cliques-of-a-given-size-k/

def k_cliques(graph):
    # 2-cliques
    cliques = [{i, j} for i, j in graph.edges() if i != j]
    k = 2

    while cliques:
        # result
        yield k, cliques

        # merge k-cliques into (k+1)-cliques
        cliques_1 = set()
        for u, v in combinations(cliques, 2):
            w = u ^ v
            if len(w) == 2 and graph.has_edge(*w):
                cliques_1.add(tuple(u | w))

        # remove duplicates
        cliques = list(map(set, cliques_1))
        k += 1


def print_cliques(graph, size_k):
    for k, cliques in k_cliques(graph):
        if k == size_k:
            print('%d-cliques = %d, %s.' % (k, len(cliques), cliques))


graph_dict = {(1, 1): [(1, 2), (1, 3), (1, 4), (2, 1)], (1, 2): [(1, 1), (1, 3), (1, 4), (2, 2)], (1, 3): [(1, 1), (1, 2), (1, 4), (2, 3)], (1, 4): [(1, 1), (1, 2), (1, 3), (2, 4)], (2, 1): [(1, 1), (2, 2), (2, 3), (2, 4)], (2, 2): [(1, 2), (2, 1), (2, 3), (2, 4)], (2, 3): [(1, 3), (2, 1), (2, 2), (2, 4)], (2, 4): [(1, 4), (2, 1), (2, 2), (2, 3)]}

graph = nx.Graph(graph_dict)
print(graph)


size_k = 3
print_cliques(graph, size_k)
