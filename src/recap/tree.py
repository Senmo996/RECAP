"""Best-first transition composition, suffix insertion and packed topology."""

from dataclasses import dataclass
import heapq

import torch

HIT_MASSES = (0.68, 0.63, 0.50, 0.43, 0.40, 0.38)


@dataclass
class RecapTree:
    tokens: list
    parents: list
    depths: list
    previous: list
    previous_previous: list
    mask: torch.Tensor
    positions: torch.Tensor
    paths: list


def _python_nodes(memory, root, previous, previous_previous, config, blocked):
    selected = {(): (root, previous, previous_previous, frozenset())}
    pending = []
    distributions = {}

    def push(path, probability):
        depth = len(path) + 1
        if depth > config.depth:
            return
        token, prev, prevprev, edges = selected[path]
        key = (token, prev, prevprev)
        if key not in distributions:
            distributions[key] = memory.lookup(
                *key, config.top_k, config.fusion == "prior_confidence")
        candidates, probabilities = distributions[key]
        seen = set()
        for rank, (candidate, score) in enumerate(zip(candidates, probabilities), 1):
            edge = (token, candidate)
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate == blocked or edge in edges:
                continue
            score = probability * HIT_MASSES[min(depth - 1, len(HIT_MASSES) - 1)] * score
            heapq.heappush(pending, (-score, (*path, rank), candidate, path, edges | {edge}))

    push((), 1.0)
    while pending and len(selected) - 1 < config.node_budget:
        negative_score, path, token, parent, edges = heapq.heappop(pending)
        parent_token, prev, _, _ = selected[parent]
        selected[path] = (token, parent_token, prev, edges)
        push(path, -negative_score)
    ordered = sorted(path for path in selected if path)
    indices = {path: i for i, path in enumerate(ordered, 1)}
    return [(selected[path][0], indices.get(path[:-1], 0), len(path), path[-1]) for path in ordered]


def build_recap_tree(memory, root, previous, previous_previous, config,
                     blocked=None, suffix=(), device="cpu", topology_cache=None):
    if config.backend == "cpp":
        from .fast_tree import load_recap_extension
        nodes = load_recap_extension().build_recap_nodes(
            root_token=root, root_previous_token=previous if previous is not None else -1,
            root_previous_previous_token=previous_previous if previous_previous is not None else -1,
            width=config.top_k, branch_width=config.top_k, depth=config.depth,
            node_budget=config.node_budget, blocked_token_id=blocked if blocked is not None else -1,
            unigram_rows=memory.rows[0], unigram_scores=memory.scores[0],
            context_rows=memory.rows[1], context_scores=memory.scores[1],
            trigram_rows=memory.rows[2], trigram_scores=memory.scores[2],
            hit_masses=HIT_MASSES, confidence_weighting=config.fusion == "prior_confidence",
        )
    else:
        nodes = _python_nodes(memory, root, previous, previous_previous, config, blocked)
    # Insert only missing edges of the retrieved continuation (up to +4 nodes).
    children = {(p, t): i for i, (t, p, _, _) in enumerate(nodes, 1)}
    parent, parent_token, edges = 0, root, set()
    for depth, token in enumerate(suffix[:config.suffix_tokens], 1):
        edge = (parent_token, token)
        if token == blocked or edge in edges or depth > config.depth:
            break
        child = children.get((parent, token))
        if child is None:
            child = len(nodes) + 1
            nodes.append((token, parent, depth, 0))
            children[(parent, token)] = child
        edges.add(edge)
        parent, parent_token = child, token
    tokens = [root] + [n[0] for n in nodes]
    parents = [0] + [n[1] for n in nodes]
    depths = [0] + [n[2] for n in nodes]
    prev = [previous] + [tokens[p] for p in parents[1:]]
    prevprev = [previous_previous] + [prev[p] for p in parents[1:]]
    key = (str(device), tuple(parents), tuple(depths))
    cached = topology_cache.get(key) if topology_cache is not None else None
    if cached is None:
        mask = [[False] * len(tokens) for _ in tokens]
        mask[0][0] = True
        paths = []
        for i in range(1, len(tokens)):
            path, node = [], i
            mask[i][0] = True
            while node:
                mask[i][node] = True
                path.append(node)
                node = parents[node]
            paths.append(path[::-1])
        cached = (torch.tensor(mask, dtype=torch.bool, device=device)[None, None],
                  torch.tensor(depths, dtype=torch.long, device=device), paths)
        if topology_cache is not None:
            if len(topology_cache) >= 256:
                topology_cache.clear()
            topology_cache[key] = cached
    return RecapTree(tokens, parents, depths, prev, prevprev, *cached)


def verify_recap_tree(predictions, tree, remaining):
    """Return the longest greedy-matching path plus one target correction."""
    best = []
    for path in tree.paths:
        accepted, parent = [], 0
        for node in path:
            if predictions[parent] != tree.tokens[node]:
                break
            accepted.append(node)
            parent = node
        if len(accepted) > len(best):
            best = accepted
    best = best[:max(remaining - 1, 0)]
    return best, [tree.tokens[n] for n in best] + [predictions[best[-1] if best else 0]]
