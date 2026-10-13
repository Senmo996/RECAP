"""Persistent token memory U and request-local contextual memories C/G."""

import math

import torch


def recap_fuse(rows, limit, confidence=True):
    """Fuse available rows in decreasing specificity order (G, C, U).

    Priors are assigned by rank among available sources, matching the paper
    and original C++ implementation. Ties retain first-occurrence order.
    """
    if not rows:
        return [], []
    if len(rows) == 1:
        return rows[0]
    scores = {}
    for prior, (tokens, probabilities) in zip((0.70, 0.20, 0.10), rows):
        weight = prior
        if confidence:
            total = sum(probabilities)
            values = [p / total for p in probabilities]
            weight *= math.exp(sum(p * math.log(p) for p in values if p > 0))
        for token, probability in zip(tokens, probabilities):
            scores[token] = scores.get(token, 0.0) + weight * probability
    selected = sorted(scores, key=lambda token: -scores[token])[:limit]
    total = sum(scores[token] for token in selected) or 1.0
    return selected, [scores[token] / total for token in selected]


class RecapMemory:
    def __init__(self, top_k=8):
        self.top_k = top_k
        self.rows = [{}, {}, {}]
        self.scores = [{}, {}, {}]

    def reset_request(self):
        for order in (1, 2):
            self.rows[order].clear()
            self.scores[order].clear()

    def reset(self):
        for rows, scores in zip(self.rows, self.scores):
            rows.clear()
            scores.clear()

    def update(self, logits, tokens, previous=None, previous_previous=None):
        """Write all verified rows, including nodes outside the accepted path.

        The first occurrence of each key wins within one target pass. A later
        pass overwrites the row. Prefill provides tokens only, seeding U.
        """
        top = logits.topk(min(self.top_k, logits.shape[-1]), dim=-1)
        probabilities = torch.softmax(top.values.float(), dim=-1)
        candidates, probabilities = top.indices.tolist(), probabilities.tolist()
        seen = [set(), set(), set()]
        for index, token in enumerate(tokens):
            keys = [int(token), None, None]
            if previous is not None and previous[index] is not None:
                keys[1] = (int(previous[index]), int(token))
                if previous_previous is not None and previous_previous[index] is not None:
                    keys[2] = (int(previous_previous[index]), *keys[1])
            for order, key in enumerate(keys):
                if key is not None and key not in seen[order]:
                    seen[order].add(key)
                    self.rows[order][key] = candidates[index]
                    self.scores[order][key] = probabilities[index]

    def seed_prompt(self, logits, tokens):
        positions = list(dict.fromkeys(tokens))
        first = {}
        for index, token in enumerate(tokens):
            first.setdefault(token, index)
        indices = torch.tensor([first[t] for t in positions], device=logits.device)
        self.update(logits.index_select(0, indices), positions)

    def lookup(self, token, previous, previous_previous, width, confidence):
        keys = [token, (previous, token), (previous_previous, previous, token)]
        found = []
        for order in (2, 1, 0):
            key = keys[order]
            if key in self.rows[order]:
                found.append((self.rows[order][key][:width], self.scores[order][key][:width]))
        return recap_fuse(found, width, confidence)
