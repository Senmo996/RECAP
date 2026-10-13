"""Request-local suffix automaton over prompt and committed token history."""

from dataclasses import dataclass


@dataclass
class _State:
    next: dict
    link: int
    length: int
    min_endpos: int


class RecapSuffixIndex:
    def __init__(self):
        self.states = [_State({}, -1, 0, 0)]
        self.tokens = [-1]
        self.last = self.max_length = self.cur_index = self.cur_length = 0

    def _transfer(self, token):
        index, length = self.cur_index, self.cur_length
        while index != 0 and token not in self.states[index].next:
            index = self.states[index].link
            length = self.states[index].length
        if token in self.states[index].next:
            index, length = self.states[index].next[token], length + 1
        else:
            index = length = 0
        self.cur_index, self.cur_length = index, length

    def update(self, tokens):
        tokens = [int(token) for token in tokens]
        for token in tokens:
            # Match against history before adding this committed token.
            self._transfer(token)
            self.max_length += 1
            cur = len(self.states)
            self.states.append(_State({}, -1, self.max_length, self.max_length))
            p = self.last
            while p != -1 and token not in self.states[p].next:
                self.states[p].next[token] = cur
                p = self.states[p].link
            if p == -1:
                self.states[cur].link = 0
            else:
                q = self.states[p].next[token]
                if self.states[p].length + 1 == self.states[q].length:
                    self.states[cur].link = q
                else:
                    source = self.states[q]
                    clone = len(self.states)
                    self.states.append(_State(source.next.copy(), source.link,
                                              self.states[p].length + 1, source.min_endpos))
                    while p != -1 and self.states[p].next.get(token) == q:
                        self.states[p].next[token] = clone
                        p = self.states[p].link
                    self.states[q].link = self.states[cur].link = clone
            self.last = cur
        self.tokens.extend(tokens)

    def continuation(self, max_tokens=4):
        """The current state already contains the root; do not consume it twice."""
        count = min(max_tokens, self.cur_length * 4)
        if count <= 0 or self.cur_index == 0:
            return []
        end = self.states[self.cur_index].min_endpos
        return self.tokens[end + 1:end + 1 + count]
