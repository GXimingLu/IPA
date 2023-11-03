import pickle
from typing import List, Optional, Set


class Literal:
    def __init__(self, tokens: List[int]):
        self.tokens = tokens
        self.pointer = -1
        self.satisfy = False

    def advance(self, word_id: int):
        # token matches the next token in constraint
        if word_id == self.tokens[self.pointer + 1]:
            self.pointer += 1
        else:
            self.pointer = -1

        if self.pointer == len(self.tokens) - 1:
            self.satisfy = True


class Clause:
    def __init__(self, idx: int, phrases: List[List[int]]):
        self.idx = idx
        self.literals = [Literal(p) for p in phrases]
        self.satisfy = False

    def advance(self, word_id: int):
        for literal in self.literals:
            literal.advance(word_id)
            if literal.satisfy:
                self.satisfy = True

    def __str__(self):
        return f'clause(id={self.idx}, phrases={[l.tokens for l in self.literals]}, satisfy={self.satisfy})'


class ConstrainedHypothesis:

    def __init__(self,
                 constraint_list: List[List[List[int]]],
                 eos_tokens: List[int]) -> None:
        self.clauses = []
        for idx, clause in enumerate(constraint_list):
            self.clauses.append(Clause(idx=idx, phrases=clause))
        self.eos_tokens = eos_tokens

    def __len__(self) -> int:
        """
        :return: The number of constraints.
        """
        return len(self.clauses)

    def __str__(self) -> str:
        return '\n'.join([str(c) for c in self.clauses])

    def num_met(self) -> int:
        """
        :return: the number of constraints that have been met.
        """
        return sum([int(c.satisfy) for c in self.clauses])

    def advance(self, word_id: int) -> 'ConstrainedHypothesis':
        obj = pickle.loads(pickle.dumps(self))

        for clause in obj.clauses:
            if clause.satisfy:
                continue
            clause.advance(word_id)

        return obj

    def avoid(self) -> Set[int]:
        """
        :return: the tokens to avoid for next generation
        """
        allowed_token, avoid_token = set(), set()
        unsatisfied_clauses = [c for c in self.clauses if not c.satisfy]
        sorted_clauses = sorted(unsatisfied_clauses, key=lambda x: x.idx)

        for j, clause in enumerate(sorted_clauses):
            assert not clause.satisfy
            for literal in clause.literals:
                assert literal.pointer < len(literal.tokens) - 1 and not literal.satisfy
                tokens = {literal.tokens[literal.pointer + 1], literal.tokens[0]}
                if j == 0:
                    allowed_token.update(tokens)
                else:
                    avoid_token.update(tokens)

        negative_token = {t for t in avoid_token if t not in allowed_token}

        if self.eos_tokens is not None and not all(c.satisfy for c in self.clauses):
            negative_token.update(self.eos_tokens)
        return negative_token


def init_batch(raw_constraints: List[List[List[List[int]]]],
               eos_tokens: List[int]) -> List[Optional[ConstrainedHypothesis]]:
    """
    :param raw_constraints: The list of clause constraints.
    :param beam_size: The beam size.
    :param eos_id: The target-language vocabulary ID of the EOS symbol.
    :param ordered: Whether enforce constraints to be satisfied in given order
    :return: A list of ConstrainedHypothesis objects (shape: (batch_size * beam_size,)).
    """
    constraints_list = [None] * len(raw_constraints)  # type: List[Optional[ConstrainedHypothesis]]
    for i, raw_list in enumerate(raw_constraints):
        constraints_list[i] = ConstrainedHypothesis(raw_list, eos_tokens)
    return constraints_list


