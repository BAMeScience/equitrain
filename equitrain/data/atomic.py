
import ast

from typing import Iterable

class AtomicNumberTable(list):

    def __init__(self, zs: list):
        super().__init__(sorted(list(zs)))


    @classmethod
    def from_zs(cls, zs : Iterable[int]):
        z_set = set()
        z_set.update(zs)
        return AtomicNumberTable(sorted(list(z_set)))


    @classmethod
    def from_str(cls, string : str):
        zs_list = ast.literal_eval(string)
        assert isinstance(zs_list, list)
        return AtomicNumberTable.from_zs(zs_list)


    def z_to_index(self, atomic_number: int) -> int:
        try:
            return self.index(atomic_number)
        except ValueError:
            raise ValueError(f"Observed atom type {atomic_number} that is not listed in the atomic numbers table.")
