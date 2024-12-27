
class AtomicNumberTable(list):

    def __init__(self, zs: list):
        super().__init__(sorted(list(zs)))

    def z_to_index(self, atomic_number: int) -> int:
        try:
            return self.index(atomic_number)
        except ValueError:
            raise ValueError(f"Observed atom type {atomic_number} that is not listed in the atomic numbers table.")
