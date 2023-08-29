import abc

class Model(abc.ABC):

    def __init__(self, dimension):
        """Initialize a model."""
        super().__init__()
        self.dimension = dimension

    @abc.abstractmethod
    def minimize(self, cc=None):
        """Minimize objective cc."""
        pass

    def solve(self, cc=None):  # Linear Optimization Oracle
        return self.minimize(cc)