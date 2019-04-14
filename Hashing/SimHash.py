import numpy as np


class SimHash(object):

    def __init__(self, D, k, preprocessor=None):
        """
            Initialize SimHash function object
        :param D: The dimension D of the raw input, or the dimension of the preprocessed input (the dimension of the
                    output of the given preprocessor)
        :param k: The dimension (length) of the binary hash code
        :param preprocessor: Optional argument. A preprocessor to preprocess the given raw input
        """
        self._D = D
        self._k = k
        self._preprocessor = preprocessor

        # Initialize hashing matrix
        self.A = np.random.randn(self._k, self._D)

    def hash(self, s, base_ten=True):
        """
            SimHash function
        :param s: Raw input
        :param base_ten: Whether to return a base-10 number of return a vector of binary values. Default to true
        :return: Binary hashed code. Return a base-10 number if base_ten is True
        """
        # Preprocess input if preprocessor is given
        if self._preprocessor is not None:
            x = self._preprocessor(s)
            assert x.shape == (self._D,), "The preprocessor must return a vector of shape (D,)"
        else:
            x = s
        # Dot product
        y = np.dot(self.A, x)
        # Round to integer
        y = (y > 0).astype(np.int0)
        # Return output
        out = np.asarray([y[-(i + 1)] * (2 ** i) for i in range(len(y))])
        if base_ten:
            return np.sum(out)
        else:
            return y
