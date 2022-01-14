from enum import Enum

class STFTBackend(str, Enum):
    """ Enumeration of supported STFT backend. """

    AUTO: str = "auto"
    TENSORFLOW: str = "tensorflow"
    LIBROSA: str = "librosa"

    @classmethod
    def resolve(cls: type, backend: str) -> str:
        # NOTE: import is resolved here to avoid performance issues on command
        #       evaluation.
        # pyright: reportMissingImports=false
        # pylint: disable=import-error
        import tensorflow as tf

        if backend not in cls.__members__.values():
            raise ValueError(f"Unsupported backend {backend}")
        if backend == cls.AUTO:
            return cls.TENSORFLOW
        return backend