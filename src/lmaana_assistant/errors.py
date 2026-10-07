"""Errors that retain their operational meaning at the API boundary."""


class DependencyUnavailable(RuntimeError):
    pass


class InferenceTimeout(RuntimeError):
    pass


class InvalidGeneration(ValueError):
    pass


class UnsupportedAudio(ValueError):
    pass
