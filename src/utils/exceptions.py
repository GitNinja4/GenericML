"""Custom exception types for the ML workflow."""


class DatasetValidationError(ValueError):
    """Raised when a dataset fails validation."""


class MissingTargetError(ValueError):
    """Raised when no target column is selected."""


class UnsupportedProblemTypeError(ValueError):
    """Raised when a problem type is unsupported."""


class ModelTrainingError(RuntimeError):
    """Raised when training fails."""


class PredictionSchemaError(ValueError):
    """Raised when new prediction data does not match the training schema."""
