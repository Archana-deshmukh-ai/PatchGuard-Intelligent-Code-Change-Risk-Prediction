"""
Custom exception hierarchy for the PatchGuard Risk Prediction Engine.
"""

class PatchGuardError(Exception):
    """Base exception for all PatchGuard errors."""
    pass

class RepositoryError(PatchGuardError):
    """Base exception for repository-related issues."""
    pass

class RepositoryNotFoundError(RepositoryError):
    """Raised when the specified repository directory does not exist."""
    pass

class InvalidGitRepositoryError(RepositoryError):
    """Raised when the specified directory is not a valid Git repository."""
    pass

class CommitError(PatchGuardError):
    """Base exception for commit-related issues."""
    pass

class CommitNotFoundError(CommitError):
    """Raised when a commit hash or ref cannot be found."""
    pass

class UnsupportedCommitTypeError(CommitError):
    """Raised when trying to predict on an unsupported commit type (e.g. merge commits)."""
    pass

class EmptyCommitError(CommitError):
    """Raised when a commit contains no file modifications to analyze."""
    pass

class FeatureExtractionError(PatchGuardError):
    """Base exception for feature extraction failures."""
    pass

class GitExtractionError(FeatureExtractionError):
    """Raised when git diff or log parsing fails during extraction."""
    pass

class ModelError(PatchGuardError):
    """Base exception for model artifact or inference issues."""
    pass

class ModelArtifactNotFoundError(ModelError):
    """Raised when the trained model artifact pickle file is missing."""
    pass

class ScalerArtifactNotFoundError(ModelError):
    """Raised when the fitted scaler artifact pickle file is missing."""
    pass

class CorruptedArtifactError(ModelError):
    """Raised when an artifact pickle file cannot be deserialized."""
    pass

class IncompatibleSchemaError(ModelError):
    """Raised when feature names or column order do not match the expected canonical schema."""
    pass

class InvalidFeatureValueError(ModelError):
    """Raised when feature values contain NaNs, infinite values, or bad types."""
    pass
