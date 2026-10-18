"""Custom exceptions, so the API can turn each kind of failure into a clear HTTP answer.

    ConfigurationError -> 503  (a setting is missing, e.g. no GOOGLE_API_KEY, no index ID)
    IngestionError     -> 422  (the file could not be read or indexed)
"""


class MeridianError(Exception):
    """Base class for every error this app raises on purpose."""


class ConfigurationError(MeridianError):
    """A required setting is missing or the selected APP_MODE cannot start."""


class IngestionError(MeridianError):
    """A document could not be loaded, chunked or indexed."""
