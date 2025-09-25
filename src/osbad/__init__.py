import importlib.metadata

try:
    # Exposing the current version in osbad
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    # Fallback for development mode
    __version__ = "0.0.0"
