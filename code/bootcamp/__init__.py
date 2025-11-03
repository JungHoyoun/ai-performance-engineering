"""Bootcamp package powering the Inference Empire game prototype."""

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """Return the package version if installed, otherwise '0.0.0-dev'."""
    try:
        return version("bootcamp")
    except PackageNotFoundError:
        return "0.0.0-dev"


__all__ = ["get_version"]
