"""Shared sklearn runtime configuration."""

from sklearn import set_config


def configure_sklearn_output() -> None:
    """Configure sklearn transformers to preserve pandas DataFrame outputs."""
    set_config(transform_output="pandas")
