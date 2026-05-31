"""Hierarchical JSON configuration loader with dot-notation access."""

import json
from typing import Any


class Config:
    """Hierarchical JSON configuration loader with dot-notation access.

    Nested dicts are recursively wrapped as Config instances, enabling
    attribute-style access at any depth (e.g. config.training.early_stopping.patience).

    Args:
        config_path: Path to the JSON configuration file.
    """

    def __init__(self, config_path: str = "config.json") -> None:
        with open(config_path) as f:
            self._set_attributes(json.load(f))

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"Config has no attribute '{name}'. " f"Check that the key exists in config.json."
        )

    def _set_attributes(self, config_data: dict[str, Any]) -> None:
        for key, value in config_data.items():
            setattr(self, key, Config.from_dict(value) if isinstance(value, dict) else value)

    def get_init_kwargs(self, key: str) -> dict[str, Any]:
        """Return a config section as a plain dict for direct **kwargs unpacking.

        Useful for passing config sections directly into Keras callbacks or
        other constructors without manually extracting each field.

        Args:
            key: Dot-separated path to a dict-valued config section,
                e.g. "training.early_stopping".

        Returns:
            The resolved section as a plain Python dict.

        Raises:
            ValueError: If any part of the path is missing or the resolved
                value is a scalar rather than a dict section.

        Example::

            EarlyStopping(**config.get_init_kwargs("training.early_stopping"))
        """
        node: Any = self
        for part in key.split("."):
            try:
                node = getattr(node, part)
            except AttributeError:
                raise ValueError(f"Config key not found: '{key}' (failed at '{part}').")
        if not isinstance(node, Config):
            raise ValueError(
                f"Config key '{key}' resolves to a scalar value, not a dict section. "
                f"Use config.{key.replace('.', '.')} to access it directly."
            )
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in vars(node).items()}

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert this Config back to a plain dict.

        Returns:
            A plain Python dict representation of this config section.
        """
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in vars(self).items()}

    @classmethod
    def from_dict(cls, data_dict: dict[str, Any]) -> "Config":
        """Create a Config instance from a plain dict without reading a file.

        Args:
            data_dict: A dict to wrap with dot-notation access.

        Returns:
            A Config instance wrapping the provided dict.
        """
        instance = cls.__new__(cls)
        instance._set_attributes(data_dict)
        return instance
