"""Component registry: a yaml node like ``{name: adamw, ...}`` -> an object.

A minimal counterpart of llmfoundry.registry: every swappable piece of the
pipeline (model, dataset, alpha/beta schedule, optimizer, lr schedule, SDE,
simulator, logger, task) is registered under a name, and the config selects it
through the ``name`` field.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, TypeVar

T = TypeVar("T")


class Registry:
    """A ``name -> factory`` mapping."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, *names: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            for n in names:
                if n in self._items and self._items[n] is not obj:
                    raise ValueError(f"{self.name}: name {n!r} is already taken by {self._items[n]!r}")
                self._items[n] = obj  # type: ignore[assignment]
            return obj

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._items:
            raise KeyError(
                f"{self.name}: unknown name {name!r}. Available: {sorted(self._items)}"
            )
        return self._items[name]

    def keys(self) -> Iterable[str]:
        return sorted(self._items)

    def build(self, cfg: Any, **extra: Any) -> Any:
        """Build an object from a config node.

        ``cfg`` is either a name string or a mapping with a required ``name`` key;
        the remaining keys are passed to the constructor as kwargs (plus ``extra``).
        """
        if isinstance(cfg, str):
            return self.get(cfg)(**extra)

        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg, dict):
            raise TypeError(f"{self.name}: expected a dict or str, got {type(cfg)}")
        kwargs = dict(cfg)
        name = kwargs.pop("name", None)
        if name is None:
            raise KeyError(f"{self.name}: config node is missing the required 'name' field: {cfg}")
        kwargs.update(extra)
        return self.get(name)(**kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, items={sorted(self._items)})"


models = Registry("models", "denoisers and AR models")
datasets = Registry("datasets", "torch Datasets")
tasks = Registry("tasks", "what the loss is computed on: diffusion | ar")
alphas = Registry("alphas", "alpha_t schedules")
betas = Registry("betas", "beta_t schedules")
optimizers = Registry("optimizers")
schedulers = Registry("schedulers", "learning rate schedules")
sdes = Registry("sdes", "SDE/ODE used for sampling")
simulators = Registry("simulators", "SDE/ODE integrators")
loggers = Registry("loggers", "where metrics are written")

__all__ = [
    "Registry",
    "models",
    "datasets",
    "tasks",
    "alphas",
    "betas",
    "optimizers",
    "schedulers",
    "sdes",
    "simulators",
    "loggers",
]
