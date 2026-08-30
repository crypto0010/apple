"""Decorator-based model registry: name -> builder."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from torch import nn

_REGISTRY: dict[str, Callable[..., nn.Module]] = {}

F = TypeVar("F", bound=Callable[..., nn.Module])


def register(name: str) -> Callable[[F], F]:
    def deco(fn: F) -> F:
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        _REGISTRY[name] = fn
        return fn
    return deco


def build(name: str, **kwargs) -> nn.Module:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return _REGISTRY[name](**kwargs)


def all_names() -> list[str]:
    return sorted(_REGISTRY.keys())
