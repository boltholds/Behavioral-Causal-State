"""Public transition contract consumed by exact inference."""
from typing import Protocol
from .simulator import Action, Variable, DeviceState, NoiseLaw

class Dynamics(Protocol):
    @property
    def noise(self) -> NoiseLaw: ...
    def reset(self, c: int, nm: int, ny: int) -> DeviceState: ...
    def advance(self, previous: DeviceState, action: Action,
                clamps: tuple[tuple[Variable,int], ...], nm: int, ny: int) -> DeviceState: ...
