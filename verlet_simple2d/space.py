from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from verlet_simple2d import DTYPE, shapes
from verlet_simple2d.handler import CollisionHandler, get_handler
from verlet_simple2d.helpers import fmt_asrt


class Space:
  def __init__(self, dt: float) -> None:
    self.kinetics: list[shapes.Body] = []
    self.statics: list[shapes.Border] = []
    self._gravity: NDArray[DTYPE] = np.array((0, -10), dtype=DTYPE)
    self.dt: DTYPE = DTYPE(dt)

    self.collision_handlers: list[CollisionHandler] = []

  class _Reverse:
    def __init__(self, space: Space) -> None: self.space = space
    def __enter__(self) -> None: self.rev()
    def __exit__(self, exc_type, exc_value, traceback) -> None: self.rev()
    def rev(self) -> None:
      for kin in self.space.kinetics:
        kin.prev_location = kin.location + (kin.location - kin.prev_location)

  def reverse(self) -> _Reverse:
    return Space._Reverse(self)
  
  @property
  def gravity(self) -> NDArray[DTYPE]:
    return self._gravity
  
  @gravity.setter
  def gravity(self, val) -> None:
    assert isinstance(val, (tuple, list, np.ndarray)), fmt_asrt('gravity', (tuple, list, np.ndarray))
    if isinstance(val, np.ndarray): self._gravity = val
    else: self._gravity = np.array(val, dtype=DTYPE)

  def add_collision_handler(self, X: shapes.Body | shapes.Border, Y: shapes.Body | shapes.Border) -> CollisionHandler:
    handler = get_handler(X, Y)
    self.collision_handlers.append(handler)
    return handler
  
  def get_collision_handler(self, X: shapes.Body | shapes.Border, Y: shapes.Body | shapes.Border) -> CollisionHandler | None:
    for handler in self.collision_handlers:
      if handler.check_types(X.collision_type, Y.collision_type):
        return handler
      if handler.check_types(type(Y), X.collision_type):
        return handler
      if handler.check_types(type(X), Y.collision_type):
        return handler
    return None

  def add_body(self, body: shapes.Body | shapes.Border) -> None:
    if body in self.kinetics or body in self.statics: return

    if isinstance(body, shapes.Body):
      for stat in self.statics:
        handler = self.get_collision_handler(body, stat)
        if handler is None:
          self.add_collision_handler(body, stat)
    
    for kin in self.kinetics:
      handler = self.get_collision_handler(body, kin)
      if handler is None:
        self.add_collision_handler(body, kin)

    if isinstance(body, shapes.Body):
      self.kinetics.append(body)
    elif isinstance(body, shapes.Border):
      self.statics.append(body)

  def remove_body(self, body: shapes.Body | shapes.Border) -> None:
    if body not in self.kinetics and body not in self.statics: return

    if isinstance(body, shapes.Body):
      self.kinetics.remove(body)
    elif isinstance(body, shapes.Border):
      self.statics.remove(body)

  def step(self) -> None:
    for kin in self.kinetics:
      kin.acceleration = self.gravity
      vel = kin.location - kin.prev_location
      kin.prev_location = kin.location
      kin.location = kin.location + vel + kin.acceleration * (self.dt*self.dt)

    for i, kin in enumerate(self.kinetics):
      for o_kin in self.kinetics[i+1:]:
        handler = self.get_collision_handler(kin, o_kin)
        if handler is None: raise ValueError('Unknown Handler')

        if handler.check(kin, o_kin):
          handler.resolve(kin, o_kin)
      
      for stat in self.statics:
        handler = self.get_collision_handler(kin, stat)
        if handler is None: raise ValueError('Unkown Handler')

        if handler.check(kin, stat):
          handler.resolve(kin, stat)
