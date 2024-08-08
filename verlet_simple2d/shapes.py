from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from verlet_simple2d import DTYPE
from verlet_simple2d.helpers import fmt_asrt


class Body:
  def __init__(self, x: float, y:float) -> None:
    self._location: NDArray[DTYPE] = np.array((x, y), dtype=DTYPE)
    self._prev_location: NDArray[DTYPE] = np.array((x, y), dtype=DTYPE)
    self._acceleration: NDArray[DTYPE] = np.array((0.0, 0.0), dtype=DTYPE)

    self._collision_type: type[Body] | int

    self._collisions: int = 0
    self.mass: DTYPE = DTYPE(1)
    # self._mass: DTYPE = DTYPE(1)
    # self._elasticity: DTYPE = DTYPE(1)
  
  @property
  def x(self) -> DTYPE:
    return self._location[0]

  @x.setter
  def x(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), fmt_asrt('x', (int, float, DTYPE)) # type: ignore[misc]
    self._location[0] = DTYPE(val)

  @property
  def y(self) -> DTYPE:
    return self._location[1]

  @y.setter
  def y(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), fmt_asrt('y', (int, float, DTYPE)) # type: ignore[misc]
    self._location[1] = DTYPE(val)

  @property
  def location(self) -> NDArray[DTYPE]:
    return self._location
  
  @location.setter
  def location(self, val) -> None:
    assert isinstance(val, (tuple, list, np.ndarray)), fmt_asrt('location', (tuple, list, np.ndarray)) 
    if isinstance(val, np.ndarray):
      self._location = val
    else:
      self._location = np.array(val, dtype=DTYPE)
    #vel = self.velocity
    #self.prev_location = self.location + (-vel)

  @property
  def acceleration(self) -> NDArray[DTYPE]:
    return self._acceleration

  @acceleration.setter
  def acceleration(self, val) -> None:
    assert isinstance(val, (tuple, list, np.ndarray)), fmt_asrt('acceleration', (tuple, list, np.ndarray))
    if isinstance(val, np.ndarray):
      self._acceleration = val
    else:
      self._acceleration = np.array(val, dtype=DTYPE)

  @property
  def prev_x(self) -> DTYPE:
    return self._prev_location[0]

  @prev_x.setter
  def prev_x(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), fmt_asrt('prev_x', (int, float, DTYPE)) # type: ignore[misc]
    self._prev_location[0] = DTYPE(val)

  @property
  def prev_y(self) -> DTYPE:
    return self._prev_location[1]

  @prev_y.setter
  def prev_y(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), fmt_asrt('prev_y', (int, float, DTYPE)) # type: ignore[misc]
    self._prev_location[1] = DTYPE(val)

  @property
  def prev_location(self) -> NDArray[DTYPE]:
    return self._prev_location
  
  @prev_location.setter
  def prev_location(self, val) -> None:
    assert isinstance(val, (tuple, list, np.ndarray)), fmt_asrt('prev_location', (tuple, list, np.ndarray)) 
    if isinstance(val, np.ndarray):
      self._prev_location = val
    else:
      self._prev_location = np.array(val, dtype=DTYPE)
  
  @property
  def collision_type(self) -> type[Body] | int:
    return self._collision_type

  @collision_type.setter
  def collision_type(self, val) -> None:
    assert isinstance(val, int), fmt_asrt('collision_type', int)
    self._collision_type = val
  
  """
  @property
  def velocity(self) -> NDArray[DTYPE]:
    return self.location - self.prev_location

  @velocity.setter
  def velocity(self, val) -> None:
    assert isinstance(val, (tuple, list, np.ndarray)), fmt_asrt('velocity', (tuple, list, np.ndarray))
    if isinstance(val, np.ndarray): self.prev_location = self.location + (-val)
    else: self.prev_location = self.location + (-np.array(val, dtype=DTYPE))
  """

  def __repr__(self) -> str:
    return f'{self.__class__.__name__} (x={self.x:.2f}, y={self.y:.2f}, \'x={self.prev_x:.2f}, \'y={self.prev_y:.2f}, velocity=({self.location[0]-self.prev_location[0]:.2f}{self.location[1]-self.prev_location[1]:.2f}), collisiontype={(self.collision_type) if isinstance(self.collision_type, int) else self.collision_type.__name__})'

class Circle(Body):
  def __init__(self, x: float, y: float, r: float) -> None:
    super().__init__(x, y)
    self._radius: DTYPE = DTYPE(r)
    self._collision_type = Circle
  
  @property
  def radius(self) -> DTYPE:
    return self._radius

  @radius.setter
  def radius(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), fmt_asrt('radius', (int, float, DTYPE)) # type: ignore[misc]
    self._radius = DTYPE(val)

class Rectangle(Body):
  def __init__(self, x: float, y: float, width: float, height: float) -> None:
    super().__init__(x, y)
    self._dims: NDArray[DTYPE] = np.array((width, height), dtype=DTYPE)
    self._ang_vel: NDArray = np.array((0,0), dtype=DTYPE)
  
  @property
  def width(self) -> DTYPE:
    return self._dims[0]
  
  @width.setter
  def width(self, val) -> None:
    assert isinstance(val, (float, int, DTYPE)), fmt_asrt('width', (float, int, DTYPE)) # type: ignore[misc]
    self._dims[0] = val

  @property
  def height(self) -> DTYPE:
    return self._dims[1]

  @height.setter
  def height(self, val) -> None:
    assert isinstance(val, (float, int, DTYPE)), fmt_asrt('height', (float, int, DTYPE)) # type: ignore[misc]
    self._dims[1] = val

  @property
  def dims(self) -> NDArray[DTYPE]:
    return self._dims
  
  @dims.setter
  def dims(self, val) -> None:
    assert isinstance(val, (tuple, list, np.ndarray)), fmt_asrt('dims', (tuple, list, np.ndarray))
    if isinstance(val, np.ndarray): self._dims = val
    else: self._dims = np.array(val, dtype=DTYPE)

class Border:
  def __init__(self, x: float, y: float, line_width: float=1) -> None:
    self._location: NDArray[DTYPE] = np.array((x, y), dtype=DTYPE)
    self.line_width: DTYPE = DTYPE(line_width)
    self._collision_type: type[Border] | int
    self._collisions: int = 0

  @property
  def x(self) -> DTYPE:
    return self._location[0]

  @x.setter
  def x(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), fmt_asrt('x', (int, float, DTYPE)) # type: ignore[misc]
    self._location[0] = DTYPE(val)

  @property
  def y(self) -> DTYPE:
    return self._location[1]

  @y.setter
  def y(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), fmt_asrt('y', (float, int, DTYPE)) # type: ignore[misc]
    self._location[1] = DTYPE(val)
  
  @property
  def location(self) -> NDArray[DTYPE]:
    return self._location
  
  @location.setter
  def location(self, val) -> None:
    assert isinstance(val, (list, tuple, np.ndarray)), fmt_asrt('location', (list, tuple, np.ndarray))
    if isinstance(val, np.ndarray): self._location = val
    else: self._location = np.ndarray(val, dtype=DTYPE)

  @property
  def collision_type(self) -> type[Border] | int:
    return self._collision_type
  
  @collision_type.setter
  def collision_type(self, val) -> None:
    assert isinstance(val, int), fmt_asrt('collision_type', int)
    self._collision_type = val

class CircleBorder(Border):
  def __init__(self, x: float, y: float, r: float, line_width: float=1) -> None:
    super().__init__(x, y, line_width)
    self._radius: DTYPE = DTYPE(r)
    self._collision_type = CircleBorder
  
  @property
  def radius(self) -> DTYPE:
    return self._radius
  
  @radius.setter
  def radius(self, val) -> None:
    assert isinstance(val, (int, float, DTYPE)), fmt_asrt('radius', (float, int, DTYPE)) # type: ignore[misc]
    self._radius = DTYPE(val)

class RectangleBorder(Border):
  def __init__(self, x: float, y: float, width: float, height: float, line_width: float) -> None:
    super().__init__(x, y, line_width)
    self._dims: NDArray[DTYPE] = np.array((width, height), dtype=DTYPE)
    self._collision_type = RectangleBorder
  
  @property
  def width(self) -> DTYPE:
    return self._dims[0]
  
  @width.setter
  def width(self, val) -> None:
    assert isinstance(val, (float, int , DTYPE)), fmt_asrt('width', (float, int, DTYPE)) # type: ignore[misc]
    self._dims[0] = DTYPE(val)

  @property
  def height(self) -> DTYPE:
    return self._dims[1]
  
  @height.setter
  def height(self, val) -> None:
    assert isinstance(val, (float, int , DTYPE)), fmt_asrt('height', (float, int, DTYPE)) # type: ignore[misc]
    self._dims[1] = DTYPE(val)

  @property 
  def dims(self) -> NDArray[DTYPE]:
    return self._dims
  
  @dims.setter
  def dims(self, val) -> None:
    assert isinstance(val, (tuple, list, np.ndarray)), fmt_asrt('dims', (tuple, list, np.ndarray))
    if isinstance(val, np.ndarray):
      self._dims = val
    else:
      self._dims = np.ndarray(val, dtype=DTYPE)
