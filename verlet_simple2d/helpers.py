from typing import Any

import numpy as np
from numpy.typing import NDArray

from verlet_simple2d import DTYPE


def fmt_asrt(name: str, types: Any | tuple[Any, ...]) -> str:
  if isinstance(types, tuple):
    for t in types: assert isinstance(t, type), 'tuple should only contain types'
    return f'{name} should be of types {[e.__name__ for e in types]}'
  return f'{name} should be of type {types}'

def closest_point(point: NDArray[DTYPE], points: list[NDArray[DTYPE]]) -> NDArray[DTYPE]:
  if len(points) == 0: raise ValueError('points must be of length >0')
  if len(points) == 1: return points[0]
  d1 = np.linalg.norm(point - points[0])
  d2 = np.linalg.norm(point - points[1])
  if d1 < d2: return points[0]
  return points[1]

def line_line_intersection(
  p1: NDArray[DTYPE], v1: NDArray[DTYPE],
  p2: NDArray[DTYPE], v2: NDArray[DTYPE]
) -> NDArray[DTYPE]:
  x1, y1 = p1
  x2, y2 = p2
  vx1, vy1 = v1
  vx2, vy2 = v2

  det = vx1 * vy2 - vx2 * vy1
  if det == 0: raise RuntimeError("BUG")
  t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / det
  return np.array((x1 + t * vx1, y1 + t * vy1))

def line_circle_intersection(
    circle_center: NDArray[DTYPE], circle_radius: DTYPE,
    line_point: NDArray[DTYPE], line_vector: NDArray[DTYPE],
) -> list[NDArray[DTYPE]]:
  a = line_vector[0]**2 + line_vector[1]**2
  b = 2 * (line_vector[0] * (line_point[0] - circle_center[0]) + line_vector[1] * (line_point[1] - circle_center[1]))
  c = (line_point[0] - circle_center[0])**2 + (line_point[1] - circle_center[1])**2 - circle_radius**2

  discriminant = b**2 - 4*a*c

  if discriminant < 0: return []
  if discriminant == 0:
    t = -b / (2*a)
    p = line_point + t * line_vector
    return [p]
  t1 = (-b + np.sqrt(discriminant)) / (2*a)
  t2 = (-b - np.sqrt(discriminant)) / (2*a)
  point1 = line_point + t1 * line_vector
  point2 = line_point + t2 * line_vector
  return [point1, point2]
