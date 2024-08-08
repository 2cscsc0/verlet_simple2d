from __future__ import annotations

from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from verlet_simple2d import DTYPE, helpers, shapes


def get_handler(X: shapes.Body | shapes.Border, Y: shapes.Body | shapes.Border) -> CollisionHandler:
  assert not (isinstance(X, shapes.Border) and isinstance(Y, shapes.Border)), 'X and Y cannot both be a border'

  match type(X), type(Y):
    case (shapes.Circle, shapes.CircleBorder) | (shapes.CircleBorder, shapes.Circle):
      return CircleCircleBorderHandler((X.collision_type, Y.collision_type))
    case (shapes.Circle, shapes.Circle):
      return CircleCircleHandler((X.collision_type, Y.collision_type))
    case (shapes.Circle, shapes.RectangleBorder) | (shapes.RectangleBorder, shapes.Circle):
      return CircleRectangleBorderHandler((X.collision_type, Y.collision_type))
  raise NotImplementedError()

class CollisionHandler:
  def __init__(self, types):
    self.types = types

  def check_types(self, type1, type2) -> bool:
    return self.types == (type1, type2) or self.types == (type2, type1)
  
  @abstractmethod
  def check(self, X, Y) -> bool: pass

  @abstractmethod
  def resolve(self, X, Y) -> None: pass

class CircleCircleHandler(CollisionHandler):
  def __init__(self, types=(shapes.Circle, shapes.Circle)) -> None:
    super().__init__(types)

  def check(self, X: shapes.Circle, Y:shapes.Circle) -> bool:
    return bool(np.linalg.norm(X.location - Y.location) < X.radius + Y.radius)
  
  def resolve(self, X: shapes.Circle, Y: shapes.Circle) -> None:
    vX = X.location - X.prev_location
    vY = Y.location - Y.prev_location

    """
    g1 = X.prev_x - Y.prev_x
    g2 = X.prev_y - Y.prev_y
    h1 = (vX[0] - vY[0])
    h2 = (vX[1] - vY[1])
    r = X.radius + Y.radius

    a = h1**2 + h2**2
    b = ((g1*h1) + (g2*h2))*2
    c = g1**2 + g2**2 - r**2

    discriminant = b**2 - 4*a*c

    if discriminant < 0:
      raise RuntimeError('BUGG')
    elif discriminant == 0:
      t = (-b) / (2*a)
    else:
      t1 = (-b + np.sqrt(discriminant)) / (2*a)
      t2 = (-b - np.sqrt(discriminant)) / (2*a)
      t = min(t1, t2)
    
    #assert t <= 1, f't should be <= 1, but is {t}'

    # location = old_location    + (velocity * t)
    X.location = X.prev_location + (vX) * t
    Y.location = Y.prev_location + (vY) * t
    """

    # commented out code should be significantly better, but not sure if it's working properly 100% of the time
    d_vec = X.location - Y.location
    d = np.linalg.norm(d_vec)
    d_vec = d_vec/d
    d_vec = d_vec * ((X.radius + Y.radius) - d)
    X.location += d_vec * 0.5
    Y.location -= d_vec * 0.5
    #END


    #tmp_v1 = c1.velocity - (2*c2.mass/(c1.mass+c2.mass)) * (np.dot(c1.velocity-c2.velocity, c1.loc-c2.loc) / np.linalg.norm(c1.loc-c2.loc)**2) * (c1.loc-c2.loc)
    #tmp_v2 = c2.velocity - (2*c1.mass/(c2.mass+c1.mass)) * (np.dot(c2.velocity-c1.velocity, c2.loc-c1.loc) / np.linalg.norm(c2.loc-c1.loc)**2) * (c2.loc-c1.loc)
    tmp_vX =  vX - (2*X.mass/(X.mass+Y.mass)) * (np.dot(vX-vY, X.location - Y.location) / np.linalg.norm(X.location-Y.location)**2) * (X.location-Y.location)
    tmp_vY =  vY - (2*Y.mass/(Y.mass+X.mass)) * (np.dot(vY-vX, Y.location - X.location) / np.linalg.norm(Y.location-X.location)**2) * (Y.location-X.location)

    tmp_vX[np.isnan(tmp_vX) | np.isinf(tmp_vX)] = DTYPE(0)
    tmp_vY[np.isnan(tmp_vY) | np.isinf(tmp_vY)] = DTYPE(0)

    """
    X.location += tmp_vX * (1-t)
    Y.location += tmp_vY * (1-t)
    """

    X.prev_location = X.location + (-tmp_vX)
    Y.prev_location = Y.location + (-tmp_vY)

    X._collisions+=1
    Y._collisions+=1

class CircleCircleBorderHandler(CollisionHandler):
  def __init__(self, types=(shapes.Circle, shapes.CircleBorder)) -> None:
    super().__init__(types)

  def check(self, X: shapes.Circle, Y: shapes.CircleBorder) -> bool:
    d = np.linalg.norm(X.location - Y.location)
    if d >= Y.radius - X.radius:
      return True
    if d > Y.radius + Y.line_width + X.radius:
      return True
    return False
  
  def resolve(self, X: shapes.Circle, Y: shapes.CircleBorder) -> None:
    x_vel = X.location - X.prev_location
    X.location = helpers.closest_point(
      X.location,
      helpers.line_circle_intersection(
        Y.location,
        Y.radius - X.radius,
        X.location,
        x_vel,
      )
    )

    t = np.linalg.norm(X.prev_location - X.location)/np.linalg.norm(x_vel)

    assert t <= 1, f't must be < 1, but is {t}'

    mirror_vec = Y.location - X.location
    mirror_vec = mirror_vec/np.linalg.norm(mirror_vec)

    x_vel -= 2*np.dot(x_vel, mirror_vec)*mirror_vec

    X.location += (1-t)*x_vel
    X.prev_location = X.location + (-x_vel)

    X._collisions+=1
    Y._collisions+=1

class CircleRectangleBorderHandler(CollisionHandler):
  def __init__(self, types=(shapes.Circle, shapes.RectangleBorder)) -> None:
    super().__init__(types)

  def closest_side(self, X: shapes.Circle, Y: shapes.RectangleBorder) -> tuple[NDArray[DTYPE], DTYPE]:
    vf, hf = DTYPE(0), DTYPE(0)
    distance_upper = Y.y + Y.height - (X.y + X.radius)
    distance_lower = (X.y - X.radius) - Y.y
    if distance_lower < distance_upper:
      vertical_distance = distance_lower 
    else:
      vertical_distance = distance_upper
      vf = DTYPE(1)
    distance_right = Y.x + Y.width - (X.x + X.radius)
    distance_left = (X.x - X.radius) - Y.x
    if distance_right < distance_left:
      horizontal_distance = distance_right
      hf = DTYPE(1)
    else:
      horizontal_distance = distance_left
    
    if vertical_distance < horizontal_distance:
      return np.array((1,0)), vf
    return np.array((0,1)), hf
  
  def check(self, X: shapes.Circle, Y: shapes.RectangleBorder):
    if (
      X.y + X.radius > Y.y + Y.height
      or X.y - X.radius < Y.y
      or X.x + X.radius > Y.x + Y.width
      or X.x - X.radius < Y.x
    ):
      return True

  def resolve(self, X: shapes.Circle, Y: shapes.RectangleBorder) -> None:
    vel = X.location - X.prev_location

    mirror_vec,f = self.closest_side(X, Y)
    mirror_line_point = (Y.location + X.radius) + (f * (Y.dims - 2*X.radius))

    X.location = helpers.line_line_intersection(
      X.location,
      vel,
      mirror_line_point,
      mirror_vec,
    )

    t = np.linalg.norm(X.prev_location - X.location)/np.linalg.norm(vel)

    #assert t <= 1, f't must be < 1, but is {t}'

    vel -= 2*np.dot(vel, mirror_vec[::-1])*mirror_vec[::-1]

    X.location += (1-t)*vel
    X.prev_location = X.location + (-vel)

    X._collisions+=1
    Y._collisions+=1
