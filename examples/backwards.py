import numpy as np
from numpy.typing import NDArray
from verlet_simple2d import DTYPE
from verlet_simple2d.space import Space
from verlet_simple2d.shapes import Circle, RectangleBorder
from verlet_simple2d.render import Renderer
from tqdm import trange

def circle_points(x: DTYPE, y: DTYPE, r: DTYPE, numpoints: int) -> NDArray[DTYPE]:
  theta = np.linspace(0, np.pi * 2, numpoints + 1)
  theta = theta[:-1]
  points = np.column_stack((x + r * np.cos(theta), y + r * np.sin(theta)))
  return points

def main() -> None:
  num_circles = 30
  frame_rate = 30.0
  step_size = 10
  circles = [Circle(0,0, 10) for _ in range(num_circles)]
  border = RectangleBorder(10, 10, 500, 500, 100)
  space = Space(1/(frame_rate * step_size))
  space.gravity = 0, 0

  cx, cy = DTYPE(260), DTYPE(260)
  points = circle_points(cx, cy, 200, num_circles)

  for c, p in zip(circles, points):
    c.x = c.prev_x = p[0]
    c.y = c.prev_y = p[1]
    vec = np.array((cx - c.x, cy - c.y), dtype=DTYPE)
    #print(c.location, c.prev_location, c.location - c.prev_location)
    c.prev_location = c.location - np.array((-c.y / np.linalg.norm(vec), c.x / np.linalg.norm(vec)), dtype=DTYPE)
    c.prev_location = c.location  + -(c.location - c.prev_location) / 60
    space.add_body(c)
  space.add_body(border)

  steps = 10000
  with space.reverse():
    for _ in (t:=trange(steps)): space.step()
  
  print(sum([k._collisions for k in space.kinetics + space.statics])/2)

  for _ in (t:=trange(steps)): space.step()
  
  r = Renderer(space, 2.0, 'test')
  try:
    r.bad_live_render()
  except KeyboardInterrupt:
    pass
  return


if __name__ == "__main__":
  main()
