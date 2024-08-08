from verlet_simple2d.space import Space
from verlet_simple2d.shapes import Circle, CircleBorder, RectangleBorder
from verlet_simple2d.render import Renderer


def test_resolve_circle_circle():
  circle1 = Circle(45, 50, 20)
  circle1.prev_location = circle1.location - (10, 0)
  circle2 = Circle(55, 50, 20)
  circle2.prev_location = circle2.location - (-10, 0)

  border = RectangleBorder(0, 0, 100, 100, 1000)

  space = Space(1/120)

  space.add_body(circle1)
  space.add_body(circle2)
  space.add_body(border)

  handler = space.get_collision_handler(circle1, circle2)

  handler.resolve(circle1, circle2)

  re = Renderer(space, 3.0)

  re.render_current_frame().save('output/img.png')

  print(circle1)
  print(circle2)
