import platform
import random
import time
from pathlib import Path
from typing import cast

import ffmpeg  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import trange

from verlet_simple2d import DTYPE, shapes
from verlet_simple2d.space import Space


class Renderer:
  def __init__(self, space: Space, scale: float, watermark:str='', step_size:int=1) -> None:
    self.space: Space = space
    self.step_size = step_size

    self.width, self.height = self._dims()
    self.scale: DTYPE = DTYPE(scale)
    self.watermark = watermark

    self.clrs: list[bytes] = [
      b'#64E619',
      b'#E6B419',
      b'#E6DA19',
      b'#19B8E6',
      b'#E6191C',
      b'#E6DA19',
      b'#1D19E6',
    ]
    self.border_clr: bytes = b'#B4B4B4'
    self.watermark_clr: bytes = b'#C3C3C3'
    self.background_clr: bytes = b'#000000'

  def _dims(self):
    w = h = 0
    for _stat in self.space.statics:
      match type(_stat):
        case shapes.CircleBorder:
          stat = cast(shapes.CircleBorder, _stat)
          if (stat.x - stat.radius) * 2 + stat.radius * 2 > w: w = (stat.x - stat.radius) * 2 + stat.radius * 2
          if (stat.y - stat.radius) * 2 + stat.radius * 2 > h: h = (stat.y - stat.radius) * 2 + stat.radius * 2
        case shapes.RectangleBorder:
          stat = cast(shapes.RectangleBorder, _stat)
          if stat.width + stat.x * 2 > w: w = stat.width + stat.x * 2
          if stat.height + stat.y * 2 > h: h = stat.height + stat.y * 2
    return int(w), int(h)
  
  def hex_to_tuple(self, clr: bytes) -> tuple[int, int, int]:
    return (
      int(clr[1:3], 16),
      int(clr[3:5], 16),
      int(clr[5:], 16),
    )

  def flipy(self, height, y) -> DTYPE:
    return height - y

  def render_current_frame(self, watermark: tuple[str, int, int] | None=None) -> Image.Image:
    r = random.Random(x=1440)
    img = Image.new(
      mode='RGB',
      size=(
        int(self.width * self.scale),
        int(self.height * self.scale)
      ),
      color=self.hex_to_tuple(self.background_clr),
    )
    draw = ImageDraw.Draw(img)

    for _stat in self.space.statics:
      match type(_stat):
        case shapes.CircleBorder:
          cborder = cast(shapes.CircleBorder, _stat)
          draw.ellipse(
            xy=(  # type: ignore[arg-type]
              (cborder.x - cborder.radius) * self.scale,
              self.flipy(self.height, cborder.y + cborder.radius) * self.scale,
              (cborder.x + cborder.radius) * self.scale,
              self.flipy(self.height, cborder.y - cborder.radius) * self.scale,
            ),
            fill=self.hex_to_tuple(self.border_clr),
          )
        case shapes.RectangleBorder:
          rborder = cast(shapes.RectangleBorder, _stat)
          draw.rectangle(
            xy=(  # type: ignore[arg-type]
              rborder.x * self.scale,
              self.flipy(self.height, rborder.y + rborder.height) * self.scale,
              (rborder.x + rborder.width) * self.scale,
              self.flipy(self.height, rborder.y) * self.scale,
            ),
            fill=self.hex_to_tuple(self.border_clr),
          )
    if watermark:
      assert len(watermark) == 3, f'{watermark=} is not of form ("watermark", x, y)'
      text, x, y = watermark
      if platform.system() == 'Windows':
        font = ImageFont.truetype('ariblk.ttf', self.height // 4)
      else:
        font = ImageFont.truetype('Arial Black.ttf', self.height // 4)
      draw.text(xy=(x,y), text=text, align='left', anchor='mm', fill=self.hex_to_tuple(self.watermark_clr), font=font)

    for _kin in self.space.kinetics:
      match type(_kin):
        case shapes.Circle:
          kin = cast(shapes.Circle, _kin)
          draw.ellipse(
            xy=(  # type: ignore[arg-type]
              (kin.x - kin.radius) * self.scale,
              self.flipy(self.height, kin.y + kin.radius) * self.scale,
              (kin.x + kin.radius) * self.scale,
              self.flipy(self.height, kin.y - kin.radius) * self.scale,
            ),
            fill=self.hex_to_tuple(r.choice(self.clrs)),
          )

    return img

  def bad_live_render(self) -> None:
    plt.ion()
    _, ax = plt.subplots()

    while 1:
      ax.clear()
      ax.imshow(self.render_current_frame())
      for _ in range(self.step_size):
        self.space.step()
      plt.draw()
      plt.pause(0.000001)
      ax.axis('off')
      ax.cla()
  
  def render(self, frame_count: int, frame_rate: float=30.0, path: str='output') -> None:
    otp = Path(path)
    if not otp.exists():
      otp.mkdir()
    elif not otp.is_dir():
      print(f'{otp.absolute()} is no valid directory')
      return
    
    proj = int(time.time())
    proj_path = otp / Path(str(proj))
    if not proj_path.exists():
      proj_path.mkdir()
    elif not proj_path.is_dir():
      print(f'{proj_path.absolute()} is no valid directory')
      return

    frames_path = Path(proj_path / 'frames')
    if not frames_path.exists():
      frames_path.mkdir()
    elif not proj_path.is_dir():
      print(f'{frames_path.absolute()} is no valid directory')
      return
    
    frame_name = 'frame' 
    extension = 'jpg'

    for i in (t:=trange(frame_count)):
      self.render_current_frame(
        (self.watermark, int(self.width * self.scale / 2), int(self.height * self.scale / 2)),
      ).save(
        f'{str(proj_path)}/frames/{frame_name}-{i:08d}.{extension}',
        quality=95,
        subsampling=0,
      )
      t.set_description('Rendering video...')

    print(f'Saved all frames in {str(otp / proj_path)}')

    video_name = 'video'
    print(f'Saving {proj_path / video_name} ...')
    video = ffmpeg.input(f'{str(proj_path)}/frames/{frame_name}-%08d.{extension}', framerate=frame_rate)
    ffmpeg.output(video, f'{proj_path / video_name}.mp4', loglevel='quiet', crf=18).run()
