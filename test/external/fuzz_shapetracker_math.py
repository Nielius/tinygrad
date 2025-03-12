import os

import random
from tinygrad.helpers import getenv, DEBUG, colored, trange
from tinygrad.shape.shapetracker import ShapeTracker
from test.external.fuzz_shapetracker import shapetracker_ops
from test.external.fuzz_shapetracker import do_permute, do_reshape_split_one, do_reshape_combine_two, do_flip, do_pad
from test.unit.test_shapetracker_math import st_equal, MultiShapeTracker
from tinygrad.shape.view import View

def fuzz_plus() -> tuple[ShapeTracker, ShapeTracker]:
  m = MultiShapeTracker([ShapeTracker.from_shape((random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)))])
  for _ in range(4): random.choice(shapetracker_ops)(m)
  backup = m.sts[0]
  m.sts.append(ShapeTracker.from_shape(m.sts[0].shape))
  for _ in range(4): random.choice(shapetracker_ops)(m)
  st_sum = backup + m.sts[1]
  return m.sts[0], st_sum

# shrink and expand aren't invertible, and stride is only invertible in the flip case
invertible_shapetracker_ops = [do_permute, do_reshape_split_one, do_reshape_combine_two, do_flip, do_pad]

def fuzz_invert() -> tuple[ShapeTracker, ShapeTracker]:
  start = ShapeTracker.from_shape((random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)))
  m = MultiShapeTracker([start])
  for _ in range(8): random.choice(invertible_shapetracker_ops)(m)
  inv = m.sts[0].invert(start.shape)
  st_sum = (m.sts[0] + inv) if inv else None
  return start, st_sum


if __name__ == "__main__":
  if seed:=getenv("SEED"): random.seed(seed)
  total = getenv("CNT", 1000)

  st1 = ShapeTracker(views=(View(shape=(10, 3, 3), strides=(2 * 7 + 3 * 11, 7, 11), offset=0, mask=None, contiguous=False),
                      View(shape=(5, ), strides=(4, ), offset=0, mask=None, contiguous=False)))
  st2 = ShapeTracker(views=(View(shape=(5, ), strides=(7 + 11, ), offset=0, mask=None, contiguous=False), ))
  print(st1.simplify())
  print(st2.simplify())
  print(st1 + st2)
  print(st_equal(st1, st2))
  exit(0)


  for fuzz in [globals()[f'fuzz_{x}'] for x in getenv("FUZZ", "invert,plus").split(",")]:
    same_but_neq = 0
    for _ in trange(total, desc=f"{fuzz}"):
      st1, st2 = fuzz()
      eq = st_equal(st1, st2)
      if getenv("CHECK_NEQ") and eq and st1.simplify() != st2.simplify():
        print(colored("same but unequal", "yellow"))
        print(st1.simplify())
        print(st2.simplify())
        same_but_neq += 1
      if DEBUG >= 1:
        print(f"EXP: {st1}")
        print(f"GOT: {st2}")
        print(colored("****", "green" if eq else "red"))
      if not eq: exit(0)
    if getenv("CHECK_NEQ"): print(f"same but unequal {same_but_neq}/{total} = {(same_but_neq/total)*100:.2f}%")


# if __name__ == "__main__":
#   if seed:=getenv("SEED"): random.seed(seed)
#   total = getenv("CNT", 1000)
#   for fuzz in [globals()[f'fuzz_{x}'] for x in getenv("FUZZ", "invert,plus").split(",")]:
#     same_but_neq = 0
#     for _ in trange(total, desc=f"{fuzz}"):
#       st1, st2 = fuzz()
#       eq = st_equal(st1, st2)
#       if getenv("CHECK_NEQ") and eq and st1.simplify() != st2.simplify():
#         print(colored("same but unequal", "yellow"))
#         print(st1.simplify())
#         print(st2.simplify())
#         same_but_neq += 1
#       if DEBUG >= 1:
#         print(f"EXP: {st1}")
#         print(f"GOT: {st2}")
#         print(colored("****", "green" if eq else "red"))
#       if not eq: exit(0)
#     if getenv("CHECK_NEQ"): print(f"same but unequal {same_but_neq}/{total} = {(same_but_neq/total)*100:.2f}%")
