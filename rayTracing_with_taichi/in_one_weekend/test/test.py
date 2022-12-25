import taichi as ti
import taichi.math as tm
from typing import Tuple

ti.init(arch=ti.gpu)


@ti.dataclass
class Base:
    pos: tm.vec3
    dir: tm.vec3


base = Base(tm.vec3(2.0, 0.0, 0.0))
# print(base.pos)
# print(base.dir)


@ti.func
def randf_range(min, max) -> ti.f32:
    return min + (max - min) * ti.random()


@ti.func
def randvec3_range(min, max) -> tm.vec3:
    return tm.vec3(randf_range(min, max), randf_range(min, max), randf_range(min, max))


@ti.func
def random_in_unit_sphere() -> tm.vec3:
    p = tm.vec3(0, 0, 0)
    while True:
        p = randvec3_range(-1.0, 1.0)
        if tm.length(p) >= 1:
            continue
        break
    return p


@ti.func
def test_base() -> Base:
    base = Base(tm.vec3(1, 0, 0), tm.vec3(0, 1, 0))
    # base.pos = tm.vec3(2, 2, 2)
    return base


@ti.func
def test_00() -> Tuple[Base, tm.vec3]:
    base = Base(tm.vec3(1, 0, 0), tm.vec3(0, 1, 0))
    # base.pos = tm.vec3(2, 2, 2)
    return base, tm.vec3(1, 2, 3)


@ti.kernel
def test():
    base, color = test_00()
    # print(base.pos)
    # print(base.dir)
    # print(color)
    a = tm.vec3(1, 2, 3)
    b = tm.vec3(2, 2, 2)
    print(a * b)


test()
