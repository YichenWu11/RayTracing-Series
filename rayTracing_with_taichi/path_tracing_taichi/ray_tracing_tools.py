import taichi as ti

PI = 3.14159265

@ti.func
def rand3():
    return ti.Vector([ti.random(), ti.random(), ti.random()])

@ti.func
def random_in_unit_sphere():
    # 在一个单位正方体里sample，当采样到的点在球内（即 p.norm() < 1.0 ）时才返回
    p = 2.0 * rand3() - ti.Vector([1, 1, 1])  # [-1, 1]
    while p.norm() >= 1.0:
        p = 2.0 * rand3() - ti.Vector([1, 1, 1])
    return p

@ti.func
def random_unit_vector():
    return random_in_unit_sphere().normalized()

@ti.func
def to_light_source(hit_point, light_source):
    return light_source - hit_point  # 从 hit_point 指向 light_source

@ti.func
def reflect(v, normal):
    return v - 2 * v.dot(normal) * normal

@ti.func
def refract(uv, n, etai_over_etat):
    # 折射
    cos_theta = min(n.dot(-uv), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.dot(r_out_perp))) * n
    return r_out_perp + r_out_parallel

@ti.func
def reflectance(cosine, ref_idx):
    # Use Schlick's approximation for reflectance.
    # reflection coefficient
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cosine), 5)

@ti.data_oriented
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
    @ti.func
    def at(self, t):
        return self.origin + t * self.direction
