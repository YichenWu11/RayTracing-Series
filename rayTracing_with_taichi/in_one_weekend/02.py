import taichi as ti
import taichi.math as tm

# yapf: disable
"""
Material:
        0 : lambertian
        1 : metal
"""

ti.init(arch=ti.gpu)

T_MIN = 0.001
T_MAX = tm.inf
SPP = 32  # samples per pixel
MAX_RAY_DEPTH = 8

image_resolution = (960, 540)
aspect_ratio = image_resolution[0] / image_resolution[1]
image_pixels = ti.Vector.field(3, float, image_resolution)


@ti.func
def near_zero(v) -> ti.i32:
    s = 1e-8
    return (v.x < s) and (v.y < s) and (v.z < s)


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
def random_unit_vec() -> tm.vec3:
    return tm.normalize(random_in_unit_sphere())


@ti.func
def random_in_hemisphere(normal) -> tm.vec3:
    in_unit_sphere = random_in_unit_sphere()
    ret = in_unit_sphere if tm.dot(in_unit_sphere, normal) > 0.0 else -in_unit_sphere
    return ret


@ti.func
def reflect(v, n) -> tm.vec3:
    return v - 2 * tm.dot(v, n) * n


@ti.dataclass
class Ray:
    origin: tm.vec3
    direction: tm.vec3

    @ti.func
    def at(r, t: float) -> tm.vec3:
        return r.origin + t * r.direction


@ti.dataclass
class Material:
    type: ti.u32
    albedo: tm.vec3


@ti.dataclass
class HitRecord:
    pos: tm.vec3
    normal: tm.vec3
    t: ti.f32
    is_hit: ti.i32
    is_front_face: ti.i32
    obj_idx: ti.u32

    @ti.func
    def set_face_normal(h, r, outward_normal):
        h.is_front_face = tm.dot(r.direction, outward_normal) < 0
        h.normal = outward_normal if h.is_front_face else -outward_normal

@ti.dataclass
class ScatterRet:
    ray: Ray
    attenuation: tm.vec3
    is_out: ti.i32


@ti.dataclass
class Sphere:
    center: tm.vec3
    radius: ti.f32
    mtl: Material
    obj_idx: ti.u32

    @ti.func
    def hit(s, r, t_min, t_max) -> HitRecord:
        record = HitRecord(r.origin, tm.vec3(0, 0, 0), t_min, False, 0)

        oc = r.origin - s.center
        a = tm.dot(r.direction, r.direction)
        half_b = tm.dot(oc, r.direction)
        c = tm.dot(oc, oc) - s.radius * s.radius
        discriminant = half_b * half_b - a * c

        return_flag = False

        if discriminant < 0:
            record.is_hit = False
        else:
            sqrtd = tm.sqrt(discriminant)
            root = (-half_b - sqrtd) / a
            if root < t_min or t_max < root:
                root = (-half_b + sqrtd) / a
                if root < t_min or t_max < root:
                    return_flag = True
            if not return_flag:
                record.t = root
                record.pos = r.at(record.t)
                record.is_hit = True
                record.obj_idx = s.obj_idx
                outward_normal = (record.pos - s.center) / s.radius
                record.set_face_normal(r, outward_normal)
        return record


    # return Ray and attenuation(Color)
    @ti.func
    def scatter(s, r_in, rec) -> ScatterRet:
        scattered = Ray(tm.vec3(0, 0, 0), tm.vec3(0, 0, 0))
        attenuation = tm.vec3(1, 1, 1)
        is_out = True

        if s.mtl.type == 0:
            scatter_dir = rec.normal + random_unit_vec()
            if near_zero(scatter_dir):
                scatter_dir = rec.normal

            scattered = Ray(rec.pos, tm.normalize(scatter_dir))
            attenuation = s.mtl.albedo
        elif s.mtl.type == 1:
            reflected = reflect(r_in.direction, rec.normal)
            scattered = Ray(rec.pos, tm.normalize(reflected))
            attenuation = s.mtl.albedo

            is_out = tm.dot(scattered.direction, rec.normal) > 0
        else:
            pass

        ret = ScatterRet(ray=scattered, attenuation=attenuation, is_out=is_out)

        return ret


@ti.dataclass
class Camera:
    origin: tm.vec3

    @ti.func
    def get_ray(c, u, v) -> Ray:
        viewport_height = 2.0
        viewport_width = aspect_ratio * viewport_height
        focal_dist = 1.0

        horizontal = tm.vec3(viewport_width, 0, 0)
        vertical = tm.vec3(0, viewport_height, 0)
        lower_left_corner = c.origin - horizontal / 2 - vertical / 2 - tm.vec3(0, 0, focal_dist)

        return Ray(c.origin, tm.normalize(lower_left_corner + u * horizontal + v * vertical - c.origin))


objects_num = 4
objects = Sphere.field(shape=objects_num)

objects[0] = Sphere(tm.vec3(0, -100.5, -1), 100, mtl=Material(0, tm.vec3(0.8, 0.8, 0.0)), obj_idx=0)
objects[1] = Sphere(tm.vec3(0, 0, -1), 0.5, mtl=Material(0, tm.vec3(0.7, 0.3, 0.3)), obj_idx=1)
objects[2] = Sphere(tm.vec3(-1, 0, -1), 0.5, mtl=Material(1, tm.vec3(0.8, 0.8, 0.8)), obj_idx=2)
objects[3] = Sphere(tm.vec3(1, 0, -1), 0.5, mtl=Material(1, tm.vec3(0.8, 0.6, 0.2)), obj_idx=3)

camera = Camera(tm.vec3(0, 0, 0))


@ti.func
def hit(ray) -> HitRecord:
    ret = HitRecord(ray.origin, tm.vec3(0, 0, 0), T_MIN, False, 0)
    record = HitRecord(ray.origin, tm.vec3(0, 0, 0), T_MIN, False, 0)
    closest_so_far = T_MAX

    for i in range(objects_num):
        record = objects[i].hit(ray, T_MIN, closest_so_far)
        if record.is_hit:
            closest_so_far = record.t
            ret = record
    return ret


@ti.func
def ray_color(ray) -> tm.vec3:
    depth = MAX_RAY_DEPTH
    attenuation = tm.vec3(1, 1, 1)
    is_out = True

    color = tm.vec3(1, 1, 1)

    for _ in range(MAX_RAY_DEPTH):
        if depth <= 0:
            color *= tm.vec3(0, 0, 0)
            break

        record = hit(ray)

        if record.is_hit:
            scatter_ret = objects[record.obj_idx].scatter(ray, record)
            ray = scatter_ret.ray
            attenuation = scatter_ret.attenuation
            is_out = scatter_ret.is_out
            if is_out:
                color *= attenuation
                depth -= 1
                continue
            else:
                color *= tm.vec3(0, 0, 0)
                break
        else:
            t = 0.5 * (ray.direction.y + 1.0)
            color *= (1.0 - t) * tm.vec3(1.0, 1.0, 1.0) + t * tm.vec3(0.5, 0.7, 1.0)
            break

    return color


@ti.kernel
def render():
    for i, j in image_pixels:
        u = (i + ti.random()) / (image_resolution[0] - 1)
        v = (j + ti.random()) / (image_resolution[1] - 1)

        for _ in range(SPP):
            ray = camera.get_ray(u, v)
            image_pixels[i, j] += ray_color(ray)

        image_pixels[i, j] /= SPP
        image_pixels[i, j] = tm.sqrt(image_pixels[i, j])


window = ti.ui.Window("InOneWeekend", image_resolution)
canvas = window.get_canvas()

while window.running:
    render()
    canvas.set_image(image_pixels)
    window.show()
