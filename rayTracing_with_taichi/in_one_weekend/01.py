import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

T_MIN = 0.001
T_MAX = tm.inf
SPP = 100  # samples per pixel

image_resolution = (960, 540)
aspect_ratio = image_resolution[0] / image_resolution[1]
image_pixels = ti.Vector.field(3, float, image_resolution)


@ti.func
def randf_range(min, max) -> ti.f32:
    return min + (max - min) * ti.random()


@ti.dataclass
class Ray:
    origin: tm.vec3
    direction: tm.vec3

    @ti.func
    def at(r, t: float) -> tm.vec3:
        return r.origin + t * r.direction


@ti.dataclass
class HitRecord:
    pos: tm.vec3
    normal: tm.vec3
    t: ti.f32
    is_hit: ti.i32
    is_front_face: ti.i32

    @ti.func
    def set_face_normal(h, r, outward_normal):
        h.is_front_face = tm.dot(r.direction, outward_normal) < 0
        h.normal = outward_normal if h.is_front_face else -outward_normal


@ti.dataclass
class Sphere:
    center: tm.vec3
    radius: ti.f32

    @ti.func
    def hit(s, r, t_min, t_max) -> HitRecord:
        record = HitRecord(r.origin, tm.vec3(0, 0, 0), t_min, False)

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
                outward_normal = (record.pos - s.center) / s.radius
                record.set_face_normal(r, outward_normal)

        return record


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

        return Ray(c.origin, lower_left_corner + u * horizontal + v * vertical - c.origin)


objects_num = 2
objects = Sphere.field(shape=objects_num)

objects[0] = Sphere(tm.vec3(0, 0, -1), 0.5)
objects[1] = Sphere(tm.vec3(0, -100.5, -1), 100)\

camera = Camera(tm.vec3(0, 0, 0))


@ti.func
def ray_color(ray) -> tm.vec3:
    t = 0.5 * (ray.direction.y + 1.0)
    ret = (1.0 - t) * tm.vec3(1.0, 1.0, 1.0) + t * tm.vec3(0.5, 0.7, 1.0)  # default

    closest_so_far = T_MAX
    for i in range(objects_num):
        rec = objects[i].hit(ray, T_MIN, closest_so_far)
        if rec.is_hit:
            # ret = 0.5 * (rec.normal + tm.vec3(1.0, 1.0, 1.0))
            ret = tm.vec3(1, 0, 0)
            closest_so_far = rec.t
            break
    return ret


@ti.kernel
def render():
    for i, j in image_pixels:
        u = (i + ti.random()) / (image_resolution[0] - 1)
        v = (j + ti.random()) / (image_resolution[1] - 1)

        for _ in range(SPP):
            ray = camera.get_ray(u, v)
            image_pixels[i, j] += ray_color(ray)

        image_pixels[i, j] /= SPP


window = ti.ui.Window("InOneWeekend", image_resolution)
canvas = window.get_canvas()

while window.running:
    render()
    canvas.set_image(image_pixels)
    window.show()
