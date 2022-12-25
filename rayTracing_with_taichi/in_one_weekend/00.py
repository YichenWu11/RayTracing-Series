import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

image_resolution = (960, 540)
aspect_ratio = image_resolution[0] / image_resolution[1]
image_pixels = ti.Vector.field(3, float, image_resolution)


@ti.dataclass
class Ray:
    origin: tm.vec3
    direction: tm.vec3

    @ti.func
    def at(r, t: float) -> tm.vec3:
        return r.origin + t * r.direction


@ti.func
def hit_sphere(center, radius, r) -> ti.i32:
    oc = r.origin - center
    a = tm.dot(r.direction, r.direction)
    b = 2.0 * tm.dot(oc, r.direction)
    c = tm.dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c
    return (discriminant > 0)


@ti.func
def hit_sphere_normal(center, radius, r) -> ti.f64:
    oc = r.origin - center
    a = tm.dot(r.direction, r.direction)
    b = 2.0 * tm.dot(oc, r.direction)
    c = tm.dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c
    ret = ti.f64(0.0)
    if discriminant < 0:
        ret = -1.0
    else:
        ret = (-b - tm.sqrt(discriminant)) / (2.0 * a)
    return ret


@ti.func
def ray_color(ray) -> tm.vec3:
    ret = tm.vec3(0, 0, 0)
    if hit_sphere(tm.vec3(0, 0, -1), 0.5, ray):
        ret = tm.vec3(1, 0, 0)
    else:
        t = 0.5 * (ray.direction.y + 1.0)
        ret = (1.0 - t) * tm.vec3(1.0, 1.0, 1.0) + t * tm.vec3(0.5, 0.7, 1.0)
    return ret


@ti.func
def ray_color_normal(ray) -> tm.vec3:
    ret = tm.vec3(0, 0, 0)
    t = hit_sphere_normal(tm.vec3(0, 0, -1), 0.5, ray)
    if t > 0.0:
        N = tm.normalize(ray.at(t) - tm.vec3(0, 0, -1))
        ret = 0.5 * tm.vec3(N.x + 1, N.y + 1, N.z + 1)
    else:
        t = 0.5 * (ray.direction.y + 1.0)
        ret = (1.0 - t) * tm.vec3(1.0, 1.0, 1.0) + t * tm.vec3(0.5, 0.7, 1.0)
    return ret


@ti.kernel
def render():
    for i, j in image_pixels:
        u = i / (image_resolution[0] - 1)
        v = j / (image_resolution[1] - 1)

        focal_dist = 1.0
        horizontal = tm.vec3(2 * aspect_ratio, 0, 0)
        vertical = tm.vec3(0, 2, 0)
        origin = tm.vec3(0, 0, 0)
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - tm.vec3(0, 0, focal_dist)

        ray = Ray(origin, lower_left_corner + u * horizontal + v * vertical - origin)

        # image_pixels[i, j] = ray_color(ray)
        image_pixels[i, j] = ray_color_normal(ray)


window = ti.ui.Window("InOneWeekend", image_resolution)
canvas = window.get_canvas()

while window.running:
    render()
    canvas.set_image(image_pixels)
    window.show()
