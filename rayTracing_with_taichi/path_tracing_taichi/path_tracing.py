import taichi as ti
import numpy as np
import argparse
from ray_tracing_tools import Ray, random_in_unit_sphere, refract, reflect, reflectance, random_unit_vector
from camera import Camera
from object import Plane, Cube, Sphere
from hittable import Hittable_list

ti.init(arch=ti.cuda)

# Canvas
aspect_ratio = 1.0
image_width = 800
image_height = int(image_width / aspect_ratio)
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))

# Rendering parameters
samples_per_pixel = 4
max_depth = 10
sample_on_unit_sphere_surface = True


@ti.kernel
def clear():
    for i, j in canvas:
        canvas[i, j] = ti.Vector([0.0, 0.0, 0.0])
    
@ti.kernel
def render():
    for i, j in canvas:
        u = (i + ti.random()) / image_width
        v = (j + ti.random()) / image_height
        color = ti.Vector([0.0, 0.0, 0.0])
        for n in range(samples_per_pixel):
            ray = camera.get_ray(u, v)
            color += ray_color(ray)
        color /= samples_per_pixel
        canvas[i, j] += color

# Path tracing
@ti.func
def ray_color(ray):
    color_buffer = ti.Vector([0.0, 0.0, 0.0])
    brightness = ti.Vector([1.0, 1.0, 1.0])
    scattered_origin = ray.origin
    scattered_direction = ray.direction
    p_RR = 0.8
    for n in range(max_depth):
        if ti.random() > p_RR:
            break
        is_hit, hit_point, hit_point_normal, front_face, material, color = scene.hit(Ray(scattered_origin, scattered_direction))
        if is_hit:
            if material == 0:
                color_buffer = color * brightness
                break
            else:
                # Diffuse
                if material == 1:
                    target = hit_point + hit_point_normal
                    if sample_on_unit_sphere_surface:
                        target += random_unit_vector()
                    else:
                        target += random_in_unit_sphere()
                    scattered_direction = target - hit_point
                    scattered_origin = hit_point
                    brightness *= color
                # Metal and Fuzz Metal
                elif material == 2 or material == 4:
                    fuzz = 0.0
                    if material == 4:
                        fuzz = 0.4
                    scattered_direction = reflect(scattered_direction.normalized(),
                                                  hit_point_normal)
                    if sample_on_unit_sphere_surface:
                        scattered_direction += fuzz * random_unit_vector()
                    else:
                        scattered_direction += fuzz * random_in_unit_sphere()
                    scattered_origin = hit_point
                    if scattered_direction.dot(hit_point_normal) < 0:
                        break
                    else:
                        brightness *= color
                # Dielectric
                elif material == 3:
                    refraction_ratio = 1.5
                    if front_face:
                        refraction_ratio = 1 / refraction_ratio
                    cos_theta = min(-scattered_direction.normalized().dot(hit_point_normal), 1.0)
                    sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
                    # total internal reflection
                    if refraction_ratio * sin_theta > 1.0 or reflectance(cos_theta, refraction_ratio) > ti.random():
                        scattered_direction = reflect(scattered_direction.normalized(), hit_point_normal)
                    else:
                        scattered_direction = refract(scattered_direction.normalized(), hit_point_normal, refraction_ratio)
                    scattered_origin = hit_point
                    brightness *= color
                brightness /= p_RR
    return color_buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Path Tracing')
    parser.add_argument(
        '--max_depth', type=int, default=10, help='max depth (default: 10)')
    parser.add_argument(
        '--samples_per_pixel', type=int, default=4, help='samples_per_pixel  (default: 4)')
    parser.add_argument(
        '--samples_in_unit_sphere', action='store_true', help='whether sample in a unit sphere')
    args = parser.parse_args()

    max_depth = args.max_depth
    samples_per_pixel = args.samples_per_pixel
    sample_on_unit_sphere_surface = not args.samples_in_unit_sphere
    scene = Hittable_list()

    """
        Build the World
    """
    # Light source
    scene.add(Sphere(center=ti.Vector([0, 5.4, -0.2]), radius=3.0, material=0, color=ti.Vector([10.0, 10.0, 10.0])))
    # scene.add(Plane(center=ti.Vector([0, 2.5, -1]), normal=ti.Vector([0.0, 1.0, 0.0]), color=ti.Vector([50.0, 50.0, 50.0]), material=0, width=0.8))

    # Ground
    # scene.add(Sphere(center=ti.Vector([0, -100.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    scene.add(Plane(center=ti.Vector([0, -0.5, -1]), normal=ti.Vector([0.0, 1.0, 0.0]), color=ti.Vector([0.8, 0.8, 0.8]), material=2))

    # ceiling
    # scene.add(Sphere(center=ti.Vector([0, 102.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    scene.add(Plane(center=ti.Vector([0, 2.5, -1]), normal=ti.Vector([0.0, -1.0, 0.0]), color=ti.Vector([0.8, 0.8, 0.8])))

    # back wall
    # scene.add(Sphere(center=ti.Vector([0, 1, 101]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    scene.add(Plane(center=ti.Vector([0, 1, 1]), normal=ti.Vector([0.0, 0.0, -1.0]), color=ti.Vector([0.8, 0.8, 0.8])))

    # right wall
    # scene.add(Sphere(center=ti.Vector([-101.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.6, 0.0, 0.0])))
    scene.add(Plane(center=ti.Vector([-1.5, 0, -1]), normal=ti.Vector([1.0, 0.0, 0.0]), color=ti.Vector([0.6, 0.0, 0.0])))

    # left wall
    # scene.add(Sphere(center=ti.Vector([101.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.0, 0.6, 0.0])))
    scene.add(Plane(center=ti.Vector([1.5, 0, -1]), normal=ti.Vector([-1.0, 0.0, 0.0]), color=ti.Vector([0.0, 0.6, 0.0])))

    # Diffuse ball
    scene.add(Sphere(center=ti.Vector([0, -0.2, -1.5]), radius=0.3, material=1, color=ti.Vector([0.8, 0.3, 0.3])))
    # Metal ball
    scene.add(Sphere(center=ti.Vector([-0.8, 0.2, -1]), radius=0.7, material=2, color=ti.Vector([0.6, 0.8, 0.8])))
    # Glass ball
    # scene.add(Sphere(center=ti.Vector([0.7, 0.0, -0.5]), radius=0.5, material=3, color=ti.Vector([1.0, 1.0, 1.0])))
    scene.add(Cube(center=ti.Vector([0.7, 0.0, -0.5]), material=1, color=ti.Vector([1.0, 1.0, 1.0]), width=1))
    # Metal ball-2
    scene.add(Sphere(center=ti.Vector([0.6, -0.3, -2.0]), radius=0.2, material=4, color=ti.Vector([0.8, 0.6, 0.2])))

    camera = Camera()  # look at [0.0, 1.0, -1.0]  look from [0.0, 1.0, -4.0]
    gui = ti.GUI("Ray Tracing", res=(image_width, image_height))
    canvas.fill(0)
    cnt = 0
    # look from
    lf_x = 0.0
    lf_y = 1.0
    lf_z = -5.0

    while gui.running:
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
                exit()
            elif e.key == 'w':
                clear()
                cnt = 0
                lf_z += 0.5
                # print("w, lf_z is ", lf_z)
            elif e.key == 's':
                clear()
                cnt = 0
                lf_z -= 0.5
                # print("s, lf_z is ", lf_z)
            elif e.key == 'a':
                clear()
                cnt = 0
                lf_x += 0.5
                # print("a, lf_x is ", lf_x)
            elif e.key == 'd':
                clear()
                cnt = 0
                lf_x -= 0.5
                # print("d, lf_x is ", lf_x)         
        # camera motion                                  
        camera.reset(ti.math.vec3(lf_x, lf_y, lf_z))
        render()
        cnt += 1
        gui.set_image(np.sqrt(canvas.to_numpy() / cnt))  # correction
        gui.show()
