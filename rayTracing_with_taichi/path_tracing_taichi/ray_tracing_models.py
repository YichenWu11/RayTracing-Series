from multiprocessing import set_forkserver_preload
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

'''
    Material
        0 : light_source
        1 : diffuse  
        2 : metal        (金属)
        3 : glass        (dielectric)
        4 : Fuzz Metal   (有光泽)
'''

# TODO:Plane是对的，Cube不对
@ti.data_oriented
class Plane:
    def __init__(self, center, normal, color):
        self.center = center
        self.normal = normal
        self.color = color
        self.material = 1

    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        is_hit = False
        front_face = False
        root = t_max
        hit_point =  ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        t = ((self.center - ray.origin).dot(self.normal)) / (self.normal.dot(ray.direction))
        if t > t_min and t < t_max:
            is_hit = True
            root = t
            hit_point = ray.at(t)
            hit_point_normal = self.normal
            if ray.direction.dot(hit_point_normal) < 0:
                front_face = True
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color


# 正方体
@ti.data_oriented
class Cube:
    def __init__(self, center, front_center, bottom_center, left_center, material, color):
        self.material = material
        self.color = color

        self.center = center
        self.front_center = front_center
        self.bottom_center = bottom_center
        self.left_center = left_center

        self.front_normal = (front_center - center).normalized()
        self.back_normal = -self.front_normal
        self.left_normal = (left_center - center).normalized()
        self.right_normal = -self.left_normal
        self.bottom_normal = (bottom_center - center).normalized()
        self.top_normal = -self.bottom_normal

        self.plane_list = [
            Plane(front_center, self.front_normal, color),
            Plane(center + (center - front_center), self.back_normal, color),
            Plane(left_center, self.left_normal, color),
            Plane(center + (center - left_center), self.right_normal, color),
            Plane(center + (center - bottom_center), self.top_normal, color),
            Plane(bottom_center, self.bottom_normal, color),
        ]

    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        closest_t = t_max
        root = 0.0
        is_hit = False
        front_face = False
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        # color = ti.Vector([0.0, 0.0, 0.0])
        for index in ti.static(range(len(self.plane_list))):
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp =  self.plane_list[index].hit(ray, t_min, closest_t)
            if is_hit_tmp:
                closest_t = root_tmp
                is_hit = is_hit_tmp
                hit_point = hit_point_tmp
                hit_point_normal = hit_point_normal_tmp
                front_face = front_face_tmp
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color

@ti.data_oriented
class Sphere:
    def __init__(self, center, radius, material, color):
        self.center = center
        self.radius = radius
        self.material = material
        self.color = color

    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        is_hit = False
        front_face = False
        root = 0.0
        hit_point =  ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        if discriminant > 0:
            sqrtd = ti.sqrt(discriminant)
            root = (-b - sqrtd) / (2 * a)
            if root < t_min or root > t_max:
                root = (-b + sqrtd) / (2 * a)
                if root >= t_min and root <= t_max:
                    is_hit = True
            else:
                is_hit = True
        if is_hit:
            hit_point = ray.at(root)
            hit_point_normal = (hit_point - self.center) / self.radius  # normalized
            # Check which side does the ray hit, we set the hit point normals always point outward from the surface
            if ray.direction.dot(hit_point_normal) < 0:
                front_face = True
            else:
                hit_point_normal = -hit_point_normal
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color

@ti.data_oriented
class Hittable_list:
    def __init__(self):
        self.objects = []
    def add(self, obj):
        self.objects.append(obj)
    def clear(self):
        self.objects = []

    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        closest_t = t_max
        is_hit = False
        front_face = False
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        color = ti.Vector([0.0, 0.0, 0.0])
        material = 1
        for index in ti.static(range(len(self.objects))):
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp =  self.objects[index].hit(ray, t_min, closest_t)
            if is_hit_tmp:
                closest_t = root_tmp
                is_hit = is_hit_tmp
                hit_point = hit_point_tmp
                hit_point_normal = hit_point_normal_tmp
                front_face = front_face_tmp
                material = material_tmp
                color = color_tmp
        return is_hit, hit_point, hit_point_normal, front_face, material, color

    @ti.func
    def hit_shadow(self, ray, t_min=0.001, t_max=10e8):
        # 是否击中光源
        is_hit_source = False
        is_hit_source_temp = False
        hitted_dielectric_num = 0
        is_hitted_non_dielectric = False
        # Compute the t_max to light source
        is_hit_tmp, root_light_source, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp = \
        self.objects[0].hit(ray, t_min)
        for index in ti.static(range(len(self.objects))):
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp =  self.objects[index].hit(ray, t_min, root_light_source)
            if is_hit_tmp:
                if material_tmp != 3 and material_tmp != 0:
                    is_hitted_non_dielectric = True
                if material_tmp == 3:
                    hitted_dielectric_num += 1
                if material_tmp == 0:
                    is_hit_source_temp = True
        if is_hit_source_temp and (not is_hitted_non_dielectric) and hitted_dielectric_num == 0:
            is_hit_source = True
        return is_hit_source, hitted_dielectric_num, is_hitted_non_dielectric


@ti.data_oriented
class Camera:
    def __init__(self, fov=60, aspect_ratio=1.0):
        # Camera parameters
        self.lookfrom = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.lookat = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.vup = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.fov = fov
        self.aspect_ratio = aspect_ratio

        self.cam_lower_left_corner = ti.Vector.field(3, dtype=ti.f32, shape=())  # 左下角
        self.cam_horizontal = ti.Vector.field(3, dtype=ti.f32, shape=())  # 水平
        self.cam_vertical = ti.Vector.field(3, dtype=ti.f32, shape=())  # 垂直
        self.cam_origin = ti.Vector.field(3, dtype=ti.f32, shape=())

    @ti.kernel
    def reset(self, new_ori : ti.types.vector(3, ti.f32)):
        self.lookfrom[None] = new_ori
        self.lookat[None] = [0.0, 1.0, -1.0]
        self.vup[None] = [0.0, 1.0, 0.0]
        theta = self.fov * (PI / 180.0)
        half_height = ti.tan(theta / 2.0)
        half_width = self.aspect_ratio * half_height
        self.cam_origin[None] = self.lookfrom[None]
        """
                            ^ (v)
                            |
                            |
                            |  
                            * —— —— —— —— > (u)  
                           /        
                          /
                         /
                        (w) 
        """
        w = (self.lookfrom[None] - self.lookat[None]).normalized()  # gaze at -y
        u = (self.vup[None].cross(w)).normalized()
        v = w.cross(u) # up at y
        self.cam_lower_left_corner[None] = ti.Vector([-half_width, -half_height, -1.0])
        self.cam_lower_left_corner[None] = self.cam_origin[None] - half_width * u - half_height * v - w
        print(self.cam_lower_left_corner[None])
        self.cam_horizontal[None] = 2 * half_width * u
        self.cam_vertical[None] = 2 * half_height * v

    @ti.func
    def get_ray(self, u, v):
        return Ray(self.cam_origin[None], self.cam_lower_left_corner[None] + u * self.cam_horizontal[None] + v * self.cam_vertical[None] - self.cam_origin[None])

