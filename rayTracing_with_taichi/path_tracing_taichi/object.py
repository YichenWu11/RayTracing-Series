import taichi as ti


'''
    Material
        0 : light_source
        1 : diffuse  
        2 : metal        (金属)
        3 : glass        (dielectric)
        4 : Fuzz Metal   (有光泽)
'''

# 平面
@ti.data_oriented
class Plane:
    def __init__(self, center, normal, color, material=1, width=5, height=5):
        self.center = center
        self.normal = normal
        self.color = color
        self.material = material
        self.width = width
        self.height = height

    @ti.func
    def is_inside_plane(self, point):
        res = False
        if self.center[0] - self.width/2 < point[0] and self.center[0] + self.width/2 > point[0]:
            if self.center[1] - self.width/2 < point[1] and self.center[1] + self.width/2 > point[1]:
                if self.center[2] - self.width/2 < point[2] and self.center[2] + self.width/2 > point[2]:
                    res = True
        return res

    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        is_hit = False
        front_face = False
        root = t_max
        hit_point =  ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        t = ((self.center - ray.origin).dot(self.normal)) / (self.normal.dot(ray.direction))
        if t > t_min and t < t_max:
            hit_point_tmp = ray.at(t)
            if self.is_inside_plane(hit_point_tmp):
                is_hit = True
                root = t
                hit_point = hit_point_tmp
                hit_point_normal = self.normal
                if ray.direction.dot(hit_point_normal) < 0:
                    front_face = True
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color


# TODO:三角形
@ti.data_oriented
class Triangle:
    def __init__(self, a, b, c, normal, color, material=1):
        ...

    
    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        ...

# 正方体
@ti.data_oriented
class Cube:
    def __init__(self, center, material, color, width=1):
        self.material = material
        self.color = color

        self.center = center
        self.front_center = (center[0], center[1], center[2]-width/2)
        self.bottom_center = (center[0], center[1]-width/2, center[2])
        self.left_center = (center[0]-width/2, center[1], center[2])

        self.front_normal = (self.front_center - center).normalized()
        self.back_normal = -self.front_normal
        self.left_normal = (self.left_center - center).normalized()
        self.right_normal = -self.left_normal
        self.bottom_normal = (self.bottom_center - center).normalized()
        self.top_normal = -self.bottom_normal
        self.width = width

        self.plane_list = [
            Plane(self.front_center, self.front_normal, color, material=self.material, width=self.width),
            Plane(center + (center - self.front_center), self.back_normal, color, material=self.material, width=self.width),
            Plane(self.left_center, self.left_normal, color, material=self.material, width=self.width),
            Plane(center + (center - self.left_center), self.right_normal, color, material=self.material, width=self.width),
            Plane(center + (center - self.bottom_center), self.top_normal, color, material=self.material, width=self.width),
            Plane(self.bottom_center, self.bottom_normal, color, material=self.material, width=self.width),
        ]

    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        closest_t = t_max
        root = 0.0
        is_hit = False
        front_face = False
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        color = ti.Vector([0.0, 0.0, 0.0])
        material = self.material

        for index in ti.static(range(len(self.plane_list))):
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp =  self.plane_list[index].hit(ray, t_min, closest_t)
            if is_hit_tmp:
                closest_t = root_tmp
                is_hit = is_hit_tmp
                hit_point = hit_point_tmp
                hit_point_normal = hit_point_normal_tmp
                front_face = front_face_tmp
                color = color_tmp
                material = material_tmp
        return is_hit, root, hit_point, hit_point_normal, front_face, material, color


# 球体
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
