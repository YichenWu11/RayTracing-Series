import taichi as ti
from ray_tracing_tools import Ray, PI

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
        # print(self.cam_lower_left_corner[None])
        self.cam_horizontal[None] = 2 * half_width * u
        self.cam_vertical[None] = 2 * half_height * v

    @ti.func
    def get_ray(self, u, v):
        return Ray(self.cam_origin[None], self.cam_lower_left_corner[None] + u * self.cam_horizontal[None] + v * self.cam_vertical[None] - self.cam_origin[None])

