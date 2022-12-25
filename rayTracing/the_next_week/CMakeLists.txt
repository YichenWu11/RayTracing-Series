cmake_minimum_required(VERSION 3.0.0)
project(thenextweek VERSION 0.1.0)

add_executable(thenextweek
               main.cpp 
               vec3.h 
               color.h
               ray.h
               hittable.h
               sphere.h
               hittable_list.h
               rtweekend.h
               camera.h
               material.h
               moving_sphere.h
               aabb.h
               bvh.h
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
