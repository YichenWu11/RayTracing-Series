cmake_minimum_required(VERSION 3.0.0)
project(inoneweekend VERSION 0.1.0)

add_executable(inoneweekend 
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
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
