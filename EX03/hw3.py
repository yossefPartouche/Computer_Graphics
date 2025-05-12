from helper_classes import *
import matplotlib.pyplot as plt

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)

            t, object = ray.nearest_intersected_object(objects)
            hitP = ray.origin + t * ray.direction

            color = np.zeros(3)
            #color = (255,255,255)
           
            # This is the main loop where each pixel color is computed.
            for c in range(3):
                color[c] = getColor(c, ambient, object, objects, lights, ray, hitP, max_depth, level=1)
            
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image

def getColor(c, ambient, object, objects, lights, ray, hitP, max_depth, level):
    # Ambiant and Emission Calculations
    color = calcEmissionColor(object) + calcAmbientColor(c, ambient, object)
    
    # Diffuse & Specular Calculation
    for light in lights:
        sj = calcShadowFactor(light, objects, hitP)
        color = color + sj * (calcDiffuseColor(c, object, objects, hitP, ray, light) + calcSpecularColor(c, object, objects,hitP,ray,light))
    
    level += 1
    
    if level > max_depth:
        return color
    
    # Reflective and Refractive components
    #r_ray = ConstructReflectiveRay(ray, objects, hitP)
    #r_hit = r_ray.nearest_intersected_object(objects)
    #color += k_r * getColor(objects, r_ray, r_hit, level)
    #t_ray = ConstructRefractiveRay(ray, objects, hitP)
    #t_hit = t_ray.nearest_intersected_object(objects)
    #color += k_t * getColor(objects, t_ray, t_hit, level)
    
    return color

def calcEmissionColor(object):
    if isinstance(object, LightSource):
        return 1
    else:
        return 0
    
def calcAmbientColor(c, ambient, object):
    return object.ambient[c] * ambient[c]

def calcShadowFactor(light, objects, hitP):
    ray = light.get_light_ray(hitP)
    if ray.nearest_intersected_object(objects) is None:
        return 1
    
    _, intersected_obj = ray.nearest_intersected_object(objects)

    if intersected_obj == light:
        return 1
    else:
        return 0

def calcDiffuseColor(c, object, objects, hitP, ray, light):
    vec_l = light.get_light_ray(hitP).direction
    vec_n = object.normal
    return object.diffuse[c] * light.get_intensity(hitP)[c] * np.dot(vec_n, vec_l)

def calcSpecularColor(c, object, objects, hitP, ray, light):
    l_hat = reflected(ray.direction, object.normal)
    v = ray.direction
    return object.specular[c] * light.get_intensity(hitP)[c] * np.dot(v, l_hat) ** object.shininess
    
# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0,1])
    lights = []
    objects = []
    return camera, lights, objects
