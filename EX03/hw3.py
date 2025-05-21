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

            obj_intersect = ray.nearest_intersected_object(objects)

            if obj_intersect is None:
                color = np.zeros(3)

            else:
                t, obj = obj_intersect
                hitP = ray.origin + t * ray.direction
           
                # This is the main loop where each pixel color is computed.
                color = getColor(ambient, obj, objects, lights, ray, hitP, max_depth, 1)
            
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image

def getColor(ambient, obj, objects, lights, ray, hitP, max_depth, level):
    hitP += 1e-2 * obj.getNormal(hitP)

    # Ambiant and Emission Calculations
    color = calcEmissionColor(obj) + calcAmbientColor(ambient, obj)
    
    # Diffuse & Specular Calculation
    for light in lights:
        sj = calcShadowFactor(light, obj, objects, hitP)
        if sj == 0.0:
            continue
        color += sj * (calcDiffuseColor(obj, hitP, light) + calcSpecularColor(obj,hitP,ray,light))
    
    level += 1
    
    if level > max_depth:
        return color
    
    # Reflective and Refractive components
    if obj.reflection > 0:
        r_ray = ConstructReflectiveRay(ray, obj, hitP)
        r_intersect = r_ray.nearest_intersected_object(objects)
        if r_intersect is not None:
            t, r_obj = r_intersect
            r_hit = r_ray.origin + t * r_ray.direction
            color += r_obj.reflection * getColor(ambient, r_obj, objects, lights, r_ray, r_hit, max_depth, level)
    
    if obj.refraction > 0:
        t_ray = ConstructRefractiveRay(ray, obj, hitP)
        t_intersect = t_ray.nearest_intersected_object(objects)    
        if t_intersect is not None:
            t, t_obj = t_intersect
            t_hit = t_ray.origin + t * t_ray.direction
            color += obj.refraction * getColor(ambient, t_obj, objects, lights, t_ray, t_hit, max_depth, level)
    return color

def calcEmissionColor(obj):
    if isinstance(obj, LightSource):
        return np.ones(3)
    else:
        return np.zeros(3)
    
def calcAmbientColor(ambient, obj):
    return obj.ambient * ambient

def calcShadowFactor(light, obj, objects, hitP):
    ray = light.get_light_ray(hitP)
    light_distance = light.get_distance_from_light(hitP)
    obj_intersect = ray.nearest_intersected_object(objects)
    if obj_intersect is None:
        return 1.0
    
    t, obj = obj_intersect

    if t < light_distance:
        return obj.refraction
    else:
        return 1.0

def calcDiffuseColor(obj, hitP, light):
    vec_l = light.get_light_ray(hitP).direction
    vec_n = obj.getNormal(hitP)
    dot_nl = max(np.dot(vec_n, vec_l), 0.0)
    return obj.diffuse * light.get_intensity(hitP) * dot_nl

def calcSpecularColor(obj, hitP, ray, light):
    vec_l = light.get_light_ray(hitP).direction
    l_hat = normalize(reflected(-vec_l, obj.getNormal(hitP)))
    v = -ray.direction
    dot_lv = max(np.dot(l_hat, v), 0.0)
    return obj.specular * light.get_intensity(hitP) * (dot_lv ** obj.shininess)
    
def ConstructReflectiveRay(ray, obj, hitP):
    return Ray(hitP, reflected(ray.direction, obj.getNormal(hitP)))

def ConstructRefractiveRay(ray, obj, hitP):
    return Ray(hitP - obj.getNormal(hitP), ray.direction)

# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0.7,1.5])

    globe = Sphere([0, 0.4, 0.2],0.5)
    globe.set_material(
        ambient=[0, 0, 0],
        diffuse=[0.1, 0.1, 0.1], 
        specular=[1, 1, 1],
        shininess=100,
        reflection=0.5,
        refraction=0.8
    )

    snow_bottom = Sphere([0, 0.4, 0.2],0.3)
    snow_bottom.set_material(
        ambient=[1, 1, 0],
        diffuse=[1, 1, 0], 
        specular=[0, 0, 0],
        shininess=100,
        reflection=0.5
    )

    plane = Plane([0,1,0], [0,-0.3,0])
    plane.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [1, 1, 1], 1000, 0.5)
    background = Plane([0,0,1], [0,0,-3])
    background.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], 1000, 0.5)


    objects = [globe, snow_bottom, plane, background]

    light1 = PointLight(intensity= np.array([1, 1, 1]),position=np.array([1,1.5,1]),kc=0.1,kl=0.1,kq=0.1)    

    lights = [light1]
    
    
    return camera, lights, objects
