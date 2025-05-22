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
    offset = obj.getNormal(hitP)
    if isinstance(obj, Sphere):
        offset *= 0.1*obj.radius
    return Ray(hitP-offset, ray.direction)


def your_own_scene():
    camera = np.array([0.5,0.1,4])

    globe = Sphere([0.05, 0, 0.1],0.93)
    globe.set_material(
        ambient=[1, 1, 1],
        diffuse=[0, 0, 0], 
        specular=[0, 0, 0],
        shininess=0,
        reflection=0.0,
        refraction=1
    )

    plane = Plane([0,1,0], [0,-0.3,0])
    plane.set_material(
        ambient=[1, 1, 1], 
        diffuse=[1, 1, 1], 
        specular=[1, 1, 1], 
        shininess=0, 
        reflection = 0
    )
    
    background = Plane([0,0,1], [0,0,-3])
    background.set_material(
        ambient=[0.05, 0.05, 0.3], 
        diffuse=[0.05, 0.1, 0.3], 
        specular=[0.1, 0.1, 0.3], 
        shininess=1000, 
        reflection=0.2)
    
    snowflakes = createSnowFlakes()
    snowMan = createSnowMan()
    xmasTree = createChristmasTree()
    smallTree = createSmallChristmasTree()

    #objects = [globe, plane, background] + snowflakes + snowMan + xmasTree + smallTree
    #objects = [globe, plane, background] + xmasTree
    #objects = [globe, plane, background] + snowMan + xmasTree
    #objects = [globe, plane, background] + snowflakes
    objects  = [globe, plane, background] + snowMan


    light1 = PointLight(intensity= np.array([1, 1, 1]),position=np.array([0,1,1]),kc=0.4,kl=0.4,kq=0.4)    
    light2 = PointLight(intensity= np.array([1, 1, 1]),position=np.array([-0.5, 0, 0]),kc=0.1,kl=0.1,kq=0.1) 

    lights =  [light1, light2]
    
    return camera, lights, objects

def createSnowFlakes():
    snowflakes = []
    num_flakes = 100  # Adjust number of snowflakes

    # Globe center and radius for reference
    globe_center = np.array([0, 0.4, 0.2])
    globe_radius = 0.5
    flake_radius = 0.01

    # Generate random snowflakes within the globe
    for i in range(num_flakes):
    # Generate a random position inside the globe
    # This uses a rejection method to ensure uniform distribution
        while True:
            # Random point in cube, then check if in sphere
            random_offset = np.random.uniform(-globe_radius, globe_radius, 3)
            if np.linalg.norm(random_offset) < globe_radius * 0.9: 
                break
        
        # Position the snowflake at globe center + offset
        position = globe_center + random_offset
        
        # Create snowflake and set material
        snowflake = Sphere(position, flake_radius)
        snowflake.set_material(
            ambient=[1, 1, 1],
            diffuse=[1, 1, 1], 
            specular=[1, 1, 1],  # Add some specular for sparkle
            shininess=100,
            reflection=0.1,      # Slight reflection for sparkle
            refraction=0.0
        )
        snowflakes.append(snowflake)

    return snowflakes

def createSnowMan():
    snowMan = []
    snow_bottom = Sphere([-0.5, -0.1, 0.2],0.2)
    snow_bottom.set_material(
        ambient=[1, 1, 1],
        diffuse=[1, 1, 1], 
        specular=[1, 1, 1],
        shininess=10,
        reflection= 0.0,
        refraction= 0.0
    )
    snowMan.append(snow_bottom)
    snow_middle = Sphere([-0.5, 0.2, 0.2],0.15)
    snow_middle.set_material(
        ambient=[1, 1, 1],
        diffuse=[1, 1, 1], 
        specular=[0, 0, 0],
        shininess=0,
        reflection= 0.0,
        refraction= 0.0
    )
    snowMan.append(snow_middle)
    snow_top = Sphere([-0.5, 0.43, 0.2],0.1)
    snow_top.set_material(
        ambient=[1, 1, 1],
        diffuse=[1, 1, 1], 
        specular=[0, 0, 0],
        shininess=0,
        reflection= 0.0,
        refraction= 0.0
    )
    snowMan.append(snow_top)
    return snowMan

def createChristmasTree():
    tree_triangles = []
    trunk_color = [0.55, 0.27, 0.07]  # Brown
    tree_color = [0.0, 0.5, 0.0]      # Green
    
    # Tree trunk (simple rectangular prism made of triangles)
    trunk_width = 0.08
    trunk_height = 0.2
    trunk_center = [0.35, -0.2, 0.2]  # Position near snowman
    
    # Create trunk using triangles (simplified as a rectangular structure)
    trunk_vertices = [
        # Front face
        [trunk_center[0]-trunk_width/2, trunk_center[1], trunk_center[2]+trunk_width/2],
        [trunk_center[0]+trunk_width/2, trunk_center[1], trunk_center[2]+trunk_width/2],
        [trunk_center[0]+trunk_width/2, trunk_center[1]+trunk_height, trunk_center[2]+trunk_width/2],
        [trunk_center[0]-trunk_width/2, trunk_center[1]+trunk_height, trunk_center[2]+trunk_width/2],
    ]
    
    # Create triangles for front face
    trunk_tri1 = Triangle(trunk_vertices[0], trunk_vertices[1], trunk_vertices[2])
    trunk_tri1.set_material(trunk_color, trunk_color, [0.1, 0.1, 0.1], 10, 0.0)
    trunk_tri2 = Triangle(trunk_vertices[0], trunk_vertices[2], trunk_vertices[3])
    trunk_tri2.set_material(trunk_color, trunk_color, [0.1, 0.1, 0.1], 10, 0.0)
    
    tree_triangles.extend([trunk_tri1, trunk_tri2])
    
    # Create the tree tiers (multiple cones stacked)
    num_tiers = 4
    tree_base = [trunk_center[0], trunk_center[1] + trunk_height, trunk_center[2]]
    max_radius = 0.3
    tree_height = 0.8
    tier_height = tree_height/num_tiers
    
    for tier in range(num_tiers):
        tier_base_y = tree_base[1] + (tier_height * tier)
        tier_top_y = tree_base[1] + (tier_height * (tier + 1))
        tier_radius = max_radius * (1 - tier/num_tiers)
        
        # Create circular arrangement of triangles
        num_sides = 8  # Octagonal pyramid for each tier
        for i in range(num_sides):
            angle1 = 2 * np.pi * i / num_sides
            angle2 = 2 * np.pi * (i+1) / num_sides
            
            # Calculate vertices for this triangle
            base1 = [
                tree_base[0] + tier_radius * np.cos(angle1),
                tier_base_y,
                tree_base[2] + tier_radius * np.sin(angle1)
            ]
            
            base2 = [
                tree_base[0] + tier_radius * np.cos(angle2),
                tier_base_y,
                tree_base[2] + tier_radius * np.sin(angle2)
            ]
            
            top = [tree_base[0], tier_top_y, tree_base[2]]
            
            # Create and add the triangle
            tree_tri = Triangle(base1, base2, top)
            tree_tri.set_material(
                tree_color, 
                tree_color, 
                [0.1, 0.1, 0.1], 
                30, 
                0.0
            )
            tree_triangles.append(tree_tri)
            if tier > 0:  # Don't add snow to bottom tier
                # Create a smaller triangle above the branch to represent snow
                snow_height = tier_height * 0.3  # Snow covers top 30% of branch
                snow_base_y = tier_top_y - snow_height
                
                # Create snow triangle (smaller than the branch triangle)
                snow_top = top
                snow_base1 = [
                    tree_base[0] + tier_radius * 0.8 * np.cos(angle1),
                    snow_base_y,
                    tree_base[2] + tier_radius * 0.8 * np.sin(angle1)
                ]
                
                snow_base2 = [
                    tree_base[0] + tier_radius * 0.8 * np.cos(angle2),
                    snow_base_y,
                    tree_base[2] + tier_radius * 0.8 * np.sin(angle2)
                ]
                
                # Create and add the snow triangle
                snow_tri = Triangle(snow_base1, snow_base2, snow_top)
                snow_tri.set_material(
                    [1.2, 1.2, 1.2], 
                    [1.2, 1.2, 1.2], 
                    [1, 1, 1],
                    50, 
                    0  
                )
                tree_triangles.append(snow_tri)
    
    return tree_triangles

def createSmallChristmasTree():
    tree_triangles = []
    trunk_color = [0.55, 0.27, 0.07]  # Same brown as original
    tree_color = [0.0, 0.5, 0.0]      # Same green as original
    
    # Smaller dimensions
    trunk_width = 0.05  # Reduced from 0.08
    trunk_height = 0.12  # Reduced from 0.2
    
    # Position on the other side of snowman
    trunk_center = [-0.1, -0.2, 0.2]  # Changed X from 0 to -0.7
    
    # Create simplified trunk (just one face)
    trunk_vertices = [
        [trunk_center[0]-trunk_width/2, trunk_center[1], trunk_center[2]+trunk_width/2],
        [trunk_center[0]+trunk_width/2, trunk_center[1], trunk_center[2]+trunk_width/2],
        [trunk_center[0]+trunk_width/2, trunk_center[1]+trunk_height, trunk_center[2]+trunk_width/2],
        [trunk_center[0]-trunk_width/2, trunk_center[1]+trunk_height, trunk_center[2]+trunk_width/2],
    ]
    
    # Create trunk triangles - same as original
    trunk_tri1 = Triangle(trunk_vertices[0], trunk_vertices[1], trunk_vertices[2])
    trunk_tri1.set_material(trunk_color, trunk_color, [0.1, 0.1, 0.1], 10, 0.0)
    trunk_tri2 = Triangle(trunk_vertices[0], trunk_vertices[2], trunk_vertices[3])
    trunk_tri2.set_material(trunk_color, trunk_color, [0.1, 0.1, 0.1], 10, 0.0)
    
    tree_triangles.extend([trunk_tri1, trunk_tri2])
    
    # Create the tree tiers - simpler structure with fewer tiers
    num_tiers = 3  # Reduced from 4
    tree_base = [trunk_center[0], trunk_center[1] + trunk_height, trunk_center[2]]
    max_radius = 0.18  # Reduced from 0.3
    tree_height = 0.45  # Reduced from 0.8
    tier_height = tree_height/num_tiers
    
    # Fewer sides for simpler geometry
    num_sides = 6  # Reduced from 8
    
    for tier in range(num_tiers):
        tier_base_y = tree_base[1] + (tier_height * tier)
        tier_top_y = tree_base[1] + (tier_height * (tier + 1))
        tier_radius = max_radius * (1 - tier/num_tiers)
        
        for i in range(num_sides):
            angle1 = 2 * np.pi * i / num_sides
            angle2 = 2 * np.pi * (i+1) / num_sides
            
            # Calculate vertices
            base1 = [
                tree_base[0] + tier_radius * np.cos(angle1),
                tier_base_y,
                tree_base[2] + tier_radius * np.sin(angle1)
            ]
            
            base2 = [
                tree_base[0] + tier_radius * np.cos(angle2),
                tier_base_y,
                tree_base[2] + tier_radius * np.sin(angle2)
            ]
            
            top = [tree_base[0], tier_top_y, tree_base[2]]
            
            # Create triangle with the same color as original
            tree_tri = Triangle(base1, base2, top)
            tree_tri.set_material(
                tree_color, 
                tree_color, 
                [0.1, 0.1, 0.1], 
                30, 
                0.0
            )
            tree_triangles.append(tree_tri)
    
    return tree_triangles