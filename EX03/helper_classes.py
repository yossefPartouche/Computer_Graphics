import numpy as np

# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    vec_L = normalize(vector)
    vec_N = normalize(axis)
    vec_R = vec_L - 2 * np.dot(vec_L, vec_N) * vec_N
     
    return vec_R

## Lights
class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):
    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = direction

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection_point):
        ##  returns ray from point to light source
        return Ray(intersection_point, -self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(self.position - intersection)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        # calculate distance between light source and intersection 
        # calculate and return the light intensity based on kc, kl, kq
        distance = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl * distance + self.kq * np.square(distance))

class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.direction = normalize(direction)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    ### implemented point to light source
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(self.position - intersection)

    def get_intensity(self, intersection):
        distance = self.get_distance_from_light(intersection)
        v = normalize(intersection - self.position)
        v_d = self.direction
        return self.intensity * np.dot(v, v_d) / (self.kc + self.kl * distance + self.kq * (distance ** 2))

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        #intersections = None
        nearest_object = None
        min_distance = np.inf

        for object in objects:
            obj_intersect = object.intersect(self)

            if obj_intersect is None:
                continue

            t, obj = obj_intersect
            
            if t < min_distance:
                nearest_object = obj
                min_distance = t
        
        if nearest_object is None:
            return None
        
        return min_distance, nearest_object


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection, refraction=0):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection
        self.refraction = refraction


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction) + 1e-6)
        if t > 0:
            return t, self
        else:
            return None
        
    def getNormal(self, _):
        return self.normal


class Triangle(Object3D):
    """
        C
        /\
       /  \
    A /____\ B

    The fornt face of the triangle is A -> B -> C.
    
    """
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        vec_ba = self.a - self.b
        vec_bc = self.c - self.b
        cross = np.cross(vec_bc, vec_ba)
        self.area2 = np.linalg.norm(cross) # 2 * area
        self.normal = normalize(cross)
        self.plane = Plane(self.normal, a)
        
    # computes normal to the trainagle surface. Pay attention to its direction!
    #def compute_normal(self):
       #return normalize(self.cross)

    def intersect(self, ray: Ray):
        obj_intersect = self.plane.intersect(ray)
        if obj_intersect is None:
            return None
        
        t, _ = obj_intersect
        point = ray.origin + t * ray.direction
        
        vec_pb = self.b - point
        vec_pc = self.c - point
        vec_pa = self.a - point
        
        alpha = np.linalg.norm(np.cross(vec_pb, vec_pc)) / self.area2
        beta = np.linalg.norm(np.cross(vec_pc, vec_pa)) / self.area2
        gamma = 1 - alpha - beta
        
        ratios = np.array([alpha, beta, gamma])
        intersected = np.all((0 <= ratios) & (ratios <= 1)) and np.isclose(ratios.sum(), 1, atol=1e-6)
        if intersected:
            return t, self
        return None
        
    def getNormal(self, _):
        return self.normal

class Diamond(Object3D):
    """     
            D
            /\*\
           /==\**\
         /======\***\
       /==========\***\
     /==============\****\
   /==================\*****\
A /&&&&&&&&&&&&&&&&&&&&\ B &&&/ C
   \==================/****/
     \==============/****/
       \==========/****/
         \======/***/
           \==/**/
            \/*/
             E 
    
    Similar to Traingle, every from face of the diamond's faces are:
        A -> B -> D
        B -> C -> D
        A -> C -> B
        E -> B -> A
        E -> C -> B
        C -> E -> A
    """
    def __init__(self, v_list):
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        l = []
        t_idx = [
                [0,1,3],
                [1,2,3],
                [0,3,2],
                [4,1,0],
                [4,2,1],
                [2,4,0]]
        
        for triangle in t_idx:
            point1 = self.v_list[triangle[0]]
            point2 = self.v_list[triangle[1]]
            point3 = self.v_list[triangle[2]]
            l.append(Triangle(point1, point2, point3))

        return l

    def apply_materials_to_triangles(self):
        for triangle in self.triangle_list:
            triangle.set_material(self.ambient, self.diffuse, self.specular, self.shininess, self.reflection, self.refraction)

    def intersect(self, ray: Ray):
        obj_intersect = ray.nearest_intersected_object(self.triangle_list)
        if obj_intersect is None:
            return None
        else:
            return obj_intersect

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        k = ray.origin - self.center
        
        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(k, ray.direction)
        c = np.dot(k, k) - self.radius ** 2

        discr = b ** 2 - 4 * a * c

        if discr < 0:
            return None
        
        t_1 = (-b + np.sqrt(discr)) / (2 * a)
        t_2 = (-b - np.sqrt(discr)) / (2 * a)

        t = min(filter(lambda t: t > 1e-6, (t_1, t_2)), default=None)

        if t is None:
            return None
        
        return t, self
    
    def getNormal(self, point):
        return normalize(point - self.center)