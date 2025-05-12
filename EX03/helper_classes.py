import numpy as np



# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    vec_L = normalize(vector)
    vec_N = normalize(axis)
    vec_R = vec_L - 2 * np.dot(vec_L, vec_N)*vec_N
     
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
        self.direction = direction
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    ### implemented point to light source
    def get_light_ray(self, intersection):
        return Ray(self.position, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        distance = self.get_distance_from_light(intersection)
        v = normalize(intersection - self.position)
        v_d = self.direction
        return self.intensity * np.dot(v, v_d) / (self.kc + self.kl * distance + self.kq * np.square(distance))

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
            if object.intersect(self) is None:
                continue

            t, _ = object.intersect(self)
            
            if t < min_distance:
                nearest_object = object
                min_distance = t
        
        if nearest_object is None:
            return None
        
        return min_distance, nearest_object


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection


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
        self.normal = self.compute_normal()
        self.plane = Plane(self.compute_normal(), a)

    # computes normal to the trainagle surface. Pay attention to its direction!
    def compute_normal(self):
        vec_ba = self.a - self.b
        vec_bc = self.c - self.b
        return normalize(np.cross(vec_bc, vec_ba))

    def intersect(self, ray: Ray):
        if self.plane.intersect(ray) is None:
            return None
        t, _ = self.plane.intersect(ray)
        point = np.array(ray.origin + t * ray.direction)
        n = self.normal * np.linalg.norm(self.normal) 
        area = np.linalg.norm(n) / 2
        vec_pb = self.b - point
        vec_pc = self.c - point
        vec_pa = self.a - point
        alpha = np.linalg.norm(np.cross(vec_pb, vec_pc)) / (2 * area)
        beta = np.linalg.norm(np.cross(vec_pc, vec_pa)) / (2 * area)
        gamma = 1 - alpha - beta
        ratios = np.array([alpha, beta, gamma])
        intersected = np.all((0 <= ratios) & (ratios <= 1)) and np.isclose(ratios.sum(), 1, atol=1e-6)
        if intersected:
            return t, self

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
            point1 = self.v_list(triangle[0])
            point2 = self.v_list(triangle[1])
            point3 = self.v_list(triangle[2])
            l.append(Triangle(point1, point2, point3))

        return l

    def apply_materials_to_triangles(self):
        for triangle in self.triangle_list:
            triangle.set_material(self.ambient, self.diffuse, self.specular, self.shininess, self.reflection)

    def intersect(self, ray: Ray):
        min_triangle = None
        min_t = np.inf
        for triangle in self.triangle_list:
            if triangle.intersect(ray) is None:
                continue
            t, _ = triangle.intersect(ray)
            if t < min_t:
                min_t = t
                min_triangle = triangle
        if min_triangle is None:
            return None
        else:
            return min_t, min_triangle

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        v_x, v_y, v_z = ray.direction[0], ray.direction[1], ray.direction[2]
        o_x, o_y, o_z = ray.origin[0], ray.origin[1], ray.origin[2]
        c_x, c_y, c_z = self.center[0], self.center[1], self.center[2]
        k_x, k_y, k_z = o_x - c_x, o_y - c_y, o_z - c_z
        
        a = v_x ** 2 + v_y ** 2 + v_z ** 2
        b = 2 * k_x * v_x + 2 *  k_y * v_y + 2 * k_z * v_z
        c = k_x ** 2 + k_y ** 2 + k_z ** 2

        discr = b ** 2 - 4 * a * c

        if discr < 0:
            return None
        
        else:
            t_1 = (-b + np.sqrt(discr)) / 2 * a
            t_2 = (-b - np.sqrt(discr)) / 2 * a

        return np.min(t_1, t_2)