from time import time
a = time()
from PIL import Image, ImageFilter
import numpy as np
def extract(cond, x):
    if isinstance(x, (int, float, complex)):
        return x
    return np.extract(cond, x)
class Ray:    
    def __init__(self,origin, dir, depth, n, reflections, transmissions, diffuse_reflections):
        self.origin  = origin   
        self.dir = dir
        self.depth = depth     
        self.n = n
        self.reflections = reflections
        self.transmissions = transmissions
        self.diffuse_reflections = diffuse_reflections
    def extract(self,hit_check):
        return Ray(self.origin.extract(hit_check), self.dir.extract(hit_check), self.depth,  self.n.extract(hit_check), self.reflections, self.transmissions,self.diffuse_reflections)
class Hit:
    def __init__(self, distance, orientation, material, collider,surface):
        self.distance = distance
        self.orientation = orientation
        self.material = material
        self.collider = collider
        self.surface = surface
        self.u = None
        self.v = None
        self.N = None
        self.point = None
    def get_uv(self):
        if self.u is None:
            self.u, self.v = self.collider.assigned_primitive.get_uv(self)
        return self.u, self.v
    def get_normal(self):
        if self.N is None:
            self.N = self.collider.get_N(self)
        return self.N
def get_raycolor(ray, scene):
    distances, hit_orientation = zip(*[s.intersect(ray.origin, ray.dir) for s in scene.collider_list])
    nearest = np.minimum.reduce(distances)
    color = vec3(0., 0., 0.)
    for (coll, dis , orient) in zip(scene.collider_list, distances, hit_orientation):
        hit_check = (nearest != FARAWAY) & (dis == nearest)
        if np.any(hit_check):
            color += coll.assigned_primitive.material.get_color(scene,  ray.extract(hit_check), Hit(extract(hit_check,dis) , extract(hit_check,orient), coll.assigned_primitive.material, coll, coll.assigned_primitive)).place(hit_check)
    return color
class Primitive:
    def __init__(self, center, material, max_ray_depth = 1, shadow = True):
        self.center = center
        self.material = material
        self.material.assigned_primitive = self
        self.shadow = shadow
        self.collider_list = []
        self.max_ray_depth = max_ray_depth
    def rotate(self, delta, u):
        u = u.normalize()
        delta = delta/180 *np.pi
        cosdelta = np.cos(delta)
        sindelta = np.sqrt(1-cosdelta**2) * np.sign(delta)
        M = np.array([[cosdelta + u.x*u.x * (1-cosdelta),      u.x*u.y*(1-cosdelta) - u.z*sindelta,         u.x*u.z*(1-cosdelta) +u.y*sindelta],[u.y*u.x*(1-cosdelta) + u.z*sindelta,        cosdelta + u.y**2 * (1-cosdelta),       u.y*u.z*(1-cosdelta) -u.x*sindelta],[u.z*u.x*(1-cosdelta) -u.y*sindelta,             u.z*u.y*(1-cosdelta) + u.x*sindelta,         cosdelta + u.z*u.z*(1-cosdelta)]])
        for c in self.collider_list:
            c.rotate(M, self.center)
class Sphere(Primitive):
    def __init__(self,center,  material, radius, max_ray_depth = 5, shadow = True):
        super().__init__(center,  material, max_ray_depth, shadow = shadow)
        self.collider_list += [Sphere_Collider(assigned_primitive = self, center = center, radius = radius)]
        self.bounded_sphere_radius = radius
    def get_uv(self, hit):
        return hit.collider.get_uv(hit)
class Plane(Primitive):
    def __init__(self,center,  material, width,height, u_axis, v_axis, max_ray_depth = 5, shadow = True):
        super().__init__(center,  material, max_ray_depth,shadow = shadow)
        self.collider_list += [Plane_Collider(assigned_primitive = self, center = center, u_axis = u_axis, v_axis = v_axis, w= width/2, h=height/2)]
        self.width = width
        self.height = height
        self.bounded_sphere_radius = np.sqrt((width/2)**2 + (height/2)**2)
    def get_uv(self, hit):
        return hit.collider.get_uv(hit)
class Collider:
    def __init__(self,assigned_primitive, center):
        self.assigned_primitive = assigned_primitive
        self.center = center
class Plane_Collider(Collider):
    def __init__(self, u_axis, v_axis, w, h, uv_shift = (0.,0.),**kwargs):
        super().__init__(**kwargs)
        self.normal = u_axis.cross(v_axis).normalize()
        self.w = w
        self.h = h
        self.u_axis = u_axis
        self.v_axis = v_axis
        self.uv_shift = uv_shift
        self.inverse_basis_matrix =  np.array([[self.u_axis.x,self.v_axis.x,self.normal.x],[self.u_axis.y,self.v_axis.y,self.normal.y],[self.u_axis.z,self.v_axis.z,self.normal.z]])
        self.basis_matrix = self.inverse_basis_matrix.T
    def intersect(self, O, D):
        NdotD = self.normal.dot(D)
        NdotD = np.where(NdotD == 0., NdotD + 0.0001, NdotD)
        NdotC_O = self.normal.dot(self.center - O)
        d =  D * NdotC_O / NdotD
        dis =  d.length()
        M_C = O +  d - self.center
        hit_inside = (np.abs(self.u_axis.dot(M_C))  <= self.w) & (np.abs(self.v_axis.dot(M_C)) <= self.h) & (NdotC_O * NdotD > 0)
        hit_UPWARDS  = (NdotD < 0)
        return np.select([hit_inside & hit_UPWARDS,hit_inside & np.logical_not(hit_UPWARDS),True] , [[dis, np.tile(UPWARDS, dis.shape) ], [dis,np.tile(UPDOWN, dis.shape)], FARAWAY])
    def rotate(self,M, center):
        self.u_axis = self.u_axis.matmul(M)
        self.v_axis = self.v_axis.matmul(M)
        self.normal = self.normal.matmul(M)
        self.center = center + (self.center-center).matmul(M)
    def get_uv(self, hit):
        M_C = hit.point - self.center
        return ((self.u_axis.dot(M_C)/self.w + 1 ) /2 + self.uv_shift[0]),((self.v_axis.dot(M_C)/self.h + 1 ) /2  + self.uv_shift[1])
    def get_Normal(self, hit):
        return self.normal
class Sphere_Collider(Collider):
    def __init__(self,  radius, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
    def intersect(self, O, D):
        b = 2 * D.dot(O - self.center)
        disc = (b ** 2) - (4 * (self.center.square_length() + O.square_length() - 2 * self.center.dot(O) - (self.radius * self.radius)))
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        NdotD = (((O + D * h) - self.center) * (1. / self.radius) ).dot(D)
        return np.select([(disc > 0) & (h > 0) & (NdotD > 0),(disc > 0) & (h > 0) & (NdotD < 0),True] , [[h, np.tile(UPDOWN, h.shape)], [h,np.tile(UPWARDS, h.shape)], FARAWAY])
    def get_Normal(self, hit):
        return (hit.point - self.center) * (1. / self.radius)
    def get_uv(self, hit):
        M_C = (hit.point - self.center) / self.radius
        return (np.arctan2(M_C.z, M_C.x) + np.pi) / (2*np.pi),(np.arcsin(M_C.y) + np.pi/2) / np.pi
class PointLight:
    def __init__(self, pos, color):
        self.pos = pos
        self.color = color
    def get_L(self):
        return (self.pos - M)*(1./(dist_light))
    def get_distance(self, M):
        return numpy.sqrt((self.pos - M).dot(self.pos - M))
    def get_irradiance(self,dist_light, NdotL):
        return self.color * NdotL/(dist_light**2.) * 100
class DirectionalLight:
    def __init__(self, Ldir, color):
        self.Ldir = Ldir
        self.color = color
    def get_L(self):
        return self.Ldir
    def get_distance(self, M):
        return SKYBOX_DISTANCE
    def get_irradiance(self, dist_light, NdotL):
        return self.color * NdotL
class Camera():
    def __init__(self, look_from, look_at, screen_width = 400 ,screen_height = 300,  field_of_view = 90., aperture = 0., focal_distance = 1.):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.aspect_ratio = float(screen_width) / screen_height
        self.look_from = look_from
        self.look_at = look_at
        self.camera_width = np.tan(field_of_view * np.pi / 180 / 2.) * 2.
        self.camera_height = self.camera_width / self.aspect_ratio
        self.cameraFwd = (look_at - look_from).normalize()
        self.cameraRight = (self.cameraFwd.cross(vec3(0., 1., 0.))).normalize()
        self.cameraUp = self.cameraRight.cross(self.cameraFwd)
        self.lens_radius = aperture / 2.
        self.focal_distance = focal_distance
        self.near = .1
        self.far = 100.
        self.x = np.linspace(-self.camera_width / 2., self.camera_width / 2., self.screen_width)
        self.y = np.linspace(self.camera_height / 2., -self.camera_height / 2., self.screen_height)
        xx, yy = np.meshgrid(self.x, self.y)
        self.x = xx.flatten()
        self.y = yy.flatten()
    def get_ray(self,n):
        x = self.x + (np.random.rand(len(self.x)) - 0.5) * self.camera_width / (self.screen_width)
        y = self.y + (np.random.rand(len(self.y)) - 0.5) * self.camera_height / (self.screen_height)
        ray_origin = self.look_from + self.cameraRight * x * self.near + self.cameraUp * y * self.near
        return Ray(origin=ray_origin, dir=(self.look_from + self.cameraUp * y * self.focal_distance +self.cameraRight * x * self.focal_distance +self.cameraFwd * self.focal_distance - ray_origin).normalize(), depth=0, n=n, reflections=0, transmissions=0, diffuse_reflections=0)
class vec3():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"
    def __add__(self, v):
        if isinstance(v, vec3):
            return vec3(self.x + v.x, self.y + v.y, self.z + v.z)
        return vec3(self.x + v, self.y + v, self.z + v)
    def __radd__(self, v):
        if isinstance(v, vec3):
            return vec3(self.x + v.x, self.y + v.y, self.z + v.z)
        return vec3(self.x + v, self.y + v, self.z + v)
    def __sub__(self, v):
        if isinstance(v, vec3):
            return vec3(self.x - v.x, self.y - v.y, self.z - v.z)
        return vec3(self.x - v, self.y - v, self.z - v)
    def __rsub__(self, v):
        if isinstance(v, vec3):
            return vec3(v.x - self.x, v.y - self.y ,  v.z - self.z)
        return vec3(v  - self.x, v  - self.y ,  v - self.z)
    def __mul__(self, v):
        if isinstance(v, vec3):
            return vec3(self.x * v.x , self.y *  v.y , self.z * v.z )
        return vec3(self.x * v, self.y * v, self.z * v)
    def __rmul__(self, v):
        if isinstance(v, vec3):
            return vec3(v.x *self.x  , v.y * self.y, v.z * self.z )
        return vec3(v *self.x  , v * self.y, v * self.z )
    def __truediv__(self, v):
        if isinstance(v, vec3):
            return vec3(self.x / v.x , self.y /  v.y , self.z / v.z )
        return vec3(self.x / v, self.y / v, self.z / v)
    def __rtruediv__(self, v):
        if isinstance(v, vec3):
            return vec3(v.x / self.x, v.y / self.y, v.z / self.z)
        return vec3(v / self.x, v / self.y, v / self.z)
    def __abs__(self):
        return vec3(np.abs(self.x), np.abs(self.y), np.abs(self.z))
    def real(v):
        return vec3(np.real(v.x), np.real(v.y), np.real(v.z))
    def imag(v):
        return vec3(np.imag(v.x), np.imag(v.y), np.imag(v.z))
    def yzx(self):
        return vec3(self.y, self.z, self.x)
    def xyz(self):
        return vec3(self.x, self.y, self.z)
    def zxy(self):
        return vec3(self.z, self.x, self.y)
    def xyz(self):
        return vec3(self.x, self.y, self.z)
    def average(self):
        return (self.x + self.y +  self.z)/3
    def matmul(self, matrix):
        if isinstance(self.x, (int, float, complex)):
            array = np.dot(matrix,self.to_array())
            return vec3(array[0],array[1],array[2])
        array = np.tensordot(matrix,self.to_array() , axes=([1,0]))
        return vec3(array[0],array[1],array[2])
    def change_basis(self, new_basis):
        return vec3(self.dot(new_basis[0]),  self.dot(new_basis[1]),   self.dot(new_basis[2]))
    def __pow__(self, a):
        return vec3(self.x**a, self.y**a, self.z**a)
    def dot(self, v):
        return self.x*v.x + self.y*v.y + self.z*v.z
    def exp(v):
        return vec3(np.exp(v.x) , np.exp(v.y) ,np.exp(v.z))
    def sqrt(v):
        return vec3(np.sqrt(v.x) , np.sqrt(v.y) ,np.sqrt(v.z))
    def to_array(self):
        return np.array([self.x , self.y , self.z])
    def cross(self, v):
        return vec3(self.y*v.z - self.z*v.y, -self.x*v.z + self.z*v.x,  self.x*v.y - self.y*v.x)
    def length(self):
        return np.sqrt(self.dot(self))
    def square_length(self):
        return self.dot(self)
    def normalize(self):
        mag = self.length()
        return self * (1.0 / np.where(mag == 0, 1, mag))
    def components(self):
        return (self.x, self.y, self.z)
    def extract(self, cond):
        return vec3(extract(cond, self.x),extract(cond, self.y),extract(cond, self.z))
    def where(cond, out_true, out_false):
        return vec3(np.where(cond, out_true.x, out_false.x),np.where(cond, out_true.y, out_false.y),np.where(cond, out_true.z, out_false.z))
    def select(mask_list, out_list):
        return vec3(np.select(mask_list, [i.x for i in out_list]),np.select(mask_list, [i.y for i in out_list]),np.select(mask_list, [i.z for i in out_list]))
    def clip(self, min, max):
        return vec3(np.clip(self.x, min, max),np.clip(self.y, min, max),np.clip(self.z, min, max))
    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r
    def repeat(self, n):
        return vec3(np.repeat(self.x , n), np.repeat(self.y , n),   np.repeat(self.z , n))
    def reshape(self, *newshape):
        return vec3(self.x.reshape(*newshape), self.y.reshape(*newshape),   self.z.reshape(*newshape))
    def shape(self, *newshape):
        if isinstance(self.x, (int, float, complex)):
            return 1
        return self.x.shape
    def mean(self, axis):
        return vec3(np.mean(self.x,axis = axis), np.mean(self.y,axis = axis),   np.mean(self.z,axis = axis))
    def __eq__(self, other):
        return (self.x == other.x)  &  (self.y == other.y) & (self.z == other.z)
class SkyBox(Primitive):
    def __init__(self, cubemap, center = vec3(0.,0.,0.), light_intensity = 0.0, blur = 0.0):
        super().__init__(center,  SkyBox_Material(cubemap, light_intensity, blur), shadow = False)
        l = SKYBOX_DISTANCE
        self.light_intensity = light_intensity
        self.collider_list += [Cuboid_Collider(assigned_primitive = self, center = center, width = 2*l, height =2*l ,length =2*l )]
    def get_uv(self, hit):
        u,v = hit.collider.get_uv(hit)
        return u/4,v/3
class Scene():
    def __init__(self, ambient_color = vec3(0.01, 0.01, 0.01), n = vec3(1.0,1.0,1.0)) :
        self.scene_primitives = []
        self.collider_list = []
        self.shadowed_collider_list = []
        self.Light_list = []
        self.importance_sampled_list = []
        self.ambient_color = ambient_color
        self.n = n
        self.importance_sampled_list = []
    def add_Camera(self, look_from, look_at, **kwargs):
        self.camera = Camera(look_from, look_at, **kwargs)
    def add_PointLight(self, pos, color):
        self.Light_list += [PointLight(pos, color)]
    def add_DirectionalLight(self, Ldir, color):
        self.Light_list += [DirectionalLight(Ldir.normalize() , color)]
    def add(self,primitive, importance_sampled = False):
        self.scene_primitives += [primitive]
        self.collider_list += primitive.collider_list
        if importance_sampled == True:
            self.importance_sampled_list += [primitive]
        if primitive.shadow == True:
            self.shadowed_collider_list += primitive.collider_list
    def add_Background(self, img, light_intensity = 0.0, blur =0.0 , spherical = False):
        primitive = None
        if spherical == False:
            primitive = SkyBox(img, light_intensity = light_intensity, blur = blur)
        else:
            primitive = Panorama(img, light_intensity = light_intensity, blur = blur)
        self.scene_primitives += [primitive]
        self.collider_list += primitive.collider_list
    def render(self, samples_per_pixel, progress_bar = False):
        color_RGBlinear = vec3(0.,0.,0.)
        if progress_bar:
            for i in range(samples_per_pixel):
                color_RGBlinear += get_raycolor(self.camera.get_ray(self.n), scene = self)
                print(i)
        else:
            for i in range(samples_per_pixel):
                color_RGBlinear += get_raycolor(self.camera.get_ray(self.n), scene = self)
        color_RGBlinear = color_RGBlinear/samples_per_pixel
        rgb_linear = color_RGBlinear.to_array()
        rgb = np.where( rgb_linear <= 0.00304,  12.92 * rgb_linear,  1.055 * np.power(rgb_linear, 1.0/2.4) - 0.055)
        rgb_max =  np.amax(rgb, axis=0)  + 0.00001
        intensity_cutoff = 1.0
        color = np.where(rgb_max > intensity_cutoff, rgb * intensity_cutoff / (rgb_max), rgb)
        img_RGB = []
        for c in color:
            img_RGB += [Image.fromarray((255 * np.clip(c, 0, 1).reshape((self.camera.screen_height, self.camera.screen_width))).astype(np.uint8), "L") ]
        return Image.merge("RGB", img_RGB)
def load_image_as_linear_sRGB(path, blur = 0.0):
    img = Image.open(path)
    if blur != 0.0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    img_array = np.asarray(img)/256.
    img_sRGB_linear_array = np.where( img_array <= 0.03928,  img_array / 12.92,  np.power((img_array + 0.055) / 1.055,  2.4))
    return img_sRGB_linear_array
class solid_color:
    def __init__(self,color):
        self.color = color
    def get_color(self, hit):
        return self.color
class image:
    def __init__(self,img, repeat = 1.0):
        self.img = load_image_as_linear_sRGB(img)
        self.repeat = repeat
    def get_color(self, hit):
        u,v = hit.get_uv()
        im = self.img[-((v * self.img.shape[0]*self.repeat ).astype(int)% self.img.shape[0]) , (u   * self.img.shape[1]*self.repeat).astype(int) % self.img.shape[1]  ].T
        color = vec3(im[0],im[1],im[2])
        return color
class Material():
    def __init__(self,normalmap = None):
        if normalmap != None:
            normalmap = load_image(normalmap)
        self.normalmap = normalmap
    def get_Normal(self, hit):
        N_coll = hit.collider.get_Normal(hit)
        if self.normalmap is not None:
            u,v = hit.get_uv()
            im = self.normalmap[-((v * self.normalmap.shape[0]*self.repeat ).astype(int)% self.normalmap.shape[0]) , (u   * self.normalmap.shape[1]*self.repeat).astype(int) % self.normalmap.shape[1]  ].T
            N_map = (vec3(im[0] - 0.5,im[1] - 0.5,im[2] - 0.5)) * 2.0
            return N_map.matmul(hit.collider.inverse_basis_matrix).normalize()*hit.orientation
        else:
            return N_coll*hit.orientation
    def set_normalmap(self, normalmap,repeat= 1.0):
        self.normalmap = load_image(normalmap)
        self.repeat = repeat
    def get_color(self, scene, ray, hit):
        pass
class SkyBox_Material(Material):
    def __init__(self, cubemap, light_intensity, blur):
        self.texture = load_image_as_linear_sRGB(cubemap)
        if light_intensity != 0.0:
            self.lightmap = load_image(cubemap)
        if blur != 0.0:
            self.blur_image = blur_skybox(load_image(cubemap), blur, cubemap)
        self.blur = blur
        self.light_intensity = light_intensity
        self.repeat = 1.0
    def get_texture_color(self, hit, ray):
        u,v = hit.get_uv()
        if (self.blur != 0.0) :
            im = self.blur_image[-((v * self.blur_image.shape[0]*self.repeat ).astype(int)% self.blur_image.shape[0]) , (u   * self.blur_image.shape[1]*self.repeat).astype(int) % self.blur_image.shape[1]  ].T
        else:
            im = self.texture[-((v * self.texture.shape[0]*self.repeat ).astype(int)% self.texture.shape[0]) , (u   * self.texture.shape[1]*self.repeat).astype(int) % self.texture.shape[1]  ].T        
        if (ray.depth != 0) and (self.light_intensity != 0.0):
            ls = self.lightmap[-((v * self.texture.shape[0]*self.repeat ).astype(int)% self.texture.shape[0]) , (u   * self.texture.shape[1]*self.repeat).astype(int) % self.texture.shape[1]  ].T       
            color = vec3(im[0] + self.light_intensity * ls[0],  im[1] + self.light_intensity * ls[1],  im[2] + self.light_intensity * ls[2])
        else:
            color = vec3(im[0] ,  im[1] ,  im[2] )
        return color
    def get_color(self, scene, ray, hit):
        hit.point = (ray.origin + ray.dir * hit.distance)
        return hit.material.get_texture_color(hit,ray)
class Cuboid(Primitive):
    def __init__(self,center,  material, width,height, length,max_ray_depth = 5, shadow = True):
        super().__init__(center,  material,  max_ray_depth, shadow = shadow)
        self.width = width
        self.height = height
        self.length = length
        self.bounded_sphere_radius = np.sqrt((self.width/2)**2 + (self.height/2)**2 + (self.length/2)**2)
        self.collider_list += [Cuboid_Collider(assigned_primitive = self, center = center, width = width, height =height ,length =length )]
    def get_uv(self, hit):
        u,v = hit.collider.get_uv(hit)
        u,v = u/4,v/3
        return u,v
class Cuboid_Collider(Collider):
    def __init__(self, width, height,length,**kwargs):
        super().__init__(**kwargs)
        self.lb = self.center - vec3(width/2, height/2, length/2)
        self.rt = self.center + vec3(width/2, height/2, length/2)
        self.lb_local_basis = self.lb
        self.rt_local_basis = self.rt
        self.width = width
        self.height = height
        self.length = length
        self.ax_w = vec3(1.,0.,0.)
        self.ax_h = vec3(0.,1.,0.)
        self.ax_l = vec3(0.,0.,1.)
        self.inverse_basis_matrix = np.array([[self.ax_w.x,       self.ax_h.x,         self.ax_l.x],
                                              [self.ax_w.y,       self.ax_h.y,         self.ax_l.y],
                                              [self.ax_w.z,       self.ax_h.z,         self.ax_l.z]])
        self.basis_matrix = self.inverse_basis_matrix.T
    def rotate(self,M, center):
        self.ax_w = self.ax_w.matmul(M)
        self.ax_h = self.ax_h.matmul(M)
        self.ax_l = self.ax_l.matmul(M)
        self.inverse_basis_matrix = np.array([[self.ax_w.x,       self.ax_h.x,         self.ax_l.x],
                                              [self.ax_w.y,       self.ax_h.y,         self.ax_l.y],
                                              [self.ax_w.z,       self.ax_h.z,         self.ax_l.z]])
        self.basis_matrix = self.inverse_basis_matrix.T
        self.lb = center + (self.lb-center).matmul(M)
        self.rt = center + (self.rt-center).matmul(M)
        self.lb_local_basis = self.lb.matmul(self.basis_matrix)
        self.rt_local_basis = self.rt.matmul(self.basis_matrix)
    def intersect(self, O, D):
        O_local_basis = O.matmul(self.basis_matrix)
        D_local_basis = D.matmul(self.basis_matrix)
        dirfrac = 1.0 / D_local_basis
        t1 = (self.lb_local_basis.x - O_local_basis.x)*dirfrac.x
        t2 = (self.rt_local_basis.x - O_local_basis.x)*dirfrac.x
        t3 = (self.lb_local_basis.y - O_local_basis.y)*dirfrac.y
        t4 = (self.rt_local_basis.y - O_local_basis.y)*dirfrac.y
        t5 = (self.lb_local_basis.z - O_local_basis.z)*dirfrac.z
        t6 = (self.rt_local_basis.z - O_local_basis.z)*dirfrac.z
        tmin = np.maximum(np.maximum(np.minimum(t1, t2), np.minimum(t3, t4)), np.minimum(t5, t6))
        tmax = np.minimum(np.minimum(np.maximum(t1, t2), np.maximum(t3, t4)), np.maximum(t5, t6))
        mask1 = (tmax < 0) | (tmin > tmax)
        mask2 = tmin < 0
        return np.select([mask1,mask2,True] , [FARAWAY , [tmax,  np.tile(UPDOWN, tmin.shape)] ,  [tmin,  np.tile(UPWARDS, tmin.shape)]])
    def get_Normal(self, hit):
        P = (hit.point-self.center).matmul(self.basis_matrix)
        absP = vec3(1./self.width, 1./self.height, 1./self.length)*np.abs(P)
        Pmax = np.maximum(np.maximum(absP.x, absP.y), absP.z)
        P.x = np.where(Pmax == absP.x, np.sign(P.x),  0.)
        P.y = np.where(Pmax == absP.y, np.sign(P.y),  0.)
        P.z = np.where(Pmax == absP.z, np.sign(P.z),  0.)
        return P.matmul(self.inverse_basis_matrix)
    def get_uv(self, hit):
        hit.N = self.get_Normal(hit)
        M_C = hit.point - self.center
        BOTTOM = (hit.N == vec3(0.,-1.,0.))
        TOP =  (hit.N == vec3(0., 1.,0.))
        RIGHT =  (hit.N == vec3(1.,0.,0.))
        LEFT = (hit.N == vec3(-1.,0.,0.) )
        FRONT = (hit.N == vec3(0.,0.,1.))
        BACK = (hit.N == vec3(0.,0.,-1.))
        u = np.select([BOTTOM , TOP,  RIGHT, LEFT , FRONT , BACK],  [((self.ax_w.dot(M_C)/self.width*2 *0.985  + 1 ) /2  + 1),   ((self.ax_w.dot(M_C)/self.width*2 *0.985  + 1 ) /2  + 1),   ((self.ax_l.dot(M_C)/self.width*2 *0.985  + 1 ) /2  + 2),    (((self.ax_l*-1).dot(M_C)/self.width*2 *0.985  + 1 ) /2  + 0),    (((self.ax_w*-1).dot(M_C)/self.width*2 *0.985  + 1 ) /2  + 3),     (( self.ax_w.dot(M_C)/self.width*2 *0.985  + 1 ) /2  + 1)])
        v = np.select([BOTTOM , TOP,  RIGHT, LEFT , FRONT , BACK],  [(((self.ax_l*-1).dot(M_C)/self.width*2 *0.985  + 1 ) /2  + 0),   ((self.ax_l.dot(M_C)/self.width*2 *0.985  + 1 ) /2  + 2),   ((self.ax_h.dot(M_C)/self.width*2 *0.985  + 1 ) /2  + 1),    (((self.ax_h).dot(M_C)/self.width*2 *0.985  + 1 ) /2  + 1),    (((self.ax_h).dot(M_C)/self.width*2 *0.985  + 1 ) /2  + 1),     (( self.ax_h.dot(M_C)/self.width*2 *0.985  + 1 ) /2  + 1)])
        return u,v
class Glossy(Material):
    def __init__(self, diff_color, roughness, spec_coeff, diff_coeff, n, **kwargs):
        super().__init__(**kwargs)
        if isinstance(diff_color, vec3):
            self.diff_texture = solid_color(diff_color)
        else:
            self.diff_texture = diff_color
        self.roughness = roughness
        self.diff_coeff = diff_coeff
        self.spec_coeff = spec_coeff
        self.n = n
    def get_color(self, scene, ray, hit):
        hit.point = (ray.origin + ray.dir * hit.distance)
        N = hit.material.get_Normal(hit)
        diff_color = self.diff_texture.get_color(hit)* self.diff_coeff
        color = scene.ambient_color * diff_color
        V = ray.dir*-1.
        nudged = hit.point + N * .000001
        for light in scene.Light_list:
            L = light.get_L()
            dist_light = light.get_distance(hit.point)
            NdotL = np.maximum(N.dot(L), 0.)
            lv = light.get_irradiance(dist_light, NdotL)
            H = (L + V).normalize()
            if not scene.shadowed_collider_list == []:
                inters = [s.intersect(nudged, L) for s in scene.shadowed_collider_list]
                light_distances, light_hit_orientation = zip(*inters)
                light_nearest = np.minimum.reduce(light_distances)
                seelight = (light_nearest >= dist_light)
            else:
                seelight = 1.
            color +=  diff_color * lv * seelight
            if self.roughness != 0.0:
                F0 = np.abs((ray.n - self.n)/(ray.n  + self.n))**2
                cosdelta = np.clip(V.dot(H), 0.0, 1.)
                F = F0 + (1. - F0)*(1.- cosdelta)**5
                a = 2./(self.roughness**2.) - 2.
                Dphong =  np.power(np.clip(N.dot(H), 0., 1.), a) * (a + 2.)/(2.*np.pi)
                color +=  F  * Dphong  /(4. * np.clip(N.dot(V) * NdotL, 0.001, 1.) ) * seelight * lv * self.spec_coeff
        if ray.depth < hit.surface.max_ray_depth:
            F0 = np.abs((scene.n - self.n)/(scene.n  + self.n))**2
            return color + (get_raycolor(Ray(nudged, (ray.dir - N * 2. * ray.dir.dot(N)).normalize(), ray.depth + 1, ray.n, ray.reflections + 1, ray.transmissions, ray.diffuse_reflections), scene))*(F0 + (1. - F0)*(1.- np.clip(V.dot(N),0.0,1.))**5)
        return color
class Mirror(Material):
    def __init__(self, reflection_coeff):
        super().__init__()
        self.reflection_coeff = reflection_coeff
    def get_color(self, scene, ray, hit):
        hit.point = (ray.origin + ray.dir * hit.distance)
        N = hit.material.get_Normal(hit)
        return get_raycolor(Ray(hit.point + N * 0.000001, (ray.dir - N * 2.0 * ray.dir.dot(N)).normalize(), ray.depth + 1, ray.n, ray.reflections + 1, ray.transmissions, ray.diffuse_reflections),scene,) * self.reflection_coeff
class Refractive(Material):
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n
    def get_color(self, scene, ray, hit):
        hit.point = (ray.origin + ray.dir * hit.distance)
        N = hit.material.get_Normal(hit)
        color = vec3(0.,0.,0.)
        V = ray.dir*-1.
        nudged = hit.point + N * .000001
        if ray.depth <hit.surface.max_ray_depth:
            n1 = ray.n
            n2 = vec3.where(hit.orientation== UPWARDS,self.n,scene.n)
            n1_div_n2 =  vec3.real(n1)/vec3.real(n2) 
            cosθi = V.dot(N)
            sin2θt = (n1_div_n2)**2 * (1.-cosθi**2)
            cosθt = vec3.sqrt(1. - (n1/n2)**2 * (1.-cosθi**2)  )
            r_per = (n1*cosθi - n2*cosθt)/(n1*cosθi + n2*cosθt)
            r_par = -1.*(n1*cosθt - n2*cosθi)/(n1*cosθt + n2*cosθi) 
            F = (np.abs(r_per)**2 + np.abs(r_par)**2)/2.
            reflected_ray_dir = (ray.dir - N * 2. * ray.dir.dot(N)).normalize()
            color += (get_raycolor(Ray(nudged, reflected_ray_dir, ray.depth + 1, ray.n, ray.reflections + 1, ray.transmissions, ray.diffuse_reflections), scene))*F
            n1_div_n2_aver = n1_div_n2.average()
            sin2θt = (n1_div_n2_aver)**2 * (1.-cosθi**2)
            non_TiR = (sin2θt <= 1.)
            if np.any(non_TiR):
                refracted_ray_dir = (ray.dir*(n1_div_n2_aver) + N*(n1_div_n2_aver * cosθi - np.sqrt(1-np.clip(sin2θt,0,1)))).normalize() 
                nudged = hit.point - N * .000001
                T = 1. - F
                refracted_color = (get_raycolor( Ray(nudged, refracted_ray_dir, ray.depth + 1, n2, ray.reflections, ray.transmissions + 1, ray.diffuse_reflections).extract(non_TiR), scene)  )*T.extract(non_TiR) 
                color += refracted_color.place(non_TiR)
            color = color *vec3.exp(-2.*vec3.imag(ray.n)*2.*np.pi/vec3(630,550,475) * 1e9* hit.distance)
        return color
UPDOWN = -1
UPWARDS = 1
FARAWAY = 1e+39
SKYBOX_DISTANCE = 1000000.0
green_glass = Refractive(n = vec3(1.5 + 4e-8j,1.5 + 0.j,1.5 + 4e-8j)) 
mirror_material = Mirror(reflection_coeff=1.0)
gold_metal = Glossy(diff_color = vec3(1., .572, .184), n = vec3(0.15+3.58j, 0.4+2.37j, 1.54+1.91j), roughness = 0.0, spec_coeff = 0.2, diff_coeff= 0.8)
bluish_metal = Glossy(diff_color = vec3(0.0, 0, 0.1), n = vec3(1.3+1.91j, 1.3+1.91j, 1.4+2.91j), roughness = 0.2,spec_coeff = 0.5, diff_coeff= 0.3)
floor =  Glossy(diff_color = image(r'c:\Users\lenovo\Downloads\Compressed\Python-Raytracer-master\sightpy\textures\checkered_floor.png', repeat = 80.),n = vec3(1.2+ 0.3j, 1.2+ 0.3j, 1.1+ 0.3j), roughness = 0.2, spec_coeff = 0.3, diff_coeff= 0.9 )
Sc = Scene(ambient_color = vec3(0.05, 0.05, 0.05))
angle = -np.pi/2 * 0.3
Sc.add_Camera(look_from = vec3(2.5*np.sin(angle), 0.25, 2.5*np.cos(angle)  -1.5 ),look_at = vec3(0., 0.25, -3.),screen_width = 500 ,screen_height = 500)
Sc.add_DirectionalLight(Ldir = vec3(0.52,0.45, -0.5),  color = vec3(0.15, 0.15, 0.15))
Sc.add(Cuboid( material = green_glass, center = vec3(0.5, 0.0001, -0.8), width = 0.9,height = 1.0, shadow=False, length = 0.4,  max_ray_depth = 5))
Sc.add(Sphere(material = gold_metal, center = vec3(-.75, .1, -3.),radius =  .6, max_ray_depth = 3))
Sc.add(Sphere(material = bluish_metal, center = vec3(1.25, .1, -3.), radius = .6, max_ray_depth = 3))
Sc.add(Plane(material = floor,  center = vec3(0, -0.5, -3.0), width = 120.0,height = 120.0, u_axis = vec3(1.0, 0, 0), v_axis = vec3(0, 0, -1.0),  max_ray_depth = 3))
Sc.add_Background(r'c:\Users\lenovo\Downloads\Compressed\Python-Raytracer-master\sightpy\backgrounds\stormydays.png')
Sc.render(samples_per_pixel = 1).show()
print(time() - a)