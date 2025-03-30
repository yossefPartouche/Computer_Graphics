import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod
from os.path import basename
from typing import List
import functools

    
def NI_decor(fn):
    def wrap_fn(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except NotImplementedError as e:
            print(e)
    return wrap_fn


class SeamImage:
    def __init__(self, img_path: str, vis_seams: bool=True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path
        
        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T
        
        """ The values in RGB range from [0,1]"""
        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()
        
        self.h, self.w = self.rgb.shape[:2]
        
        try:
            """ The values in GREYSCALE range from [0,1]"""
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices 
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))
        self.idx_map = np.arange(self.h * self.w).reshape(self.h, self.w) #Combind index map

    @NI_decor
    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3) 
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vector matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
            explicitly before the conversion.
            - np.pad might be useful
        """
        # Pad the image with 0.5
        #rgb_img_padded = np.pad(np_img, ((1,1),(1,1),(0,0)), mode='constant',constant_values=0.5)
        #plt.imshow(rgb_img_padded)
        #print("below should be an image")
        grey_img = np.dot(np_img, self.gs_weights)
        return grey_img
        raise NotImplementedError("TODO: Implement SeamImage.rgb_to_grayscale")

    @NI_decor
    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            - In order to calculate a gradient of a pixel, only its neighborhood is required.
            - keep in mind that values must be in range [0,1]
            - np.gradient or other off-the-shelf tools are NOT allowed, however feel free to compare yourself to them
        """
        """ create an matrix of the same size as our greyscale image
        """
        greyscale = self.resized_gs[..., 0]
        # create gradient matrices
        # will produce matrices containing values like this[[[0.], [0.] [0.] -- DON'T WORRY ABOUT THIS
        Gradient_x = np.zeros_like(greyscale)
        Gradient_y = np.zeros_like(greyscale)
        """ For x-axis Energy we need to calculate the difference between the pixel and its neighbor below
        if we think about it, we only are able to identify a horizontal edge by checking vertical difference"""
        """ start|  |  |  |y | x | end """
        Gradient_x[-1] = np.abs(greyscale[-1] - greyscale[-2])
        Gradient_x[:-1] = np.abs(greyscale[:-1]- greyscale[1:])

        #print("Gradient_x", Gradient_x)

        Gradient_y[:, 1:] = np.abs(greyscale[:, 1:] - greyscale[:, :-1])
        Gradient_y[:, 0] = np.abs(greyscale[:, 0] - greyscale[:, 1])

        #print("Gradient_y", Gradient_y)
        # grad_horz_f[:, :-1] = self.gs[:, 1:] - self.gs[:, :-1]
        # grad_vert_f[:-1, :] = self.gs[1:, :] - self.gs[:-1, :]
        #^^^ we we're calculating the energy of the image of the whole image
        """ LESSON [:, 0] - select rows from [row __ :(to)row (excl) __, col 0 ]"""
        """" Calculate the absolute gradient magnitude whilst keeping the values in the range of 0 to 1 """
        abs_grad = np.sqrt(Gradient_x** 2 + Gradient_y** 2)
        #print("abs_grad", abs_grad)
        return abs_grad
        raise NotImplementedError("TODO: Implement SeamImage.calc_gradient_magnitude")


    def update_ref_mat(self):
        for i, s in enumerate(self.seam_history[-1]):
            #self.idx_map[i, s:] += 1
                # Shift the indices in idx_map_v and idx_map_h to the left for pixels to the right of the seam
                self.idx_map_v[i, s:self.w - 1] = self.idx_map_v[i, s + 1:self.w]
                self.idx_map_h[i, s:self.w - 1] = self.idx_map_h[i, s + 1:self.w]
        # Reduce the width of the maps to match the new image width
        self.idx_map_v = self.idx_map_v[:, :self.w - 1]
        self.idx_map_h = self.idx_map_h[:, :self.w - 1]

        #self.idx_map_h[i, s:] +=

    def reinit(self):
        """
        Re-initiates instance and resets all variables.
        """
        self.__init__(img_path=self.path)

    @staticmethod
    def load_image(img_path, format='RGB'):
        return np.asarray(Image.open(img_path).convert(format)).astype('float32') / 255.0

    def paint_seams(self):
        #print("Sesam history", self.seam_history)
        for s in self.seam_history:
            for i, s_i in enumerate(s):
                self.cumm_mask[self.idx_map_v[i,s_i], self.idx_map_h[i,s_i]] = False
        self.cumm_mask = np.squeeze(self.cumm_mask)
        cumm_mask_rgb = np.stack([self.cumm_mask] * 3, axis=2)
        #print("cumm_mask_rgb", cumm_mask_rgb.shape)
        self.seams_rgb = np.where(cumm_mask_rgb, self.seams_rgb, [1,0,0])

        #self.mask[np.arange(self.h), seam] = False
        #threeD_mask = np.stack([self.mask] * 3, axis=2)
        #self.w -= 1
        #self.resized_gs = self.resized_gs[self.mask].reshape(self.h, self.w,1)
        #self.resized_rgb = self.resized_rgb[threeD_mask].reshape(self.h, self.w,3)
        

    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seams to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, mask) where:
                - E is the gradient magnitude matrix
                - mask is a boolean matrix for removed seams
            iii) find the best seam to remove and store it
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the chosen seam (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you wish, but it needs to support:
            - removing seams a couple of times (call the function more than once)
            - visualize the original image with removed seams marked in red (for comparison)
        """
        for _ in tqdm(range(num_remove)):
            self.E = self.calc_gradient_magnitude()
            self.mask = np.ones_like(self.E, dtype=bool)

            seam = self.find_minimal_seam()
            self.seam_history.append(seam)
            self.remove_seam(seam)
            self.update_ref_mat()
            if self.vis_seams:
                self.paint_seams()
                # Show the image with the seam marked in red

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the seam with the minimal energy.
        Returns:
            The found seam, represented as a list of indexes
        """
        return GreedySeamImage.find_minimal_seam()
        raise NotImplementedError("TODO: Implement SeamImage.find_minimal_seam in one of the subclasses")


    @NI_decor
    def remove_seam(self, seam: List[int]):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using:
        3d_mask = np.stack([1d_mask] * 3, axis=2)
        ...and then use it to create a resized version.

        :arg seam: The seam to remove
        """
        # Update the mask based on the seam
        #self.mask[np.arange(self.h), seam] = False

        # Apply the mask to create the resized RGB image
        #self.resized_rgb = np.delete(self.rgb, seam, axis=1)

        # Update the grayscale version of the resized image
       # self.resized_gs = self.rgb_to_grayscale(self.resized_rgb)
        # Update the width of the image
        #self.w -= 1
        self.mask[np.arange(self.h), seam] = False
        threeD_mask = np.stack([self.mask] * 3, axis=2)
        self.resized_gs = self.resized_gs[self.mask].reshape(self.h, self.w-1,1)
        self.resized_rgb = self.resized_rgb[threeD_mask].reshape(self.h, self.w-1,3)
        self.w -= 1


    @NI_decor
    def rotate_mats(self, clockwise: bool):
        """
        Rotates the matrices either clockwise or counter-clockwise.
        """
        count =1 if clockwise else 3
        self.resized_rgb = np.rot90(self.resized_rgb, count)
        self.resized_gs = np.rot90(self.resized_gs , count)
        self.E = np.rot90(self.E, count)
        self.h, self.w = self.resized_rgb.shape[:2]
        #self.resized_rgb = np.rot90(self.resized_rgb, count)

        # Rotate idx_map and other related attributes if they exist
        """"
        if hasattr(self, 'idx_map'):
            self.idx_map = np.rot90(self.idx_map, count)
        if hasattr(self, 'idx_map_h'):
            self.idx_map_h = np.rot90(self.idx_map_h, count)
        if hasattr(self, 'idx_map_v'):
            self.idx_map_v = np.rot90(self.idx_map_v, count)
        if hasattr(self, 'cumm_mask'):
            self.cumm_mask = np.rot90(self.cumm_mask, count)
        """

    @NI_decor
    def seams_removal_vertical(self, num_remove: int):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """
        self.seams_removal(num_remove)

    @NI_decor
    def seams_removal_horizontal(self, num_remove: int):
        """ Removes num_remove horizontal seams by rotating the image, removing vertical seams, and restoring the original rotation.

        Parameters:
            num_remove (int): number of horizontal seam to be removed
        """
        self.rotate_mats(clockwise=True)
        self.seams_removal(num_remove)
        self.rotate_mats(clockwise=False)

    """
    BONUS SECTION
    """

    @NI_decor
    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seams to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition")

    @NI_decor
    def seams_addition_horizontal(self, num_add: int):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_add (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_horizontal")

    @NI_decor
    def seams_addition_vertical(self, num_add: int):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """

        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_vertical")

class GreedySeamImage(SeamImage):
    """Implementation of the Seam Carving algorithm using a greedy approach"""
    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the minimal seam by using a greedy algorithm.

        Guidelines & hints:
        The first pixel of the seam should be the pixel with the lowest cost.
        Every row chooses the next pixel based on which neighbor has the lowest cost.
        # Initialize the path array with the index of the minimum cost pixel in the first row
        """
        Greedy_seam = np.zeros(self.h, dtype=int)
        Greedy_seam[0] = np.argmin(self.E[0])
        #print(Greedy_seam[0])
        for i in range(1,self.h):
            j = Greedy_seam[i - 1]
            #print("j", j)
            left = j - 1 if j > 0 else j 
            #print("left", left)
            right = j + 1 if j < self.w-1 else j 
            #print("right", right)
            next_triple_E = [(self.E[i, left], left), (self.E[i, j], j),  (self.E[i, right], right)]
            min_energy, next_indx = min(next_triple_E, key=lambda x: x[0])
            #print("next_indx", next_indx)
            Greedy_seam[i] = next_indx
        
        #print("Greedy_seam", Greedy_seam.shape)
        return Greedy_seam
        raise NotImplementedError("TODO: Implement GreedySeamImage.find_minimal_seam")


class DPSeamImage(SeamImage):
    """
    Implementation of the Seam Carving algorithm using dynamic programming (DP).
    """
    def __init__(self, *args, **kwargs):
        """ DPSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the minimal seam by using dynamic programming.

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (M, backtracking matrix) where:
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
        """
        raise NotImplementedError("TODO: implement DPSeamImage.find_minimal_seam")

    @NI_decor
    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        raise NotImplementedError("TODO: Implement DPSeamImage.calc_M")

    def init_mats(self):
        self.M = self.calc_M()
        self.backtrack_mat = np.zeros_like(self.M, dtype=int)

    @staticmethod
    @jit(nopython=True)
    def calc_bt_mat(M, E, GS, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.

        Recommended parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            E: np.ndarray (float32) of shape (h,w)
            GS: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a reference type. Changing it here may affect it on the outside.
        """
        raise NotImplementedError("TODO: Implement DPSeamImage.calc_bt_mat")
        h, w = M.shape


def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    raise NotImplementedError("TODO: Implement scale_to_shape")


def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """
    raise NotImplementedError("TODO: Implement resize_seam_carving")


def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)
    ###Your code here###
    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org
    scaled_x_grid = [get_scaled_param(x,in_width,out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y,in_height,out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid,dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid,dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:,x1s] * dx + (1 - dx) * image[y1s][:,x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:,x1s] * dx + (1 - dx) * image[y2s][:,x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)
    return new_image


