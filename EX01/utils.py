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
        
        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()
        
        self.h, self.w = self.rgb.shape[:2]
        
        try:
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

    @NI_decor
    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3) 
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        # Pad the RGB image with 0.5
        np_img = np.pad(np_img, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0.5)
        # Convert the RGB image to grayscale
        greyscale_img = np.dot(np_img, self.gs_weights)
        # Remove the padding
        greyscale_img = greyscale_img[1:-1, 1:-1]
        # return the grayscale image as float32
        return greyscale_img.astype('float32')

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
        # Extract the single grayscale channel from the (h, w, 1) image.
        greyscale = self.resized_gs[..., 0]  # Now grey is (h, w)
        # Create two zero matrices with the same shape as the grayscale image
        horizontal_diff = np.zeros_like(greyscale)
        vertical_diff = np.zeros_like(greyscale)
        # Calculate the derivative of the grayscale image in the x and y directions
        horizontal_diff[:, :-1] = greyscale[:, 1:] - greyscale[:, :-1] # chnaged to use resized gs 
        vertical_diff[:-1, :] = greyscale[1:, :] - greyscale[:-1, :]
        # Calculate the gradient magnitude and scale it to the range [0,1]
        gradient = np.sqrt(horizontal_diff ** 2 + vertical_diff ** 2)
        # Normalize the gradient magnitude to the range [0,1]
        gradient = np.clip(gradient, 0, 1)
        # return the gradient magnitude image without the padded values
        return gradient.astype('float32')

    def update_ref_mat(self):
        seam = self.seam_history[-1]
        for i, seam_idx in enumerate(seam):
            self.idx_map[i, seam_idx:] = np.roll(self.idx_map[i, seam_idx:], -1)
            

    def reinit(self):
        """
        Re-initiates instance and resets all variables.
        """
        self.__init__(img_path=self.path)

    @staticmethod
    def load_image(img_path, format='RGB'):
        return np.asarray(Image.open(img_path).convert(format)).astype('float32') / 255.0

    def paint_seams(self):
        for s in self.seam_history:
            for i, s_i in enumerate(s):
                self.cumm_mask[self.idx_map_v[i,s_i], self.idx_map_h[i,s_i]] = False
        new_mask = np.squeeze(self.cumm_mask, axis=2)
        cumm_mask_rgb = np.stack([new_mask] * 3, axis=2) # resulted in # (h, w, 3, 1) -> changed to (h, w, 3)
        self.seams_rgb = np.where(cumm_mask_rgb, self.seams_rgb, [1,0,0])

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
            # init/update matrices
            self.E = self.calc_gradient_magnitude()
            self.mask = np.ones_like(self.E, dtype=bool)
            # find the best seam to remove and store it
            seam = self.find_minimal_seam()
            self.seam_history.append(seam)
            # update the index mapping for removed seam
            if self.vis_seams:
                self.update_ref_mat()
                self.paint_seams()
                self.seam_history = []
            self.remove_seam(seam)



    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the seam with the minimal energy.
        Returns:
            The found seam, represented as a list of indexes
        """
        raise NotImplementedError("TODO: Implement SeamImage.find_minimal_seam")


    @NI_decor
    def remove_seam(self, seam: List[int]):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using:
        3d_mak = np.stack([1d_mask] * 3, axis=2)
        ...and then use it to create a resized version.

        :arg seam: The seam to remove
        """
        # Set the values of the seam to False
        for i, s_i in enumerate(seam):
            self.mask[i, s_i] = False
        # decrease the width of the image
        self.w -= 1
        # Extend the seam mask to support 3 channels
        threeD_mask = np.stack([self.mask] * 3, axis=2)
        # Remove the seam from the grayscale image and rgb images
        self.resized_gs = self.resized_gs[self.mask].reshape((self.h, self.w, 1))
        self.resized_rgb = self.resized_rgb[threeD_mask].reshape((self.h, self.w, 3))        
        
    @NI_decor
    def rotate_mats(self, clockwise: bool):
        """
        Rotates the matrices either clockwise or counter-clockwise.
        """
        # define the number of times to rotate the matrices
        count = -1 if clockwise else 1 # np.rot90 rotates counter-clockwise
        # Rotate the original RGB image
        self.rgb = np.rot90(self.rgb, count)
        # Rotate the grayscale image
        self.gs = np.rot90(self.gs, count)
        # Rotate the gradient matrix
        self.E = np.rot90(self.E, count)
        # Rotate the resized RGB image
        self.resized_rgb = np.rot90(self.resized_rgb, count)
        # Rotate the resized grayscale image
        self.resized_gs = np.rot90(self.resized_gs, count)
        # Update the height and width of the image
        self.h, self.w = self.w, self.h
        # Update the index mapping for the vertical seams
        self.idx_map_v = np.rot90(self.idx_map_v, -count)
        # Update the index mapping for the horizontal seams 
        self.idx_map_h = np.rot90(self.idx_map_h, count)
        self.idx_map_h, self.idx_map_v = self.idx_map_v, self.idx_map_h
        # Update the cumulative mask
        self.cumm_mask = np.rot90(self.cumm_mask, count)
        # Update the seam visualization
        self.seams_rgb = np.rot90(self.seams_rgb, count)


    @NI_decor
    def seams_removal_vertical(self, num_remove: int):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of vertical seams to be removed
        """
        self.idx_map = self.idx_map_h
        self.seams_removal(num_remove)

    @NI_decor
    def seams_removal_horizontal(self, num_remove: int):
        """ Removes num_remove horizontal seams by rotating the image, removing vertical seams, and restoring the original rotation.

        Parameters:
            num_remove (int): number of horizontal seam to be removed
        """
        # Rotate the matrices clockwise
        self.rotate_mats(clockwise=True)
        self.idx_map = self.idx_map_h
        # Remove the vertical seams
        self.seams_removal(num_remove)
        # Restore the original rotation
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
        """
        """"
        # Initialize the seam array
        local_seam = np.empty(self.h, dtype=int)
        original_seam = np.empty(self.h, dtype=int)
        # Find the first pixel of the seam
        local_seam[0] = np.argmin(self.E[0])
        original_seam[0] = self.idx_map_h[0, local_seam[0]]
        # Find the next pixels of the seam
        for i in range(1,self.h):
            prev_idx = local_seam[i - 1]
            # Find the indices of the neighbors
            left = prev_idx - 1 if prev_idx - 1 >= 0 else prev_idx
            right = prev_idx + 1 if prev_idx + 1 < self.w else prev_idx
            values = np.array([left, prev_idx, right])
            # Find the costs of the neighbors
            values_energies = self.E[i, values]
            # Find the index of the neighbor with the lowest cost
            local_seam[i] = values[np.argmin(values_energies)]
            original_seam[i] = self.idx_map_h[i, local_seam[i]]
        return original_seam
        """
        seam = []
        seam.append(np.argmin(self.E[0]))
        for i in range(1, self.h):
            # Find the indices of the neighbors
            prev_idx = seam[-1]
            start = max(0, prev_idx - 1)
            end = min(self.w - 1, prev_idx + 1)
            # Find the costs of the neighbors
            values = self.E[i, start:end + 1]
            # Find the index of the neighbor with the lowest cost
            seam.append(start + np.argmin(values))
        return seam


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
           # Compute the cost matrix M, which updates the backtracking matrix
        self.M = self.calc_M()
        # Initialize the seam array
        seam = np.empty(self.h, dtype=int)
        # Find the index of the minimum cost in the last row
        seam[-1] = np.argmin(self.M[-1])
        # Backtrack to find the indices of the seam
        for i in range(self.h - 2, -1, -1):
            seam[i] = self.backtrack_mat[i + 1, seam[i + 1]]
        return seam

    @NI_decor
    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)
        """
        # Initialize the backtracking matrix
        self.backtrack_mat = np.zeros((self.h, self.w), dtype=int)
        # Initialize the cost matrix M
        m = np.zeros_like(self.E)
        m[0] = self.E[0]
        # Compute the cummulative cost of for each row 
        #for i in range(1, self.h):
        #    for j in range(self.w):
        #        c_center = np.abs(self.gs[i, j + 1] - self.gs[i, j - 1])
        #        c_left =  c_center + np.abs(self.gs[i - 1, j] - self.gs[i, j - 1])
        #        c_right = c_center + np.abs(self.gs[i - 1, j] - self.gs[i, j + 1])
        #        m_left = m[i - 1, j - 1] + c_left
        #        m_center = m[i - 1, j] + c_center
        #        m_right = m[i - 1, j + 1] + c_right
        #        m[i, j] = self.E[i, j] + min(m_left, m_center, m_right)
        # using np.roll:
        for i in range(1, self.h):
            m_prev = m[i - 1]
            row_gs = self.resized_gs[i, :, 0] # Now row_gs is (w,)
            prev_row_gs = self.resized_gs[i - 1, :, 0] # Now prev_row_gs is (w,)
            # Compute forward looking cost using np.roll
            c_center = np.abs(np.roll(row_gs, -1) - np.roll(row_gs, 1))
            c_left = c_center + np.abs(prev_row_gs - np.roll(row_gs, 1))
            c_right = c_center + np.abs(prev_row_gs - np.roll(row_gs, -1))
            # Shift previous cumulative cost to get neighbor costs
            m_left = np.roll(m_prev, 1)
            m_right = np.roll(m_prev, -1)
            # Avoid wrap-around by setting boundaries to infinity
            m_left[0] = np.inf
            m_right[-1] = np.inf
            m_center = m_prev
            # Compute the 3 possible cumulative costs
            cost_left = m_left + c_left
            cost_center = m_center + c_center
            cost_right = m_right + c_right
            # Stack the costs to find the minimum
            costs = np.stack([cost_left, cost_center, cost_right], axis=0)
            # Find the index of the minimum cost (0: left, 1: center, 2: right)
            min_idx = np.argmin(costs, axis=0)
            # Update the backtracking matrix 
            # If candidate is 0, previous index is j-1; if 1, it's j; if 2, it's j+1.
            for j in range(self.w):
                if min_idx[j] == 0:
                    self.backtrack_mat[i, j] = j - 1
                elif min_idx[j] == 1:
                    self.backtrack_mat[i, j] = j
                else:
                    self.backtrack_mat[i, j] = j + 1
            # Update the cumulative cost matrix
            m[i] = self.E[i] + np.min(costs, axis=0)
        return m

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
     # Implemented in the calc_M method
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


