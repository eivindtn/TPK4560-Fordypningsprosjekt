import numpy as np
import glob
a = np.array([1,1,1])
cols = 9
rows = 6
images = glob.glob('captured_images/*.png')
zdf = glob.glob('zdf/*.zdf')
method = 'center'
camera_intrinsics = np.array([[2767.193359, 0 , 942.942505], [0, 2766.449119, 633.898],[0, 0, 1]])
distortion_coeffsients = np.array([[-0.26513, 0.282772, 0.000767328, 0.000367037,0]])
flags = None
projector_res_width = 1024
projector_res_height = 768
square_size = 90
interpolate_method = 'nearest'
interpolate_number = 100