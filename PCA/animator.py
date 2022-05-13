__author__ = "Garrett Dal Byrd"
__version__ = "0.0.1"
__email__ = "gbyrd4@vols.utk.edu"
__status__ = "Development"


from os import listdir, curdir, chdir, mkdir
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from numpy import genfromtxt
import h5py
from PIL import Image
import sys
import imageio


def animate(seed, run_number, output_directory=".", image_filetype="tiff"):
    """
    

    Parameters:
        seed                (int):              The seed of the model. (Model must already have been run; i.e., this directory must already exist.)
        run_number          (int):              The specific run the user seeks to visualize.
        output_directory    (str, optional):    The directory which the resulting image/gif will be exported to. By default, exports into the directory of the selected run.
        image_filetype      (str, optional):    The output image filetype. Tiff by default.
    """
    if isinstance(seed, int):
        seed = str(seed)
    elif isinstance(seed, str):
        pass
    else:
        raise ValueError(
            f"Expected parameter 'seed' as type str or int; got {type(seed).__name__}."
        )

    if isinstance(run_number, int):
        run_number = str(run_number)
    elif isinstance(run_number, str):
        pass
    else:
        raise ValueError(
            f"Expected parameter 'run_number' as type str or int; got {type(run_number).__name__}."
        )

    def determine_dim(dataset):
        '''Returns the dimension of the data array.'''
        if dataset.shape[1] == 1 and dataset.shape[2] == 1:
            return 1
        elif dataset.shape[2] == 1:
            return 2
        else:
            return 3

    def to_rgb(pixel):
        '''Used to convert binary data into an 8-bit channel.'''
        if pixel == 0:
            return 255
        if pixel == 1:
            return 0

    def produce_1dim_image(data):
        '''Used to produce visualization for 1-dimensional data.'''
        data = np.squeeze(data)
        pixel_data = np.asarray(np.vectorize(to_rgb)(data))
        red_band = np.full(pixel_data.shape, 255)
        final_pixel_data = np.dstack((red_band, pixel_data, pixel_data))
        image = Image.fromarray(final_pixel_data.astype("uint8"), mode="RGB")
        image.save(f"{seed}_run_{run_number}.{image_filetype}")

    def produce_2dim_animation(data):
        '''Used to produce visualization for 2-dimensional data.'''
        data = np.squeeze(data)
        pixel_data = np.asarray(np.vectorize(to_rgb)(data))
        try:
            mkdir("gif_frames")
        except FileExistsError:
            pass
        chdir("gif_frames")
        red_band = np.full(pixel_data.shape, 255)
        for i in range(data.shape[2]):
            final_pixel_data = np.dstack(
                (red_band[:, :, i], pixel_data[:, :, i], pixel_data[:, :, i])
            )
            image = Image.fromarray(final_pixel_data.astype("uint8"), mode="RGB")
            image.save(f"{seed}_run_{run_number}_slice_{i}.{image_filetype}")
        # Create gif
        gif_images = []
        for i in range(run_length):
            file_name = f"{seed}_run_{run_number}_slice_"
            gif_images.append(imageio.imread(f"{file_name}{i}.{image_filetype}"))
        imageio.mimwrite(f"{curdir}/animation.gif", gif_images, fps=10)

    def produce_3dim_animation(data):
        '''Used to produce visualization for 3-dimensional data.'''
        pass

    if output_directory == ".":
        chdir("runs")
        chdir(seed)
        chdir(f"run_{run_number}")
    else:
        chdir(output_directory)

    parameters = genfromtxt("parameters.csv", delimiter=",")[1]
    x_length = int(parameters[1])
    y_length = int(parameters[2])
    z_length = int(parameters[3])
    init_infection = int(parameters[4])
    local_prob = float(parameters[5])
    global_prob = float(parameters[6])
    run_length = int(parameters[7])
    moore = parameters[9]
    population = x_length * y_length * z_length

    with h5py.File("data.hdf5", "r") as file:
        data = np.asarray(file["default"])

    data_dim = determine_dim(data)
    np.set_printoptions(threshold=sys.maxsize)

    if data_dim == 1:
        produce_1dim_image(data)
    elif data_dim == 2:
        produce_2dim_animation(data)
    elif data_dim == 3:
        produce_3dim_animation(data)
