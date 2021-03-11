import numpy as np

from skimage.draw import draw
from skimage._shared.utils import warn
from skimage.util import random_noise

import remove_data as rmd 
import collections

Label = collections.namedtuple('Label', 'category, x1, x2, y1, y2')
Point = collections.namedtuple('Point', 'row, column')
ShapeProperties = collections.namedtuple('ShapeProperties',
                                         'min_size, max_width, max_height, color, groundpos')
ShapeSize  = collections.namedtuple('ShapeSize', 'type, width, height')
ImageShape = collections.namedtuple('ImageShape', 'nrows, ncols, depth')



def _generate_ground_mask(height, img_shape):
    """Generate a mask for a filled ground.

    The ground starts from the bottom of the image and 
    cover all its width

    Parameters
    ----------
    height: height of the ground
    img_shape: size of the image

    Returns
    -------
    ground_mask: mask of the ground position

    """

    # Size of the image
    nrows = img_shape.nrows
    ncols = img_shape.ncols

    # Draw the mask
    ground_mask = draw.polygon([
        nrows-height,
        nrows,
        nrows,
        nrows-height,
    ], [
        0,
        0,
        ncols,
        ncols,
    ]
    )

    return ground_mask



def _select_random_size(image, shape, shapetype, random):
    """ Select a random size and return a shape """

    if (shapetype == 'rectangle'):
        # Do we have enough space to fit width while being larger than min size
        available_width = shape.max_width

        if available_width < shape.min_size:
            raise ArithmeticError('cannot fit shape to image')

        # Do we have enough space to fit height while being larger than min size
        available_height = shape.max_height

        if available_height < shape.min_size:
            raise ArithmeticError('cannot fit shape to image')

        # Pick random widths and select height
        width = random.randint(shape.min_size, available_width + 1)

        shape_size = ShapeSize(
            'rectangle', width, shape.groundpos
        )

    elif (shapetype == 'triangle'):
        # Do we have enough space to fit triangle height
        available_height = shape.max_height

        if available_height < shape.min_size:
            raise ArithmeticError('cannot fit shape to image')

        # Select height for the triangle
        triangle_height = random.randint(shape.min_size, available_height + 1)
        side = shape.max_width

        shape_size = ShapeSize(
            'triangle', side, triangle_height
        )

    elif (shapetype == 'circle'):
        # Do we have enough space to fit circle radius
        available_radius = min(shape.max_height, shape.max_width)

        if available_radius < shape.min_size:
            raise ArithmeticError('cannot fit shape to image')

        # Select radius for the circle

        radius = random.randint(shape.min_size, available_radius + 1)

        shape_size = ShapeSize(
            'circle', radius, radius
        )


    return shape_size


def _generate_rectangle_mask(topleft, groundpos, image, rect_size, random):
    """Generate a mask for a filled rectangle shape.

    The height and width of the rectangle are generated randomly.

    Parameters
    ----------
    topleft : Point
        The row and column of the top left corner of the rectangle.
    image : ImageShape
        The size of the image into which this shape will be fit.
    shape : ShapeProperties
        The minimum and maximum size and color of the shape to fit.
    random : np.random.RandomState
        The random state to use for random sampling.

    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates (x_0, y_0). This usually means the image dimensions are too
        small or shape dimensions too large.

    Returns
    -------
    label : Label
        A tuple specifying the category of the shape, as well as its x1, x2, y1
        and y2 bounding box coordinates.
    indices: 2-D array
        A mask of indices that the shape fills.
    """

    # Rectangle has to reach ground
    rectangle = draw.polygon([
        topleft.row,
        groundpos,
        groundpos,
        topleft.row,
    ], [
        topleft.column,
        topleft.column,
        topleft.column + rect_size.width,
        topleft.column + rect_size.width,
    ])

    return rectangle


def _generate_circle_mask(center, image, circ_size, random):
    """Generate a mask for a filled circle shape.

    The radius of the circle is generated randomly.

    Parameters
    ----------
    center : Point
        The row and column of the top left corner of the rectangle.
    image : ImageShape
        The size of the image into which this shape will be fit.
    shape : ShapeProperties
        The minimum and maximum size and color of the shape to fit.
    random : np.random.RandomState
        The random state to use for random sampling.

    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates (x_0, y_0). This usually means the image dimensions are too
        small or shape dimensions too large.

    Returns
    -------
    label : Label
        A tuple specifying the category of the shape, as well as its x1, x2, y1
        and y2 bounding box coordinates.
    indices: 2-D array
        A mask of indices that the shape fills.
    """
    
    radius = circ_size.width

    circle = draw.circle(center.row, center.column, radius)

    return circle


def _generate_triangle_mask(point, image, shape, random):
    """Generate a mask for a filled equilateral triangle shape.

    The length of the sides of the triangle is generated randomly.

    Parameters
    ----------
    point : Point
        The row and column of the top left corner of the rectangle.
    image : ImageShape
        The size of the image into which this shape will be fit.
    shape : ShapeProperties
        The minimum and maximum size and color of the shape to fit.
    random : np.random.RandomState
        The random state to use for random sampling.

    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates (x_0, y_0). This usually means the image dimensions are too
        small or shape dimensions too large.

    Returns
    -------
    label : Label
        A tuple specifying the category of the shape, as well as its x1, x2, y1
        and y2 bounding box coordinates.
    indices: 2-D array
        A mask of indices that the shape fills.
    """

    # Draw the triangle
    triangle = draw.polygon([
        point.row,
        point.row - shape.height,
        point.row,
    ], [
        point.column,
        point.column + shape.width // 2,
        point.column + shape.width,
    ])

    return triangle


# Allows lookup by key as well as random selection.
SHAPE_GENERATORS = dict(
    ground=_generate_ground_mask,
    rectangle=_generate_rectangle_mask,
    circle=_generate_circle_mask,
    triangle=_generate_triangle_mask)
SHAPE_CHOICES = list(SHAPE_GENERATORS.values())


def _generate_random_color(gray, min_pixel_intensity, random):
    """Generates a random color array.

    Parameters
    ----------
    gray : bool
        If `True`, the color will be a scalar, else a 3-element array.
    min_pixel_intensity : [0-255] int
        The lower bound for the pixel values.
    random : np.random.RandomState
        The random state to use for random sampling.

    Raises
    ------
    ValueError
        When the min_pixel_intensity is not in the interval [0, 255].

    Returns
    -------
    color : scalar or array
        If gray is True, a single random scalar in the range of
        [min_pixel_intensity, 255], else an array of three elements, each in
        the range of [min_pixel_intensity, 255].

    """
    if not (0 <= min_pixel_intensity <= 255):
        raise ValueError('Minimum intensity must be in interval [0, 255]')
    if gray:
        return random.randint(min_pixel_intensity, 256)
    return random.randint(min_pixel_intensity, 256, size=3)


def generate_data(
        image_shape,
        max_shapes,
        min_shapes=1,
        mingroundheight=2,
        maxgroundheight=None,
        min_size=2,
        max_size=None,
        gray=False,
        min_pixel_intensity=0,
        allow_overlap=False,
        numtrials=100,
        random_seed=None,
        gauss_noise_var=0.05,
        sp_noise_amount=0.25
    ):
    """Generate an image with random shapes, labeled with bounding boxes.

    The image is populated with random shapes with random sizes, random
    locations, and random colors, with or without overlap.

    Shapes have random (row, col) starting coordinates and random sizes bounded
    by `min_size` and `max_size`. It can occur that a randomly generated shape
    will not fit the image at all. In that case, the algorithm will try again
    with new starting coordinates a certain number of times. However, it also
    means that some shapes may be skipped altogether. In that case, this
    function will generate fewer shapes than requested.

    Parameters
    ----------
    image_shape : (int, int)
        The number of rows and columns of the image to generate.
    max_shapes : int
        The maximum number of shapes to (attempt to) fit into the shape.
    min_shapes : int
        The minimum number of shapes to (attempt to) fit into the shape.
    min_size : int
        The minimum dimension of each shape to fit into the image.
    max_size : int
        The maximum dimension of each shape to fit into the image.
    gray : bool
        If `True`, generate 1-D monochrome images, else 3-D RGB images.
    shape : {rectangle, circle, triangle, None} str
        The name of the shape to generate or `None` to pick random ones.
    min_pixel_intensity : [0-255] int
        The minimum pixel value for colors.
    allow_overlap : bool
        If `True`, allow shapes to overlap.
    numtrials : int
        How often to attempt to fit a shape into the image before skipping it.
    seed : int
        Seed to initialize the random number generator.
        If `None`, a random seed from the operating system is used.

    Returns
    -------
    image : uint8 array
        An image with the fitted shapes.
    labels : list
        A list of Label namedtuples, one per shape in the image.

    Examples
    --------
    >>> import skimage.data
    >>> image, labels = skimage.data.generate_shapes((32, 32), max_shapes=3)
    >>> image # doctest: +SKIP
    array([
       [[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]]], dtype=uint8)
    >>> labels # doctest: +SKIP
    [Label(category='circle', x1=22, x2=25, y1=18, y2=21),
     Label(category='triangle', x1=5, x2=13, y1=6, y2=13)]
    """

    # Control size
    if min_size > image_shape[0] or min_size > image_shape[1]:
        raise ValueError('Minimum dimension must be less than ncols and nrows')
    max_size = max_size or max(image_shape[0], image_shape[1])

    # Set random 
    random = np.random.RandomState(random_seed)

    # Prepare output image
    image_shape = ImageShape(
        nrows=image_shape[0], ncols=image_shape[1], depth=1 if gray else 3
    )

    image = np.ones(image_shape, dtype=np.uint8) * 255

    # Create image of labels
    labels = np.zeros([image_shape.nrows, image_shape.ncols], dtype=np.uint8)
    filled = np.zeros(image_shape, dtype=bool)

    # Select size of the ground
    ground_generator = SHAPE_GENERATORS['ground']
    
    groundheight = np.random.randint(mingroundheight, maxgroundheight)
    groundpos = image_shape.nrows - groundheight

    # Create the ground
    indices = _generate_ground_mask(groundheight, image_shape)

    # Check if there is an overlap where the mask is nonzero.
    labels[indices] = 1
    filled[indices] = True
    image[indices] = [128, 128, 128]

    # Create random rectangles on the ground
    numrect = random.randint(0, max_shapes)

    for rect in range(numrect):

        created_rect = False

        for _ in range(numtrials):

            # Pick start coordinates.
            column  = random.randint(image_shape.ncols)
            row     = random.randint(groundpos)
            topleft = Point(row, column)

            # Define properties of the shape
            shape = ShapeProperties(
                min_size,
                image_shape.ncols - column,
                image_shape.nrows - row,
                [255, 0, 0],
                groundpos
            )

            # Randomly select the size of the rectangle
            try:
                rect_size = _select_random_size(image, shape, 'rectangle', random)
            except ArithmeticError:
                # Couldn't fit the shape, skip it.
                continue

            # Create the rectangle
            indices = _generate_rectangle_mask(topleft, groundpos, image_shape, rect_size, random)  

            # Check if there is an overlap where the mask is nonzero.
            if allow_overlap or not filled[indices].any():
                filled[indices] = True
                labels[indices] = 2
                image[indices]  = [255, 0, 0]
                created_rect = True
                break
            else:
                warn('Could not fit any shapes to image, '
                        'consider reducing the minimum dimension')


        # Create random triangles on the rectangle
        if created_rect:
            for _ in range(numtrials):

                # Pick start coordinates.
                shape = ShapeProperties(
                    min_size,
                    rect_size.width,
                    topleft.row,
                    [0, 0, 255],
                    groundpos
                )

                # Randomly select the size of the rectangle
                try:
                    tri_size = _select_random_size(image, shape, 'triangle', random)
                except ArithmeticError:
                    # Couldn't fit the shape, skip it.
                    continue

                indices = _generate_triangle_mask(topleft, image_shape, tri_size, random)

                # Check if there is an overlap where the mask is nonzero.
                if allow_overlap or not filled[indices].any():
                    filled[indices] = True
                    labels[indices] = 4
                    image[indices]  = [0, 0, 255]

                    break
                else:
                    warn('Could not fit any shapes to image, '
                            'consider reducing the minimum dimension')


    # Create random circles 
    numcircles = random.randint(0, max_shapes)

    for _ in range(numcircles):
        for _ in range(numtrials):

            # Pick start coordinates.
            column  = random.randint(image_shape.ncols)
            row     = random.randint(groundpos)
            center  = Point(row, column)

            # Circle - minimum radius
            min_radius = max(min_size/2, np.abs(row-groundpos) + 1)

            # Pick start coordinates.
            shape = ShapeProperties(
                min_radius,
                min(center.row, image_shape.nrows - center.row),
                min(center.column, image_shape.ncols - center.column),
                [0, 255, 0],
                groundpos
            )

            # Randomly select the size of the rectangle
            try:
                circ_size = _select_random_size(image, shape, 'circle', random)
            except ArithmeticError:
                # Couldn't fit the shape, skip it.
                continue

            indices = _generate_circle_mask(center, image_shape, circ_size, random)

            # Remove useless indices
            if filled[indices].any() and not allow_overlap:
                del_idx = np.argwhere(~filled[indices])
                del_idx = np.array(del_idx)
                del_idx = np.unique(del_idx[:,0])
                indices = np.array(indices)
                indices = indices[:,tuple(del_idx)]
                indices = tuple(indices)

            # Check if there is an overlap where the mask is nonzero.
            if allow_overlap or not filled[indices].any():
                filled[indices] = True
                labels[indices] = 3
                image[indices]  = [0, 255, 0]
                break
            else:
                warn('Could not fit any shapes to image, '
                        'consider reducing the minimum dimension')


    # Disturb the ground truth to create noisy data
    noisy_img = np.copy(image)
    noisy_img = random_noise(noisy_img, mode='gaussian', var=gauss_noise_var)

    # Salt and peper noise
    sp_noise = np.zeros([image_shape.nrows, image_shape.ncols], np.bool)
    sp_noise = random_noise(sp_noise, mode='salt', amount=sp_noise_amount)
    sp_noise = np.transpose(np.argwhere(sp_noise))

    noisy_img[sp_noise[0,:], sp_noise[1,:],:] = np.zeros([1,1,3])

    # Remove bits of the image
    rmd_idx = rmd.remove_shapes(
        [image_shape.nrows, image_shape.ncols], 
        max_shapes=10, min_size=2, max_size=50, 
        allow_overlap=False
    )
    noisy_img[rmd_idx[0,:], rmd_idx[1,:],:] = np.zeros([1,1,3])



    return labels, image, noisy_img

