import png
import pyspng

def read_png(image_path):
    with open(image_path, 'rb') as fin:
        image = pyspng.load(fin.read())
    if image.ndim == 3 and image.shape[-1] > 3:
        image = image[..., :3]
    return image


def write_png(image, image_path):
    if image.dtype == 'uint8':
        bitdepth = 8
    elif image.dtype == 'uint16':
        bitdepth = 16
    else:
        raise ValueError(f'Invalid datatype {image.dtype}')
    grayscale = (image.ndim == 2)
    with open(image_path, 'wb') as f:
        writer = png.Writer(width=image.shape[1], height=image.shape[0], bitdepth=bitdepth, greyscale=grayscale)
        if grayscale:
            image_list = image.tolist()
        else:
            image_list = image.reshape(-1, image.shape[1]*image.shape[2]).tolist()
        writer.write(f, image_list)
