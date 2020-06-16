import numpy as np
from numpy import pi, squeeze
import os
import random

from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import LabelSet, ColumnDataSource

from imageio import imread
import PIL.Image
import PIL.ImageOps

from werkzeug.utils import secure_filename

print(hasattr(PIL.ImageOps, 'exif_transpose'))

# function for use

ALLOWED_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpeg', 'gif'])
LETTER_SET = list(set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
IMAGE_LABELS = ['Not Popular', 'Popular']


def is_allowed_file(filename):
    """ Checks if a filename's extension is acceptable """
    allowed_ext = filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return '.' in filename and allowed_ext


def generate_random_name(filename):
    """ Generate a random name for an uploaded file. """
    ext = filename.split('.')[-1]
    rns = [random.randint(0, len(LETTER_SET) - 1) for _ in range(3)]
    chars = ''.join([LETTER_SET[rn] for rn in rns])

    new_name = "{new_fn}.{ext}".format(new_fn=chars, ext=ext)
    new_name = secure_filename(new_name)

    return new_name

def load_and_prepare(filepath):
    """ Load and prepares an image data for prediction """
    image_data = imread(filepath)[:, :, :3]
    
    image_data = image_data / 255.
    image_data = image_data.reshape((-1, 150, 150, 3))
    return image_data

def generate_barplot(emotions):
    """ Generates script and `div` element of bar plot of predictions using
    Bokeh
    """
    print(emotions)
    plot = figure(x_range=list(emotions.keys()), plot_height=400, plot_width=600)
    plot.vbar(x=list(emotions.keys()), top=list(emotions.values()), width=0.8, color=(81,91,212))
    plot.xaxis.major_label_orientation = 0 
    plot.xaxis.major_label_text_font_size = "15pt"
    plot.xaxis.major_label_standoff = 15

    plot.yaxis.major_label_text_font_size = "10pt"

    return components(plot)


def make_thumbnail(filepath):
    """ Converts input image to 150px by 150px thumbnail if not that size
    and save it back to the source file """
    img = PIL.Image.open(filepath)
    img = PIL.ImageOps.exif_transpose(img)
    thumb = None
    w, h = img.size

    # if it is exactly 150x150, do nothing
    if w == 150 and h == 150:
        return True

    # if the width and height are equal, scale down
    if w == h:
        thumb = img.resize((150, 150), PIL.Image.BICUBIC)
        thumb.save(filepath)
        return True

    # when the image's width is smaller than the height
    if w < h:
        # scale so that the width is 128px
        ratio = w / 150.
        w_new, h_new = 150, int(h / ratio)
        thumb = img.resize((w_new, h_new), PIL.Image.BICUBIC)

        # crop the excess
        top, bottom = 0, 0
        margin = h_new - 150
        top, bottom = margin // 2, 150 + margin // 2
        box = (0, top, 150, bottom)
        cropped = thumb.crop(box)
        cropped.save(filepath)
        return True

    # when the image's height is smaller than the width
    if h < w:
        # scale so that the height is 128px
        ratio = h / 150.
        w_new, h_new = int(w / ratio), 150
        thumb = img.resize((w_new, h_new), PIL.Image.BICUBIC)

        # crop the excess
        left, right = 0, 0
        margin = w_new - 150
        left, right = margin // 2, 150 + margin // 2
        box = (left, 0, right, 150)
        cropped = thumb.crop(box)
        cropped.save(filepath)
        return True
    return False
