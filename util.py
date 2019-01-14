import asyncio
import base64
import time
from threading import Thread

import imageio
from pathlib import Path
from io import BytesIO
import requests
import pandas as pd
import numpy as np
from skimage import io, img_as_float32, img_as_ubyte, transform
from skimage.color import rgb2gray
import warnings
import configparser
import string
import unicodedata
from typing import List

from starlette.websockets import WebSocketState, WebSocket

# load API config from 'API_config.ini'

config = configparser.ConfigParser()
config.read('API_config.ini')
_URL = config['API']['URL']
_KEY = config['API']['KEY']


def load_image(fname: str, size: int = None) -> np.array:
    """
    Loads an image file and transforms it into float representation [0.0; 1.0]. Optionally resizes the image.

    Args:
        fname (str): filename or path to the image.
        size (int):  optionally, resize the image to size x size.

    Returns:
        np.array: the loaded image, dtype is float with values between 0.0 and 1.0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        image = io.imread(fname)
        image = img_as_float32(image)
        if size is not None:
            image = transform.resize(image, (size, size), mode='constant', anti_aliasing=True)
    return image


def save_image(fname, image) -> None:
    """
    Saves an image to the specified filename.

    Args:
        fname (str): the destination filename
        image (np.array): the image to be saved

    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        image = img_as_ubyte(image)
        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=-1)
        io.imsave(fname, image)


def save_gif_from_images(dest_fname: str, images: List[np.array]) -> None:
    """
    Saves a GIF from a series of images.

    Args:
        dest_fname (str): the destination filename
        images (List[np.array]: a list containing the images in order.

    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        images = [img_as_ubyte(img) for img in images]
        imageio.mimsave(dest_fname, images)


def save_gif_from_image_folder(src: str, dest: str) -> None:
    """
    Saves a GIF from a folder containing images (sorted by filename).
    Args:
        src: the path to read images from
        dest: the destination filename to save the gif

    """
    image_dir = Path(src)
    filenames = sorted(list(image_dir.glob('*.png')), key=lambda x: int(x.stem))

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        imageio.mimsave(dest, images)


def image_to_grayscale(image: np.array):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return rgb2gray(image)


def write_to_log(fname: str, message: str) -> None:
    """
    Writes a string to a text file.

    Args:
        fname (str): filename of the log file to write to
        message (str): the string to write
    """
    with open(fname, 'a+') as f:
        f.write(f'{message}\n')


def send_query(image: np.array) -> (np.array, np.array):
    """
    Sends an image to the API.

    Args:
        image (numpy.array): The image to be sent. Shape must be either (64, 64) if grayscale or (64, 64, 3) if RGB.

    Returns:
        The predicted class labels (np.array), the corresponding confidence values (np.array):

    Raises:
        ValueError: invalid image, ConnectionError: bad API status code
    """
    if image.shape not in {(64, 64), (64, 64, 3)}:
        raise ValueError(f'invalid image with shape {image.shape}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        image = img_as_ubyte(image)
    # create file object from image
    image_file = BytesIO()
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        io.imsave(image_file, image)
    image_file.seek(0)

    # send post requests
    response = requests.post(
        _URL, params={'key': _KEY}, files={'image': image_file})

    if response.status_code == 429 or response.status_code == 400:
        print('\t\t-API rate limit reached, waiting...')
        time.sleep(10)
        return send_query(image)

    # process response
    if response.status_code == 200:  # OK
        predictions = response.json()
        df = pd.DataFrame(predictions)
        return df['class'].values, df['confidence'].values
    else:
        print(response.status_code)
        print(response.content)
        raise ConnectionError('bad API status code')


def get_confidence(image: np.array, target_class: str = None) -> float:
    """
    Get the APIs confidence value for an image.

    Args:
        image (numpy.array): The image to be evaluated.
                             Shape must be either (64, 64) if grayscale or (64, 64, 3) if RGB.
        target_class (str): Class label for which to get the confidence value (must match the API labelling)
                            Default: None - get the highest confidence without respect to any class label

    Returns:
        float:  The confidence value of the API for the given class label. If class label is None, the highest confidence value
                without respect to any class label is returned.
                If the API does not return a confidence for the target class, returns 1e-10
    """
    labels, confs = send_query(image)
    if target_class is None:
        return confs[0]
    if target_class in labels:
        return float(confs[np.where(labels == target_class)])
    else:
        return 1e-10


def query_yes_no(question: str) -> bool:
    """
    Asks the user a yes/no question via input() and returns the answer
    Args:
        question (str): The question to ask

    Returns:
        bool: True if answer is 'yes', False if answer is 'no'
    """

    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}

    while True:
        print(question + ' [y/n]')
        choice = input().lower()
        if choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def clean_filename(filename: str, replace: str = ' '):
    """
    Cleans a string to be a valid filename

    Args:
        filename: The filename to clean
        replace: any characters to be replaced with underscores

    Returns:
        str: a valid filename string
    """
    whitelist = "-_.() %s%s" % (string.ascii_letters, string.digits)
    char_limit = 255
    # replace spaces
    for r in replace:
        filename = filename.replace(r, '_')

    # keep only valid ascii chars
    cleaned_filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode()

    # keep only whitelisted chars
    cleaned_filename = ''.join(c for c in cleaned_filename if c in whitelist)
    if len(cleaned_filename) > char_limit:
        print(
            "Warning, filename truncated because it was over {}. Filenames may no longer be unique".format(char_limit))
    return cleaned_filename[:char_limit]


"""
Server utils
"""


def load_img_to_bytes(file_path: str) -> BytesIO:
    """
    Load an Image at the given path in to an ByteIO Object

    Args:
        file_path: Recursiv path to the image, starting at the main directory of the repo

    Returns:
        ByteIO: Images
    """
    # Load Image from given Filepath
    img = io.imread(file_path)

    img = transform.resize(img, (64, 64))

    image = img_as_ubyte(img)

    image_file = BytesIO()

    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    io.imsave(image_file, image)
    image_file.seek(0)
    return image_file


def send_message(websocket: WebSocket, message: str) -> None:
    """
    Send a text message to the given websocket
    In an different Thread

    Args:
        websocket: Client with will be messaged
        message: Message to be send
    """
    Thread(target=__send_message, args=(websocket, message)).start()


def __send_message(websocket: WebSocket, message: str) -> None:
    """
    Send a text message to the given websocket

    Args:
        websocket: Client with will be messaged
        message: Message to be send
    """
    if websocket.client_state == WebSocketState.CONNECTED:
        asyncio.run(websocket.send_json({'data': message}))


def send_img(websocket: WebSocket, image_path: str, final: bool) -> None:
    """
    Send a image to the given websocket
    In an different Thread

    Args:
        websocket: Client with will be messaged
        image_path: recursiv path, starting from the main directory
        final: is it the last image send?
    """
    Thread(target=__send_img, args=(websocket, image_path, final)).start()


def __send_img(websocket: WebSocket, image_path: str, final: bool) -> None:
    """
    Send a image to the given websocket

    Args:
        websocket: Client with will be messaged
        image_path: recursiv path, starting from the main directory
        final: is it the last image send?
    """
    data_to_send = base64.b64encode(load_img_to_bytes(image_path).getvalue()).decode('utf-8')
    data_to_send = 'data:image/png;base64,' + data_to_send

    if websocket.client_state == WebSocketState.CONNECTED:
        asyncio.run(websocket.send_json({'done': final, 'image': data_to_send}))


def send_gif(websocket: WebSocket, image_path: str, final: bool) -> None:
    """
    Send a gif to the given websocket
    In an different Thread

    Args:
        websocket: Client with will be messaged
        image_path: recursiv path, starting from the main directory
        final: is it the last image send?
    """
    Thread(target=__send_gif, args=(websocket, image_path, final)).start()


def __send_gif(websocket: WebSocket, image_path: str, final: bool) -> None:
    """
    Send a gif to the given websocket

    Args:
        websocket: Client with will be messaged
        image_path: recursiv path, starting from the main directory
        final: is it the last image send?
    """
    with open(image_path, 'rb') as img_file:
        img = img_file.read()

        data_to_send = base64.b64encode(img).decode('utf-8')
        data_to_send = 'data:image/gif;base64,' + data_to_send

        if websocket.client_state == WebSocketState.CONNECTED:
            asyncio.run(websocket.send_json({'done': final, 'image': data_to_send}))


def send_numpy_array_as_image(websocket, image: np.ndarray, final) -> None:
    """
    Send a gif to the given websocket

    Args:
        websocket (websocket): Client with will be messaged
        image (np.array): recursiv path, starting from the main directory
        final (bool): is it the last image send?
        """
    Thread(target=__send_numpy_array_as_image, args=(websocket, image, final)).start()


def __send_numpy_array_as_image(websocket, image: np.ndarray, final) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        image = img_as_ubyte(image)
        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=-1)

        image_bytes = BytesIO()
        io.imsave(image_bytes, image)

        data_to_send = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
        data_to_send = 'data:image/png;base64,' + data_to_send

        if websocket.client_state == WebSocketState.CONNECTED:
            asyncio.run(websocket.send_json({'done': final, 'image': data_to_send}))
