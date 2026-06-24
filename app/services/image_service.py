import base64
from io import BytesIO
from PIL import Image
import numpy as np


def image_to_base64(image_array: np.ndarray) -> str:

    image = Image.fromarray(image_array)

    buffered = BytesIO()

    image.save(buffered, format="JPEG")

    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_str
