"""Convert a white-on-black drawing into an MNIST-sized tensor."""
import base64
import binascii
from io import BytesIO

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


def prepare_drawing(data_url: str) -> torch.Tensor:
    if not data_url.startswith("data:image/png;base64,"):
        raise ValueError("Please send a PNG drawing.")
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
        with Image.open(BytesIO(raw)) as source:
            if source.format != "PNG" or max(source.size) > 1024:
                raise ValueError("Drawing must be a PNG no larger than 1024 pixels.")
            # Composite transparency onto black before converting to brightness.
            rgba = source.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "black")
            background.alpha_composite(rgba)
            image = background.convert("L")
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Could not read this drawing as a PNG.") from error

    bounds = image.point(lambda value: 255 if value > 20 else 0).getbbox()
    if bounds is None:
        return torch.zeros(1, 28, 28)
    image = image.crop(bounds)
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28))
    canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    # MNIST centres the digit's mass, rather than only its bounding box.
    yy, xx = np.indices(pixels.shape)
    dx = round(13.5 - float((pixels * xx).sum() / pixels.sum()))
    dy = round(13.5 - float((pixels * yy).sum() / pixels.sum()))
    shifted = Image.new("L", (28, 28))
    shifted.paste(canvas, (dx, dy))
    return torch.from_numpy(np.array(shifted, dtype=np.float32) / 255.0).unsqueeze(0)
