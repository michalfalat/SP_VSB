import cv2


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    if image is None:
        return image
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
