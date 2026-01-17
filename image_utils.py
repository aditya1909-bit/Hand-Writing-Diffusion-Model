from PIL import Image


def resize_with_pad(image, size, fill=255):
    target_h, target_w = size
    resized = image.copy()
    resized.thumbnail((target_w, target_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), color=(fill, fill, fill))
    paste_x = (target_w - resized.width) // 2
    paste_y = (target_h - resized.height) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas
