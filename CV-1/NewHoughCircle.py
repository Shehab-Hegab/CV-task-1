from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
import myCanny
from collections import defaultdict
import numpy as np


def Hough_circle(img,edged):
    # Output image:
    img = Image.fromarray(img) 
    output_image = Image.new("RGB", img.size)
    output_image.paste(img)
    draw_result = ImageDraw.Draw(output_image)

    # Find circles
    rmin = 18
    rmax = 20
    steps = 100
    threshold = 0.4

    points = []
    for r in range(rmin, rmax + 1):
        for t in range(steps):
            points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

    acc = defaultdict(int)
    
    #indices = np.where(edged != [0])
    #coordinates = zip(indices[0], indices[1])
    #edgedSet = list(coordinates)
    for x in range (edged.shape[1]):
        for y in range(edged.shape[0]):
            if(edged.item(y,x) !=0):
                for r,dx,dy in points:
                    a = x - dx
                    b = y - dy
                    acc[(a, b, r)] += 1

    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            print(v / steps, x, y, r)
            circles.append((x, y, r))

    for x, y, r in circles:
        draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255,0,0,0))

    # Save output image
    output_image.save("result.png")
    return np.array(output_image)