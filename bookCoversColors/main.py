from PIL import ImageColor
import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw
import numpy as np
import tkinter as tk
from tkinter import filedialog
import colorsys
# from prompt_toolkit import prompt


def rgb_to_hex(rgb):
    """
    Convert RGB to HEX color code.

    Args:
        rgb: A tuple of the 3 RGB values

    Returns:
        (str): HEX code value

    """
    return '%02x%02x%02x' % rgb


def palette(data_colors, square_size, **kwargs):
    """

    Args:
        data_colors (DataFrame): each row is a color with its RGB values
        square_size (int): size of the colored squares

    Keyword Args:
        covers (list[Image]): list of the covers to show on the right of the
        palette
        count (list[int]): list of numbers to show below the colored squares,
        here representing the number of pixels that are closest to the above
        color

    Returns:
        palette (Image): resulting image

    """
    palette = Image.new("RGB", (data_colors.shape[0] * square_size,
                                square_size), "#ffffff")
    offset_x = 0
    for color in data_colors.itertuples():
        square = ImageDraw.Draw(palette)
        # hex colors
        if len(color) == 2:
            square.rectangle([(offset_x, 0),
                              (offset_x + square_size, square_size)],
                             fill="#" + str(color[1:]))
        else:
            square.rectangle([(offset_x, 0),
                              (offset_x + square_size, square_size)],
                             fill="#" + rgb_to_hex(tuple(color[1:])))
        offset_x += square_size

    if kwargs.get("covers") is not None:
        for cover in kwargs.get("covers"):
            palette2 = Image.new("RGB",
                                 (palette.width+20+cover.width,
                                  max(palette.height, cover.height)),
                                 "#ffffff")
            palette2.paste(palette, (0, 0))
            palette2.paste(cover, (palette2.width-cover.width, 0))
            palette = palette2.copy()

    if kwargs.get("count") is not None:
        if palette.height == square_size:
            palette2 = Image.new("RGB",
                                 (palette.width,
                                  palette.height+20),
                                 "#ffffff")
            palette2.paste(palette, (0, 0))
            palette = palette2.copy()
        t = ImageDraw.Draw(palette)
        for i in range(len(kwargs.get("count"))):
            fill = "#000"
            if np.where(np.array(kwargs.get("count")) ==
                        np.amax(np.array(kwargs.get("count"))))[0][0] == i:
                fill = "#f00"
            t.text((i * square_size, square_size),
                   str(kwargs.get("count")[i]),
                   fill=fill)

    return palette


def dominant_colors_clean(colors, round_to_multiple):
    """
    Rounds the RGB values to the closest multiple, eliminates the duplicates,
    and removes the colors too dark or too light.

    Args:
        colors (DataFrame): each row is a color with its RGB values
        round_to_multiple (int): closest multiple to round up to

    Returns:
        (DataFrame): each row is a color with its RGB values, after treatment
    """
    colors = round_to_multiple * round(colors / round_to_multiple)
    colors.drop_duplicates()

    dominant = []
    black = 0
    white = 0
    for i, r, g, b in colors.itertuples():
        if not (r, g, b) <= (60, 60, 60):
            if not (r, g, b) >= (230, 230, 230):
                dominant.append([r, g, b])
            else:
                white += 1
        else:
            black += 1

    if white > 0:
        colors.append([255, 255, 255])
    if black > 0:
        colors.append([0, 0, 0])
    return pd.DataFrame(colors)


def get_image(url, max_width):
    """
    Gets image from URL.

    Args:
        url (str): URL to the picture
        max_width (int): resize image if above maximum width

    Returns:
        (Image): image from the URL
    """
    image = Image.open(requests.get(url, stream=True).raw)
    image = image.convert("RGB")
    if image.size[0] > max_width:
        factor = max_width / image.size[0]
        image = image.resize((int(image.size[0]*factor),
                              int(image.size[1]*factor)))
    return image


def dominant_colors(im, max_num_colors, mode):
    """
    Finds the dominant colors in an image with K-means clustering.

    Args:
        im (Image): image to analyze
        max_num_colors (int): maximum number of colors to find
        mode (str):  use "RGB" or "HSV" colors

    Returns:
        dominant_colors (DataFrame): each row is a dominant color in the
        image with its RGB values
        colors (DataFrame): each row is a pixel of the image with its RGB or
        HSV values
    """
    if mode == "HSV":
        colors = pd.DataFrame(columns=["red", "green", "blue"])
    else:
        colors = pd.DataFrame(columns=["hue", "value", "saturation"])

    for i in range(im.size[1]):
        for j in range(im.size[0]):
            p = im.getpixel((i, j))
            if mode == "HSV":
                p = colorsys.rgb_to_hsv(p[0], p[1], p[2])
            colors.loc[colors.shape[0]] = p

    scaler = MinMaxScaler()
    c_scale = pd.DataFrame(scaler.fit_transform(colors))
    km = KMeans(n_clusters=max_num_colors)
    km.fit_predict(c_scale)
    dominant_colors = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=["red", "green", "blue"])
    if mode == "HSV":
        for i, h, s, v in dominant_colors.itertuples():
            rgb = colorsys.hsv_to_rgb(h, s, v)
            dominant_colors.loc[i] = rgb
    dominant_colors = dominant_colors_clean(dominant_colors, 10).astype(int)
    return dominant_colors, colors


# def addColors(row, file, colors, start):
#     for i in range(colors.shape[1]):
#         file.iat[row, start] = colors.iloc[0, i]
#         file.iat[row, start + 1] = colors.iloc[1, i]
#         file.iat[row, start + 2] = colors.iloc[2, i]
#         file.iat[row, start + 3] = "#" + rgb_to_hex((colors.iloc[0, i],
#         colors.iloc[1, i], colors.iloc[2, i]))
#         start = start + 4


def get_file():
    """
    Shows dialogue window to open the CSV file.

    Returns:
        file (str): directory of the file
    """
    root = tk.Tk()
    root.withdraw()
    file = filedialog.askopenfilename(
        initialdir="C:/Users/MainFrame/Desktop/",
        title="Open CSV file",
        filetypes=(("CSV Files", "*.csv"),)
    )
    return file


def get_urls_from_file():
    """
    Get the URLs from a file.

    Returns:
        urls (DataFrame): column of URLs from the CSV file
    """
    file = pd.read_csv(get_file())
    column = prompt("Number of the column containing the covers :\n")
    urls = file.iloc[:, int(column) - 1]


def add_color_columns(file, max_num_colors, mode):
    """
    Add new columns to the copy of the original file.

    Args:
        file (DataFrame): opened file
        max_num_colors (int): maximum number of colors to find in the pictures
        mode (str):  use "RGB" or "HSV" colors

    Returns:

    """
    last_column = file.shape[1]
    empty_column = [None] * file.shape[0]
    # if mode == "RGB":
    #     mode = ["RGB", "red", "green", "blue"]
    # else:
    #     mode = ["HSV", "hue", "value", "saturation"]

    # for i in range(max_num_colors):
    #     file["Color " + str(i) + " : RGB"] = empty_column
    #     file["Color " + str(i) + " : red"] = empty_column
    #     file["Color " + str(i) + " : green"] = empty_column
    #     file["Color " + str(i) + " : blue"] = empty_column
    #     file["Color " + str(i) + " : Hex code"] = empty_column

for j in range(urls.size):
    addColors(j, file, dominantColors(getImage(urls[j]), 4), lastColumn)

hexColors = pd.DataFrame(file.loc[:, "Color 1 : Hex code"].copy())
hue = []
value = []
saturation = []
notionEq = []

for i in range(hexColors.size):
    r = ImageColor.getcolor(hexColors.iloc[i, 0], "RGB")[0]
    g = ImageColor.getcolor(hexColors.iloc[i, 0], "RGB")[1]
    b = ImageColor.getcolor(hexColors.iloc[i, 0], "RGB")[2]
    hsv = colorsys.rgb_to_hsv(r, g, b)
    hue.append(hsv[0])
    value.append(hsv[1])
    saturation.append(hsv[2])
    notionEq.append("$$\color{" + str(hexColors.iloc[i, 0]) + "}\\" + str(hexColors.iloc[i, 0]) + "$$")

file["Hue"] = hue
file["Value"] = value
file["Saturation"] = saturation
file["Notion equation"] = notionEq

byHue = file.loc[:, ["Color 1 : Hex code", "Hue", "Saturation"]].copy()
index = np.arange(0, byHue.shape[0], 1)
byHue["ogIndex"] = index
for i in range(byHue.shape[0]):
    byHue.loc[i, "Hue"] = 5 * round((byHue.loc[i, "Hue"] * 100) / 5)
byHue = byHue.sort_values(by="Hue").reset_index(drop=True)

j = 0
k = 0
hue = byHue.loc[0, "Hue"]
change = False
for i in range(1, byHue.shape[0]):
    if hue != byHue.loc[i, "Hue"]:
        replace = byHue.loc[j:k, :].sort_values(ascending=change, by=["Saturation"]).copy()
        byHue.iloc[j:k + 1, :] = replace.to_numpy().tolist()
        change = not change
        j = i
        k = i
        hue = byHue.loc[j, "Hue"]
    else:
        k += 1

orderRainbow = [None] * byHue.shape[0]
for i in range(0, byHue.shape[0]):
    orderRainbow[byHue.loc[i, ["ogIndex"]][0]] = i

file["Rainbow order"] = orderRainbow

file.to_csv('colors.csv')
