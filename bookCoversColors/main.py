import traceback

import pandas as pd
from requests.exceptions import MissingSchema
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from PIL import Image as PilImage
from PIL import ImageDraw as PilImageDraw
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import filedialog, scrolledtext
import colorsys
import time
from concurrent.futures import as_completed
from requests_futures.sessions import FuturesSession
from sklearn.metrics.pairwise import euclidean_distances
import requests

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

    palette = PilImage.new("RGB", (data_colors.shape[0] * square_size,
                                square_size), "#ffffff")
    offset_x = 0
    for color in data_colors.itertuples():
        square = PilImageDraw.Draw(palette)
        # hex colors
        if len(color) == 2:
            square.rectangle((offset_x, 0, offset_x + square_size, square_size),
                             fill="#" + str(color[1:]))
        else:
            square.rectangle((offset_x, 0, offset_x + square_size, square_size),
                             fill="#" + rgb_to_hex(tuple(color[1:])))
        offset_x += square_size

    if kwargs.get("covers") is not None:
        for cover in kwargs.get("covers"):
            palette2 = PilImage.new("RGB",
                                 (palette.width+20+cover.width,
                                  max(palette.height, cover.height)),
                                 "#ffffff")
            palette2.paste(palette, (0, 0))
            palette2.paste(cover, (palette2.width-cover.width, 0))
            palette = palette2.copy()

    if kwargs.get("count") is not None:
        if palette.height == square_size:
            palette2 = PilImage.new("RGB",
                                 (palette.width,
                                  palette.height+20),
                                 "#ffffff")
            palette2.paste(palette, (0, 0))
            palette = palette2.copy()
        t = PilImageDraw.Draw(palette)
        total = sum(kwargs.get("count"))
        for i in range(len(kwargs.get("count"))):
            fill = "#000"
            if np.where(np.array(kwargs.get("count")) ==
                        np.amax(np.array(kwargs.get("count"))))[0][0] == i:
                fill = "#f00"
            percent = (kwargs.get("count")[i]*100)/total
            t.text((i * square_size, square_size),
                   str(round(percent)) + "%",
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


def get_images(urls, max_width):
    """
    Gets images from URLs.

    Args:
        urls (:obj:`list` of str): URLs to the pictures
        max_width (int): resize image if above maximum width

    Returns:
        (:obj:`list` of Image): images from the URLs
    """
    urls = np.array(urls)
    images = np.array([None]*len(urls))
    with FuturesSession() as session:
        futures = [session.get(urls[i], stream=True) for i in range(len(urls))]
        for future in as_completed(futures):
            ind = np.where(urls == future.result().url)[0][0]
            im = PilImage.open(future.result().raw).convert("RGB")
            factor = max_width / im.size[0]
            images[ind] = im.resize((int(im.size[0]*factor),
                                     int(im.size[1]*factor)))

    return images
    # image = Image.open(session.get(url, stream=True).raw)
    # image = image.convert("RGB")
    # if image.size[0] > max_width:
    #
    #     image = image.resize((int(image.size[0]*factor),
    #                           int(image.size[1]*factor)))
    # return image


def dominant_colors(im, max_num_colors, mode):
    """
    Finds the dominant colors in an image with K-means clustering.

    Args:
        im (Image): image to analyze

        max_num_colors (int): maximum number of colors to find

        mode (str):  use "RGB" or "HSV" colors

    Returns:
        dominant_colors (:obj:`list` of DataFrame): list of DataFrame where
            each row is a dominant color in the image with its RGB values,
            based on RGB values, HSV values, or both

        colors (:obj:`list` of DataFrame): list of DataFrame where each row
            is a pixel of the image with its RGB or HSV values
    """
    colors = []
    r = np.array(im.getdata())
    if mode == "RGB":
        colors.append(pd.DataFrame(r, columns=["red", "green", "blue"]))
    elif mode == "HSV":
        r = [colorsys.rgb_to_hsv(r[i][0], r[i][1], r[i][2])
             for i in range(r.shape[0])]
        colors.append(pd.DataFrame(r, columns=["hue", "saturation", "value"]))
    else:
        colors.append(pd.DataFrame(r, columns=["red", "green", "blue"]))
        r = [colorsys.rgb_to_hsv(r[i][0], r[i][1], r[i][2])
             for i in range(r.shape[0])]
        colors.append(pd.DataFrame(r, columns=["hue", "saturation", "value"]))

    scaler = MinMaxScaler()
    dominant_colors = []
    for i in range(len(colors)):
        c_scale = pd.DataFrame(scaler.fit_transform(colors[i]))
        km = KMeans(n_clusters=max_num_colors)
        km.fit_predict(c_scale)
        dominant_colors.append(pd.DataFrame(
            scaler.inverse_transform(km.cluster_centers_),
            columns=["red", "green", "blue"]))
        if colors[i].columns.tolist() == ["hue", "saturation", "value"]:
            for ind, h, s, v in dominant_colors[i].itertuples():
                rgb = colorsys.hsv_to_rgb(h, s, v)
                dominant_colors[i].loc[ind] = rgb
        dominant_colors[i] = \
            dominant_colors_clean(dominant_colors[i], 10).astype(int)
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
    root = Tk()
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
    # url = "C:/Users/charo/Downloads/Export-e8d05104-a27b-4b14-ace2" \
    #       "-f7ddae1ea7c7/Bookshelf cafd6.csv"
    url = get_file()
    file = pd.read_csv(url)
    column = input("Number of the column containing the covers :\n")
    # column = 4
    urls = file.iloc[:, int(column) - 1]
    return urls


def analyse(urls):
    """
    Analyses each cover at the URL and finds their RGB, HSV and dominant colors.

    Args:
        urls (:obj:`list` of str): list of the covers' URLs

    Returns:
        dominant_colors_all (:obj:`list` of :obj:`list` of DataFrame): list
            for each cover of lists containing a DataFrame of the dominant
            colors found with RGB mode and HSV mode

        colors_all (:obj:`list` of :obj:`list` of DataFrame): list for each
            cover of lists containing a DataFrame of all the pixels in RGB
            and HSV

        covers (:obj:`list` of Image): list of all the covers
    """

    covers = get_images(urls, 50)
    result = np.array(list(map(dominant_colors,
                               covers,
                               [4]*len(urls),
                               ["both"]*len(urls)
                               )
                           )
                      )
    print("done")
    dominant_colors_all = result[:, 0]
    colors_all = result[:, 1]

    # for i in range(len(urls)):
    #     # print("analyse #" + str(i))
    #     # dom[0] = RGB dom
    #     # colors[0] = RGB
    #     dom, colors = dominant_colors(covers[i], 4, "both")
    #     dominant_colors_all.append(dom)
    #     .append(colors)

    return dominant_colors_all, colors_all, covers


def count_colors(dominants, colors, cover):
    """
    For each dominant color, counts how many are closest to it.

    Args:
        dominants (DataFrame): each row is a dominant color with its RGB values
        colors (DataFrame): each row is a pixel from the cover with its RGB
            values
        cover (Image): image of the cover

    Returns:
        count (:obj:`list` of int): number of pixels closest to each dominant
            color
        dom_color (:obj:`list` of int): RGB values of the dominant color
        replaced_cover (Image): cover replaced with the dominant colors

    """
    replaced_cover = cover.copy()
    dist_min = np.argmin(euclidean_distances(colors, dominants), axis=1)
    colors_replaced = map(list(map(tuple, np.array(dominants))).__getitem__,
                          dist_min)
    replaced_cover.putdata(list(colors_replaced))
    count = [(dist_min == i).sum() for i in range(dominants.shape[0])]
    dom_color = dominants.iloc[np.argmax(count), :]

    return count, dom_color, replaced_cover


def get_order_rainbow(hsv):
    hue = pd.DataFrame([v for i, v in hsv.itertuples()],
                       columns=["h", "s", "v"])
    hue.reset_index(inplace=True)
    hue.loc[:, "h"] = 5 * round((hue.loc[:, "h"] * 100) / 5)

    # sort by hue
    hue = hue.sort_values(by="h").reset_index(drop=True)

    # for each hue, sort saturation alternatively
    hue_value = hue.loc[0, "h"]
    start = 0
    next_hue = np.where(np.array(hue.loc[:, "h"]) > hue_value)[0]
    change = True

    while next_hue.size != 0:
        i = next_hue[0]
        replace = hue.loc[start:i - 1, :].sort_values(ascending=change,
                                                      by=["s"]).copy()
        hue.loc[start:i - 1, :] = np.array(replace)
        hue.reset_index(drop=True, inplace=True)
        change = not change
        start = i
        hue_value = hue.loc[i, "h"]
        next_hue = np.where(np.array(hue.loc[:, "h"]) > hue_value)[0]

    # for each saturation, sort value alternatively
    hue.loc[:, "s"] = 5 * round((hue.loc[:, "s"] * 100) / 5)
    s_value = hue.loc[0, "s"]
    start = 0
    next_s = np.where(np.array(hue.loc[:, "s"]) > s_value)[0]
    change = True
    while next_s.size != 0:
        i = next_s[0]
        replace = hue.loc[start:i - 1, :].sort_values(ascending=change,
                                                      by=["v"]).copy()
        hue.loc[start:i - 1, :] = np.array(replace)
        hue.reset_index(drop=True, inplace=True)
        change = not change
        start = i
        s_value = hue.loc[i, "s"]
        next_s = np.where(np.array(hue.loc[:, "s"]) > s_value)[0]
    hue.reset_index(inplace=True)
    hue = hue.sort_values(by=["index"])

    return np.array(hue.iloc[:, 0]).astype(int)


def path_to_cover(path):
    return '<img src="' + path + '" width="50px">'


def path_to_palette(path):
    return '<img src="' + path + '">'


def main():

    start_time = time.perf_counter()

    # get list of URLs to the covers
    urls = get_urls_from_file().tolist()[:1]
    n = len(urls)

    palette_square_size = 50
    palette_with_cover = True
    palette_with_replaced_cover = True
    palette_with_count = True
    path_palette_rgb = "palettes_covers"
    path_palette_hsv = "palettes_covers"

    # s False if we already have a file of results
    s = True
    if s:

        dominant_colors_all, colors_all, covers = analyse(urls)

        # dominant_colors_all[x][0] = RGB dom of x
        # dominant_colors_all[x][1] = HSV dom of x
        # colors_all[x][0] = RGB of x
        # colors_all[x][1] = HSV of x
        results_rgb = pd.DataFrame(columns=["Cover", "Result RGB", "RGB", "HSV",
                                            "Hex"])
        results_hsv = pd.DataFrame(columns=["Cover", "Result HSV", "RGB", "HSV",
                                            "Hex"])

        # for each url
        for i in range(n):
            count_rgb, dom_color_rgb, replaced_cover_rgb = \
                count_colors(dominant_colors_all[i][0],
                             colors_all[i][0],
                             covers[i])

            count_hsv, dom_color_hsv, replaced_cover_hsv = \
                count_colors(dominant_colors_all[i][1],
                             colors_all[i][0],
                             covers[i])

            p_rgb = palette(dominant_colors_all[i][0],
                            palette_square_size,
                            covers=[covers[i], replaced_cover_rgb],
                            count=count_rgb)

            p_hsv = palette(dominant_colors_all[i][1],
                            palette_square_size,
                            covers=[covers[i], replaced_cover_hsv],
                            count=count_hsv)

            p_rgb.save(path_palette_rgb + "/" + str(i) + "_rgb.jpg")
            p_hsv.save(path_palette_hsv + "/" + str(i) + "_hsv.jpg")

            results_rgb.loc[results_rgb.shape[0]] = \
                [urls[i],
                 path_palette_rgb + "/" + str(i) + "_rgb.jpg",
                 dom_color_rgb.tolist(),
                 list(colorsys.rgb_to_hsv(dom_color_rgb[0],
                                          dom_color_rgb[1],
                                          dom_color_rgb[2])),
                 "#" + rgb_to_hex((dom_color_rgb[0],
                                   dom_color_rgb[1],
                                   dom_color_rgb[2]))]

            results_hsv.loc[results_hsv.shape[0]] = \
                [urls[i],
                 path_palette_rgb + "/" + str(i) + "_hsv.jpg",
                 dom_color_hsv.tolist(),
                 list(colorsys.rgb_to_hsv(dom_color_hsv[0],
                                          dom_color_hsv[1],
                                          dom_color_hsv[2])),
                 "#" + rgb_to_hex((dom_color_hsv[0],
                                   dom_color_hsv[1],
                                   dom_color_hsv[2]))]

        rgb_hsv = pd.DataFrame(results_hsv.loc[:, "HSV"].copy())
        hsv_hsv = pd.DataFrame(results_hsv.loc[:, "HSV"].copy())

    else:
        results_rgb = pd.read_csv('results_rgb.csv')
        rgb_hsv = pd.DataFrame(results_rgb.loc[:, "HSV"].copy())
        results_hsv = pd.read_csv('results_hsv.csv')
        hsv_hsv = pd.DataFrame(results_hsv.loc[:, "HSV"].copy())
        for i, s in rgb_hsv.itertuples():
            rgb_hsv.iloc[i, 0] = list(map(float, s[1:len(s)-1].split(', ')))
        for i, s in hsv_hsv.itertuples():
            hsv_hsv.iloc[i, 0] = list(map(float, s[1:len(s)-1].split(', ')))

    results_rgb["orderRainbow"] = get_order_rainbow(rgb_hsv)
    results_hsv["orderRainbow"] = get_order_rainbow(hsv_hsv)
    results_rgb = results_rgb.sort_values(by=["orderRainbow"])
    results_hsv = results_hsv.sort_values(by=["orderRainbow"])

    results_rgb.to_csv('results_rgb.csv')
    results_hsv.to_csv('results_hsv.csv')

    format_rgb = {"Cover": path_to_cover,
                  "Result RGB": path_to_palette}

    format_hsv = {"Cover": path_to_cover,
                  "Result HSV": path_to_palette}

    results_rgb.to_html('results_rgb.html', escape=False,
                        formatters=format_rgb)
    results_hsv.to_html('results_hsv.html', escape=False,
                        formatters=format_hsv)

    end_time = time.perf_counter()
    print(f"end : {end_time - start_time:0.6f}")
    print("palette")

    palette_all = PilImage.new("RGB", (n * 3, 100), "#ffffff")
    offset_x = 0
    images = get_images(results_hsv.loc[:, "Cover"], 50)
    for im in images:
        palette_all.paste(im.resize((3, 100)), (offset_x, 0))
        offset_x += 3

    palette_all.save(path_palette_hsv + "/all_covers_hsv.jpg")


def donothing():
    return


from pandastable import Table, TableModel


def open_csv(frame):
    # name = get_file()
    name = "C:/Users/charo/Downloads/Export-e8d05104-a27b-4b14-ace2" \
           "-f7ddae1ea7c7/Bookshelf cafd6.csv"
    file = pd.read_csv(name)
    pt = Table(frame)


def main_window():
    root = Tk()
    menubar = Menu(root)
    file_menu = Menu(menubar, tearoff=0)
    file_menu.add_command(label="Open", command=donothing)
    file_menu.add_command(label="Save", command=donothing)
    file_menu.add_command(label="Save as...", command=donothing)
    file_menu.add_separator()
    file_menu.add_command(label="Close", command=donothing)
    menubar.add_cascade(label="File", menu=file_menu)
    root.config(menu=menubar)

    frame = Frame(root)
    frame.grid(row=0, column=0)
    name = "C:/Users/charo/Downloads/Export-e8d05104-a27b-4b14-ace2" \
           "-f7ddae1ea7c7/Bookshelf cafd6.csv"
    pt = Table(frame, dataframe=pd.read_csv(name))
    pt.show()

    frame2 = Frame(root)
    frame2.grid(row=0, column=1)

    Label(frame2, text='Column').grid(row=0, column=0)

    values = list(pt.model.df.columns)
    variable = StringVar()
    option_menu = OptionMenu(
        frame2,
        variable,
        *values,
    )
    option_menu.grid(row=0, column=1)

    progress = scrolledtext.ScrolledText(
        frame2,
        wrap=tk.WORD,
        width=30,
        height=10)

    progress.grid(row=1, columnspan=3)
    #progress.configure(state="disabled")

    Button(frame2, text="OK",
           command=lambda: check_column(
               variable.get(),
               pt.model.df,
               progress
           )).grid(row=0, column=2)

    root.mainloop()


import requests
from io import BytesIO


def check_column(col, df, progress):
    try:
        url = df.loc[0, col]
        response = requests.get(url)
        im = PilImage.open(BytesIO(response.content))
    except MissingSchema:
        progress.insert(
            tk.INSERT,
            "The column should include links to pictures.\n")
        return False
    except Exception as e:
        progress.insert(
            tk.INSERT,
            "An error occurred.\n")
        progress.insert(
            tk.INSERT,
            "   " + type(e).__name__ + "\n")
        return False
    else:
        progress.insert(tk.INSERT, "OK\n")
        return True



if __name__ == "__main__":
    main_window()
