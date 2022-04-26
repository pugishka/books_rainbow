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
from io import BytesIO
from pandastable import Table, TableModel
import pdfkit


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

        mode (:obj:`list` of str):  use "RGB" or "HSV" colors

    Returns:
        dominant_colors (:obj:`list` of DataFrame): list of DataFrame where
            each row is a dominant color in the image with its RGB values,
            based on RGB values, HSV values, or both

        colors (:obj:`list` of DataFrame): list of DataFrame where each row
            is a pixel of the image with its RGB or HSV values
    """
    colors = []
    r = np.array(im.getdata())
    for m in mode:
        if m == "RGB":
            colors.append(pd.DataFrame(r, columns=["red", "green", "blue"]))
        elif m == "HSV":
            r = [colorsys.rgb_to_hsv(r[i][0], r[i][1], r[i][2])
                 for i in range(r.shape[0])]
            colors.append(pd.DataFrame(r, columns=["hue",
                                                   "saturation",
                                                   "value"]))

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


def save_results_to():
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Save resulting files",
    )
    return folder


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


def analyse(urls, mode, max_n_palette):
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
                               [max_n_palette]*len(urls),
                               [mode]*len(urls)
                               )
                           )
                      )
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


def palette_path_save(i, mode, folder):
    name_file = str(i) + "_" + mode + ".jpg"
    path = folder + "/palettes/" + mode + "/" + name_file
    return path


def open_csv(frame):
    # name = get_file()
    name = "C:/Users/charo/Downloads/Export-e8d05104-a27b-4b14-ace2" \
           "-f7ddae1ea7c7/Bookshelf cafd6.csv"
    file = pd.read_csv(name)
    pt = Table(frame)


def start_analysis(
        urls,
        progress,
        palette_square_size,
        palette_with_cover,
        palette_with_replaced_cover,
        palette_with_count,
        mode,
        generate_html,
        generate_csv,
        generate_pdf,
        generate_palettes,
        max_n_palette
):

    # start_time = time.perf_counter()

    #n = len(urls)
    n = 5
    folder_files = save_results_to()
    m = []
    if mode[0]:
        m.append("RGB")
    if mode[1]:
        m.append("HSV")
    mode = m

    # if mode = "both
    # dominant_colors_all[x][0] = dominant colors analysed in RGB for x
    # dominant_colors_all[x][1] = dominant colors analysed in HSV for x
    # colors_all[x][0] = RGB pixels of x
    # colors_all[x][1] = HSV pixels of x
    progress.insert(tk.INSERT, "Finding colors...\n")
    dominant_colors_all, colors_all, covers = analyse(urls, mode, max_n_palette)
    progress.insert(tk.INSERT, "Done\n")

    results = []
    nb_mode = len(mode)
    columns = ["Cover"]
    if generate_palettes:
        columns.append("Result ")
    columns.extend(["RGB", "HSV", "Hex"])
    if generate_palettes:
        for m in mode:
            c = columns
            c[1] += m
            results.append(pd.DataFrame(columns=c))
    else:
        for m in mode:
            results.append(pd.DataFrame(columns=columns))

    for j in range(nb_mode):
        progress.insert(
            tk.INSERT,
            "Finding dominant colors for " + mode[j] + "...\n"
        )
        for i in range(n):
            progress.insert(
                tk.INSERT,
                u"\nImage number " + str(i+1) + " out of " + str(n)
            )
            count = []
            dom_color = []
            replaced_cover = []
            count_m, dom_color_m, replaced_cover_m = \
                count_colors(dominant_colors_all[i][j],
                             colors_all[i][j],
                             covers[i])
            count.append(count_m)
            dom_color.append(dom_color_m)
            replaced_cover.append(replaced_cover_m)

            if generate_palettes:
                covers_arg = []
                count_arg = []
                if palette_with_count:
                    count_arg = count_m
                if palette_with_cover:
                    covers_arg.append(covers[i])
                if palette_with_replaced_cover:
                    covers_arg.append(replaced_cover_m)
                pal = palette(
                    dominant_colors_all[i][j],
                    palette_square_size,
                    covers=covers_arg,
                    count=count_arg
                )
                palette_dir = palette_path_save(i, mode[j], folder_files)
                pal.save(palette_dir)

                results[j].loc[results[j].shape[0]] = [
                    urls[i],
                    palette_dir,
                    dom_color_m.tolist(),
                    list(colorsys.rgb_to_hsv(*dom_color_m)),
                    "#" + rgb_to_hex(tuple(dom_color_m))
                ]
            else:
                results[j].loc[results[j].shape[0]] = [
                    urls[i],
                    dom_color_m.tolist(),
                    list(colorsys.rgb_to_hsv(*dom_color_m)),
                    "#" + rgb_to_hex(tuple(dom_color_m))
                ]

            progress.delete("end-1l", "end")

        progress.insert(
            tk.INSERT,
            "Finding dominant colors for " + mode[j] + " : Done\n"
        )

        progress.insert(
            tk.INSERT,
            "Finding rainbow order...\n"
        )
        hsv = pd.DataFrame(results[j].loc[:, "HSV"].copy())

        results[j]["orderRainbow"] = get_order_rainbow(hsv)
        # results[j] = results[j].sort_values(by=["orderRainbow"])

        progress.insert(
            tk.INSERT,
            "Finding rainbow order : Done\n"
        )

        progress.insert(
            tk.INSERT,
            "Generating files...\n"
        )

        if generate_csv:
            csv_dir = folder_files + "/results_" + mode[j] + ".csv"
            results[j].to_csv(csv_dir)

        if generate_palettes:
            format_mode = {"Cover": path_to_cover,
                           "Result " + mode[j]: path_to_palette}
        else:
            format_mode = {"Cover": path_to_cover}

        html_dir = folder_files + "/results_" + mode[j] + ".html"

        if generate_html:
            results[j].to_html(
                html_dir,
                escape=False,
                formatters=format_mode
            )

        if generate_pdf:
            if not generate_html:
                html_dir = results[j].to_html(
                    escape=False,
                    formatters=format_mode
                )
            pdf_dir = folder_files + "/results_" + mode[j] + ".pdf"
            pdfkit.from_file(
                html_dir,
                pdf_dir)

        progress.insert(
            tk.INSERT,
            "Generating files : Done\n"
        )

        progress.insert(
            tk.INSERT,
            "\nResults are ready !\n"
        )

    # end_time = time.perf_counter()
    # print(f"end : {end_time - start_time:0.6f}")
