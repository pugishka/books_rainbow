import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from PIL import Image as PilImage
from PIL import ImageDraw as PilImageDraw
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import colorsys
from concurrent.futures import as_completed
from requests_futures.sessions import FuturesSession
from sklearn.metrics.pairwise import euclidean_distances
import pdfkit

config = pdfkit.configuration(
    wkhtmltopdf='C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe')

# Style for the HTML and PDF files
STYLE_HTML = """
<style>
  table { border-collapse: collapse }
  th, td { padding: 5px 20px }
  tr { text-align: center !important }
  thead { background-color: #def; height: 50px }
</style>
    """


def rgb_to_hex(rgb):
    """
    Convert RGB to HEX color code.

    Parameters
    ----------
    rgb : tuple
        A tuple of the 3 RGB values

    Returns
    -------
    str
        HEX code value

    """
    return '%02x%02x%02x' % rgb


def palette(data_colors, square_size, **kwargs):
    """
    Creates a palette image.

    Parameters
    ----------
    data_colors : pandas.DataFrame
        Each row is a color with its RGB values or single HEX value.
    square_size : int
        Size of the colored squares.

    Other Parameters
    ----------------
    covers : list of Image
        List of the covers to show on the right of the palette.
    count : list of int
        List of numbers representing the number of pixels that are closest
        to the corresponding color in data_color.

    Returns
    -------
    palette : Image
        resulting image

    """

    palette = PilImage.new(
        "RGB",
        (data_colors.shape[0] * square_size, square_size),
        "#ffffff"
    )

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
            palette2 = PilImage.new(
                "RGB",
                (palette.width+20+cover.width,
                 max(palette.height, cover.height)),
                "#ffffff"
            )
            palette2.paste(palette, (0, 0))
            palette2.paste(cover, (palette2.width-cover.width, 0))
            palette = palette2.copy()

    if kwargs.get("count") is not None:
        if palette.height == square_size:
            palette2 = PilImage.new(
                "RGB",
                (palette.width, palette.height+20),
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
    and removes colors too dark or too light.

    Parameters
    ----------
    colors : pandas.DataFrame
        Each row is a color with its RGB values.
    round_to_multiple : int
        Closest multiple to round up to.

    Returns
    -------
    pandas.DataFrame
        Each row is a color with its RGB values.

    """

    colors = round_to_multiple * (colors.divide(round_to_multiple)).round()
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


def get_images(progress, urls, max_width):
    """
    Gets images from URLs.

    Parameters
    ----------
    progress : tkinter.ScrolledText
        ScrolledText widget to show progress.
    urls : list of str
        URLs to the pictures.
    max_width : int
        Resize image if above maximum width.

    Returns
    -------
    images : list of Image
        Images from the URLs.

    """

    progress_update(
        progress,
        "\nRetrieving images from list of URLs...\n"
    )
    count = 1
    urls = np.array(urls)
    images = np.array([None]*len(urls))
    with FuturesSession() as session:
        futures = [session.get(urls[i], stream=True) for i in range(len(urls))]
        for future in as_completed(futures):
            progress.delete("end-1l", "end")
            progress_update(
                progress,
                u"\nRetrieved " + str(count) + " covers out of " +
                str(len(urls))
            )
            count += 1
            ind = np.where(urls == future.result().url)[0][0]
            im = PilImage.open(future.result().raw).convert("RGB")
            factor = max_width / im.size[0]
            images[ind] = im.resize((int(im.size[0]*factor),
                                     int(im.size[1]*factor)))

    progress_update(
        progress,
        "\nRetrieving images from list of URLs : Done\n"
    )
    return images


def dominant_colors(im, max_num_colors, mode):
    """
    Finds the dominant colors in an image with K-means clustering.

    Parameters
    ----------
    im : Image
        Image to analyze.
    max_num_colors : int
        Maximum number of colors to find.
    mode : list of str
        List of modes to use to retrieve colors from the picture. Currently
        "RGB" and/or "HSV" can be used.

    Returns
    -------
    dominant_colors : list of pandas.DataFrame
        List of DataFrames for each mode where each row is a dominant color in
        the image with its RGB values
    colors : list of pandas.DataFrame
        List of DataFrames for each mode where each row is a pixel of the image
        with its values in the corresponding mode.

    """

    colors = []
    r = np.array(im.getdata())
    for m in mode:
        if m == "RGB":
            colors.append(pd.DataFrame(r, columns=["red", "green", "blue"]))
        elif m == "HSV":
            r = [colorsys.rgb_to_hsv(r[i][0], r[i][1], r[i][2])
                 for i in range(r.shape[0])]
            df = pd.DataFrame(r, columns=["hue", "saturation", "value"])
            colors.append(df)

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


def get_file():
    """
    Opens dialogue window to open a CSV file.

    Returns
    -------
    file : str
        Directory of the file

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
    """
    Opens dialogue window to choose a directory for saved files.

    Returns
    -------
    folder : str
        The chosen directory.
    """

    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Save resulting files",
    )
    return folder


def analyse(progress, urls, mode, max_n_palette):
    """
    Analyses each cover at the URL and gets their colors and dominant colors.

    Parameters
    ----------
    progress : tkinter.ScrolledText
        ScrolledText widget to show progress.
    urls : list of str
        List of the covers' URLs.
    mode : list of str
        List of modes to use to retrieve colors from the picture. Currently
        "RGB" and/or "HSV" can be used.
    max_n_palette : int
        Average number of dominant colors to find in a picture.

    Returns
    -------
    dominant_colors_all : list of lists of pandas.DataFrame
        For each cover, for each mode, a DataFrame where rows are the dominant
        colors found in the corresponding mode.

    colors_all : list of lists of pandas.DataFrame
        For each cover, for each mode, a DataFrame where rows are the values
        of the pixels in the corresponding mode.

    covers : list of Image
        List of all the covers.

    """

    covers = get_images(progress, urls, 50)
    progress_update(
        progress,
        "\nFinding colors...\n"
    )
    result = []
    for i in range(len(urls)):
        progress.delete("end-1l", "end")
        progress_update(
            progress,
            u"\nCover #" + str(i+1) + " out of " + str(len(urls))
        )
        result.append(dominant_colors(covers[i], max_n_palette, mode))
    result = np.array(result, dtype=object)
    dominant_colors_all = result[:, 0]
    colors_all = result[:, 1]
    progress_update(
        progress,
        "\nFinding colors : Done\n"
    )
    return dominant_colors_all, colors_all, covers


def count_colors(dominants, colors, cover):
    """
    For each dominant color, counts how many are closest to it.

    Parameters
    ----------
    dominants : pandas.DataFrame
        Each row is a dominant color with its RGB values.
    colors : pandas.DataFrame
        Each row is a pixel from the cover with its values.
    cover : Image
        Image of the cover.

    Returns
    -------
    count : list of int
        Number of pixels closest to each dominant color.
    dom_color : list of int
        RGB values of the dominant color.
    replaced_cover : Image
        Cover replaced with the dominant colors.

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
    """
    Orders colors into a rainbow.

    Parameters
    ----------
    hsv : pandas.DataFrame
        Each row is a list of the HSV values of a dominant color.

    Returns
    -------
    ndarray
        List of indexes for the rainbow order.

    """
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
    """
    Format path to cover to HTML.

    Parameters
    ----------
    path : str
        Path to the cover.

    Returns
    -------
    str
        HTML format for the cover.

    """
    return '<img src="' + path + '" width="50px">'


def path_to_palette(path):
    """
    Format path to palette to HTML.

    Parameters
    ----------
    path : str
        Path to the palette.

    Returns
    -------
    str
        HTML format for the palette.

    """
    return '<img src="' + path + '">'


def palette_path_save(i, mode, folder):
    """
    Creates path for saving palettes.

    Parameters
    ----------
    i : int
        Palette index.
    mode : str
        Mode used to retrieve colors.
    folder : str
        Path to folder where results will be saved.

    Returns
    -------
    path : str
        Path to palette.

    """
    name_file = str(i) + "_" + mode + ".jpg"
    path = folder + "/palettes/" + mode + "/" + name_file
    return path


def progress_update(progress, text):
    """
    Update progress text.

    Parameters
    ----------
    progress : tkinter.ScrolledText
        ScrolledText widget to show progress.
    text : str
        Text to insert in the text field.

    """
    progress.insert(
        tk.INSERT,
        text,
    )
    progress.see("end")


def start_analysis(
        start,
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
    """
    Main function to analyse the covers.

    Parameters
    ----------
    start : tkinter.BooleanVar()
        Set it to False to indicate the end of the Thread.

    urls : list of str
        URLs of the covers.

    progress : tkinter.ScrolledText
        ScrolledText widget to show progress.

    palette_square_size : int
        Size of the squares in the palettes.

    palette_with_cover : bool
        Add the cover to the palette ?

    palette_with_replaced_cover : bool
        Add the cover replaced with its dominant colors to the palette ?

    palette_with_count : bool
        Add the % of each dominant color to the palette ?

    mode : list of bool
        For each mode, used it to analyse if True, skip if False.

    generate_html : bool
        Generate the HTML file ?

    generate_csv : bool
        Generate the CSV file ?

    generate_pdf : bool
        Generate the PDF file ?

    generate_palettes : bool
        Generate the palettes ?

    max_n_palette : int
        Average number of dominant colors to find for each cover.
    """

    urls = urls[:10]
    n = len(urls)
    folder_files = save_results_to()
    m = []
    if mode[0]:
        m.append("RGB")
    if mode[1]:
        m.append("HSV")
    mode = m

    # dominant_colors_all[x][0] = dominant colors analysed in RGB for x
    # dominant_colors_all[x][1] = dominant colors analysed in HSV for x
    # colors_all[x][0] = RGB pixels of x
    # colors_all[x][1] = HSV pixels of x
    dominant_colors_all, colors_all, covers = \
        analyse(progress, urls, mode, max_n_palette)

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
        results.append(pd.DataFrame(columns=columns))

    for j in range(nb_mode):
        progress_update(
            progress,
            "\nFinding dominant colors for " + mode[j] + "...\n"
        )
        for i in range(n):
            progress.delete("end-1l", "end")
            progress_update(
                progress,
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
                    [round(
                        list(colorsys.rgb_to_hsv(*dom_color_m))[i],
                        3
                    ) for i in range(
                            len(list(colorsys.rgb_to_hsv(*dom_color_m)))
                        )
                    ],
                    round(*list(colorsys.rgb_to_hsv(*dom_color_m)), 3),
                    "#" + rgb_to_hex(tuple(dom_color_m))
                ]
            else:
                results[j].loc[results[j].shape[0]] = [
                    urls[i],
                    dom_color_m.tolist(),
                    [round(
                        list(colorsys.rgb_to_hsv(*dom_color_m))[i],
                        3
                    ) for i in range(
                        len(list(colorsys.rgb_to_hsv(*dom_color_m)))
                    )
                    ],
                    "#" + rgb_to_hex(tuple(dom_color_m))
                ]

        progress_update(
            progress,
            "\nFinding dominant colors for " + mode[j] + " : Done\n"
        )

        progress_update(
            progress,
            "\nFinding rainbow order...\n"
        )
        hsv = pd.DataFrame(results[j].loc[:, "HSV"].copy())

        results[j]["Rainbow order"] = get_order_rainbow(hsv)
        # results[j] = results[j].sort_values(by=["orderRainbow"])

        progress_update(
            progress,
            "Finding rainbow order : Done\n"
        )

        progress_update(
            progress,
            "\nGenerating files...\n"
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
            file_html = results[j].to_html(
                escape=False,
                formatters=format_mode
            )
            file_html = STYLE_HTML + file_html
            html_open = open(html_dir, "w")
            html_open.write(file_html)
            html_open.close()

        if generate_pdf:
            pdf_dir = folder_files + "/results_" + mode[j] + ".pdf"
            if not generate_html:
                file_html = results[j].to_html(
                    escape=False,
                    formatters=format_mode
                )
                file_html = STYLE_HTML + file_html
                pdfkit.from_string(
                    file_html,
                    pdf_dir,
                    configuration=config
                )
            else:
                pdfkit.from_file(
                    html_dir,
                    pdf_dir,
                    configuration=config
                )

        progress_update(
            progress,
            "Generating files : Done\n"
        )

        progress_update(
            progress,
            "\n--- Results are ready ! ---\n"
        )

    start.set(False)
