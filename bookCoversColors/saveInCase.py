def main():

    start_time = time.perf_counter()

    # get list of URLs to the covers
    urls = get_urls_from_file().tolist()[:1]
    n = len(urls)

    palette_square_size = 50
    palette_with_cover = True
    palette_with_replaced_cover = True
    palette_with_count = True

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

    palette_all = PilImage.new("RGB", (n * 3, 100), "#ffffff")
    offset_x = 0
    images = get_images(results_hsv.loc[:, "Cover"], 50)
    for im in images:
        palette_all.paste(im.resize((3, 100)), (offset_x, 0))
        offset_x += 3

    palette_all.save(path_palette_hsv + "/all_covers_hsv.jpg")