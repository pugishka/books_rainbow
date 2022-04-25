from tkinter import filedialog

import requests
from io import BytesIO
from PIL import Image as PilImage
from requests.exceptions import MissingSchema
import tkinter as tk


def check_column(col, df, progress, start):
    try:
        progress.delete(1.0, tk.END)
        url = df.loc[0, col]
        response = requests.get(url)
        im = PilImage.open(BytesIO(response.content))
    except MissingSchema:
        progress.insert(
            tk.INSERT,
            "The column should include links to pictures.\n")
        start.set(False)
    except Exception as e:
        progress.insert(
            tk.INSERT,
            "An error occurred.\n")
        progress.insert(
            tk.INSERT,
            "   " + type(e).__name__ + "\n")
        start.set(False)
    else:
        progress.insert(tk.INSERT, "Starting\n")
        start.set(True)


def get_file(url):
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
    url.set(file)
