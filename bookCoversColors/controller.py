from tkinter import filedialog
import requests
from io import BytesIO
from PIL import Image as PilImage
from requests.exceptions import MissingSchema
import tkinter as tk


def check_params(**kwargs):
    """
    Checks if the selected options are correct.

    Parameters
    ----------
    **kwargs
        The options to verify.

    """
    col = kwargs["col"]
    df = kwargs["table"]
    csv = kwargs["csv"]
    html = kwargs["html"]
    pdf = kwargs["pdf"]
    rgb = kwargs["mode"][0]
    hsv = kwargs["mode"][1]
    progress = kwargs["progress"]
    start = kwargs["start"]

    try:
        progress.delete(1.0, tk.END)
        url = df.loc[0, col]
        response = requests.get(url)
        im = PilImage.open(BytesIO(response.content))
    except KeyError:
        progress.insert(
            tk.INSERT,
            "Choose a column.")
    except MissingSchema:
        progress.insert(
            tk.INSERT,
            "The column should include links to pictures.")
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
        if not csv.get() and not html.get() and not pdf.get():
            progress.insert(tk.INSERT, "Select at least one file to generate")
        elif not rgb.get() and not hsv.get():
            progress.insert(tk.INSERT, "Select at least one mode")
        else:
            progress.insert(tk.INSERT, "Starting...\n")
            start.set(True)


def get_file(url):
    """
    Shows dialogue window to open the CSV file.

    Parameters
    ----------
    url : StringVar
        Variable of the link to the CSV file.

    """
    root = tk.Tk()
    root.withdraw()
    file = filedialog.askopenfilename(
        initialdir="C:/Users/MainFrame/Desktop/",
        title="Open CSV file",
        filetypes=(("CSV Files", "*.csv"),)
    )
    url.set(file)


