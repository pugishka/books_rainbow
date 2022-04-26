from tkinter import filedialog
import requests
from io import BytesIO
from PIL import Image as PilImage
from requests.exceptions import MissingSchema
import tkinter as tk


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
    progress.configure(state='normal')
    progress.insert(
        tk.INSERT,
        text,
    )
    progress.see("end")
    progress.configure(state='disabled')


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
    # pdf = kwargs["pdf"]
    rgb = kwargs["mode"][0]
    hsv = kwargs["mode"][1]
    progress = kwargs["progress"]
    start = kwargs["start"]

    try:
        progress.configure(state='normal')
        progress.delete(1.0, tk.END)
        progress.configure(state='disabled')
        url = df.loc[0, col]
        response = requests.get(url)
        im = PilImage.open(BytesIO(response.content))
    except KeyError:
        progress_update(
            progress,
            "Choose a column."
        )
    except MissingSchema:
        progress_update(
            progress,
            "The column should include links to pictures."
        )
        start.set(False)
    except Exception as e:
        progress_update(
            progress,
            "An error occurred.\n"
        )
        progress_update(
            progress,
            "   " + type(e).__name__ + "\n"
        )
        start.set(False)
    else:
        if not csv.get() and not html.get():  # and not pdf.get():
            progress_update(progress, "Select at least one file to generate")
        elif not rgb.get() and not hsv.get():
            progress_update(progress, "Select at least one mode")
        else:
            progress_update(progress, "Starting...\n")
            start.set(True)


def get_file(progress, url):
    """
    Shows dialogue window to open the CSV file.

    Parameters
    ----------
    progress : tkinter.ScrolledText
        ScrolledText widget to show progress.
    url : StringVar
        Variable of the link to the CSV file.

    """
    try:
        root = tk.Tk()
        root.withdraw()
        file = filedialog.askopenfilename(
            initialdir="C:/Users/MainFrame/Desktop/",
            title="Open CSV file",
            filetypes=(("CSV Files", "*.csv"),)
        )
    except Exception as e:
        progress.configure(state='normal')
        progress.delete(1.0, tk.END)
        progress.configure(state='disabled')
        progress_update(
            progress,
            "An error occurred.\n"
        )
        progress_update(
            progress,
            "   " + type(e).__name__ + "\n"
        )
    else:
        url.set(file)



