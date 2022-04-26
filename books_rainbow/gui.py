import tkinter as tk
from tkinter import *
from tkinter import scrolledtext
from tkinter import _setit
from tkinter.ttk import Separator
import pandas as pd
from pandastable import Table, TableModel
import books_rainbow.controller as con
import books_rainbow.main as main
import threading


class App(tk.Tk):
    """
    GUI class.

    Attributes
    ----------
    HSV : BooleanVar
        Evaluate in HSV ?
    RGB : BooleanVar
        Evaluate in RGB ?
    all_columns : list of str
        Columns in the opened CSV.
    column_covers : StringVar
        Selected column.
    file_menu : Menu
        Top bar menu.
    frame_info : Frame
        Frame containing all the options.
    generate_csv : BooleanVar
        Generate CSV file ?
    generate_html : BooleanVar
        Generate HTML file ?
    generate_palettes : BooleanVar
        Generate palette files ?
    generate_pdf : BooleanVar
        Generate PDF file ? (Removed)
    max_n_palette : DoubleVar
        Average number of dominant colors to find.
    ok : Button
        Button to confirm.
    option_menu : OptionMenu
        Menu to choose from the different columns.
    palette_square_size : DoubleVar
        Size of the squares in the palettes.
    palette_with_count : BooleanVar
        Add percentage of each dominant color to palette ?
    palette_with_cover : BooleanVar
        Add cover to palette ?
    palette_with_replaced_cover : BooleanVar
        Add cover replaced with dominant colors to palette ?
    progress : ScrolledText
        Text field to show progress.
    start : BooleanVar
        True if the analysis can start.
    table : Table
        Opened CSV table.
    thread_exe : Thread
        Thread to execute functions from main.py.
    url : StringVar
        Link to the CSV file.

    Methods
    -------
    __init__:
        Constructor.
    __start_analyse__:
        Create Thread to analyse the table.
    switch_state:
        Enable or disable some options.
    __create_menu__:
        Constructor for the top bar menu.
    __add_w_info__:
        Constructor for the list of option widgets.
    __add_info__:
        Constructor for the options.
    update_table:
        Update after a CSV file is opened.

    """
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.start = BooleanVar(self)
        self.start.set(False)

        self.start.trace(
            'w',
            lambda *x: self.__start_analyse__(),
        )

        self.wm_title("Book cover sort")
        self.config(
            padx=30,
            pady=10
        )
        self.resizable(True, False)
        self.minsize(900, 0)

        self.__create_menu__()

        frame_left = Frame(self)
        frame_left.grid(row=0, column=0, sticky="nwes")
        frame_left.config(
            padx=30,
            pady=50
        )

        frame_table = Frame(frame_left)
        frame_table.config(
            background="white",
            width=400,
        )
        frame_table.pack(fill="both", expand=True)
        # url = "C:/Users/charo/Downloads/Export-e8d05104-a27b-4b14-ace2" \
        #       "-f7ddae1ea7c7/Bookshelf cafd6.csv"

        self.table = Table(
            frame_table,
            width=400,
        )

        self.frame_info = Frame(self)
        self.frame_info.grid(row=0, column=1, sticky="we")
        self.frame_info.config(
            # width=100,
        )
        self.__add_info__(self.frame_info)

        # self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        # self.col

    def __start_analyse__(self):
        if self.start.get():
            self.ok.config(
                state="disabled"
            )
            self.file_menu.entryconfig("Open", state="disabled")
            self.thread_exe = threading.Thread(
                target=main.start_analysis,
                args=(
                    self.start,
                    list(self.table.model.df[self.column_covers.get()]),
                    self.progress,
                    int(self.palette_square_size.get()),
                    self.palette_with_cover.get(),
                    self.palette_with_replaced_cover.get(),
                    self.palette_with_count.get(),
                    [self.RGB.get(), self.HSV.get()],
                    self.generate_html.get(),
                    self.generate_csv.get(),
                    #self.generate_pdf.get(),
                    self.generate_palettes.get(),
                    int(self.max_n_palette.get())
                ),
                daemon=True
            )
            # self.thread_exe = threading.Thread(
            #     target=main.test,
            #     args=[self.start]
            # )
            self.thread_exe.start()
        else:
            self.ok.config(
                state="normal"
            )
            self.file_menu.entryconfig("Open", state="normal")

    @staticmethod
    def switch_state(*args):
        for w in args[1:]:
            if w.winfo_class() == "Scale":
                if not args[0]:
                    state_w = ["disabled", "#aaa", "#dedede"]
                else:
                    state_w = ["normal", "#000", "#ccc"]
                w.configure(
                    state=state_w[0],
                    fg=state_w[1],
                    troughcolor=state_w[2]
                )
            elif w.winfo_class() == "Checkbutton":
                if not args[0]:
                    w.configure(
                        state="disabled"
                    )
                else:
                    w.configure(
                        state="normal"
                    )

    def __create_menu__(self):
        menubar = Menu(self)
        self.file_menu = Menu(
            menubar,
            tearoff=0
        )
        self.url = StringVar(self)
        self.file_menu.add_command(
            label="Open",
            command=lambda: [con.get_file(self.progress, self.url),
                             self.update_table()]
        )
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Close", command=self.quit)
        menubar.add_cascade(label="File", menu=self.file_menu)
        self.config(menu=menubar)

    def __add_w_info__(self, frame):
        self.column_covers = StringVar()
        self.palette_square_size = DoubleVar()
        self.palette_with_cover = BooleanVar()
        self.palette_with_replaced_cover = BooleanVar()
        self.palette_with_count = BooleanVar()
        self.RGB = BooleanVar()
        self.HSV = BooleanVar()
        self.generate_csv = BooleanVar()
        self.generate_html = BooleanVar()
        #self.generate_pdf = BooleanVar()
        self.max_n_palette = DoubleVar()
        self.generate_palettes = BooleanVar()
        self.generate_palettes.set(True)
        self.generate_palettes.trace(
            "w",
            callback=lambda *x: self.switch_state(
                self.generate_palettes.get(),
                widgets["Squares"]["arg"]["en_dis"],
                widgets["Original cover"]["arg"]["en_dis"],
                widgets["Replaced cover"]["arg"]["en_dis"],
                widgets["Count"]["arg"]["en_dis"],
            )
        )
        self.all_columns = []

        w_col = 2
        widgets = {
            "Column": {
                "text": "Column containing the links to the covers",
                "type": "menu",
                "arg": {
                    "row": 0,
                    "values": self.all_columns
                },
                "variable": self.column_covers
            },
            "Squares": {
                "text": "Palette squares size (px)",
                "type": "slider",
                "arg": {
                    "row": 1,
                    "from": 1,
                    "to": 100
                },
                "variable": self.palette_square_size
            },
            "Original cover": {
                "text": "Add original cover to palette ?",
                "type": "yesno",
                "arg": {
                    "row": 2,
                },
                "variable": self.palette_with_cover
            },
            "Replaced cover": {
                "text": "Add posterized cover to palette ?",
                "type": "yesno",
                "arg": {
                    "row": 3,
                },
                "variable": self.palette_with_replaced_cover
            },
            "Count": {
                "text": "Add % of each dominant color to palette ?",
                "type": "yesno",
                "arg": {
                    "row": 4
                },
                "variable": self.palette_with_count
            },
            "Modes": {
                "text": "Color mode(s) to use for analysis",
                "type": "choices",
                "arg": {
                    "row": 5,
                    "choices": {
                        "RGB": self.RGB,
                        "HSV": self.HSV
                    }
                },
            },
            "Files": {
                "text": "Files to generate",
                "type": "choices",
                "arg": {
                    "row": 6,
                    "choices": {
                        "CSV": self.generate_csv,
                        "HTML": self.generate_html,
                        #"PDF": self.generate_pdf
                    }
                },
            },
            "Colors": {
                "text": "Number of colors to find "
                        "(some palettes might have less)",
                "type": "slider",
                "arg": {
                    "row": 7,
                    "from": 1,
                    "to": 10
                },
                "variable": self.max_n_palette
            },
            "Palettes": {
                "text": "Generate palettes files ?",
                "type": "yesno",
                "arg": {
                    "row": 8,
                },
                "variable": self.generate_palettes
            },
        }
        n_sep = 0

        for name in widgets:
            label = Label(
                frame,
                text=widgets[name]["text"],
                wraplength=200,
                justify=RIGHT
            )
            label.grid(
                row=widgets[name]["arg"]["row"] + n_sep,
                column=0,
                sticky="e"
            )

            if widgets[name]["type"] == "menu":
                self.option_menu = OptionMenu(
                    frame,
                    widgets[name]["variable"],
                    "",
                    *widgets[name]["arg"]["values"]
                )
                self.option_menu.grid(
                    row=widgets[name]["arg"]["row"] + n_sep,
                    column=w_col,
                    sticky="we",
                )
                self.option_menu.config(
                    anchor="w"
                )

            elif widgets[name]["type"] == "slider":
                s = Scale(
                    frame,
                    from_=widgets[name]["arg"]["from"],
                    to=widgets[name]["arg"]["to"],
                    orient="horizontal",
                    variable=widgets[name]["variable"],
                    fg="#000",
                    troughcolor="#ccc"
                )
                s.grid(
                    row=widgets[name]["arg"]["row"] + n_sep,
                    column=w_col,
                    sticky="we"
                )
                widgets[name]["arg"]["en_dis"] = s

            elif widgets[name]["type"] == "yesno":
                c = Checkbutton(
                    frame,
                    variable=widgets[name]["variable"],
                    onvalue=True,
                    offvalue=False,
                )
                c.grid(
                    row=widgets[name]["arg"]["row"] + n_sep,
                    column=w_col,
                    sticky="w"
                )
                widgets[name]["arg"]["en_dis"] = c

            elif widgets[name]["type"] == "choices":
                choices = Frame(
                    frame,
                    borderwidth=2,
                    relief=GROOVE
                )
                choices.grid(
                    row=widgets[name]["arg"]["row"] + n_sep,
                    column=w_col,
                    sticky="we",
                    pady=5
                )
                for i in widgets[name]["arg"]["choices"]:
                    c = Checkbutton(
                        choices,
                        text=i,
                        variable=widgets[name]["arg"]["choices"][i],
                        onvalue=True,
                        offvalue=False
                    )
                    c.pack(anchor="w")

            # Separator(frame, orient='horizontal').grid(
            #     row=widgets[name]["arg"]["row"] + n_sep,
            #     columnspan=2,
            #     pady=(40, 10),
            #     sticky="we"
            # )
            # n_sep += 1

        self.ok = Button(
            frame,
            text="OK",
            command=lambda: con.check_params(
                col=self.column_covers.get(),
                table=self.table.model.df,
                csv=self.generate_csv,
                html=self.generate_html,
                #pdf=self.generate_pdf,
                mode=[self.RGB, self.HSV],
                progress=self.progress,
                start=self.start
            )
        )

        self.ok.grid(
            row=len(widgets) + n_sep + 1,
            columnspan=3,
            column=0,
            sticky="we",
            pady=(30, 0)
        )

        Separator(frame, orient='vertical').grid(
            column=1,
            row=0,
            sticky='ns',
            rowspan=len(widgets) + n_sep + 1,
            padx=10,
        )

    def __add_info__(self, frame):

        lf = LabelFrame(frame, text="Options")
        choice_column = Frame(lf)
        self.__add_w_info__(choice_column)

        choice_column.pack(
            fill="x",
            # expand=True,
            pady=20,
            padx=20
        )

        lf.pack(fill="x")

        # choice_column.grid_columnconfigure(0, weight=1)
        choice_column.grid_columnconfigure(2, weight=1)

        Separator(self, orient='horizontal').grid(
            row=1,
            columnspan=2,
            pady=(40, 10),
            sticky="we"
        )

        Label(self, text='Progress and error messages :').grid(
            row=2,
            columnspan=2,
            sticky="w"
        )

        self.progress = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            # width=40,
            height=10,
            borderwidth=0,
            state="disabled"
        )
        self.progress.grid(
            row=3,
            columnspan=2,
            pady=10,
            sticky="wes"
        )

    def update_table(self):
        self.table.updateModel(TableModel(pd.read_csv(self.url.get())))
        self.table.config(
            width=100,
            height=100,
        )
        self.table.show()

        self.column_covers.set('')
        self.all_columns.clear()
        new_choices = list(self.table.model.df.columns)
        self.option_menu["menu"].delete(0, "end")
        for choice in new_choices:
            self.option_menu["menu"].add_command(
                label=choice,
                command=_setit(self.column_covers, choice)
            )

