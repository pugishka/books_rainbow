import tkinter as tk
from tkinter import *
from tkinter import filedialog, scrolledtext, ttk, _setit
from tkinter.ttk import Separator

import pandas as pd
from pandastable import Table, TableModel
import controller as con
import main


class App(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.start = BooleanVar(self)
        self.start.set(False)

        self.start.trace(
            'w',
            lambda: main.start_analysis(
                list(self.table.model.df.loc[self.variable.get()]),
                self.progress,
                self.palette_square_size,
                self.palette_with_cover,
                self.palette_with_replaced_cover,
                self.palette_with_count,
                [self.RGB, self.HSV],
                self.generate_html,
                self.generate_csv,
                self.generate_pdf,
                self.generate_palettes,
                self.max_n_palette
            )
        )

        self.wm_title("Book cover sort")
        self.config(
            padx=30,
            pady=10
        )
        self.resizable(False, False)

        self.__create_menu__()

        frame_left = Frame(self)
        frame_left.grid(row=0, column=0)
        frame_left.config(
            padx=30,
            pady=50
        )

        frame_table = Frame(frame_left)
        frame_table.config(
            background="white",
            width=400,
            height=300,
        )
        frame_table.pack()
        # url = "C:/Users/charo/Downloads/Export-e8d05104-a27b-4b14-ace2" \
        #       "-f7ddae1ea7c7/Bookshelf cafd6.csv"

        self.table = Table(
            frame_table,
            width=400,
            height=300,
        )

        frame_info = Frame(self)
        frame_info.grid(row=0, column=1)
        frame_info.config(
            width=400,
        )
        self.__add_info__(frame_info)

    def __create_menu__(self):
        menubar = Menu(self)
        file_menu = Menu(menubar, tearoff=0)
        self.url = StringVar(self)
        file_menu.add_command(
            label="Open",
            command=lambda: [con.get_file(self.url),
                             self.update_table()]
        )
        file_menu.add_command(label="Save")
        file_menu.add_command(label="Save as...")
        file_menu.add_separator()
        file_menu.add_command(label="Close")
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    def __add_info__(self, frame):

        lf = LabelFrame(frame, text="Options")
        choice_column = Frame(lf)

        # column
        Label(choice_column, text='Column')\
            .grid(row=0, column=0, sticky="we")
        self.values = []
        self.variable = StringVar(self)
        self.option_menu = OptionMenu(
            choice_column,
            self.variable,
            "",
            *self.values
        )
        self.option_menu.grid(
            row=0,
            column=1,
            sticky="we",
        )
        self.option_menu.config(
            anchor="w"
        )

        # palette
        Label(choice_column, text='Palette squares size (px)')\
            .grid(row=1, column=0, sticky="we", pady=(15, 0))

        self.palette_square_size = DoubleVar()
        s1 = Scale(
            choice_column,
            from_=1,
            to=100,
            orient="horizontal",
            variable=self.palette_square_size
        )
        s1.grid(row=1, column=1, sticky="we")

        # original cover ?
        Label(choice_column, text='Add original cover to palette ?')\
            .grid(row=2, column=0, sticky="we")

        self.palette_with_cover = BooleanVar()
        c1 = Checkbutton(
            choice_column,
            variable=self.palette_with_cover,
            onvalue=True,
            offvalue=False
        )
        c1.grid(row=2, column=1, sticky="we")

        # replaced cover ?
        Label(choice_column, text='Add posterized cover to palette ?')\
            .grid(row=3, column=0, sticky="we")

        self.palette_with_replaced_cover = BooleanVar()
        c2 = Checkbutton(
            choice_column,
            variable=self.palette_with_replaced_cover,
            onvalue=True,
            offvalue=False
        )
        c2.grid(row=3, column=1, sticky="we")

        # add count ?
        Label(choice_column, text='Add % of each dominant color to palette ?')\
            .grid(row=4, column=0, sticky="we")

        self.palette_with_count = BooleanVar()
        c3 = Checkbutton(
            choice_column,
            variable=self.palette_with_count,
            onvalue=True,
            offvalue=False
        )
        c3.grid(row=4, column=1, sticky="we")

        # which modes
        Label(choice_column, text='Color mode(s) to use for analysis')\
            .grid(row=5, column=0, sticky="nwe")
        modes = Frame(choice_column)
        modes.grid(row=5, column=1, sticky="nwe")

        self.RGB = BooleanVar()
        self.HSV = BooleanVar()
        c4 = Checkbutton(
            modes,
            text="RGB",
            variable=self.RGB,
            onvalue=True,
            offvalue=False
        )
        c4.pack(anchor="w")
        c5 = Checkbutton(
            modes,
            text="HSV",
            variable=self.HSV,
            onvalue=True,
            offvalue=False
        )
        c5.pack(anchor="w")

        # what files
        Label(choice_column, text='Files to generate')\
            .grid(row=6, column=0, sticky="nwe")
        files = Frame(choice_column)
        files.grid(row=6, column=1, sticky="nwe")

        self.generate_csv = BooleanVar()
        self.generate_html = BooleanVar()
        self.generate_pdf = BooleanVar()
        c6 = Checkbutton(
            files,
            text="CSV",
            variable=self.generate_csv,
            onvalue=True,
            offvalue=False
        )
        c6.pack(anchor="w")
        c7 = Checkbutton(
            files,
            text="HTML",
            variable=self.generate_html,
            onvalue=True,
            offvalue=False
        )
        c7.pack(anchor="w")
        c8 = Checkbutton(
            files,
            text="PDF",
            variable=self.generate_pdf,
            onvalue=True,
            offvalue=False
        )
        c8.pack(anchor="w")

        # number of colors
        Label(
            choice_column,
            text='Number of colors to find\n(some palettes might have less)',
            justify=LEFT)\
            .grid(row=7, column=0, sticky="we", pady=(15, 0))

        self.max_n_palette = DoubleVar()
        s2 = Scale(
            choice_column,
            from_=1,
            to=10,
            orient="horizontal",
            variable=self.max_n_palette
        )
        s2.grid(row=7, column=1, sticky="nwe")

        # generate palettes ?
        Label(choice_column, text='Generate palettes files ?')\
            .grid(row=8, column=0, sticky="we")

        self.generate_palettes = BooleanVar()
        c9 = Checkbutton(
            choice_column,
            variable=self.palette_with_count,
            onvalue=True,
            offvalue=False
        )
        c9.grid(row=8, column=1, sticky="we")

        self.progress = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            # width=40,
            height=10,
            borderwidth=0
        )

        ok = Button(
            choice_column,
            text="OK",
            command=lambda: con.check_column(
                self.variable.get(),
                self.table.model.df,
                self.progress,
                self.start
            )
        )

        ok.grid(row=9, column=0, sticky="we", pady=(20, 0))

        choice_column.pack(
            # fill="x",
            # expand=True,
            pady=20,
            padx=20
        )
        #TODO
        choice_column.columnconfigure(0, weight=1)
        choice_column.columnconfigure(1, weight=5)

        lf.pack()

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

        self.variable.set('')
        self.values.clear()
        new_choices = list(self.table.model.df.columns)
        self.option_menu["menu"].delete(0, "end")
        for choice in new_choices:
            self.option_menu["menu"].add_command(
                label=choice,
                command=_setit(self.variable, choice)
            )


if __name__ == "__main__":
    testObj = App()
    testObj.mainloop()
