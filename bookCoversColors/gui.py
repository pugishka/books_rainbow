import tkinter as tk
from tkinter import *
from tkinter import filedialog, scrolledtext, ttk, _setit
import pandas as pd
from pandastable import Table, TableModel
import controller as con


class App(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.wm_title("Book cover sort")

        self.__create_menu__()

        frame_table = Frame(self)
        frame_table.grid(row=0, column=0)
        frame_table.config(width=1000, height=1000)
        # url = "C:/Users/charo/Downloads/Export-e8d05104-a27b-4b14-ace2" \
        #       "-f7ddae1ea7c7/Bookshelf cafd6.csv"

        self.table = Table(frame_table)

        frame_info = Frame(self)
        frame_info.grid(row=0, column=1)
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
        Label(frame, text='Column').grid(row=0, column=0)

        self.values = []
        self.variable = StringVar(self)
        self.option_menu = OptionMenu(
            frame,
            self.variable,
            "",
            *self.values,
        )
        self.option_menu.grid(row=0, column=1)

        progress = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            width=30,
            height=10)

        progress.grid(row=1, columnspan=3)

        Button(frame, text="OK",
               command=lambda: con.check_column(
                   self.variable.get(),
                   progress
               )).grid(row=0, column=2)

    def update_table(self):
        self.table.updateModel(TableModel(pd.read_csv(self.url.get())))
        self.table.grid(
            row=0,
            column=0,
            # TODO
            sticky="s"
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
