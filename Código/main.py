import tkinter as tk
from tkinter import filedialog
import csv

selected_columns = []
target_column = ""
column_names = []
execution_model = ""

def open_file():
    filepath = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    if filepath:
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            # Leer la primera fila del archivo CSV que contiene los nombres de las columnas
            global column_names
            column_names = next(reader)
            show_column_names(column_names)

def show_column_names(columns):
    # Eliminar cualquier widget existente debajo del botón
    for widget in window.winfo_children():
        widget.destroy()

    # Botón para seleccionar otro archivo CSV
    open_button = tk.Button(window, text="Seleccionar otro archivo CSV", command=open_file)
    open_button.pack()

    # Mostrar los nombres de las columnas
    for i, column in enumerate(columns):
        label = tk.Label(window, text=column)
        label.pack()
        label.place(x=20, y=40 + i * 20)

    # Botón para seleccionar variable objetivo
    target_button = tk.Button(window, text="Seleccionar variable objetivo", command=lambda: select_target_column(columns))
    target_button.pack()
    target_button.place(x=20, y=40 + len(columns) * 20)

    # Botón para seleccionar variables a eliminar
    delete_button = tk.Button(window, text="Seleccionar variables a eliminar", command=lambda: select_columns_to_delete(columns))
    delete_button.pack()
    delete_button.place(x=200, y=40 + len(columns) * 20)

    # Botón de siguiente
    if target_column != "":
        next_button = tk.Button(window, text="Siguiente", command=show_execution_window)
        next_button.pack()

def select_target_column(columns):
    global target_column
    global selected_columns
    if target_column in selected_columns:
        selected_columns.remove(target_column)
    show_selected_columns(columns)

def select_columns_to_delete(columns):
    global target_column
    global selected_columns
    if target_column in selected_columns:
        selected_columns.remove(target_column)
    show_selected_columns(columns, delete_mode=True)

def show_selected_columns(columns, delete_mode=False):
    # Eliminar cualquier widget existente debajo de los botones
    for widget in window.winfo_children():
        if isinstance(widget, tk.Button) or isinstance(widget, tk.Label):
            widget.destroy()

    # Mostrar los nombres de las columnas como botones y resaltados en función de la selección
    for i, column in enumerate(columns):
        if delete_mode and column in selected_columns:
            button = tk.Button(window, text=column, command=lambda c=column: select_column(c, delete_mode))
            button.pack()
            button.place(x=20, y=40 + i * 20)
            button.config(bg="red")

        elif column in selected_columns:
            label = tk.Label(window, text=column)
            label.pack()
            label.place(x=20, y=40 + i * 20)
            label.config(bg="red")

        elif column == target_column:
            label = tk.Label(window, text=column)
            label.pack()
            label.place(x=20, y=40 + i * 20)
            label.config(bg="blue")

        else:
            button = tk.Button(window, text=column, command=lambda c=column: select_column(c, delete_mode))
            button.pack()
            button.place(x=20, y=40 + i * 20)

    # Mostrar el botón "OK" si se han seleccionado variables a eliminar o una variable objetivo
    if selected_columns or target_column:
        ok_button = tk.Button(window, text="OK", command=update_main_window)
        ok_button.pack()
        ok_button.place(x=20, y=40 + len(columns) * 20)

def select_column(column, delete_mode):
    global selected_columns
    global target_column

    if delete_mode:
        if column == target_column:
            target_column = ""
        elif column in selected_columns:
            selected_columns.remove(column)
        else:
            selected_columns.append(column)
    else:
        if column == target_column:
            target_column = ""
        elif column in selected_columns:
            selected_columns.remove(column)
        else:
            target_column = column

    show_selected_columns(column_names, delete_mode=delete_mode)

def update_main_window():
    global selected_columns
    global target_column

    # Eliminar cualquier widget existente debajo del botón "OK"
    for widget in window.winfo_children():
        widget.destroy()

    # Mostrar los nombres de las columnas en la ventana principal y resaltados en función de la selección
    columns = 0
    for i, column in enumerate(column_names):
        label = tk.Label(window, text=column)
        label.pack()
        label.place(x=20, y=40 + i * 20)

        if column in selected_columns:
            label.config(bg="red")
        elif column == target_column:
            label.config(bg="blue")
        columns+=1

    # Botón para seleccionar otro archivo CSV
    open_button = tk.Button(window, text="Seleccionar otro archivo CSV", command=open_file)
    open_button.pack()

    # Botón para seleccionar variable objetivo
    target_button = tk.Button(window, text="Seleccionar variable objetivo", command=lambda: select_target_column(column_names))
    target_button.pack()
    target_button.place(x=20, y=40 + columns * 20)

    # Botón para seleccionar variables a eliminar
    delete_button = tk.Button(window, text="Seleccionar variables a eliminar", command=lambda: select_columns_to_delete(column_names))
    delete_button.pack()
    delete_button.place(x=200, y=40 + columns * 20)

    # Botón de siguiente
    next_button = tk.Button(window, text="Siguiente", command=show_execution_window)
    next_button.pack()
    next_button.place(x=20, y=60 + columns * 20)

def show_execution_window():
    global execution_model

    # Eliminar cualquier widget existente debajo del botón de siguiente
    for widget in window.winfo_children():
        widget.destroy()

    # Mostrar los nombres de las variables seleccionadas
    for i, column in enumerate(selected_columns):
        label = tk.Label(window, text=column)
        label.pack()
        label.place(x=20, y=40 + i * 20)

    # Botón para seleccionar el modelo de ejecución
    model_button_a = tk.Button(window, text="Modelo A", command=lambda: select_execution_model("A"))
    model_button_a.pack()
    model_button_a.place(x=20, y=40 + len(selected_columns) * 20)

    model_button_b = tk.Button(window, text="Modelo B", command=lambda: select_execution_model("B"))
    model_button_b.pack()
    model_button_b.place(x=120, y=40 + len(selected_columns) * 20)

    model_button_c = tk.Button(window, text="Modelo C", command=lambda: select_execution_model("C"))
    model_button_c.pack()
    model_button_c.place(x=220, y=40 + len(selected_columns) * 20)

    # Botón para crear la ejecución
    create_execution_button = tk.Button(window, text="Crear ejecución", command=create_execution)
    create_execution_button.pack()
    create_execution_button.place(x=20, y=40 + (len(selected_columns) + 1) * 20)

def select_execution_model(model):
    global execution_model
    execution_model = model

def create_execution():
    # Aquí puedes agregar la lógica para crear la ejecución con el modelo seleccionado
    print("Ejecución creada con el modelo", execution_model)

# Crear la ventana principal
window = tk.Tk()

# Botón para abrir el archivo CSV
open_button = tk.Button(window, text="Seleccionar archivo CSV", command=open_file)
open_button.pack()

#Iniciamos el bucle principal de la ventana:

# Iniciar el bucle principal de la ventana
window.mainloop()
