import tkinter as tk
from typing_extensions import Literal
import util.util_ventana as util_ventana

class FormularioInfoDesign(tk.Toplevel):
    def __init__(self) -> None:
        super().__init__()
        self.config_window()
        self.contruirWidget()
        #self.miembros()

    def config_window(self):
        # Configuración inicial de la ventana
        self.title('SEUN: Sobre Nosotros')
        self.iconbitmap("SEUN SIC\imagenes\logo.ico")
        w, h = 450, 300        
        util_ventana.centrar_ventana(self, w, h)     
    
    def contruirWidget(self):         
        self.labelVersion = tk.Label(self, text="Version : 1.1.0")
        self.labelVersion.config(fg="#000000", font=("Roboto", 15), pady=10, width=40)
        self.labelVersion.pack()

        self.labelAutor = tk.Label(self, text="Autores: Los Tricentenarios")
        #self.labelAutor = tk.Label(self, text="Daniel\n"+"Alvaro\n"+"Cindy\n"+"Pamela\n"+"Leandro\n")
        self.labelAutor.config(fg="#000000", font=("Roboto", 15), pady=10, width=80)
        self.labelAutor.pack()
        self.labelMiembros = tk.Label(self, text="- Daniel Portillo\n"+"- Alvaro Ceballos\n"+"- Cindy Saminez\n"+"- Pamela Marroquin\n"+"- Leandro Revolorio\n")
        self.labelMiembros.config(fg="#000000", font=("Roboto", 15), pady=10, width=100)
        self.labelMiembros.pack()