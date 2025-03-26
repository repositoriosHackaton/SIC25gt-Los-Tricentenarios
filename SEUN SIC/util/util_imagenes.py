from PIL import Image, ImageTk

def leer_imagen(path, size): 
        return ImageTk.PhotoImage(Image.open(path).resize(size, Image.ADAPTIVE))  

