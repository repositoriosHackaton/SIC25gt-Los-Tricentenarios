import tkinter as tk
from config import COLOR_CUERPO_PRINCIPAL
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np


def distancia_euclidiana(p1, p2):
    d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return d


class FormularioPrograma():
    def __init__(self, panel_principal, logo):
        self.panel_principal = panel_principal
        self.current_letter = None

        # Configurar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Panel principal
        self.panel_contenedor = tk.Frame(
            panel_principal, bg=COLOR_CUERPO_PRINCIPAL)
        self.panel_contenedor.pack(fill='both', expand=True)

        # Barra superior para el título
        self.barra_superior = tk.Frame(self.panel_contenedor)
        self.barra_superior.pack(side=tk.TOP, fill=tk.X, expand=False)

        # Título
        self.labelTitulo = tk.Label(
            self.barra_superior,
            text="Programa de Señas",
            fg="#222d33",
            font=("Roboto", 30),
            bg=COLOR_CUERPO_PRINCIPAL
        )
        self.labelTitulo.pack(side=tk.TOP, fill='both', expand=True)

        # Panel inferior para contenido
        self.panel_inferior = tk.Frame(self.panel_contenedor)
        self.panel_inferior.pack(side=tk.BOTTOM, fill='both', expand=True)


        self.panel_camara = tk.Frame(self.panel_inferior, bg='black')
        self.panel_camara.pack(side=tk.LEFT, fill='both', expand=True)

 
        self.panel_controles = tk.Frame(
            self.panel_inferior, bg=COLOR_CUERPO_PRINCIPAL, width=300)
        self.panel_controles.pack(side=tk.RIGHT, fill='both', expand=False)

        # Label para la cámara
        self.label_camara = tk.Label(self.panel_camara)
        self.label_camara.pack(fill='both', expand=True)


        self.text_panel = tk.Text(
            self.panel_controles,
            height=10,
            width=30,
            bg='#f0f0f0',
            font=('Roboto', 12)
        )
        self.text_panel.pack(pady=10, padx=10, fill='x')

        botones = [
            ("Agregar letra", '#4CAF50', self.agregar_letra),
            ("Agregar espacio", '#2196F3', self.agregar_espacio),
            ("Salto de línea", '#FF9800', self.agregar_salto)
        ]

        for texto, color, comando in botones:
            btn = tk.Button(
                self.panel_controles,
                text=texto,
                command=comando,
                bg=color,
                fg='white',
                relief=tk.FLAT
            )
            btn.pack(fill='x', padx=10, pady=5, ipady=5)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.update_camera()

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                    self.current_letter = self.detect_letter(
                        hand_landmarks, frame.shape, frame)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            self.label_camara.imgtk = imgtk
            self.label_camara.configure(image=imgtk)

        self.panel_principal.after(10, self.update_camera)

    def detect_letter(self, hand_landmarks, image_shape, image):
        image_height, image_width, _ = image_shape

        # Obtener las coordenadas de los puntos de referencia
        index_finger_tip = (int(hand_landmarks.landmark[8].x * image_width),
                            int(hand_landmarks.landmark[8].y * image_height))
        index_finger_pip = (int(hand_landmarks.landmark[6].x * image_width),
                            int(hand_landmarks.landmark[6].y * image_height))

        thumb_tip = (int(hand_landmarks.landmark[4].x * image_width),
                     int(hand_landmarks.landmark[4].y * image_height))
        thumb_pip = (int(hand_landmarks.landmark[2].x * image_width),
                     int(hand_landmarks.landmark[2].y * image_height))

        middle_finger_tip = (int(hand_landmarks.landmark[12].x * image_width),
                             int(hand_landmarks.landmark[12].y * image_height))
        middle_finger_pip = (int(hand_landmarks.landmark[10].x * image_width),
                             int(hand_landmarks.landmark[10].y * image_height))

        ring_finger_tip = (int(hand_landmarks.landmark[16].x * image_width),
                           int(hand_landmarks.landmark[16].y * image_height))
        ring_finger_pip = (int(hand_landmarks.landmark[14].x * image_width),
                           int(hand_landmarks.landmark[14].y * image_height))

        pinky_tip = (int(hand_landmarks.landmark[20].x * image_width),
                     int(hand_landmarks.landmark[20].y * image_height))
        pinky_pip = (int(hand_landmarks.landmark[18].x * image_width),
                     int(hand_landmarks.landmark[18].y * image_height))

        wrist = (int(hand_landmarks.landmark[0].x * image_width),
                 int(hand_landmarks.landmark[0].y * image_height))

        ring_finger_pip2 = (int(hand_landmarks.landmark[5].x * image_width),
                            int(hand_landmarks.landmark[5].y * image_height))

        current_letter = None

        # Letra A
        if abs(thumb_tip[0] - thumb_pip[0]) > 40 and \
                thumb_tip[1] < thumb_pip[1] and \
                index_finger_tip[1] > index_finger_pip[1] and \
                middle_finger_tip[1] > middle_finger_pip[1] and \
                ring_finger_tip[1] > ring_finger_pip[1] and \
                pinky_tip[1] > pinky_pip[1]:
            current_letter = 'A'
            # Dibujar la letra en la camara
            cv2.putText(image, 'A', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

        # Letra B
        elif index_finger_pip[1] - index_finger_tip[1] > 0 and pinky_pip[1] - pinky_tip[1] > 0 and \
                middle_finger_pip[1] - middle_finger_tip[1] > 0 and ring_finger_pip[1] - ring_finger_tip[1] > 0 and \
                middle_finger_tip[1] - ring_finger_tip[1] < 0 and abs(thumb_tip[1] - ring_finger_pip2[1]) < 40:
            current_letter = 'B'
            cv2.putText(image, 'B', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

        # Letra C
        elif abs(index_finger_tip[1] - thumb_tip[1]) < 50 and \
                abs(middle_finger_tip[1] - thumb_tip[1]) < 50 and \
                abs(ring_finger_tip[1] - thumb_tip[1]) < 50 and \
                abs(pinky_tip[1] - thumb_tip[1]) < 50 and \
                index_finger_tip[0] < index_finger_pip[0] and \
                middle_finger_tip[0] < middle_finger_pip[0] and \
                ring_finger_tip[0] < ring_finger_pip[0] and \
                pinky_tip[0] < pinky_pip[0]:
            current_letter = 'C'
            cv2.putText(image, 'C', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

            # Letra D
        elif distancia_euclidiana(thumb_tip, middle_finger_tip) < 65 and \
                distancia_euclidiana(thumb_tip, ring_finger_tip) < 65 and \
                pinky_pip[1] - pinky_tip[1] < 0 and \
                index_finger_pip[1] - index_finger_tip[1] > 0:
            current_letter = 'D'
            cv2.putText(image, 'D', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

            # Letra E
        elif index_finger_pip[1] - index_finger_tip[1] < 0 and pinky_pip[1] - pinky_tip[1] < 0 and \
                middle_finger_pip[1] - middle_finger_tip[1] < 0 and ring_finger_pip[1] - ring_finger_tip[1] < 0 and \
                abs(index_finger_tip[1] - thumb_tip[1]) < 20 and \
                thumb_tip[1] - index_finger_tip[1] > 0 and \
                thumb_tip[1] - middle_finger_tip[1] > 0 and \
                thumb_tip[1] - ring_finger_tip[1] > 0 and \
                thumb_tip[1] - pinky_tip[1] > 0:
            current_letter = 'E'
            cv2.putText(image, 'E', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

            # Letra F
        elif pinky_pip[1] - pinky_tip[1] > 0 and middle_finger_pip[1] - middle_finger_tip[1] > 0 and \
                ring_finger_pip[1] - ring_finger_tip[1] > 0 and index_finger_pip[1] - index_finger_tip[1] < 0 and \
                abs(thumb_pip[1] - thumb_tip[1]) > 0 and distancia_euclidiana(index_finger_tip, thumb_tip) < 65:
            current_letter = 'F'
            cv2.putText(image, 'F', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

        # Letra G
        elif abs(index_finger_tip[0] - thumb_tip[0]) < 50 and \
                abs(index_finger_tip[1] - thumb_tip[1]) < 50 and \
                index_finger_tip[0] < index_finger_pip[0] and \
                thumb_tip[0] < thumb_pip[0] and \
                middle_finger_tip[1] > middle_finger_pip[1] and \
                ring_finger_tip[1] > ring_finger_pip[1] and \
                pinky_tip[1] > pinky_pip[1]:
            current_letter = 'G'
            cv2.putText(image, 'G', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

        # Letra H
        elif abs(index_finger_tip[1] - middle_finger_tip[1]) < 30 and \
                index_finger_tip[1] < index_finger_pip[1] and \
                middle_finger_tip[1] < middle_finger_pip[1] and \
                ring_finger_tip[1] > ring_finger_pip[1] and \
                pinky_tip[1] > pinky_pip[1] and \
                thumb_tip[1] > thumb_pip[1]:
            current_letter = 'H'
            cv2.putText(image, 'H', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

        # Letra J
        elif abs(pinky_tip[0] - pinky_pip[0]) > 20 and \
                abs(pinky_tip[1] - pinky_pip[1]) < 20:
            current_letter = 'J'
            cv2.putText(image, 'J', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

        # Letra I
        elif pinky_tip[1] < pinky_pip[1]:
            current_letter = 'I'
            cv2.putText(image, 'I', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

        # Letra K
        elif abs(index_finger_tip[1] - middle_finger_tip[1]) < 30 and \
                index_finger_tip[1] < index_finger_pip[1] and \
                middle_finger_tip[1] < middle_finger_pip[1] and \
                ring_finger_tip[1] > ring_finger_pip[1] and \
                pinky_tip[1] > pinky_pip[1] and \
                abs(thumb_tip[0] - middle_finger_pip[0]) < 30 and \
                thumb_tip[1] > middle_finger_pip[1]:
            current_letter = 'K'
            cv2.putText(image, 'K', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

        # Letra L
        elif abs(index_finger_tip[0] - thumb_tip[0]) < 50 and \
                index_finger_tip[0] < index_finger_pip[0] and \
                thumb_tip[0] < thumb_pip[0] and \
                middle_finger_tip[0] > middle_finger_pip[0] and \
                ring_finger_tip[0] > ring_finger_pip[0] and \
                pinky_tip[0] > pinky_pip[0]:
            current_letter = 'L'
            cv2.putText(image, 'L', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

        elif abs(index_finger_tip[1] - middle_finger_tip[1]) < 50 and \
                index_finger_tip[1] > index_finger_pip[1] and \
                middle_finger_tip[1] > middle_finger_pip[1] and \
                ring_finger_tip[1] < ring_finger_pip[1] and \
                thumb_tip[1] > thumb_pip[1]:
            current_letter = 'N'
            cv2.putText(image, 'N', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

        # Letra M
        elif index_finger_tip[1] > index_finger_pip[1] and \
                middle_finger_tip[1] > middle_finger_pip[1] and \
                ring_finger_tip[1] > ring_finger_pip[1] and \
                distancia_euclidiana(thumb_tip, thumb_pip) < 30:
            current_letter = 'M'
            cv2.putText(image, 'M', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

        # Letra P
        elif abs(middle_finger_tip[0] - thumb_tip[0]) < 20 and \
                abs(middle_finger_tip[1] - thumb_tip[1]) < 20 and \
                index_finger_tip[1] > index_finger_pip[1] and \
                middle_finger_tip[1] > middle_finger_pip[1] and \
                thumb_tip[1] > thumb_pip[1]:
            current_letter = 'P'
            cv2.putText(image, 'P', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

        elif distancia_euclidiana(thumb_tip, index_finger_tip) < 38 and \
                distancia_euclidiana(thumb_tip, middle_finger_tip) < 38:
            current_letter = 'O'
            cv2.putText(image, 'O', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)
       # Letra Q
        elif thumb_tip[1] > thumb_pip[1] and \
                middle_finger_tip[1] > middle_finger_pip[1]:
            current_letter = 'Q'
            cv2.putText(image, 'Q', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

        # Letra T
        elif thumb_tip[1] < thumb_pip[1] and \
                index_finger_tip[1] > index_finger_pip[1] and \
                middle_finger_tip[1] > middle_finger_pip[1] and \
                ring_finger_tip[1] > ring_finger_pip[1] and \
                pinky_tip[1] > pinky_pip[1]:
            current_letter = 'T'
            cv2.putText(image, 'T', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

        # Letra W
        elif index_finger_tip[1] < index_finger_pip[1] and \
                middle_finger_tip[1] < middle_finger_pip[1] and \
                ring_finger_tip[1] < ring_finger_pip[1] and \
                pinky_tip[1] > pinky_pip[1] and \
                thumb_tip[1] > thumb_pip[1]:
            current_letter = 'W'
            cv2.putText(image, 'W', (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0, (20, 219, 26), 10)

        return current_letter

    def agregar_letra(self):
        if self.current_letter:
            self.text_panel.insert(tk.END, self.current_letter)

    def agregar_espacio(self):
        self.text_panel.insert(tk.END, ' ')

    def agregar_salto(self):
        self.text_panel.insert(tk.END, '\n')

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
