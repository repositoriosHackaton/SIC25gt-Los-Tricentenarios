from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button  # Widget para el botón
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import mediapipe as mp


def distancia_euclidiana(p1, p2):
    d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return d


def draw_bounding_box(image, hand_landmarks):
    image_height, image_width, _ = image.shape
    x_min, y_min = image_width, image_height
    x_max, y_max = 0, 0

    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        if x < x_min:
            x_min = x
        if y < y_min:
            y_min = y
        if x > x_max:
            x_max = x
        if y > y_max:
            y_max = y

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


class MainApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        self.image_widget = Image()
        self.layout.add_widget(self.image_widget)


        self.text_panel = TextInput(
            readonly=True, 
            multiline=True,  
            size_hint=(1, 0.3), 
            background_color=(0.9, 0.9, 0.9, 1),  
            foreground_color=(0, 0, 0, 1),  
            font_size=20 
        )
        self.layout.add_widget(self.text_panel)


        self.boton_agregar = Button(
            text="Agregar letra",
            size_hint=(1, 0.1),  
            background_color=(0.2, 0.6, 1, 1), 
            color=(1, 1, 1, 1) 
        )
        
        self.boton_agregar.bind(on_press=self.agregar_letra)
        self.layout.add_widget(self.boton_agregar)

        self.boton_espacio = Button(
            text="Agregar Espacio", 
            size_hint=(1, 0.1),
            background_color=(0.5, 0.6, 1, 1),
            color=(1, 1, 1, 1)
        )
        
        self.boton_espacio.bind(on_press=self.agregar_espacio)
        self.layout.add_widget(self.boton_espacio)


        self.boton_salto = Button(
            text="Agregar Salto de linea",
            size_hint=(1, 0.1),
            background_color=(0.1, 0.6, 1, 1),
            color=(1, 1, 1, 1)
        )
        
        self.boton_salto.bind(on_press=self.agregar_salto)
        self.layout.add_widget(self.boton_salto)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )

        self.current_letter = None

        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS

        return self.layout

    def update(self, dt):
        success, image = self.cap.read()
        if not success:
            return

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )


                draw_bounding_box(image, hand_landmarks)

                self.current_letter = self.detect_letter(
                    hand_landmarks, image.shape, image)

        buf = cv2.flip(image, 0).tobytes()
        texture = Texture.create(
            size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture


    def detect_letter(self, hand_landmarks, image_shape, image):
        image_height, image_width, _ = image_shape


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

    def agregar_letra(self, instance):
        if self.current_letter is not None:
            self.text_panel.text += self.current_letter 
            self.current_letter = None  

    def agregar_espacio(self, instance):
        self.text_panel.text += " "

    def agregar_salto(self, instance):
        self.text_panel.text += "\n"


    def on_stop(self):
        self.cap.release()

if __name__ == '__main__':
    MainApp().run()
