import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from threading import Thread
from datetime import datetime

# Variables globales
eye_closed = False
blink_count = 0
interruption_count = 0
lost_pupil_frames = 0
blink_warning_persistent = False
fixation_warning_persistent = False
blink_warning_acknowledged = False
timestamps = []
pupil_x_positions = []
pupil_y_positions = []
start_time = time.time()
fixation_detected = False
fixation_start_time = None
last_event_time = None
last_blink_time = None
blink_reset_time = time.time()

# Umbrales y parámetros
blink_threshold_min = 0.1
blink_threshold_max = 0.4
fixation_threshold = 20
fixation_tolerance = 25
interruption_threshold = 1.0
max_graph_points = 100
max_lost_frames = 3

# Archivo de resumen
txt_filename = "resultados_fatiga_visual.txt"

def generate_summary():
    end_time = time.time()
    total_time = end_time - start_time
    total_time_minutes = int(total_time // 60)
    total_time_seconds = int(total_time % 60)

    with open(txt_filename, mode="w", encoding="utf-8") as file:  # 🔹 Se agrega encoding="utf-8" para que se guarde el .txt sin problemas de tíldes.
        file.writelines([
            "Resultados de la Detección de Pupilas\n",
            "-------------------------------------\n",
            f"Fecha y Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"Tiempo total de ejecución: {total_time_minutes} minutos y {total_time_seconds} segundos\n",
            f"Total de Parpadeos Detectados: {blink_count}\n",
            f"Interrupciones Detectadas: {interruption_count}\n",
            f"Fijaciones Detectadas: {1 if fixation_detected else 0}\n",
            "-------------------------------------\n\n",
            "Posiciones de la Pupila:\n"
        ])
        
        if pupil_x_positions and pupil_y_positions:
            file.writelines(f"  {i + 1}: X = {x}, Y = {y}\n" for i, (x, y) in enumerate(zip(pupil_x_positions, pupil_y_positions)))
        else:
            file.write("  No se registraron posiciones de la pupila.\n")

        file.write("\nFin del Reporte\n")



def smooth_data(data, kernel_size=5):
    if len(data) < kernel_size:
        return np.array(data)  # Convertimos a numpy para evitar errores de tipo
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(data, kernel, mode='valid')


def draw_rounded_rectangle(img, top_left, bottom_right, radius, color, thickness):
    """Dibuja un rectángulo con bordes redondeados"""
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)

    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)



def draw_text_with_shadow(img, text, position, font, scale, color, thickness):
    """Dibuja texto con sombra para mejor visibilidad"""
    x, y = position
    cv2.putText(img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2)  # Sombra negra
    cv2.putText(img, text, (x, y), font, scale, color, thickness)



def get_dynamic_text_color():
    """Cambia el color del texto si hay advertencias"""
    if blink_warning_persistent:
        return (0, 0, 255)  # Rojo
    elif fixation_warning_persistent:
        return (128, 0, 128)  # Morado
    else:
        return (0, 255, 0)  # Verde



def plot_all_graphs():
    global blink_warning_persistent, fixation_warning_persistent, blink_warning_acknowledged
    global blink_count, blink_reset_time

    plt.ion()
    fig = plt.figure(figsize=(10, 9))
    fig.patch.set_facecolor('#f2f2f2')

    # --- Área del contador de parpadeos y botón Reset ---
    ax_blinks = plt.axes([0.35, 0.85, 0.3, 0.1])  # Ajustar tamaño y posición
    ax_blinks.axis("off")
    text_blinks = ax_blinks.text(0.4, 0.5, f"Parpadeos: {blink_count}",
                                 fontsize=20, fontweight='bold', color='blue', ha='center', va='center')

    # --- Botón Reset Parpadeos al lado del contador ---
    reset_button_ax = plt.axes([0.65, 0.87, 0.15, 0.06])  # Ajuste de posición
    reset_button = Button(reset_button_ax, 'Reset Parpadeos', color='#f2f2f2', hovercolor='lightcoral')

    def reset_blinks(event):
        global blink_count
        blink_count = 0
        text_blinks.set_text(f"Parpadeos: {blink_count}")
        fig.canvas.draw_idle()

    reset_button.on_clicked(reset_blinks)  # Vincular botón con función

    # --- Área para las advertencias ---
    ax_warnings = plt.axes([0.1, 0.75, 0.8, 0.1])
    ax_warnings.axis("off")
    text_blink_warning = ax_warnings.text(0.5, 0.8, "", fontsize=14, fontweight='bold', color='red', ha='center', va='center')
    text_fixation_warning = ax_warnings.text(0.5, 0.1, "", fontsize=14, fontweight='bold', color='purple', ha='center', va='center')


    # --- Botones OK Parpadeo y OK Fijación ---
    blink_button_ax = plt.axes([0.2, 0.65, 0.25, 0.05])
    fixation_button_ax = plt.axes([0.55, 0.65, 0.25, 0.05])
    blink_button = Button(blink_button_ax, 'OK Parpadeo', color='#f2f2f2', hovercolor='lightgreen')
    fixation_button = Button(fixation_button_ax, 'OK Fijación', color='#f2f2f2', hovercolor='lightgreen')

    def clear_blink_warning(event):
        global blink_warning_acknowledged, blink_reset_time
        blink_warning_acknowledged = True
        blink_reset_time = time.time()
        text_blink_warning.set_text("")
        fig.canvas.draw_idle()

    def clear_fixation_warning(event):
        global fixation_warning_persistent, fixation_detected, fixation_start_time
        fixation_warning_persistent = False
        fixation_detected = False
        fixation_start_time = None
        text_fixation_warning.set_text("")
        fig.canvas.draw_idle()

    blink_button.on_clicked(clear_blink_warning)
    fixation_button.on_clicked(clear_fixation_warning)

    # --- Título de los gráficos ---
    ax_title = plt.axes([0.1, 0.55, 0.8, 0.05])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, "MOVIMIENTOS DE LA PUPILA EN TIEMPO REAL", fontsize=14, fontweight='bold', color='black', ha='center', va='center')

    # --- Gráfico de movimiento en X ---
    ax2 = plt.axes([0.1, 0.30, 0.8, 0.2])
    ax2.set_facecolor("#e6e6e6")
    ax2.set_title("Movimiento en X", fontsize=10)
    ax2.set_ylabel("Posición X (px)", fontsize=10)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    line_x, = ax2.plot([], [], color='blue', linewidth=1)

    # --- Gráfico de movimiento en Y ---
    ax3 = plt.axes([0.1, 0., 0.8, 0.2])
    ax3.set_facecolor("#e6e6e6")
    ax3.set_title("Movimiento en Y", fontsize=10)
    ax3.set_ylabel("Posición Y (px)", fontsize=10)
    ax3.set_xlabel("Tiempo (s)", fontsize=10)
    ax3.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    line_y, = ax3.plot([], [], color='orange', linewidth=1)

    while True:
        elapsed_time = time.time() - blink_reset_time
        if elapsed_time >= 60 and blink_count < 50:
            blink_warning_persistent = True
        else:
            blink_warning_persistent = False

        # --- Actualizar advertencias ---
        if blink_warning_persistent:
            text_blink_warning.set_text("¡ATENCIÓN! FRECUENCIA DE PARPADEO BAJA.\nCONSIDERE TOMARSE UN DESCANSO\nY REALIZAR LA TÉCNICA 20-20-20")
        else:
            text_blink_warning.set_text("")

        if fixation_detected and not fixation_warning_persistent:
            fixation_warning_persistent = True
        if fixation_warning_persistent:
            text_fixation_warning.set_text("¡FIJACIÓN VISUAL EN PANTALLA EXCESIVA!\nCONSIDERE TOMARSE UN DESCANSO")
        else:
            text_fixation_warning.set_text("")

        # --- Actualizar contador de parpadeos ---
        text_blinks.set_text(f"Parpadeos: {blink_count}")

        # --- Actualizar gráficos ---
        if timestamps:
            current_time = np.array(timestamps[-max_graph_points:]) - timestamps[0]
            smooth_x = smooth_data(pupil_x_positions[-max_graph_points:])
            smooth_y = smooth_data([-y for y in pupil_y_positions[-max_graph_points:]])

            length = min(len(current_time), len(smooth_x), len(smooth_y))
            line_x.set_data(current_time[-length:], smooth_x[-length:])
            line_y.set_data(current_time[-length:], smooth_y[-length:])

            ax2.set_xlim(current_time[0], current_time[-1] if len(current_time) > 1 else 1)
            ax3.set_xlim(current_time[0], current_time[-1] if len(current_time) > 1 else 1)

            ax2.set_ylim(min(smooth_x[-length:]) - 5, max(smooth_x[-length:]) + 5)
            ax3.set_ylim(min(smooth_y[-length:]) - 5, max(smooth_y[-length:]) + 5)

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)



def detect_pupil_in_roi(frame, gray, roi):
    global eye_closed, blink_count, interruption_count, lost_pupil_frames, last_event_time, last_blink_time
    global fixation_start_time, fixation_detected

    roi_x, roi_y, roi_w, roi_h = roi
    roi_gray = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Mejorar contraste y eliminar ruido
    roi_gray = cv2.equalizeHist(roi_gray)
    blurred = cv2.GaussianBlur(roi_gray, (7, 7), 0)

    # Umbralización inversa para destacar pupilas
    _, thresholded = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Umbralizado ROI", cv2.resize(thresholded, (frame.shape[1], frame.shape[0])))

    # Encontrar contornos
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_pupil = None

    # Parámetros de detección
    min_radius = 8
    max_radius = 18
    min_area_green = 40
    max_area = 900
    min_area_yellow = 30
    edge_margin = 12

    # Detección de pupilas claramente visibles (verde)
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area_green <= area <= max_area:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if min_radius <= radius <= max_radius and edge_margin < x < roi_w - edge_margin and edge_margin < y < roi_h - edge_margin:
                mask = np.zeros_like(roi_gray)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_intensity = cv2.mean(roi_gray, mask=mask)[0]
                if mean_intensity < 55:
                    area_ratio = area / (np.pi * (radius ** 2))
                    if 0.55 < area_ratio < 1.4:
                        detected_pupil = (int(x) + roi_x, int(y) + roi_y)
                        cv2.circle(frame, detected_pupil, int(radius), (0, 255, 0), 2)
                        break

    # Detección de pupilas parcialmente visibles (amarillo)
    if detected_pupil is None:
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area_yellow <= area <= max_area:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if min_radius <= radius <= max_radius and edge_margin < x < roi_w - edge_margin and edge_margin < y < roi_h - edge_margin:
                    mask = np.zeros_like(roi_gray)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    mean_intensity = cv2.mean(roi_gray, mask=mask)[0]
                    if mean_intensity < 70:
                        area_ratio = area / (np.pi * (radius ** 2))
                        if 0.5 < area_ratio < 1.5:
                            detected_pupil = (int(x) + roi_x, int(y) + roi_y)
                            cv2.circle(frame, detected_pupil, int(radius), (0, 255, 255), 2)
                            break

    # Manejo de parpadeos e interrupciones
    current_time = time.time()
    if detected_pupil is None:
        lost_pupil_frames += 1

        #  FILTRO DE MICROCORTES: Ignorar pérdidas menores a max_lost_frames  
        if lost_pupil_frames > max_lost_frames:
            if not eye_closed:
                eye_closed = True
                last_event_time = current_time
    else:
        if eye_closed:
            duration = current_time - last_event_time

            #  FILTRO DE MICROCORTES: Si la pupila reaparece rápido, ignoramos
            if lost_pupil_frames <= max_lost_frames:
                eye_closed = False
                lost_pupil_frames = 0
            else:
                # 🔹 Comparar con la nueva posición  
                if len(pupil_y_positions) >= 2:
                    last_y = np.mean(pupil_y_positions[-2:])
                    last_x = np.mean(pupil_x_positions[-2:])
                elif pupil_y_positions:
                    last_y = pupil_y_positions[-1]
                    last_x = pupil_x_positions[-1]
                else:
                    last_y, last_x = detected_pupil[1], detected_pupil[0]

                vertical_displacement = abs(detected_pupil[1] - last_y)
                horizontal_displacement = abs(detected_pupil[0] - last_x)

                # 🔹 Ajustamos los umbrales de movimiento para no perder parpadeos rápidos
                vertical_threshold = 25
                horizontal_threshold = 35

                # Confirmación extra de microcortes
                if lost_pupil_frames <= max_lost_frames + 1:
                    eye_closed = False
                    lost_pupil_frames = 0
                else:
                    if blink_threshold_min - 0.05 <= duration <= blink_threshold_max + 0.1:
                        blink_count += 1
                        last_blink_time = current_time
                    elif duration > interruption_threshold:
                        interruption_count += 1

            eye_closed = False

        lost_pupil_frames = 0

        # Guardar posición detectada para próximas comparaciones
        timestamps.append(time.time())
        pupil_x_positions.append(detected_pupil[0])
        pupil_y_positions.append(detected_pupil[1])

    #  Lógica de fijación visual
    if len(pupil_x_positions) > 1 and len(pupil_y_positions) > 1:
        last_x, last_y = pupil_x_positions[-2], pupil_y_positions[-2]
        current_x, current_y = detected_pupil if detected_pupil else (last_x, last_y)
        movement = np.sqrt((current_x - last_x) ** 2 + (current_y - last_y) ** 2)

        if movement <= 15:  # 🔹 Se bajó de 25 a 15 para mejorar la detección de fijación
            if fixation_start_time is None:
                fixation_start_time = current_time
            elif current_time - fixation_start_time >= fixation_threshold:
                fixation_detected = True
        else:
            if fixation_detected and (current_time - fixation_start_time) < fixation_threshold + 3:
                fixation_detected = True  # 🔹 Permite que la fijación dure un poco más antes de desactivarse
            else:
                fixation_start_time = None
                fixation_detected = False

    return frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

    roi_x, roi_y, roi_w, roi_h = 300, 200, 150, 100
    roi = (roi_x, roi_y, roi_w, roi_h)

    try:
        Thread(target=plot_all_graphs, daemon=True).start()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = detect_pupil_in_roi(frame, gray, roi)

            # Dibujar ROI con bordes redondeados
            draw_rounded_rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), 20, (255, 255, 255), 2)

            # Obtener dimensiones de la imagen
            h, w, _ = frame.shape

            # Calcular la posición para centrar el texto
            text = "POSICIONE SU OJO EN EL CENTRO DEL MARCO BLANCO"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = (w - text_size[0]) // 2  # Centrar horizontalmente
            text_y = 40  # Mantenerlo en la parte superior

            # Dibujar el texto centrado en la parte superior con el color del área marcada (azul)
            draw_text_with_shadow(frame, text, (text_x, text_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # 🔵 Azul

            # Obtener color dinámico para los contadores
            blink_color = (0, 255, 0)  # 🟢 Verde normal
            interruption_color = (0, 255, 255)  # 🟡 Amarillo fijo

            if blink_warning_persistent:
                blink_color = (0, 0, 255)  # 🔴 Rojo si hay advertencia de parpadeo
            elif fixation_warning_persistent:
                blink_color = (128, 0, 128)  # 🟣 Morado si hay fijación visual

            # Dibujar contadores en la esquina inferior izquierda con colores dinámicos
            draw_text_with_shadow(frame, f"Parpadeos: {blink_count}", 
                                  (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blink_color, 2)

            draw_text_with_shadow(frame, f"Interrupciones: {interruption_count}", 
                                  (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, interruption_color, 2)  # 🟡 Siempre amarillo

            # Mostrar la ventana con la imagen procesada
            cv2.imshow("Detección de Pupilas", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        generate_summary()

if __name__ == "__main__":
    main()
