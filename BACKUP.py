import cv2
import numpy as np
from threading import Thread
import time

# Variables globales
centroid_position = None
previous_position = None
movement_multiplier = 2
movement_threshold = 10
fixation_threshold = 5
fixation_detected = False
last_fixation_start = None
saccadic_movement = False
last_saccadic_time = 0
saccadic_duration = 2
last_detection_time = time.time()
interruption_threshold = 2
interruption_detected = False
exit_flag = False

# Contadores
fixation_count = 0
saccadic_count = 0
regression_count = 0
visible_pupil_count = 0
partially_visible_pupil_count = 0

# Ventana negra para representar el movimiento del centroide
def draw_centroid_movement():
    global previous_position, saccadic_movement, last_saccadic_time
    global fixation_detected, last_fixation_start, interruption_detected, exit_flag
    global fixation_count, saccadic_count, regression_count
    window_size = 500
    point_radius = 5
    regression_detected = False
    last_regression_time = 0
    regression_duration = 2
    horizontal_threshold = 5

    canvas = np.zeros((window_size, window_size, 3), dtype=np.uint8)

    while not exit_flag:
        start_time = time.time()
        display_canvas = canvas.copy()

        if time.time() - last_detection_time > interruption_threshold:
            interruption_detected = True
        else:
            interruption_detected = False

        if not interruption_detected and centroid_position is not None:
            x, y = centroid_position
            roi_width, roi_height = 150, 100
            x_scaled = int((x - 300) / roi_width * window_size)
            y_scaled = int((y - 200) / roi_height * window_size)

            dx, dy = 0, 0

            if previous_position is not None:
                px, py = previous_position
                dx = x_scaled - px
                dy = y_scaled - py

                if abs(dx) > movement_threshold:
                    if dx > 0 and abs(dy) <= horizontal_threshold:
                        regression_detected = True
                        last_regression_time = time.time()
                        regression_count += 1  # Incrementar contador de regresiones
                        cv2.line(display_canvas, (px, py), (x_scaled, y_scaled), (255, 0, 0), 4)
                    else:
                        cv2.line(display_canvas, (px, py), (x_scaled, y_scaled), (0, 0, 255), 4)
                        saccadic_movement = True
                        last_saccadic_time = time.time()
                        saccadic_count += 1  # Incrementar contador de sacádicos
                        fixation_detected = False
                        last_fixation_start = None
                else:
                    saccadic_movement = False

            if previous_position is not None:
                if abs(dx) <= movement_threshold and abs(dy) <= movement_threshold:
                    if last_fixation_start is None:
                        last_fixation_start = time.time()
                    elif time.time() - last_fixation_start >= fixation_threshold:
                        if not fixation_detected:
                            fixation_count += 1  # Incrementar contador de fijaciones
                        fixation_detected = True
                else:
                    fixation_detected = False
                    last_fixation_start = None

            cv2.circle(display_canvas, (x_scaled, y_scaled), point_radius, (0, 255, 0), -1)
            previous_position = (x_scaled, y_scaled)

        def draw_top_centered_text(canvas, text, color, position):
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (window_size - text_size[0]) // 2
            text_y = position
            cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if time.time() - last_regression_time <= regression_duration:
            draw_top_centered_text(display_canvas, "REGRESION LINEAL", (255, 0, 0), 50)
        elif not interruption_detected and (saccadic_movement or (time.time() - last_saccadic_time) <= saccadic_duration):
            draw_top_centered_text(display_canvas, "MOVIMIENTO SACADICO", (0, 0, 255), 50)
        if not interruption_detected and fixation_detected:
            draw_top_centered_text(display_canvas, "FIJACIÓN PROLONGADA", (255, 255, 255), 50)
        if interruption_detected:
            draw_top_centered_text(display_canvas, "INTERRUPCION DETECTADA", (0, 255, 255), 50)

        # Mostrar contadores
        draw_top_centered_text(display_canvas, f"Fijaciones: {fixation_count}", (255, 255, 255), 100)
        draw_top_centered_text(display_canvas, f"Sacadicos: {saccadic_count}", (255, 255, 255), 140)
        draw_top_centered_text(display_canvas, f"Regresiones: {regression_count}", (255, 255, 255), 180)

        cv2.imshow("Movimiento del Centroide", display_canvas)

        elapsed_time = time.time() - start_time
        wait_time = max(1, int(10 - elapsed_time * 1000))
        if cv2.waitKey(wait_time) & 0xFF == 27:
            exit_flag = True
            break

    cv2.destroyWindow("Movimiento del Centroide")


# Función para detectar la pupila y calcular su centroide
def detect_pupil_in_roi(frame, gray, roi):
    global centroid_position, last_detection_time
    global visible_pupil_count, partially_visible_pupil_count

    roi_x, roi_y, roi_w, roi_h = roi
    roi_gray = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    roi_gray = cv2.equalizeHist(roi_gray)
    blurred = cv2.GaussianBlur(roi_gray, (7, 7), 0)
    _, thresholded = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Umbralizado ROI", cv2.resize(thresholded, (400, 300)))

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_pupil = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if 40 < area < 800:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if 5 < radius < 18 and 10 < x < roi_w - 10 and 10 < y < roi_h - 10:
                mask = np.zeros_like(roi_gray)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_intensity = cv2.mean(roi_gray, mask=mask)[0]
                if mean_intensity < 50:
                    area_ratio = area / (np.pi * (radius ** 2))
                    if 0.7 < area_ratio < 1.3:
                        detected_pupil = (int(x) + roi_x, int(y) + roi_y)
                        cv2.circle(frame, detected_pupil, int(radius), (0, 255, 0), 2)
                        visible_pupil_count += 1  # Incrementar contador de pupilas visibles
                        break

    if detected_pupil is None:
        for contour in contours:
            area = cv2.contourArea(contour)
            if 30 < area < 800:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if 5 < radius < 18 and 10 < x < roi_w - 10 and 10 < y < roi_h - 10:
                    mask = np.zeros_like(roi_gray)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    mean_intensity = cv2.mean(roi_gray, mask=mask)[0]
                    if mean_intensity < 60:
                        area_ratio = area / (np.pi * (radius ** 2))
                        if 0.6 < area_ratio < 1.4:
                            detected_pupil = (int(x) + roi_x, int(y) + roi_y)
                            cv2.circle(frame, detected_pupil, int(radius), (0, 255, 255), 2)
                            partially_visible_pupil_count += 1  # Incrementar contador de pupilas parcialmente visibles
                            break

    if detected_pupil is not None:
        centroid_position = detected_pupil
        last_detection_time = time.time()

    return frame


# Función principal
def main():
    global exit_flag
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

    print("Iniciando detección de pupilas...")  # Mensaje de inicio

    roi_x, roi_y, roi_w, roi_h = 300, 200, 150, 100
    roi = (roi_x, roi_y, roi_w, roi_h)

    # Iniciar el hilo para el análisis del movimiento del centroide
    thread = Thread(target=draw_centroid_movement, daemon=True)
    thread.start()

    while not exit_flag:
        ret, frame = cap.read()
        
        if not ret:
            print("No se pudo obtener un cuadro de la cámara.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = detect_pupil_in_roi(frame, gray, roi)

        # Dibujar la ROI en el video
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
        cv2.imshow("Detección de Pupilas", frame)

        # Detectar si se presiona ESC (tecla 27) para salir
        if cv2.waitKey(1) & 0xFF == 27:
            exit_flag = True
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Ejecución finalizada.")  # Mensaje de cierre

if __name__ == "__main__":
    main()  # ✅ Corrección del error en el nombre
