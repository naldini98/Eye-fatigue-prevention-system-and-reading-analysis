import cv2
import numpy as np
from threading import Thread
import time
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime



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
interruption_active = False
start_time = time.time()  # 🔹 Registrar el tiempo de inicio correctamente




# Contadores
fixation_count = 0
saccadic_count = 0
regression_count = 0
visible_pupil_count = 0
partially_visible_pupil_count = 0
interruption_count = 0  # Contador de interrupciones detectadas




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




def draw_top_centered_text(canvas, text, color, position, align_left=False):
    """Dibuja texto en la imagen usando PIL para soportar tildes y caracteres especiales."""
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("arial.ttf", 24)  # 🔹 Usa una fuente que soporte caracteres especiales
    except IOError:
        font = ImageFont.load_default()  # 🔹 Si no encuentra la fuente, usa la predeterminada

    text_size = draw.textbbox((0, 0), text, font=font)  # Obtiene el tamaño del texto
    text_width = text_size[2] - text_size[0]
    
    if align_left:
        text_x = 10  # 🔹 Si es alineado a la izquierda, pone el texto cerca del borde izquierdo
    else:
        text_x = (canvas.shape[1] - text_width) // 2  # 🔹 Si no, lo centra

    text_y = position
    draw.text((text_x, text_y), text, font=font, fill=color)  # Dibuja el texto con PIL

    return np.array(pil_img)  # 🔹 Devuelve la imagen con el texto agregado




# Ventana negra para representar el movimiento del centroide

def draw_centroid_movement():

    global previous_position, saccadic_movement, last_saccadic_time
    global fixation_detected, last_fixation_start, interruption_detected, exit_flag
    global fixation_count, saccadic_count, regression_count

    # 🔹 Aumentar el tamaño de la ventana negra (antes 500x500, ahora 800x800)
    window_size = 800  
    point_radius = 8  # Ajustar tamaño del punto para que sea más visible en ventana grande
    regression_detected = False
    last_regression_time = 0
    regression_duration = 2
    horizontal_threshold = 5

    # Crear lienzo negro más grande
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

            # 🔹 Ajustar la escala de los movimientos para adaptarse al nuevo tamaño
            x_scaled = int((x - 300) / roi_width * window_size)
            y_scaled = int((y - 200) / roi_height * window_size)

            dx, dy = 0, 0

            if previous_position is not None:
                px, py = previous_position
                dx = x_scaled - px
                dy = y_scaled - py

                if abs(dx) > movement_threshold:

                    #EVALUAMOS SI ES REGRESIÓN
                
                    if dx > 0 and abs(dy) <= horizontal_threshold:
                        regression_detected = True
                        last_regression_time = time.time()
                        regression_count += 1
                        cv2.line(display_canvas, (px, py), (x_scaled, y_scaled), (255, 0, 0), 4)
                    
                    #SINO ES SACÁDICO

                    else:
                        cv2.line(display_canvas, (px, py), (x_scaled, y_scaled), (0, 0, 255), 4)
                        saccadic_movement = True
                        last_saccadic_time = time.time()
                        saccadic_count += 1
                        fixation_detected = False
                        last_fixation_start = None
                else:
                    saccadic_movement = False

                #EVALUACIÓN DE FIJACIÓN

            if previous_position is not None:

                if abs(dx) <= movement_threshold and abs(dy) <= movement_threshold:
                    if last_fixation_start is None:
                        last_fixation_start = time.time()
                    elif time.time() - last_fixation_start >= fixation_threshold:
                        if not fixation_detected:
                            fixation_count += 1
                        fixation_detected = True
                else:
                    fixation_detected = False
                    last_fixation_start = None

            cv2.circle(display_canvas, (x_scaled, y_scaled), point_radius, (0, 255, 0), -1)
            previous_position = (x_scaled, y_scaled)

        # Mostrar las advertencias con PIL
        if time.time() - last_regression_time <= regression_duration:
            display_canvas = draw_top_centered_text(display_canvas, "REGRESIÓN LINEAL", (255, 0, 0), 80)
        elif not interruption_detected and (saccadic_movement or (time.time() - last_saccadic_time) <= saccadic_duration):
            display_canvas = draw_top_centered_text(display_canvas, "MOVIMIENTO SACÁDICO", (0, 0, 255), 80)
        if not interruption_detected and fixation_detected:
            display_canvas = draw_top_centered_text(display_canvas, "FIJACIÓN PROLONGADA", (255, 255, 255), 80)
        if interruption_detected:
            display_canvas = draw_top_centered_text(display_canvas, "INTERRUPCIÓN DETECTADA", (0, 255, 255), 80)

        # Mostrar contadores con texto más grande y posición ajustada
        display_canvas = draw_top_centered_text(display_canvas, f"Fijaciones: {fixation_count}", (255, 255, 255), window_size - 150, align_left=True)
        display_canvas = draw_top_centered_text(display_canvas, f"Sacádicos: {saccadic_count}", (255, 255, 255), window_size - 110, align_left=True)
        display_canvas = draw_top_centered_text(display_canvas, f"Regresiones: {regression_count}", (255, 255, 255), window_size - 70, align_left=True)

        cv2.imshow("Movimiento del Centroide", display_canvas)

        elapsed_time = time.time() - start_time
        wait_time = max(1, int(10 - elapsed_time * 1000))
        if cv2.waitKey(wait_time) & 0xFF == 27:
            exit_flag = True
            break

    cv2.destroyWindow("Movimiento del Centroide")




#FUNCION DE DETECCIÓN DE LA PUPILA.

def detect_pupil_in_roi(frame, gray, roi):
    global centroid_position, last_detection_time
    global visible_pupil_count, partially_visible_pupil_count
    global interruption_count, interruption_active

    roi_x, roi_y, roi_w, roi_h = roi
    roi_gray = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Aplicar ecualización del histograma y suavizado
    roi_gray = cv2.equalizeHist(roi_gray)
    blurred = cv2.GaussianBlur(roi_gray, (7, 7), 0)

    # Aplicar umbralización inversa
    _, thresholded = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)

    # 🔹 Obtener el tamaño de la ventana principal para ajustar la umbralización
    frame_height, frame_width = frame.shape[:2]  # Tamaño de la ventana principal

    # 🔹 Redimensionar la imagen umbralizada al tamaño de la ventana principal
    thresholded_resized = cv2.resize(thresholded, (frame_width, frame_height))

    # Mostrar la ventana umbralizada con el tamaño corregido
    cv2.imshow("Umbralizado ROI", thresholded_resized)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_pupil = None


    # Detectar pupilas **visibles** (verde)

    for contour in contours:

        area = cv2.contourArea(contour)
        if 40 < area < 800:  # Filtrar por tamaño
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if 5 < radius < 18 and 10 < x < roi_w - 10 and 10 < y < roi_h - 10:  # Verificar límites

                mask = np.zeros_like(roi_gray)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_intensity = cv2.mean(roi_gray, mask=mask)[0]

                #Verificamos que sea circular el área de la pupilla. Evitamos áreas irregulares
                if mean_intensity < 50:
                    area_ratio = area / (np.pi * (radius ** 2))
                    if 0.7 < area_ratio < 1.3:  # Validar relación de área
                        detected_pupil = (int(x) + roi_x, int(y) + roi_y)
                        cv2.circle(frame, detected_pupil, int(radius), (0, 255, 0), 2)  # Verde
                        visible_pupil_count += 1  # ✅ Incrementar contador de pupilas visibles
                        break

    # 🔹Detectar pupilas amarrirllas. Si no se detectó una pupila **clara**, buscar pupilas **parcialmente visibles** (amarillo)

    if detected_pupil is None:

        for contour in contours:

            #Relajamos los criterios de intensidad y forma para detectar parciamente
            area = cv2.contourArea(contour)
            if 30 < area < 800:  # Tamaño más permisivo
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if 5 < radius < 18 and 10 < x < roi_w - 10 and 10 < y < roi_h - 10:
                    mask = np.zeros_like(roi_gray)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    mean_intensity = cv2.mean(roi_gray, mask=mask)[0]
                    if mean_intensity < 60:
                        area_ratio = area / (np.pi * (radius ** 2))
                        if 0.6 < area_ratio < 1.4:
                            detected_pupil = (int(x) + roi_x, int(y) + roi_y)
                            cv2.circle(frame, detected_pupil, int(radius), (0, 255, 255), 2)  # Amarillo
                            partially_visible_pupil_count += 1  # ✅ Incrementar contador de pupilas parcialmente visibles
                            break


    # 🔹 Manejo de interrupciones

    if detected_pupil is None:
        if time.time() - last_detection_time > interruption_threshold:
            if not interruption_active:  # Evitar contar múltiples veces
                interruption_detected = True
                interruption_count += 1  # ✅ Aumentar contador de interrupciones
                interruption_active = True
        else:
            interruption_detected = False
    else:
        interruption_active = False
        centroid_position = detected_pupil
        last_detection_time = time.time()

    return frame




from datetime import datetime

def evaluate_reading_efficiency(total_time):
    """Evalúa la eficiencia lectora basada en los datos recopilados"""
    global fixation_count, saccadic_count, regression_count, interruption_count
    
    sacadicos_per_second = saccadic_count / total_time if total_time > 0 else 0
    regression_percentage = (regression_count / saccadic_count) * 100 if saccadic_count > 0 else 0
    interruptions_per_minute = (interruption_count / (total_time / 60)) if total_time > 0 else 0

    analysis = []
    efficiency_level = "Buena eficiencia lectora"

    # Evaluación de sacádicos
    if sacadicos_per_second < 2:
        analysis.append("⚠️ Baja cantidad de sacádicos: posible lectura lenta o dificultades de atención.")
        efficiency_level = "Eficiencia lectora baja"
    elif sacadicos_per_second > 3.5:
        analysis.append("⚠️ Sacádicos excesivos: posible lectura superficial sin buena comprensión.")

    # Evaluación de regresiones
    if regression_percentage > 20:
        analysis.append("⚠️ Exceso de regresiones: puede indicar dificultades en la comprensión lectora.")
        efficiency_level = "Eficiencia lectora baja"

    # Evaluación de interrupciones
    if interruptions_per_minute > 3:
        analysis.append("⚠️ Muchas interrupciones detectadas: posible distracción frecuente o problemas con la visibilidad.")

    # Evaluación de fijaciones
    if fixation_count > saccadic_count:
        analysis.append("⚠️ Exceso de fijaciones prolongadas: puede afectar la fluidez de lectura.")

    # Generar texto final
    report = [
        "Evaluación de la Eficiencia Lectora",
        f"Tiempo total de lectura: {int(total_time // 60)} min {int(total_time % 60)} seg",
        f"Sacádicos por segundo: {sacadicos_per_second:.2f}",
        f"Porcentaje de regresiones: {regression_percentage:.2f}%",
        f"Interrupciones por minuto: {interruptions_per_minute:.2f}",
        f"Nivel de eficiencia lectora: {efficiency_level}",
        "---------------------------------------------",
    ] + analysis

    return "\n".join(report)




def save_results():
    global fixation_count, saccadic_count, regression_count, interruption_count, start_time

    # 🔹 Asegurar que el tiempo total de ejecución se calcula correctamente
    total_time = time.time() - start_time  # ✅ Ahora `start_time` está correctamente inicializado
    total_time_str = f"{int(total_time // 60)} minutos y {int(total_time % 60)} segundos"

    # 🔹 Evitar división por 0 en los cálculos
    if total_time > 0:
        sacadicos_por_segundo = saccadic_count / total_time
        interrupciones_por_minuto = (interruption_count / total_time) * 60
    else:
        sacadicos_por_segundo = 0
        interrupciones_por_minuto = 0


    porcentaje_regresiones = (regression_count / max(saccadic_count, 1)) * 100  # Evitar división por 0

    # 🔹 Generar la evaluación según los valores obtenidos
    nivel_eficiencia = "Buena eficiencia lectora"
    observaciones = []

    if sacadicos_por_segundo > 3:  # 🔹 Un umbral realista para lectura eficiente
        observaciones.append("⚠️ Sacádicos excesivos: posible lectura superficial sin buena comprensión.")

    if porcentaje_regresiones > 20:
        observaciones.append("⚠️ Muchas regresiones detectadas: posible dificultad de comprensión.")

    if interrupciones_por_minuto > 5:
        observaciones.append("⚠️ Muchas interrupciones detectadas: posible distracción frecuente.")

    # 🔹 Guardar en archivo de texto
    results = [
        "Resultados de la Detección de Pupilas",
        f"Fecha y Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Tiempo total de ejecución: {total_time_str}",
        f"Total de Fijaciones: {fixation_count}",
        f"Total de Sacádicos: {saccadic_count}",
        f"Total de Regresiones: {regression_count}",
        f"Total de Interrupciones: {interruption_count}",
        "-------------------------------------------",
        "Evaluación de la Eficiencia Lectora",
        f"Tiempo total de lectura: {total_time_str}",
        f"Sacádicos por segundo: {sacadicos_por_segundo:.2f}",
        f"Porcentaje de regresiones: {porcentaje_regresiones:.2f}%",
        f"Interrupciones por minuto: {interrupciones_por_minuto:.2f}",
        f"Nivel de eficiencia lectora: {nivel_eficiencia}",
        "-------------------------------------------",
    ]

    results.extend(observaciones)

    with open("resultados_análisis_lectura.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(results))

    print("✅ Resultados guardados correctamente en 'resultados_deteccion.txt'")







# Función principal

def main():
    global exit_flag
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

    roi_x, roi_y, roi_w, roi_h = 300, 200, 150, 100
    roi = (roi_x, roi_y, roi_w, roi_h)

    Thread(target=draw_centroid_movement, daemon=True).start()

    while not exit_flag:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = detect_pupil_in_roi(frame, gray, roi)

        # Dibujar el marco blanco redondeado
        draw_rounded_rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), 20, (255, 255, 255), 2)
        # Calcular la posición del texto para centrarlo
        text = "POSICIONE SU OJO EN EL CENTRO DEL MARCO BLANCO"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = 40  # Ubicación en la parte superior

        # Dibujar el texto con sombra
        draw_text_with_shadow(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        draw_text_with_shadow(frame, f"Interrupciones: {interruption_count}", 
                      (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Amarillo

        cv2.imshow("Detección de Pupilas", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            exit_flag = True
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    save_results()  # Asegura que se llame esta función
