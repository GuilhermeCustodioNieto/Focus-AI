import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import threading

# ── Configurações ──────────────────────────────────────────
TEMPO_DISTRACACAO = 5      # segundos distraído antes de alertar
EAR_LIMIAR        = 0.22   # abaixo disso = olhos fechados
YAW_LIMIAR        = 25     # graus de rotação lateral da cabeça
PITCH_LIMIAR      = 20     # graus de inclinação vertical

# ── Inicialização ──────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('voice', 'brazil')  # tenta voz PT-BR

# ── Função de alerta por voz (em thread separada) ──────────
def falar(texto):
    def _falar():
        engine.say(texto)
        engine.runAndWait()
    threading.Thread(target=_falar, daemon=True).start()

# ── Cálculo do EAR (Eye Aspect Ratio) ─────────────────────
def calcular_ear(landmarks, indices_olho, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices_olho]
    # distâncias verticais
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # distância horizontal
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

# Índices dos olhos no MediaPipe Face Mesh
OLHO_ESQ = [362, 385, 387, 263, 373, 380]
OLHO_DIR = [33,  160, 158,  133, 153, 144]

# ── Estimativa de pose da cabeça ───────────────────────────
PONTOS_3D = np.array([
    [0.0,   0.0,   0.0],    # ponta do nariz
    [0.0,  -330.0, -65.0],  # queixo
    [-225.0, 170.0, -135.0],# canto esq do olho
    [225.0,  170.0, -135.0],# canto dir do olho
    [-150.0,-150.0, -125.0],# canto esq da boca
    [150.0, -150.0, -125.0] # canto dir da boca
], dtype=np.float64)

IDX_POSE = [1, 152, 263, 33, 287, 57]  # landmarks correspondentes

def estimar_pose(landmarks, w, h):
    pts_2d = np.array([
        [landmarks[i].x * w, landmarks[i].y * h] for i in IDX_POSE
    ], dtype=np.float64)

    focal = w
    centro = (w / 2, h / 2)
    matriz_cam = np.array([
        [focal, 0,     centro[0]],
        [0,     focal, centro[1]],
        [0,     0,     1]
    ], dtype=np.float64)
    dist_coefs = np.zeros((4, 1))

    _, rot_vec, _ = cv2.solvePnP(
        PONTOS_3D, pts_2d, matriz_cam, dist_coefs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rot_mat)
    pitch, yaw, roll = angles
    return pitch, yaw, roll

# ── Loop principal ─────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

tempo_distracado_inicio = None
ultimo_alerta = 0
focado = True
total_frames = 0
frames_focado = 0

print("Sistema iniciado! Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # espelha
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = face_mesh.process(rgb)

    total_frames += 1
    status = "FOCADO"
    cor_status = (34, 197, 94)   # verde
    motivo = ""

    if resultado.multi_face_landmarks:
        lm = resultado.multi_face_landmarks[0].landmark

        # Calcula EAR
        ear_esq = calcular_ear(lm, OLHO_ESQ, w, h)
        ear_dir = calcular_ear(lm, OLHO_DIR, w, h)
        ear_medio = (ear_esq + ear_dir) / 2.0

        # Estima pose
        pitch, yaw, roll = estimar_pose(lm, w, h)

        # ── Avalia foco ────────────────────────────────────
        olhos_fechados = ear_medio < EAR_LIMIAR
        cabeca_virada  = abs(yaw)   > YAW_LIMIAR
        cabeca_baixo   = abs(pitch) > PITCH_LIMIAR

        if olhos_fechados:
            status = "DISTRAIDO"
            motivo = "Olhos fechados"
            cor_status = (60, 60, 220)
        elif cabeca_virada:
            status = "DISTRAIDO"
            motivo = f"Cabeca virada ({yaw:.0f}g)"
            cor_status = (60, 60, 220)
        elif cabeca_baixo:
            status = "DISTRAIDO"
            motivo = f"Olhando pra baixo ({pitch:.0f}g)"
            cor_status = (60, 60, 220)

        # Métricas na tela
        cv2.putText(frame, f"EAR: {ear_medio:.2f}", (10, h-70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
        cv2.putText(frame, f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}", (10, h-45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    else:
        status = "AUSENTE"
        motivo = "Rosto nao detectado"
        cor_status = (100, 100, 100)

    # ── Controle de tempo de distração ────────────────────
    agora = time.time()
    if status == "FOCADO":
        tempo_distracado_inicio = None
        focado = True
        frames_focado += 1
    else:
        if tempo_distracado_inicio is None:
            tempo_distracado_inicio = agora
        tempo_dist = agora - tempo_distracado_inicio

        # Alerta após X segundos distraído
        if tempo_dist >= TEMPO_DISTRACACAO and (agora - ultimo_alerta) > 10:
            ultimo_alerta = agora
            if status == "AUSENTE":
                falar("Você saiu. Volte aos estudos!")
            elif "Olhos" in motivo:
                falar("Atenção! Seus olhos estão fechados.")
            else:
                falar("Atenção! Você está distraído. Foque nos estudos.")

        # Barra de distração
        barra_w = int((tempo_dist / TEMPO_DISTRACACAO) * 200)
        barra_w = min(barra_w, 200)
        cv2.rectangle(frame, (10, h-20), (210, h-8), (50,50,50), -1)
        cv2.rectangle(frame, (10, h-20), (10+barra_w, h-8), (60,60,220), -1)
        cv2.putText(frame, f"Dist: {tempo_dist:.1f}s", (215, h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

    # ── Porcentagem de foco ────────────────────────────────
    pct = int((frames_focado / total_frames) * 100) if total_frames else 0

    # ── HUD na tela ────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.putText(frame, status, (12, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, cor_status, 2)
    if motivo:
        cv2.putText(frame, motivo, (200, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
    cv2.putText(frame, f"Foco: {pct}%", (w-120, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 1)

    cv2.imshow("Detector de Foco", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nSessão encerrada. Foco médio: {pct}%")