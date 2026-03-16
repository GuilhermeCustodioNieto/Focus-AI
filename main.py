import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import threading
from PIL import Image, ImageDraw
import pystray

# ══════════════════════════════════════════════════════════
#  CONFIGURAÇÕES — ajuste à vontade
# ══════════════════════════════════════════════════════════
TEMPO_DISTRACACAO   = 8     # segundos distraído antes do 1º alerta
INTERVALO_ALERTA    = 15    # segundos entre alertas repetidos
EAR_LIMIAR          = 0.18  # mais baixo = menos sensível a piscadas

# Tolerância EXTRA ao redor da posição calibrada (graus)
# Aumente se ainda travar como distraído ao olhar pro monitor
TOLERANCIA_YAW      = 30
TOLERANCIA_PITCH    = 22

# Segundos coletando dados na calibração
SEGUNDOS_CALIBRACAO = 5

# ══════════════════════════════════════════════════════════
#  ESTADO GLOBAL
# ══════════════════════════════════════════════════════════
rodando        = True
pausado        = False
status_atual   = "Iniciando..."
pct_foco_atual = 0

# Valores calibrados (centro natural da cabeça do usuário)
yaw_base   = 0.0
pitch_base = 0.0
calibrado  = False

# ══════════════════════════════════════════════════════════
#  TTS
# ══════════════════════════════════════════════════════════
engine = pyttsx3.init()
engine.setProperty('rate', 155)
tts_lock = threading.Lock()

def falar(texto):
    def _falar():
        with tts_lock:
            engine.say(texto)
            engine.runAndWait()
    threading.Thread(target=_falar, daemon=True).start()

# ══════════════════════════════════════════════════════════
#  MEDIAPIPE
# ══════════════════════════════════════════════════════════
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

OLHO_ESQ = [362, 385, 387, 263, 373, 380]
OLHO_DIR = [33,  160, 158, 133, 153,  144]

PONTOS_3D = np.array([
    [0.0,    0.0,    0.0  ],
    [0.0,   -330.0, -65.0 ],
    [-225.0, 170.0, -135.0],
    [225.0,  170.0, -135.0],
    [-150.0,-150.0, -125.0],
    [150.0, -150.0, -125.0],
], dtype=np.float64)
IDX_POSE = [1, 152, 263, 33, 287, 57]

def calcular_ear(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

def estimar_pose(landmarks, w, h):
    pts_2d = np.array([[landmarks[i].x * w, landmarks[i].y * h]
                       for i in IDX_POSE], dtype=np.float64)
    focal = w
    cam   = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)
    _, rot_vec, _ = cv2.solvePnP(PONTOS_3D, pts_2d, cam, np.zeros((4,1)),
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rot_mat)
    return angles  # pitch, yaw, roll

# ══════════════════════════════════════════════════════════
#  ALERTA VISUAL — janela vermelha piscando
# ══════════════════════════════════════════════════════════
alerta_ativo = False
alerta_lock  = threading.Lock()

def mostrar_alerta_visual(mensagem):
    """Abre uma janela de alerta vermelha em cima de tudo."""
    global alerta_ativo
    with alerta_lock:
        if alerta_ativo:
            return
        alerta_ativo = True

    def _janela():
        global alerta_ativo
        nome = "⚠ FOCO PERDIDO"
        cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(nome, 600, 200)
        # Tenta colocar no centro da tela
        cv2.moveWindow(nome, 660, 440)
        cv2.setWindowProperty(nome, cv2.WND_PROP_TOPMOST, 1)

        inicio = time.time()
        fase   = 0
        while time.time() - inicio < 4.0:
            # Pisca entre vermelho escuro e vermelho vivo
            fase = 1 - fase
            cor_bg = (0, 0, 200) if fase else (0, 0, 120)
            tela   = np.full((200, 600, 3), cor_bg, dtype=np.uint8)

            # Ícone "!"
            cv2.circle(tela, (60, 100), 40, (0, 0, 255), -1)
            cv2.putText(tela, "!", (48, 118),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255,255,255), 3)

            # Texto principal
            cv2.putText(tela, "VOCE ESTA DISTRAIDO", (115, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)
            cv2.putText(tela, mensagem, (115, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 180), 2)
            cv2.putText(tela, "Volte aos estudos!", (115, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow(nome, tela)
            cv2.waitKey(400)

        cv2.destroyWindow(nome)
        alerta_ativo = False

    threading.Thread(target=_janela, daemon=True).start()

def disparar_alerta(motivo):
    falar("Atenção! Você está distraído. Volte aos estudos.")
    mostrar_alerta_visual(motivo)

# ══════════════════════════════════════════════════════════
#  CALIBRAÇÃO
# ══════════════════════════════════════════════════════════
def tela_calibracao(frame, progresso, total):
    """Desenha a tela de calibração sobre o frame."""
    overlay = frame.copy()
    h, w    = frame.shape[:2]

    # Fundo escuro semitransparente
    cv2.rectangle(overlay, (0,0), (w,h), (10,10,10), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Caixa central
    bx, by, bw, bh = w//2 - 280, h//2 - 100, 560, 200
    cv2.rectangle(frame, (bx,by), (bx+bw, by+bh), (30,30,30), -1)
    cv2.rectangle(frame, (bx,by), (bx+bw, by+bh), (80,80,80), 2)

    cv2.putText(frame, "CALIBRACAO", (bx+170, by+45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
    cv2.putText(frame, "Olhe normalmente para o monitor", (bx+55, by+85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)
    cv2.putText(frame, "e fique quieto por alguns segundos.", (bx+65, by+113),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200,200,200), 1)

    # Barra de progresso
    barra_x = bx + 30
    barra_y = by + 145
    barra_w = bw - 60
    pct_w   = int((progresso / total) * barra_w)
    cv2.rectangle(frame, (barra_x, barra_y), (barra_x+barra_w, barra_y+18), (50,50,50), -1)
    cv2.rectangle(frame, (barra_x, barra_y), (barra_x+pct_w,   barra_y+18), (34,197,94), -1)
    cv2.putText(frame, f"{int(progresso/total*100)}%",
                (barra_x + barra_w//2 - 15, barra_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return frame

# ══════════════════════════════════════════════════════════
#  BANDEJA DO SISTEMA
# ══════════════════════════════════════════════════════════
def criar_icone(cor_hex):
    img  = Image.new("RGBA", (64,64), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    r    = int(cor_hex[1:3],16)
    g    = int(cor_hex[3:5],16)
    b    = int(cor_hex[5:7],16)
    draw.ellipse([4,4,60,60], fill=(r,g,b,255))
    return img

def atualizar_icone(tray):
    while rodando:
        if pausado:
            tray.icon  = criar_icone("#888888")
            tray.title = "Focus AI — Pausado"
        elif not calibrado:
            tray.icon  = criar_icone("#f59e0b")
            tray.title = "Focus AI — Calibrando..."
        elif "FOCADO" in status_atual:
            tray.icon  = criar_icone("#22c55e")
            tray.title = f"Focus AI — Focado ({pct_foco_atual}%)"
        elif "AUSENTE" in status_atual:
            tray.icon  = criar_icone("#888888")
            tray.title = "Focus AI — Sem rosto"
        else:
            tray.icon  = criar_icone("#ef4444")
            tray.title = f"Focus AI — {status_atual}"
        time.sleep(2)

def pausar_retomar(icon, item):
    global pausado
    pausado = not pausado

def recalibrar(icon, item):
    global calibrado
    calibrado = False

def abrir_janela(icon, item):
    cv2.namedWindow("Focus AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Focus AI", 640, 480)

def encerrar(icon, item):
    global rodando
    rodando = False
    icon.stop()

def iniciar_tray():
    menu = pystray.Menu(
        pystray.MenuItem("Mostrar câmera",    abrir_janela),
        pystray.MenuItem("Pausar / Retomar",  pausar_retomar),
        pystray.MenuItem("Recalibrar",        recalibrar),
        pystray.MenuItem("Encerrar",          encerrar),
    )
    tray = pystray.Icon("FocusAI", criar_icone("#f59e0b"), "Focus AI", menu)
    threading.Thread(target=atualizar_icone, args=(tray,), daemon=True).start()
    tray.run()

# ══════════════════════════════════════════════════════════
#  LOOP PRINCIPAL DE VISÃO
# ══════════════════════════════════════════════════════════
def loop_visao():
    global rodando, calibrado, yaw_base, pitch_base
    global status_atual, pct_foco_atual

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("Focus AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Focus AI", 640, 480)
    cv2.setWindowProperty("Focus AI", cv2.WND_PROP_VISIBLE, 0)

    # — variáveis de calibração
    cal_amostras   = []
    cal_inicio     = None

    # — variáveis de sessão
    tempo_dist_inicio = None
    ultimo_alerta     = 0
    total_frames      = 0
    frames_focado     = 0

    while rodando:
        ret, frame = cap.read()
        if not ret:
            break

        if pausado:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = face_mesh.process(rgb)

        # ── FASE DE CALIBRAÇÃO ─────────────────────────────
        if not calibrado:
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                pitch, yaw, _ = estimar_pose(lm, w, h)

                if cal_inicio is None:
                    cal_inicio = time.time()

                elapsed = time.time() - cal_inicio
                cal_amostras.append((yaw, pitch))

                frame = tela_calibracao(frame, elapsed, SEGUNDOS_CALIBRACAO)

                if elapsed >= SEGUNDOS_CALIBRACAO:
                    yaw_base   = float(np.median([s[0] for s in cal_amostras]))
                    pitch_base = float(np.median([s[1] for s in cal_amostras]))
                    calibrado  = True
                    cal_amostras.clear()
                    cal_inicio = None
                    print(f"[Calibração] yaw_base={yaw_base:.1f}  pitch_base={pitch_base:.1f}")
                    falar("Calibração concluída. Bons estudos!")
            else:
                frame = tela_calibracao(frame, 0, SEGUNDOS_CALIBRACAO)
                # Reseta se perdeu o rosto durante calibração
                cal_inicio   = None
                cal_amostras = []

            cv2.imshow("Focus AI", frame)
            cv2.waitKey(1)
            continue

        # ── FASE NORMAL ────────────────────────────────────
        total_frames += 1
        status     = "FOCADO"
        cor_status = (34, 197, 94)
        motivo     = ""

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            ear   = (calcular_ear(lm, OLHO_ESQ, w, h) +
                     calcular_ear(lm, OLHO_DIR, w, h)) / 2
            pitch, yaw, _ = estimar_pose(lm, w, h)

            # Diferença em relação à posição calibrada
            delta_yaw   = abs(yaw   - yaw_base)
            delta_pitch = abs(pitch - pitch_base)

            olhos_fechados = ear         < EAR_LIMIAR
            cabeca_virada  = delta_yaw   > TOLERANCIA_YAW
            cabeca_baixo   = delta_pitch > TOLERANCIA_PITCH

            if olhos_fechados:
                status, motivo, cor_status = "DISTRAIDO", "Olhos fechados", (60,60,220)
            elif cabeca_virada:
                status, motivo, cor_status = "DISTRAIDO", f"Cabeca virada ({delta_yaw:.0f}g)", (60,60,220)
            elif cabeca_baixo:
                status, motivo, cor_status = "DISTRAIDO", f"Inclinacao ({delta_pitch:.0f}g)", (60,60,220)

            cv2.putText(frame,
                        f"EAR:{ear:.2f}  dYaw:{delta_yaw:.1f}  dPitch:{delta_pitch:.1f}",
                        (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150,150,150), 1)
        else:
            status, motivo, cor_status = "AUSENTE", "Rosto nao detectado", (100,100,100)

        # ── Alerta repetido ────────────────────────────────
        agora = time.time()
        if status == "FOCADO":
            tempo_dist_inicio = None
            frames_focado    += 1
        else:
            if tempo_dist_inicio is None:
                tempo_dist_inicio = agora
            tempo_dist = agora - tempo_dist_inicio

            if tempo_dist >= TEMPO_DISTRACACAO and (agora - ultimo_alerta) >= INTERVALO_ALERTA:
                ultimo_alerta = agora
                msg_motivo = motivo if motivo else status
                disparar_alerta(msg_motivo)

            # Barra de distração
            bw = min(int((tempo_dist / TEMPO_DISTRACACAO) * 200), 200)
            cv2.rectangle(frame, (10, h-32), (210, h-20), (40,40,40), -1)
            cv2.rectangle(frame, (10, h-32), (10+bw, h-20), (60,60,220), -1)

        pct            = int((frames_focado / total_frames) * 100) if total_frames else 0
        pct_foco_atual = pct
        status_atual   = status

        # HUD
        cv2.rectangle(frame, (0,0), (w,50), (20,20,20), -1)
        cv2.putText(frame, status,           (12,  33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, cor_status,    2)
        cv2.putText(frame, motivo,           (200, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
        cv2.putText(frame, f"Foco: {pct}%",  (w-120,33),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 1)

        cv2.imshow("Focus AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rodando = False
            break
        elif key == ord('h'):
            vis = cv2.getWindowProperty("Focus AI", cv2.WND_PROP_VISIBLE)
            cv2.setWindowProperty("Focus AI", cv2.WND_PROP_VISIBLE, 0 if vis > 0 else 1)
        elif key == ord('c'):
            calibrado = False

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSessão encerrada. Foco médio: {pct_foco_atual}%")

# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Focus AI iniciando...")
    print("Atalhos: [H] mostrar/esconder  [C] recalibrar  [Q] encerrar")
    threading.Thread(target=loop_visao, daemon=True).start()
    iniciar_tray()