"""SMILE TRAIN LAB — 카메라 선택 수정본.
실행하면 CAMERA SELECT 창이 먼저 열립니다. 영문 입력 상태에서:

  N / Space: 다음 카메라

  Enter: 현재 영상의 카메라를 선택하고 작품 시작

  Q / Esc: 종료
외장 카메라 렌즈를 잠깐 가려서 해당 영상인지 확인한 후 Enter를 누르세요.
카메라 번호는 내장/외장 구분이 아니며 실행할 때마다 영상으로 확인합니다.
"""
import cv2
import mediapipe as mp
import numpy as np
import time
import subprocess
from functools import lru_cache
import os
from PIL import ImageFont, ImageDraw, Image
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
model_path = os.path.expanduser('~/Desktop/face_landmarker.task')
options = FaceLandmarkerOptions(

    base_options=BaseOptions(model_asset_path=model_path),

    running_mode=VisionRunningMode.VIDEO,

    output_face_blendshapes=True
)
missions = [

    {

        "name": "DOPAMINE SHOT", 

        "guide": "볼 광대를 45도 올리고 눈꼬리를 부드럽게 접어 활짝 웃으세요!", 

        "target": 0.55

    },

    {

        "name": "SOCIAL DISTANCE", 

        "guide": "입술 힘을 빼고, 입꼬리 끝만 가볍게 올리는 깔끔한 비즈니스 미소.", 

        "target": 0.2

    },

    {

        "name": "CYNICAL BITTER", 

        "guide": "양쪽 균형을 깨고, 한쪽 입꼬리만 픽 올려 비웃어보세요.", 

        "target": 0.22

    },

    {

        "name": "TRUST VITAMIN", 

        "guide": "입술 주변 힘을 풀고 좌우 균형 있게 정중한 미소를 만드세요.", 

        "target": 0.4

    }
]
@lru_cache(maxsize=64)
def get_valid_font(size, use_heavy=False):

    font_paths = [

        "/System/Library/Fonts/AppFonts/AppleSDGothicNeo.ttc",

        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",

        "/System/Library/Fonts/STHeiti Light.ttc",

        "/Library/Fonts/Arial Unicode.ttf"

    ]



    for path in font_paths:

        if os.path.exists(path):

            try:

                if "AppleSDGothicNeo" in path and use_heavy:

                    try:

                        return ImageFont.truetype(path, size, index=1)

                    except Exception:

                        return ImageFont.truetype(path, size)

                return ImageFont.truetype(path, size)

            except Exception:

                continue

    return ImageFont.load_default()
CAMERA_SCAN_LIMIT = 8
CAMERA_FRAME_TIMEOUT = 5.0
def read_camera_frame(cap):

    """프레임이 없거나 카메라 연결이 끊어지면 None을 반환합니다."""

    try:

        ok, frame = cap.read()

    except cv2.error:

        return None

    if not ok or frame is None or frame.size == 0:

        return None

    return frame
def camera_choice_image(frame, index):

    """카메라 자체 영상과 선택 안내를 표시합니다."""

    canvas = np.full((650, 960, 3), (18, 18, 20), dtype=np.uint8)

    if frame is not None:

        canvas[110:650] = letterbox_frame(cv2.flip(frame, 1), 960, 540)

        title = f"CAMERA {index}  |  {frame.shape[1]} x {frame.shape[0]}"

    else:

        title = f"CAMERA {index}  |  WAITING FOR VIDEO..."

    cv2.putText(canvas, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,

                0.75, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(canvas, "ENTER: use this camera   N / SPACE: next   Q / ESC: quit",

                (20, 62), cv2.FONT_HERSHEY_SIMPLEX,

                0.60, (0, 220, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Cover the external camera lens briefly to verify this view.",

                (20, 91), cv2.FONT_HERSHEY_SIMPLEX,

                0.58, (190, 190, 190), 1, cv2.LINE_AA)

    return canvas
def find_working_camera():

    """OpenCV의 실제 영상을 확인한 카메라만 선택합니다.



    system_profiler 출력 순서를 OpenCV 인덱스로 변환하지 않습니다.

    인덱스 0도 외장일 수 있으므로 모든 후보를 같은 방식으로 확인합니다.

    Enter를 누르기 전에는 카메라를 확정하지 않습니다.

    """

    window_name = 'CAMERA SELECT'

    print("\n[카메라 선택] 외장 카메라를 맥북에 연결하고 다른 촬영 앱을 종료하세요.")

    print("미리보기 창을 클릭하고 키보드를 영문으로 전환하세요.")

    print("N / Space: 다음 카메라 | Enter: 현재 카메라 선택 | Q / Esc: 종료")

    print("외장 카메라 렌즈를 잠깐 가려 영상이 가려지는지 확인하세요.")



    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(window_name, 960, 650)

    try:

        for index in range(CAMERA_SCAN_LIMIT):

            print(f"[확인] OpenCV 카메라 번호 {index}")

            cap = None

            selected = False

            try:

                try:

                    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)

                except cv2.error as exc:

                    print(f"[건너뜀] {index}번 카메라 열기 실패: {exc}")

                    continue

                if not cap.isOpened():

                    continue



                # MJPG/해상도/FPS를 강제하지 않고 카메라 기본 모드로 먼저 확인합니다.

                # 본 화면은 기존 letterbox_frame으로 1920x1080에 맞춥니다.

                last_frame_at = time.monotonic()

                while True:

                    frame = read_camera_frame(cap)

                    now = time.monotonic()

                    if frame is not None:

                        last_frame_at = now

                    elif now - last_frame_at >= CAMERA_FRAME_TIMEOUT:

                        print(f"[건너뜀] {index}번 카메라에서 영상을 받지 못했습니다.")

                        break



                    cv2.imshow(window_name, camera_choice_image(frame, index))

                    key = cv2.waitKey(20) & 0xFF

                    try:

                        closed = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) == 0

                    except cv2.error:

                        closed = False

                    if key in (27, ord('q'), ord('Q')) or closed:

                        print("[종료] 카메라 선택을 취소했습니다.")

                        return None

                    if key in (ord('n'), ord('N'), 32):

                        break

                    if key in (10, 13) and frame is not None:

                        selected = True

                        print(f"[선택 완료] {index}번 카메라로 시작합니다.")

                        # 확인한 연결을 그대로 반환하여 다시 열 때의 번호 변경을 피합니다.

                        return cap

            finally:

                if cap is not None and not selected:

                    cap.release()

    finally:

        try:

            cv2.destroyWindow(window_name)

        except cv2.error:

            pass



    print("[중단] 선택한 카메라가 없습니다. 자동으로 다른 카메라를 선택하지 않습니다.")

    print("외장 카메라를 맥북에 직접 연결한 뒤 QuickTime의 '새로운 동영상 녹화'에서 확인하세요.")

    print("QuickTime에서 외장 영상이 보이면 QuickTime을 종료하고 이 프로그램을 다시 실행하세요.")

    return None
def draw_text(img, text, pos, size, color, use_heavy=False, center_x_in_width=None):
    """Draw only the affected text rectangle instead of copying a full HD frame."""
    if not text:
        return img
    if not img.flags.writeable:
        img = img.copy()
    font = get_valid_font(size, use_heavy=use_heavy)
    x, y = pos
    bbox = font.getbbox(text)
    if center_x_in_width is not None:
        x = (center_x_in_width - (bbox[2] - bbox[0])) // 2
    pad = 5
    x1 = max(0, x + bbox[0] - pad)
    y1 = max(0, y + bbox[1] - pad)
    x2 = min(img.shape[1], x + bbox[2] + pad + 2)
    y2 = min(img.shape[0], y + bbox[3] + pad + 2)
    if x2 <= x1 or y2 <= y1:
        return img
    patch = Image.fromarray(img[y1:y2, x1:x2].copy())
    draw = ImageDraw.Draw(patch)
    px, py = x - x1, y - y1
    if use_heavy:
        for ox, oy in ((1,0),(0,1),(1,1)):
            draw.text((px+ox,py+oy), text, font=font, fill=color)
    draw.text((px,py), text, font=font, fill=color)
    img[y1:y2, x1:x2] = np.asarray(patch)
    return img

def draw_bold_start_title(img, text, y=310, size=138):
    """Render the opening title in moderately bold black, centered."""
    path = "/System/Library/Fonts/AppFonts/AppleSDGothicNeo.ttc"
    font = None
    if os.path.exists(path):
        candidates = []
        for index in range(20):
            try:
                candidate = ImageFont.truetype(path, size, index=index)
                family, style = candidate.getname()
                if "Bold" in style and "Heavy" not in style and "Black" not in style:
                    candidates.append((0, candidate))
                elif "Heavy" in style or "Black" in style:
                    candidates.append((1, candidate))
            except (OSError, IndexError, ValueError):
                break
        if candidates:
            font = sorted(candidates, key=lambda item: item[0])[0][1]
    if font is None:
        font = get_valid_font(size, use_heavy=True)
    image = Image.fromarray(img)
    painter = ImageDraw.Draw(image)
    stroke = 4
    bbox = painter.textbbox((0, 0), text, font=font, stroke_width=stroke)
    text_width = bbox[2] - bbox[0]
    x = (img.shape[1] - text_width) // 2 - bbox[0]
    painter.text((x, y), text, font=font, fill=(0, 0, 0),
                 stroke_width=stroke, stroke_fill=(0, 0, 0))
    return np.array(image, copy=True)


def draw_round_rectangle(img, pt1, pt2, color, radius=30, thickness=-1):

    img_pil = Image.fromarray(img)

    draw = ImageDraw.Draw(img_pil)

    if thickness == -1:

        draw.rounded_rectangle([pt1[0], pt1[1], pt2[0], pt2[1]], radius=radius, fill=color)

    else:

        draw.rounded_rectangle([pt1[0], pt1[1], pt2[0], pt2[1]], radius=radius, outline=color, width=thickness)

    return np.array(img_pil)
def draw_tech_frame(img, w, h, color):

    pad = 50

    length = 35

    cv2.line(img, (pad, pad), (pad + length, pad), color, 3, cv2.LINE_AA)

    cv2.line(img, (pad, pad), (pad, pad + length), color, 3, cv2.LINE_AA)

    cv2.line(img, (w - pad, pad), (w - pad - length, pad), color, 3, cv2.LINE_AA)

    cv2.line(img, (w - pad, pad), (w - pad, pad + length), color, 3, cv2.LINE_AA)

    cv2.line(img, (pad, h - pad), (pad + length, h - pad), color, 3, cv2.LINE_AA)

    cv2.line(img, (pad, h - pad), (pad, h - pad - length), color, 3, cv2.LINE_AA)

    cv2.line(img, (w - pad, h - pad), (w - pad - length, h - pad), color, 3, cv2.LINE_AA)

    cv2.line(img, (w - pad, h - pad), (w - pad, h - pad - length), color, 3, cv2.LINE_AA)
def draw_gauge(img, value, target, pos, size=(480, 28)):

    x, y = pos

    w, h = size

    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1, cv2.LINE_AA)

    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), 1, cv2.LINE_AA)

    fill_w = int(w * min(value, 1.0))

    color = (0, 255, 255) if value >= target else (255, 0, 180)

    if fill_w > 4:

        cv2.rectangle(img, (x + 2, y + 2), (x + fill_w - 2, y + h - 2), color, -1, cv2.LINE_AA)

    target_x = int(x + w * target)

    cv2.line(img, (target_x, y - 6), (target_x, y + h + 6), (255, 255, 255), 2, cv2.LINE_AA)
def get_easy_realtime_hint(stage_idx, shapes):

    l_s = shapes.get('mouthSmileLeft', 0)

    r_s = shapes.get('mouthSmileRight', 0)

    s_v = (l_s + r_s) / 2

    eye_squint = (shapes.get('eyeSquintLeft', 0) + shapes.get('eyeSquintRight', 0)) / 2

    mouth_pucker = shapes.get('mouthPucker', 0)

    asymmetry = abs(l_s - r_s)



    if stage_idx == 0:

        if s_v < 0.55:

            return "[볼 광대 UP] 입꼬리를 귀 쪽 방향으로 쓱 끌어올리세요!"

        elif eye_squint < 0.2:

            return "[눈꼬리 접기] 눈가에 힘을 살짝 빼고 눈웃음을 만드세요!"

        return ""



    elif stage_idx == 1:

        if s_v < 0.05:

            return "[입꼬리 슬쩍] 미소를 살짝 지어보세요."

        elif s_v > 0.4:

            return "[과도함 주의] 입을 너무 크게 벌리지 말고 가볍게 올리세요."

        elif mouth_pucker > 0.3:

            return "[입술 힘 빼기] 입술 오므림을 풀고 입꼬리 끝만 톡 올리세요."

        return ""



    elif stage_idx == 2:

        if asymmetry < 0.22:

            return "[한쪽 입꼬리] 한쪽 입꼬리에만 픽 힘을 주고 당겨보세요."

        return ""



    elif stage_idx == 3:

        if s_v < 0.2:

            return "[미소 형성] 입꼬리를 조금 더 끌어올리세요."

        elif s_v > 0.5:

            return "[과도함 주의] 너무 크게 웃지 말고 정중하게 조절하세요."

        elif asymmetry > 0.15:

            return "[좌우 균형] 한쪽으로 쏠리지 않게 양쪽 입꼬리를 균일하게!"

        elif mouth_pucker > 0.25:

            return "[입술 완화] 입술 자체 힘을 빼고 편안하게 당기세요."

        return ""



    return ""
def get_lab_report(idx, score, extra):

    if idx == 0:

        return "- 행복 수치 정상\n완벽한 도파민 제어 및 분출 능력입니다." if score > 0.55 else "- 행복 결핍 상태\n뇌 활성화를 위해 일일 미소 루틴을 강제합니다."

    if idx == 1:

        return "- 사회적 가면 적합\n비즈니스 환경에 가장 최적화된 생존형 미소입니다." if 0.05 <= score <= 0.4 else "- 감정 왜곡 실패.\n포커페이스 훈련을 권장합니다."

    if idx == 2:

        return "- 치명적 쓴맛\n냉소와 매력을 동시에 보여주는 고난도 연출입니다." if extra > 0.22 else "- 근육 대칭 오류.\n한쪽 안면만 독립적으로 조절하는 연습이 필요합니다."

    if idx == 3:

        return "- 면접 프리패스.\n상대방의 신뢰를 200% 갈취하는 고품격 미소입니다." if 0.2 <= score <= 0.5 else "- 근육 경직 발생\n안면 스트레칭 및 가면 유연성 강화를 권장합니다."
def check_pass(idx, score, extra):

    if idx == 0: return score > 0.55

    if idx == 1: return 0.05 <= score <= 0.4

    if idx == 2: return extra > 0.22

    if idx == 3: return 0.2 <= score <= 0.5
def calculate_total_score(results, extra_results):

    scores = []

    s0 = min(results[0] / 0.55, 1.0) if results[0] >= 0.55 else (results[0] / 0.55)

    scores.append(s0)

    dist = abs(results[1] - 0.2)

    s1 = max(0.0, 1.0 - (dist / 0.2))

    scores.append(s1)

    s2 = min(extra_results[2] / 0.22, 1.0) if extra_results[2] >= 0.22 else (extra_results[2] / 0.22)

    scores.append(s2)

    dist3 = abs(results[3] - 0.4)

    s3 = max(0.0, 1.0 - (dist3 / 0.4))

    scores.append(s3)

    return int(np.mean(scores) * 100)
def get_diagnosis_rank(score):

    if score >= 85: return "가면 조작 마스터 (LEVEL: MAX)"

    elif score >= 60: return "평범한 현대인 에디션 (LEVEL: 03)"

    elif score >= 35: return "고장난 감정 시뮬레이터 (LEVEL: 01)"

    else: return "감정 투명 인간 (LEVEL: 00)"
def apply_creepy_distortion(img, face_landmarks):
    """Restore the original strong four-point distortion using a bounded ROI.

    Original: full-screen 30% red wash, four independent 45px oscillations,
    and a soft convex-hull mask. Compute remapping only near the face.
    """
    if not face_landmarks:
        return img
    h, w = img.shape[:2]
    landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks], dtype=np.int32)
    # The original effect tinted the *entire* screen, not only the face.
    cv2.addWeighted(img, 0.7, (np.zeros_like(img) + (0, 0, 255)).astype(np.uint8), 0.3, 0, dst=img)
    # Preserve original 60px falloff, 45px maximum displacement and 51px mask blur.
    margin = 200
    x1 = max(0, int(landmarks[:, 0].min()) - margin)
    y1 = max(0, int(landmarks[:, 1].min()) - margin)
    x2 = min(w, int(landmarks[:, 0].max()) + margin)
    y2 = min(h, int(landmarks[:, 1].max()) + margin)
    if x2 <= x1 or y2 <= y1:
        return img
    roi = img[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    local_pts = landmarks - np.array([x1, y1], dtype=np.int32)
    mask = np.zeros((rh, rw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(local_pts), 255)
    mask = cv2.GaussianBlur(mask, (51, 51), 0).astype(np.float32) / 255.0
    yy, xx = np.indices((rh, rw), dtype=np.float32)
    bx = np.zeros((rh, rw), dtype=np.float32)
    by = np.zeros((rh, rw), dtype=np.float32)
    t = time.time()
    for pair in ([33, 133], [362, 263], [1, 2], [0, 17]):
        cx, cy = np.mean(local_pts[pair], axis=0)
        dx, dy = xx - cx, yy - cy
        falloff = np.exp(-(dx * dx + dy * dy) / (60.0 ** 2))
        bx += falloff * 45.0 * np.sin(dy / 8.0 + t * 10.0)
        by += falloff * 45.0 * np.cos(dx / 8.0 + t * 10.0)
    warped = cv2.remap(roi, xx + bx, yy + by, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    alpha = mask[:, :, None]
    roi[:] = np.clip(warped * alpha + roi * (1.0 - alpha), 0, 255).astype(np.uint8)
    return img

def letterbox_frame(frame, target_w=1920, target_h=1080, bg_color=(10, 10, 12)):

    orig_h, orig_w, _ = frame.shape

    scale = min(target_w / orig_w, target_h / orig_h)

    new_w = int(orig_w * scale)

    new_h = int(orig_h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    dx = (target_w - new_w) // 2

    dy = (target_h - new_h) // 2

    canvas[dy:dy+new_h, dx:dx+new_w] = resized

    return canvas
def crop_face_smart(pt_bgr, landmarks_list):

    if not landmarks_list or pt_bgr is None or pt_bgr.size == 0:

        return pt_bgr

    h, w, _ = pt_bgr.shape

    pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks_list])

    x_min, y_min = np.min(pts, axis=0)

    x_max, y_max = np.max(pts, axis=0)

    fw = x_max - x_min

    fh = y_max - y_min

    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2

    side = int(max(fw, fh) * 1.5)

    nx1 = max(0, cx - side // 2)

    ny1 = max(0, cy - int(side * 0.55))

    nx2 = min(w, nx1 + side)

    ny2 = min(h, ny1 + side)

    return pt_bgr[ny1:ny2, nx1:nx2]
def generate_print_card(name, photos, results, extra_results, total_score, landmark_history):

    try:

        card = np.zeros((1480, 1000, 3), dtype=np.uint8)

        card = draw_round_rectangle(card, (50, 50), (950, 165), (255, 255, 255), radius=45)

        score_str = f"{total_score}P"

        cv2.circle(card, (890, 107), 42, (20, 20, 20), -1)

        img_pil = Image.fromarray(card)

        draw = ImageDraw.Draw(img_pil)

        font_p = get_valid_font(28, use_heavy=True)

        bbox_p = font_p.getbbox(score_str)

        pw_s = bbox_p[2] - bbox_p[0]

        ph_s = bbox_p[3] - bbox_p[1]

        px = 890 - (pw_s // 2)

        py = 107 - (ph_s // 2) - 4

        draw.text((px+1, py), score_str, font=font_p, fill=(255, 255, 255))

        draw.text((px, py), score_str, font=font_p, fill=(255, 255, 255))

        card = np.array(img_pil)

        card = draw_text(card, name.upper(), (95, 84), 42, (0, 0, 0), use_heavy=True)

        positions = [(50, 215), (510, 215), (50, 565), (510, 565)]

        pw, ph = 440, 315

        pink = (180, 0, 255)

        for idx, pt in enumerate(photos):

            if idx >= len(positions): break

            x, y = positions[idx]

            lms = landmark_history[idx] if idx < len(landmark_history) else None

            cropped_pt = crop_face_smart(pt, lms)

            if cropped_pt is None or cropped_pt.size == 0:

                cropped_pt = pt

            ph_h, ph_w, _ = cropped_pt.shape

            scale = min(pw / ph_w, ph / ph_h)

            nw, nh = int(ph_w * scale), int(ph_h * scale)

            rsz = cv2.resize(cropped_pt, (nw, nh), interpolation=cv2.INTER_CUBIC)

            cx = x + (pw - nw) // 2

            cy = y + (ph - nh) // 2

            card = draw_round_rectangle(card, (x, y), (x+pw, y+ph), (37, 35, 35), radius=25)

            mask = np.zeros((ph, pw), dtype=np.uint8)

            mask_pil = Image.fromarray(mask)

            m_draw = ImageDraw.Draw(mask_pil)

            m_draw.rounded_rectangle([0, 0, pw, ph], radius=25, fill=255)

            mask = np.array(mask_pil)

            sub_canvas = card[y:y+ph, x:x+pw].copy()

            sub_canvas[cy-y:cy-y+nh, cx-x:cx-x+nw] = rsz

            idx_y, idx_x = np.where(mask == 255)

            card[y + idx_y, x + idx_x] = sub_canvas[idx_y, idx_x]

            cl = 24

            cv2.line(card, (x, y+2), (x+cl, y+2), pink, 3, cv2.LINE_AA)

            cv2.line(card, (x+2, y), (x+2, y+cl), pink, 3, cv2.LINE_AA)

            cv2.line(card, (x+pw, y+2), (x+pw-cl, y+2), pink, 3, cv2.LINE_AA)

            cv2.line(card, (x+pw-2, y), (x+pw-2, y+cl), pink, 3, cv2.LINE_AA)

            cv2.line(card, (x, y+ph-2), (x+cl, y+ph-2), pink, 3, cv2.LINE_AA)

            cv2.line(card, (x+2, y+ph), (x+2, y+ph-cl), pink, 3, cv2.LINE_AA)

            cv2.line(card, (x+pw, y+ph-2), (x+pw-cl, y+ph-2), pink, 3, cv2.LINE_AA)

            cv2.line(card, (x+pw-2, y+ph), (x+pw-2, y+ph-cl), pink, 3, cv2.LINE_AA)



        card = draw_round_rectangle(card, (50, 925), (950, 1430), (255, 255, 255), radius=35)

        cv2.line(card, (70, 1175), (930, 1175), (228, 225, 225), 2, cv2.LINE_AA)

        cv2.line(card, (500, 955), (500, 1400), (228, 225, 225), 2, cv2.LINE_AA)

        text_slots = [((80, 955), 0), ((540, 955), 1), ((80, 1205), 2), ((540, 1205), 3)]

        font_m = get_valid_font(25, use_heavy=True)

        font_s_wrap = get_valid_font(20)

        for pos_n, idx in text_slots:

            card = draw_text(card, missions[idx]['name'], pos_n, 25, (30, 30, 30), use_heavy=True)

            bbox_m = font_m.getbbox(missions[idx]['name'])

            mission_w = bbox_m[2] - bbox_m[0]

            status_x = pos_n[0] + mission_w + 22

            is_pass = check_pass(idx, results[idx], extra_results[idx])

            status_txt = "[PASSED]" if is_pass else "[FAILED]"

            status_color = (255, 160, 0) if is_pass else (140, 0, 255)

            card = draw_text(card, status_txt, (status_x, pos_n[1]), 21, status_color, use_heavy=True)

            raw_report = get_lab_report(idx, results[idx], extra_results[idx])

            lines = raw_report.split('\n')

            current_y = pos_n[1] + 52

            max_width = 390

            for l_idx, ln in enumerate(lines):

                if l_idx == 0:

                    card = draw_text(card, ln, (pos_n[0], current_y), 20, (50, 50, 50))

                    current_y += 35

                else:

                    wrapped_line = ""

                    for word in list(ln):

                        test_line = wrapped_line + word

                        bbox_w = font_s_wrap.getbbox(test_line)

                        if (bbox_w[2] - bbox_w[0]) > max_width:

                            card = draw_text(card, wrapped_line, (pos_n[0], current_y), 19, (100, 100, 100))

                            current_y += 30

                            wrapped_line = word

                        else:

                            wrapped_line = test_line

                    if wrapped_line:

                        card = draw_text(card, wrapped_line, (pos_n[0], current_y), 19, (100, 100, 100))

                        current_y += 30

        clean_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c==' ']).rstrip()

        filename = os.path.expanduser(f"~/Desktop/{clean_name.replace(' ', '_')}_report.png")

        cv2.imwrite(filename, card)

        return card, filename

    except Exception as e:

        fallback_card = np.zeros((1480, 1000, 3), dtype=np.uint8)

        return fallback_card, os.path.expanduser("~/Desktop/fallback_report.png")
# Mission-screen-based walkthrough. Live camera remains visible; the mission timer
# does not start until the visitor finishes the walkthrough.
TUTORIAL_PAGES = ("FACE", "POINT", "TIMER", "ANALYSIS")

def draw_mission_top_ui(canvas, stage_idx, gauge_value=0.0, passed=False):
    """Three floating rounded HUD pills over the full camera image."""
    pink, white, black = (237, 0, 210), (255, 255, 255), (0, 0, 0)
    # Three independent dark translucent pills; the camera remains visible behind them.
    boxes = [(35, 30, 500, 165), (520, 30, 1375, 165), (1395, 30, 1885, 165)]
    for x1,y1,x2,y2 in boxes:
        region = canvas[y1:y2,x1:x2]
        overlay = np.full_like(region,(16,16,18))
        mask = np.zeros((y2-y1,x2-x1),np.uint8)
        cv2.rectangle(mask,(44,0),(x2-x1-44,y2-y1),255,-1)
        cv2.rectangle(mask,(0,44),(x2-x1,y2-y1-44),255,-1)
        for cx in (44,x2-x1-44):
            for cy in (44,y2-y1-44):
                cv2.circle(mask,(cx,cy),44,255,-1)
        blended = cv2.addWeighted(overlay,.83,region,.17,0)
        region[mask>0] = blended[mask>0]
    canvas = draw_text(canvas, f"CHAPTER {stage_idx+1:02d}.", (66, 47), 27, pink, use_heavy=True)
    canvas = draw_text(canvas, missions[stage_idx]['name'], (66, 88), 40, white, use_heavy=True)
    canvas = draw_text(canvas, "POINT", (551, 45), 25, pink, use_heavy=True)
    guide = missions[stage_idx]['guide']
    font_size = 25
    while get_valid_font(font_size).getbbox(guide)[2] > 795 and font_size > 18:
        font_size -= 1
    canvas = draw_text(canvas, guide, (551, 93), font_size, white)
    canvas = draw_text(canvas, "LIVE ANALYSIS", (1425, 45), 25, pink, use_heavy=True)
    gx, gy, gw, gh = 1427, 105, 420, 22
    canvas = draw_round_rectangle(canvas, (gx,gy), (gx+gw,gy+gh), (90,90,90), radius=11)
    fill_w = int(gw * max(0.0, min(1.0, gauge_value)))
    if fill_w > 0:
        canvas = draw_round_rectangle(canvas, (gx,gy), (gx+fill_w,gy+gh), black if passed else pink, radius=min(11,max(1,fill_w//2)))
    tx = gx + int(gw * missions[stage_idx]['target'])
    cv2.line(canvas, (tx,gy-6), (tx,gy+gh+6), white, 2, cv2.LINE_AA)
    return canvas


_tutorial_brightness_cache = {}
def draw_tutorial_screen(page_index, name, camera_frame, width=1920, height=1080):
    """Spotlight the exact same floating HUD used by the missions."""
    pink, white = (237, 0, 210), (255,255,255)
    canvas = letterbox_frame(cv2.flip(camera_frame, 1), width, height)
    canvas = draw_mission_top_ui(canvas, 0, .38, False)
    # Timer is a separate camera overlay, not a fourth top HUD pill.
    cv2.circle(canvas, (1800, 305), 63, (0,0,0), -1, cv2.LINE_AA)
    canvas = draw_text(canvas, "10", (1765, 267), 54, white, use_heavy=True)
    targets = [(690,235,1230,815), (520,30,1375,165),
               (1718,225,1880,390), (1395,30,1885,165)]
    titles = ["01 / FACE DETECTION", "02 / MISSION GUIDE", "03 / TIME LIMIT", "04 / LIVE ANALYSIS"]
    subtitles = ["카메라를 정면으로 바라봐 주세요.", "POINT에서 표정 힌트를 확인하세요.",
                 "각 미션의 제한 시간은 10초입니다.", "게이지에서 표정 변화를 확인하세요."]
    notes = ["화면 중앙에 얼굴을 맞춰주세요.", "4가지 표정 미션이 순서대로 진행됩니다.",
             "각 챕터 안내 5초 후 미션이 자동 시작됩니다.", "미션이 끝나면 결과 카드가 출력됩니다."]
    positions = [(1250,460), (650,215), (1100,300), (1190,220)]
    x1,y1,x2,y2 = targets[page_index]
    cache_key = (page_index,width,height)
    if cache_key not in _tutorial_brightness_cache:
        mask = np.zeros((height,width),dtype=np.uint8)
        cv2.rectangle(mask,(x1,y1),(x2,y2),255,-1)
        feather = cv2.GaussianBlur(mask,(0,0),sigmaX=18,sigmaY=18)
        brightness = .12 + .88*(feather.astype(np.float32)/255.0)
        brightness[mask==255] = 1.0
        _tutorial_brightness_cache[cache_key] = np.uint8(brightness*255)
    brightness = _tutorial_brightness_cache[cache_key]
    canvas = cv2.multiply(canvas,cv2.merge((brightness,brightness,brightness)),scale=1/255.0)
    cv2.rectangle(canvas,(x1,y1),(x2,y2),pink,3,cv2.LINE_AA)
    ix,iy=positions[page_index]
    canvas=draw_text(canvas,titles[page_index],(ix,iy),29,pink,use_heavy=True)
    canvas=draw_text(canvas,subtitles[page_index],(ix,iy+49),29,white,use_heavy=True)
    canvas=draw_text(canvas,notes[page_index],(ix,iy+100),23,(215,215,215))
    footer = "[ENTER] 훈련 시작   [← / BACKSPACE] 이전" if page_index==3 else "[ENTER] 다음   [← / BACKSPACE] 이전"
    canvas=draw_text(canvas,f"{page_index+1:02d} / 04",(1660,975),23,(215,215,215))
    canvas=draw_text(canvas,footer,(1340,1025),19,(215,215,215))
    return canvas

def smoothstep(value):
    """Ease the fades in and out rather than changing opacity linearly."""
    t = max(0.0, min(1.0, value))
    return t * t * (3.0 - 2.0 * t)


def draw_welcome_screen(name, camera_frame, elapsed, width=1920, height=1080):
    """Welcome on the EXACT name-entry background, then a gentle crossfade."""
    # This matches the INPUT screen's NumPy/OpenCV BGR background exactly.
    pink = (210, 0, 237)
    base = np.full((height, width, 3), pink, dtype=np.uint8)

    # 0–0.8s text appears, 0.8–2.7s stays, 2.7–3.7s fades away.
    fade_in = smoothstep(elapsed / 0.8)
    fade_out = 1.0 - smoothstep((elapsed - 2.7) / 1.0)
    opacity = min(fade_in, fade_out)
    text_layer = base.copy()
    text_layer = draw_text(text_layer, "TRAIN YOUR SMILE", (0, 315), 38,
                           (0, 0, 0), use_heavy=True, center_x_in_width=width)
    greeting = f"{name.upper()}님, 웃음 훈련을"
    greeting_size = 78
    font = get_valid_font(greeting_size, use_heavy=True)
    while font.getbbox(greeting)[2] - font.getbbox(greeting)[0] > width - 160 and greeting_size > 35:
        greeting_size -= 3
        font = get_valid_font(greeting_size, use_heavy=True)
    text_layer = draw_text(text_layer, greeting, (0, 435), greeting_size,
                           (0, 0, 0), use_heavy=True, center_x_in_width=width)
    text_layer = draw_text(text_layer, "시작해 볼까요?", (0, 540), 78,
                           (0, 0, 0), use_heavy=True, center_x_in_width=width)
    text_layer = draw_text(text_layer, "잠시 후 사용 방법을 안내해 드릴게요.", (0, 715), 31,
                           (45, 45, 45), center_x_in_width=width)
    base = cv2.addWeighted(text_layer, opacity, base, 1.0 - opacity, 0)

    # After the words are gone, gently transition the pink screen into the camera.
    # 3.7–5.0s: no abrupt simultaneous text and camera movement.
    dissolve = smoothstep((elapsed - 3.7) / 1.3)
    if dissolve > 0:
        tutorial = draw_tutorial_screen(0, name, camera_frame, width, height)
        base = cv2.addWeighted(tutorial, dissolve, base, 1.0 - dissolve, 0)
    return base

# Full-screen chapter break; no face analysis or mission timer runs here.
MISSION_INTRO_LABELS = ("활짝 웃기", "비즈니스 미소", "한쪽 입꼬리로 비웃기", "정중한 미소")

def draw_mission_intro(stage_idx, width=1920, height=1080):
    canvas = np.full((height, width, 3), (210, 0, 237), dtype=np.uint8)
    white = (255, 255, 255)
    pink = (210, 0, 237)
    pale = (235, 155, 255)
    canvas = draw_text(canvas, "SMILE TRAIN LAB", (130, 80), 36, white, use_heavy=True)
    cv2.line(canvas, (475, 112), (1470, 112), white, 2, cv2.LINE_AA)
    canvas = draw_text(canvas, f"CHAPTER {stage_idx+1:02d} / 04", (1515, 83), 30, white)
    for cx, cy in ((72,62),(1848,62),(72,1018),(1848,1018)):
        cv2.line(canvas,(cx-20,cy),(cx+20,cy),white,1,cv2.LINE_AA)
        cv2.line(canvas,(cx,cy-20),(cx,cy+20),white,1,cv2.LINE_AA)
    canvas = draw_round_rectangle(canvas,(150,328),(366,382),white,radius=27)
    canvas = draw_text(canvas,f"MISSION {stage_idx+1:02d}.",(172,333),30,pink,use_heavy=True)
    name=missions[stage_idx]['name']
    size=92
    while get_valid_font(size,True).getbbox(name)[2]>1060 and size>30: size-=2
    canvas=draw_text(canvas,name,(146,410),size,white,use_heavy=True)
    labels=("활짝 웃으세요.","비즈니스 미소를 지으세요.","한쪽 입꼬리로 비웃으세요.","정중하게 미소 지으세요.")
    guides=("양쪽 입꼬리를 최대한 올려주세요.","입꼬리 끝만 가볍게 올려주세요.","한쪽 입꼬리만 살짝 올려주세요.","양쪽 입꼬리를 균형 있게 올려주세요.")
    canvas=draw_text(canvas,labels[stage_idx],(150,570),54,white)
    canvas=draw_text(canvas,guides[stage_idx],(150,650),34,white)
    canvas=draw_text(canvas,"MISSION STARTS IN",(0,280),28,white,center_x_in_width=3000)
    canvas=draw_text(canvas,"G E T   R E A D Y",(0,840),26,white,center_x_in_width=3000)
    return canvas

def draw_intro_countdown(canvas, remaining, progress):
    white=(255,255,255)
    cx,cy=1500,572
    for i in range(48):
        theta=np.deg2rad(-90+i*360/48)
        p1=(int(cx+206*np.cos(theta)),int(cy+206*np.sin(theta)))
        p2=(int(cx+228*np.cos(theta)),int(cy+228*np.sin(theta)))
        color=white if i/48<progress else (224,77,242)
        cv2.line(canvas,p1,p2,color,10,cv2.LINE_AA)
    number=str(remaining)
    font=get_valid_font(216,True)
    box=font.getbbox(number)
    canvas=draw_text(canvas,number,(cx-(box[2]+box[0])//2,cy-(box[3]+box[1])//2),216,white,use_heavy=True)
    return canvas

cap = find_working_camera()
if cap is None:

    exit()
SCREEN_W, SCREEN_H = 1920, 1080
app_mode, user_name, current_stage = "INPUT", "", 0
tutorial_page = 0
welcome_start = 0.0
results, extra_results, captured_photos, captured_landmarks = [], [], [], []
stage_start = 0
MISSION_TIME = 10
MISSION_INTRO_TIME = 5.0
intro_canvas = None
intro_start = 0.0
total_score = 0
diagnosis_rank = ""
card_image = None
card_filename = ""
result_sub_mode = "REPORT"
sub_mode_start = 0
cv2.namedWindow('SMILE TRAIN LAB', cv2.WINDOW_NORMAL)
cv2.resizeWindow('SMILE TRAIN LAB', SCREEN_W, SCREEN_H)
last_camera_frame_at = time.monotonic()
last_analysis_ts = 0
last_tink_at = 0.0
with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        loop_start = time.time()

        frame = read_camera_frame(cap)

        if frame is None:

            if cv2.waitKey(1) & 0xFF == 27:

                break

            if time.monotonic() - last_camera_frame_at >= CAMERA_FRAME_TIMEOUT:

                print("[중단] 선택한 카메라의 영상이 끊겼습니다. USB 연결을 확인하고 다시 실행하세요.")

                break

            time.sleep(0.01)

            continue

        last_camera_frame_at = time.monotonic()

        if app_mode == "INPUT":
            frame = np.full((SCREEN_H, SCREEN_W, 3), (210, 0, 237), dtype=np.uint8)
            frame = draw_bold_start_title(frame, "TRAIN YOUR SMILE", y=310, size=138)
            frame = draw_text(frame, "SYSTEM SYNIC ACCESS", (0, 465), 30, (0, 0, 0), center_x_in_width=SCREEN_W)
            frame = draw_text(frame, "SUBJECT ID.", (0, 630), 30, (0, 0, 0), center_x_in_width=SCREEN_W)
            frame = draw_round_rectangle(frame, (580, 690), (1340, 805), (0, 0, 0), radius=56, thickness=3)
            frame = draw_text(frame, user_name if user_name else "이름을 입력해주세요.", (0, 718), 45,
                              (255, 255, 255) if user_name else (110, 110, 110), center_x_in_width=SCREEN_W)
            frame = draw_text(frame, "영어로 입력해주세요.", (0, 826), 25, (95, 95, 95), center_x_in_width=SCREEN_W)
            frame = draw_text(frame, "PRESS [ENTER] TO START", (0, 991), 25, (75, 75, 75), center_x_in_width=SCREEN_W)
            cv2.imshow('SMILE TRAIN LAB', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            elif key == 13 and user_name:
                app_mode = "WELCOME"
                welcome_start = time.monotonic()
                tutorial_page = 0
                current_stage, results, extra_results, captured_photos, captured_landmarks = 0, [], [], [], []
                os.system('afplay /System/Library/Sounds/Glass.aiff &')
            elif key in [8, 127]: user_name = user_name[:-1]
            elif 32 <= key <= 126: user_name += chr(key)

        elif app_mode == "WELCOME":
            elapsed_welcome = time.monotonic() - welcome_start
            frame = draw_welcome_screen(user_name, frame, elapsed_welcome, SCREEN_W, SCREEN_H)
            cv2.imshow('SMILE TRAIN LAB', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if elapsed_welcome >= 5.0:
                app_mode = "TUTORIAL"
                tutorial_page = 0

        elif app_mode == "TUTORIAL":
            frame = draw_tutorial_screen(tutorial_page, user_name, frame, SCREEN_W, SCREEN_H)
            cv2.imshow('SMILE TRAIN LAB', frame)
            key = cv2.waitKeyEx(20)
            if key == 27:
                break
            elif key in (13, 10, 32, 2555904, 65363):  # Enter, Space, right arrow
                if tutorial_page < len(TUTORIAL_PAGES) - 1:
                    tutorial_page += 1
                else:
                    app_mode = "MISSION_INTRO"
                    intro_canvas = None
            elif key in (2424832, 65361, 81, 8, 127):  # left arrow (macOS/Linux) or Backspace
                tutorial_page = max(0, tutorial_page - 1)

        elif app_mode == "MISSION_INTRO":
            if intro_canvas is None:
                intro_canvas = draw_mission_intro(current_stage, SCREEN_W, SCREEN_H)
                intro_start = time.monotonic()
            elapsed_intro = time.monotonic() - intro_start
            remaining = max(1, int(np.ceil(MISSION_INTRO_TIME - elapsed_intro)))
            progress = max(0.0, min(1.0, 1.0 - elapsed_intro / MISSION_INTRO_TIME))
            intro_frame = draw_intro_countdown(intro_canvas.copy(), remaining, progress)
            cv2.imshow('SMILE TRAIN LAB', intro_frame)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                break
            if time.monotonic() - intro_start >= MISSION_INTRO_TIME:
                app_mode = "MISSION"
                stage_start = time.time()
                os.system('afplay /System/Library/Sounds/Glass.aiff &')

        elif app_mode == "MISSION":

            frame = cv2.flip(frame, 1)

            raw_cam_frame = frame.copy()

            frame = letterbox_frame(frame, SCREEN_W, SCREEN_H)

            h, w, _ = frame.shape

            analysis_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            analysis_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(analysis_frame))
            analysis_ts = max(int(time.monotonic()*1000), last_analysis_ts+1)
            last_analysis_ts = analysis_ts
            result = landmarker.detect_for_video(mp_image, analysis_ts)

            s_v, l_s, r_s, face_lms = 0, 0, 0, None

            shapes = {}

            if result.face_blendshapes:

                shapes = {s.category_name: s.score for s in result.face_blendshapes[0]}

                l_s, r_s = shapes.get('mouthSmileLeft', 0), shapes.get('mouthSmileRight', 0)

                s_v = (l_s + r_s) / 2

                face_lms = result.face_landmarks[0]

            elapsed = time.time() - stage_start

            rem = max(0, MISSION_TIME - elapsed)

            cur = missions[current_stage]

            if s_v < cur['target'] and face_lms:

                frame = apply_creepy_distortion(frame, face_lms)

                if time.monotonic() - last_tink_at >= 1.5:
                    last_tink_at = time.monotonic()
                    subprocess.Popen(['afplay','/System/Library/Sounds/Tink.aiff'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Full-screen camera, with three floating rounded HUD sections.
            asymmetry_now = abs(l_s - r_s)
            gauge_value = asymmetry_now if current_stage == 2 else s_v
            mission_passed_now = bool(face_lms) and check_pass(current_stage, s_v, asymmetry_now)
            frame = draw_mission_top_ui(frame, current_stage, gauge_value, mission_passed_now)
            # Timer over camera, top right
            cv2.circle(frame, (1800, 305), 63, (0, 0, 0), -1, cv2.LINE_AA)
            # Center the timer text precisely using its rendered bounding box.
            timer_pil = Image.fromarray(frame)
            timer_draw = ImageDraw.Draw(timer_pil)
            timer_font = get_valid_font(55, use_heavy=True)
            timer_text = str(int(np.ceil(rem)))
            timer_box = timer_draw.textbbox((0, 0), timer_text, font=timer_font)
            timer_x = 1800 - (timer_box[0] + timer_box[2]) / 2
            timer_y = 305 - (timer_box[1] + timer_box[3]) / 2
            timer_draw.text((round(timer_x), round(timer_y)), timer_text,
                            font=timer_font, fill=(255, 255, 255))
            frame = np.array(timer_pil)
            easy_hint = get_easy_realtime_hint(current_stage, shapes) if face_lms else "얼굴을 화면 중앙에 맞추세요."
            if easy_hint:
                # PIL uses RGB: (0, 0, 255) is blue. Render at 2x resolution
                # so a 1px stroke at 2x becomes a 0.5px white outline.
                hint_pil = Image.fromarray(frame)
                hint_font = get_valid_font(38, use_heavy=True)
                measure = ImageDraw.Draw(hint_pil)
                hint_bbox = measure.textbbox((0, 0), easy_hint, font=hint_font)
                hint_width = hint_bbox[2] - hint_bbox[0]
                hint_x = (w - hint_width) // 2
                hint_y = h // 2 - 25
                scale = 2
                # Draw only the hint's local bounding region at high resolution.
                pad = 4
                left = max(0, hint_x - pad)
                top = max(0, hint_y + hint_bbox[1] - pad)
                right = min(w, hint_x + hint_bbox[2] + pad)
                bottom = min(h, hint_y + hint_bbox[3] + pad)
                if right > left and bottom > top:
                    region = hint_pil.crop((left, top, right, bottom))
                    region_hi = region.resize((region.width * scale, region.height * scale), Image.Resampling.BILINEAR)
                    draw_hi = ImageDraw.Draw(region_hi)
                    font_hi = get_valid_font(38 * scale, use_heavy=True)
                    draw_hi.text(((hint_x - left) * scale, (hint_y - top) * scale),
                                 easy_hint, font=font_hi, fill=(0, 0, 255),
                                 stroke_width=1, stroke_fill=(255, 255, 255))
                    region = region_hi.resize(region.size, Image.Resampling.BILINEAR)
                    hint_pil.paste(region, (left, top))
                frame = np.array(hint_pil)

            if rem <= 0:

                results.append(s_v)

                extra_results.append(abs(l_s - r_s))

                captured_photos.append(raw_cam_frame)

                captured_landmarks.append(face_lms)

                current_stage += 1

                if current_stage >= len(missions):

                    app_mode = "RESULT"

                    result_sub_mode = "REPORT"

                    sub_mode_start = time.time()

                    total_score = calculate_total_score(results, extra_results)

                    diagnosis_rank = get_diagnosis_rank(total_score)

                    os.system('afplay /System/Library/Sounds/Ping.aiff &')

                else:
                    app_mode = "MISSION_INTRO"
                    intro_canvas = None

            cv2.imshow('SMILE TRAIN LAB', frame)

            if cv2.waitKey(1) & 0xFF == 27: break



        elif app_mode == "RESULT":

            w, h = SCREEN_W, SCREEN_H

            canvas = np.full((h, w, 3), (10, 10, 12), dtype=np.uint8)

            if result_sub_mode == "REPORT":
                canvas[:] = (210, 0, 237)
                canvas = draw_text(canvas, f"TRAINING REPORT / SUBJECT : {user_name.upper()}", (64, 60), 58, (0, 0, 0), use_heavy=True)
                canvas = draw_text(canvas, f"TOTAL SCORE : {total_score} P  |  {diagnosis_rank}", (64, 160), 37, (255, 255, 255), use_heavy=True)
                for i, r in enumerate(results):
                    y_pos = 270 + i * 177
                    canvas = draw_round_rectangle(canvas, (50, y_pos), (1505, y_pos+160), (255, 255, 255), radius=28)
                    canvas = draw_round_rectangle(canvas, (50, y_pos), (1505, y_pos+160), (0, 0, 0), radius=28, thickness=3)
                    canvas = draw_text(canvas, f"CHAPTER {i+1:02d}. {missions[i]['name']}", (85, y_pos+24), 30, (125, 125, 125))
                    raw_rep = get_lab_report(i, r, extra_results[i]).replace('\n', ' ')
                    canvas = draw_text(canvas, raw_rep, (85, y_pos+88), 30, (0, 0, 0))
                    cv2.line(canvas, (1505, y_pos+80), (1630, y_pos+80), (0, 0, 0), 3)
                    status = "PASSED" if check_pass(i, r, extra_results[i]) else "FAILED"
                    canvas = draw_text(canvas, status, (1650, y_pos+57), 36, (0, 0, 0))
                elapsed = time.time() - sub_mode_start

                wait = int(10 - elapsed)

                canvas = draw_text(canvas, f"GENERATING CARD IN {max(0, wait)}S", (w-320, h-60), 20, (80, 80, 80))

                cv2.imshow('SMILE TRAIN LAB', canvas)

                if wait <= 0:

                    card_image, card_filename = generate_print_card(user_name, captured_photos, results, extra_results, total_score, captured_landmarks)

                    try:

                        os.system(f"lpr '{card_filename}' &")

                    except:

                        pass

                    result_sub_mode = "PREVIEW"

                    sub_mode_start = time.time()

                    os.system('afplay /System/Library/Sounds/Ping.aiff &')



            elif result_sub_mode == "PREVIEW":

                canvas[:] = (5, 5, 5)

                if card_image is not None and card_image.size > 0:

                    card_h, card_w, _ = card_image.shape

                    preview_scale = min(1800 / card_w, 980 / card_h)

                    nw, nh = int(card_w * preview_scale), int(card_h * preview_scale)

                    card_resized = cv2.resize(card_image, (nw, nh), interpolation=cv2.INTER_LINEAR)

                    cx = (SCREEN_W - nw) // 2

                    cy = (SCREEN_H - nh) // 2

                    canvas[cy:cy+nh, cx:cx+nw] = card_resized

                elapsed = time.time() - sub_mode_start

                wait = int(8 - elapsed)

                canvas = draw_text(canvas, "SENDING TO CANON SELPHY CP1500 프린터 전송 중...", (50, h-55), 22, (180, 0, 255))

                canvas = draw_text(canvas, f"SYSTEM REBOOT IN {max(0, wait)}S", (w-300, h-55), 20, (100, 100, 100))

                cv2.imshow('SMILE TRAIN LAB', canvas)

                if wait <= 0:

                    app_mode = "INPUT"

                    user_name = ""

            if cv2.waitKey(30) & 0xFF == 27: break

        time_diff = time.time() - loop_start

        if time_diff < 0.033:

            time.sleep(0.033 - time_diff)
cap.release()
cv2.destroyAllWindows() 
