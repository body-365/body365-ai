# rehab_analysis.py
# Body365 데이터 기반 재활 분석 시스템

from dataclasses import dataclass, field
from pathlib import Path
import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_KOREAN_FONT = r"C:\Windows\Fonts\malgun.ttf"  # 맑은 고딕

# ── 이미지 입출력 (X-ray 등) ─────────────────────────────────────────────────
def load_image_cv2(image_path: str | Path) -> np.ndarray:
    """
    .jpg / .png 파일을 OpenCV (BGR ndarray) 형식으로 읽어온다.
    한글 경로 호환을 위해 np.fromfile + cv2.imdecode 사용.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"지원하지 않는 이미지 형식입니다: {path.suffix} (지원: .jpg, .png)")

    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지를 디코딩할 수 없습니다: {path}")
    return img


def load_image_pil(image_path: str | Path) -> Image.Image:
    """.jpg / .png 파일을 Pillow Image 객체로 읽어온다."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"지원하지 않는 이미지 형식입니다: {path.suffix} (지원: .jpg, .png)")
    return Image.open(path)


def save_image_cv2(image: np.ndarray, output_path: str | Path) -> Path:
    """OpenCV ndarray를 .jpg/.png로 저장. 한글 경로 호환."""
    path = Path(output_path)
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"지원하지 않는 출력 형식입니다: {path.suffix} (지원: .jpg, .png)")
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"이미지 인코딩 실패: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))
    return path.resolve()


def annotate_image(
    image: str | Path | np.ndarray,
    target_xy: tuple[int, int],
    label: str,
    text_xy: tuple[int, int] | None = None,
    arrow_color: tuple[int, int, int] = (0, 0, 255),    # BGR (빨강)
    box_color: tuple[int, int, int] = (0, 0, 255),
    text_color: tuple[int, int, int] = (255, 255, 255),
    font_path: str | None = None,
    font_size: int = 24,
    box_padding: int = 8,
    arrow_thickness: int = 2,
) -> np.ndarray:
    """
    이미지의 target_xy 지점을 화살표로 가리키고 label 텍스트 박스를 그린다.

    - image       : 파일 경로 또는 BGR ndarray. ndarray 입력 시 원본은 변경하지 않고 복사본 사용.
    - target_xy   : 화살촉이 닿을 좌표 (x, y) — 예: 견봉 아래 공간 위치
    - label       : 박스 안에 표시할 한글/영문 텍스트
    - text_xy     : 텍스트 박스 좌상단 좌표. None이면 target_xy의 좌상단 ~80px 떨어진 곳에 자동 배치
    - 색상은 모두 BGR (OpenCV 컨벤션)

    반환: 주석이 그려진 BGR ndarray.
    """
    if isinstance(image, (str, Path)):
        img = load_image_cv2(image)
    else:
        img = image.copy()

    h, w = img.shape[:2]
    tx, ty = int(target_xy[0]), int(target_xy[1])

    resolved_font = font_path or (DEFAULT_KOREAN_FONT if Path(DEFAULT_KOREAN_FONT).exists() else None)
    try:
        font = ImageFont.truetype(resolved_font, font_size) if resolved_font else ImageFont.load_default()
    except OSError:
        font = ImageFont.load_default()

    # 텍스트 크기 측정
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = measure.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    box_w = text_w + box_padding * 2
    box_h = text_h + box_padding * 2

    # 박스 위치 자동 산출 (지정 안 됐을 때): target 좌상단 60px 오프셋, 화면 밖으로 안 나가게 클램프
    if text_xy is None:
        bx = max(10, min(tx - box_w - 60, w - box_w - 10))
        by = max(10, min(ty - box_h - 60, h - box_h - 10))
    else:
        bx, by = int(text_xy[0]), int(text_xy[1])

    # 화살표 시작점: 박스 모서리 중 target에 가장 가까운 코너
    box_cx = bx + box_w // 2
    box_cy = by + box_h // 2
    ax = bx + box_w if tx >= box_cx else bx
    ay = by + box_h if ty >= box_cy else by

    # 1) 화살표 (박스 아래에 깔리도록 먼저)
    cv2.arrowedLine(img, (ax, ay), (tx, ty), arrow_color, arrow_thickness, tipLength=0.15)

    # 2) 채워진 박스
    cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), box_color, thickness=-1)

    # 3) 한글 텍스트는 PIL로 (cv2.putText는 한글 미지원)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    text_rgb = (text_color[2], text_color[1], text_color[0])  # BGR → RGB
    draw.text(
        (bx + box_padding - bbox[0], by + box_padding - bbox[1]),
        label,
        fill=text_rgb,
        font=font,
    )
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


@dataclass
class Annotation:
    """다중 주석 입력 단위. target_xy, label은 필수, 나머지는 점별 스타일."""
    target_xy: tuple[int, int]
    label: str
    text_xy: tuple[int, int] | None = None
    arrow_color: tuple[int, int, int] = (0, 0, 255)
    box_color: tuple[int, int, int] = (0, 0, 255)
    text_color: tuple[int, int, int] = (255, 255, 255)
    font_size: int = 24


def _resolve_font(font_path: str | None, font_size: int):
    path = font_path or (DEFAULT_KOREAN_FONT if Path(DEFAULT_KOREAN_FONT).exists() else None)
    try:
        return ImageFont.truetype(path, font_size) if path else ImageFont.load_default()
    except OSError:
        return ImageFont.load_default()


def _measure_box(label: str, font, box_padding: int) -> tuple[int, int]:
    bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), label, font=font)
    return (bbox[2] - bbox[0] + box_padding * 2, bbox[3] - bbox[1] + box_padding * 2)


def _overlap_area(a: tuple, b: tuple) -> int:
    """a, b: (x1, y1, x2, y2). 교집합 면적."""
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih


def _point_in_rect(pt: tuple[int, int], rect: tuple, padding: int = 4) -> bool:
    px, py = pt
    x1, y1, x2, y2 = rect
    return (x1 - padding) <= px <= (x2 + padding) and (y1 - padding) <= py <= (y2 + padding)


def _find_box_position(
    target_xy: tuple[int, int],
    box_w: int,
    box_h: int,
    image_w: int,
    image_h: int,
    placed_rects: list,
    other_targets: list,
    offset: int = 60,
) -> tuple[int, int]:
    """
    target 주변 8방향 후보 중 다른 박스/타깃과 충돌이 가장 적은 위치를 반환.
    완벽 후보(점수 0)를 만나면 즉시 반환.
    """
    tx, ty = target_xy
    candidates = [
        (tx - box_w - offset, ty - box_h - offset),  # ↖
        (tx - box_w // 2,     ty - box_h - offset),  # ↑
        (tx + offset,         ty - box_h - offset),  # ↗
        (tx - box_w - offset, ty - box_h // 2),      # ←
        (tx + offset,         ty - box_h // 2),      # →
        (tx - box_w - offset, ty + offset),          # ↙
        (tx - box_w // 2,     ty + offset),          # ↓
        (tx + offset,         ty + offset),          # ↘
    ]

    best_pos = None
    best_score = float("inf")
    penalty = box_w * box_h  # 타깃 점이 박스 안에 들어올 때 큰 페널티

    for cx, cy in candidates:
        cx = max(10, min(cx, image_w - box_w - 10))
        cy = max(10, min(cy, image_h - box_h - 10))
        rect = (cx, cy, cx + box_w, cy + box_h)

        score = 0
        for prect in placed_rects:
            score += _overlap_area(rect, prect)
        for pt in other_targets:
            if _point_in_rect(pt, rect):
                score += penalty

        if score < best_score:
            best_score = score
            best_pos = (cx, cy)
            if score == 0:
                break

    return best_pos


def annotate_image_multi(
    image: str | Path | np.ndarray,
    annotations: list,
    font_path: str | None = None,
    box_padding: int = 8,
    arrow_thickness: int = 2,
) -> np.ndarray:
    """
    여러 좌표에 한 번에 화살표 + 텍스트 박스 주석을 단다.
    text_xy가 None인 항목은 8방향 후보를 평가해 기존 박스/타깃과 충돌이 가장 적은 위치에 자동 배치한다.

    - annotations : Annotation 객체 또는 dict의 리스트.
        예) [
            {"target_xy": (520, 320), "label": "충돌증후군 위험 구역"},
            Annotation(target_xy=(220, 450), label="극상근 부착부",
                       arrow_color=(0, 255, 255), box_color=(0, 128, 255)),
        ]
    - font_path / box_padding / arrow_thickness : 모든 주석에 공통 적용되는 값.

    반환: 모든 주석이 누적 적용된 BGR ndarray.
    """
    if isinstance(image, (str, Path)):
        out = load_image_cv2(image)
    else:
        out = image.copy()

    h, w = out.shape[:2]

    normalized: list[Annotation] = []
    for i, ann in enumerate(annotations):
        if isinstance(ann, dict):
            ann = Annotation(**ann)
        elif not isinstance(ann, Annotation):
            raise TypeError(
                f"annotations[{i}]는 Annotation 또는 dict여야 합니다 (받은 타입: {type(ann).__name__})"
            )
        normalized.append(ann)

    target_points = [a.target_xy for a in normalized]

    # 1단계: 모든 박스 위치를 먼저 결정 (충돌 회피)
    placed_rects: list[tuple] = []
    resolved_xy: list[tuple[int, int]] = []
    for idx, ann in enumerate(normalized):
        font = _resolve_font(font_path, ann.font_size)
        box_w, box_h = _measure_box(ann.label, font, box_padding)

        if ann.text_xy is not None:
            bx, by = int(ann.text_xy[0]), int(ann.text_xy[1])
        else:
            others = target_points[:idx] + target_points[idx + 1:]
            bx, by = _find_box_position(ann.target_xy, box_w, box_h, w, h, placed_rects, others)

        placed_rects.append((bx, by, bx + box_w, by + box_h))
        resolved_xy.append((bx, by))

    # 2단계: 결정된 위치로 실제 그리기
    for ann, xy in zip(normalized, resolved_xy):
        out = annotate_image(
            out,
            target_xy=ann.target_xy,
            label=ann.label,
            text_xy=xy,
            arrow_color=ann.arrow_color,
            box_color=ann.box_color,
            text_color=ann.text_color,
            font_path=font_path,
            font_size=ann.font_size,
            box_padding=box_padding,
            arrow_thickness=arrow_thickness,
        )
    return out


def calibrate_mm_per_pixel(
    p1: tuple[int, int],
    p2: tuple[int, int],
    known_mm: float,
) -> float:
    """
    이미지 내 두 픽셀 좌표 사이의 실제 길이(mm)를 알 때 픽셀당 mm 환산 계수 산출.
    예) X-ray에 1cm 기준 마커를 찍어둔 경우, 마커 양 끝점의 픽셀 좌표와 known_mm=10 입력.
    """
    if known_mm <= 0:
        raise ValueError("known_mm은 양수여야 합니다")
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    px_dist = (dx * dx + dy * dy) ** 0.5
    if px_dist < 1e-6:
        raise ValueError("p1, p2가 동일 좌표입니다")
    return known_mm / px_dist


def measure_joint_space(
    image: str | Path | np.ndarray,
    roi: tuple[int, int, int, int],
    mm_per_pixel: float,
    edge_orientation: str = "horizontal",
    line_color: tuple[int, int, int] = (0, 255, 255),
    label_prefix: str = "관절 간격",
    font_path: str | None = None,
    font_size: int = 26,
) -> tuple[np.ndarray, float]:
    """
    ROI 내에서 두 뼈 경계선을 자동 탐지해 관절 간격(mm)을 측정하고 시각화한다.

    ⚠️  측정 보조 도구이며 진단용이 아닙니다.
        결과는 ROI 설정/영상 품질에 민감합니다. 임상 결정 전 반드시 시각 검수가 필요합니다.

    - roi              : (x, y, w, h) — 관절 부위 사각형
    - mm_per_pixel     : 픽셀→mm 환산 계수. calibrate_mm_per_pixel()로 사전 산출
    - edge_orientation : "horizontal" (관절선 가로, 예: 무릎 정면 X-ray)
                         "vertical"   (관절선 세로)

    반환: (주석이 그려진 BGR ndarray, 측정된 mm 값)
    """
    if mm_per_pixel <= 0:
        raise ValueError("mm_per_pixel은 양수여야 합니다")
    if edge_orientation not in ("horizontal", "vertical"):
        raise ValueError("edge_orientation은 'horizontal' 또는 'vertical'이어야 합니다")

    if isinstance(image, (str, Path)):
        img = load_image_cv2(image)
    else:
        img = image.copy()

    H, W = img.shape[:2]
    x, y, w, h = roi
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))

    roi_img = img[y:y + h, x:x + w]
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if roi_img.ndim == 3 else roi_img

    # 대비 강화 (CLAHE) — 저대비 X-ray에서 경계 식별 향상
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 1D 명도 프로파일
    if edge_orientation == "horizontal":
        profile = enhanced.mean(axis=1).astype(np.float32)  # 행별 평균 (y축 따라)
    else:
        profile = enhanced.mean(axis=0).astype(np.float32)  # 열별 평균 (x축 따라)

    # 평활화
    k = max(5, len(profile) // 15)
    if k % 2 == 0:
        k += 1
    profile_s = cv2.GaussianBlur(profile.reshape(-1, 1), (1, k), 0).flatten()

    grad = np.gradient(profile_s)

    # ROI 상반/하반에서 각각 최대 명도(=뼈 중심) 위치 추정
    mid = len(profile_s) // 2
    peak1 = int(np.argmax(profile_s[:mid]))
    peak2 = mid + int(np.argmax(profile_s[mid:]))
    if peak2 - peak1 < 4:
        raise RuntimeError(
            "두 뼈 띠를 분리 검출할 수 없습니다. ROI가 관절을 사이에 두고 두 뼈를 모두 포함하는지 확인해 주세요."
        )

    # 두 봉우리 사이에서 valley와 가장 가파른 기울기 탐색 (배경 어둠에 휘둘리지 않도록 범위 제한)
    valley = peak1 + int(np.argmin(profile_s[peak1:peak2 + 1]))
    if valley <= peak1 or valley >= peak2:
        raise RuntimeError("관절 간격(밝→어두→밝 패턴)을 찾지 못했습니다.")

    upper_idx = peak1 + int(np.argmin(grad[peak1:valley + 1]))     # 밝→어두 가장 가파른 곳
    lower_idx = valley + int(np.argmax(grad[valley:peak2 + 1]))    # 어두→밝 가장 가파른 곳

    px_distance = abs(lower_idx - upper_idx)
    if px_distance < 2:
        raise RuntimeError("검출된 두 경계가 너무 가깝습니다. ROI/이미지 품질을 확인해 주세요.")
    mm_distance = px_distance * mm_per_pixel

    out = img.copy()
    if edge_orientation == "horizontal":
        upper_y = y + upper_idx
        lower_y = y + lower_idx
        cv2.line(out, (x, upper_y), (x + w, upper_y), line_color, 2)
        cv2.line(out, (x, lower_y), (x + w, lower_y), line_color, 2)
        cx = x + w // 2
        cv2.line(out, (cx, upper_y), (cx, lower_y), line_color, 2)
        cv2.line(out, (cx - 8, upper_y), (cx + 8, upper_y), line_color, 2)
        cv2.line(out, (cx - 8, lower_y), (cx + 8, lower_y), line_color, 2)
        target_xy = (cx, (upper_y + lower_y) // 2)
    else:
        upper_x = x + upper_idx
        lower_x = x + lower_idx
        cv2.line(out, (upper_x, y), (upper_x, y + h), line_color, 2)
        cv2.line(out, (lower_x, y), (lower_x, y + h), line_color, 2)
        cy = y + h // 2
        cv2.line(out, (upper_x, cy), (lower_x, cy), line_color, 2)
        cv2.line(out, (upper_x, cy - 8), (upper_x, cy + 8), line_color, 2)
        cv2.line(out, (lower_x, cy - 8), (lower_x, cy + 8), line_color, 2)
        target_xy = ((upper_x + lower_x) // 2, cy)

    label = f"{label_prefix}: {mm_distance:.2f} mm"
    out = annotate_image(
        out,
        target_xy=target_xy,
        label=label,
        arrow_color=line_color,
        box_color=line_color,
        font_path=font_path,
        font_size=font_size,
    )
    return out, mm_distance


def get_image_info(image_path: str | Path) -> dict:
    """이미지 파일의 기본 정보(경로, 크기, 모드/채널)를 반환."""
    img = load_image_pil(image_path)
    cv_img = load_image_cv2(image_path)
    return {
        "path": str(Path(image_path).resolve()),
        "format": img.format,
        "mode": img.mode,
        "size": img.size,                        # (width, height)
        "channels": cv_img.shape[2] if cv_img.ndim == 3 else 1,
        "dtype": str(cv_img.dtype),
    }


# ── 나이/성별 기준 근육량 (kg) ────────────────────────────────────────────────
MUSCLE_REFERENCE = {
    "남": {(0, 29): 34, (30, 39): 33, (40, 49): 31, (50, 59): 29, (60, 200): 26},
    "여": {(0, 29): 24, (30, 39): 23, (40, 49): 22, (50, 59): 20, (60, 200): 18},
}


# ── 입력 데이터 모델 ──────────────────────────────────────────────────────────
@dataclass
class PatientData:
    name: str
    age: int
    gender: str                              # "남" / "여"
    muscle_mass_kg: float                    # 근육량 (kg)
    joint_space_mm: float                    # 엑스레이 관절 간격 (mm)
    medications: list = field(default_factory=list)  # 복용 약물
    shockwave_shots: int = 0                 # 충격파 타수
    shockwave_intensity_bar: float = 0.0    # 충격파 강도 (bar)
    manual_care_minutes: int = 0            # 수기케어 시간 (분)
    body_weight_kg: float = 65.0            # 체중 (운동부하 계산용)
    hyaluronic_injection_count: int = 0    # 무릎 히알루론산 주사 횟수 (0~3, 1세트=3회)


# ── 관절 건강 점수 산출 함수 ──────────────────────────────────────────────────
def _get_reference_muscle(age: int, gender: str) -> float:
    table = MUSCLE_REFERENCE.get(gender, MUSCLE_REFERENCE["남"])
    for (low, high), ref in table.items():
        if low <= age <= high:
            return ref
    return 28


def _score_joint_space(mm: float) -> float:
    """관절 간격(mm) → 0~50점  (간격이 넓을수록 건강)"""
    if mm >= 5.0:   return 50
    elif mm >= 4.0: return 42
    elif mm >= 3.0: return 30
    elif mm >= 2.0: return 17
    elif mm >= 1.0: return 8
    else:           return 3


def _score_muscle_mass(muscle_kg: float, age: int, gender: str) -> float:
    """근육량 → 0~50점  (기준 대비 비율)"""
    ref = _get_reference_muscle(age, gender)
    ratio = (muscle_kg / ref) * 100

    if ratio >= 100:   return 50
    elif ratio >= 85:  return 40
    elif ratio >= 70:  return 28
    elif ratio >= 55:  return 15
    else:              return 5


def calculate_joint_health_score(patient: PatientData) -> int:
    """
    엑스레이 관절 간격 + 근육량 기반으로 관절 건강 점수 산출 (1~100점).
    관절 간격 50점 + 근육량 50점으로 구성.
    """
    joint  = _score_joint_space(patient.joint_space_mm)
    muscle = _score_muscle_mass(patient.muscle_mass_kg, patient.age, patient.gender)
    return max(1, min(100, round(joint + muscle)))


# ── 처방 추천 ─────────────────────────────────────────────────────────────────
@dataclass
class Prescription:
    score: int
    grade: str
    exercise_load_kg: float
    exercise_description: str
    shockwave_shots: int
    shockwave_intensity_bar: float
    notes: str


def _injection_load_multiplier(injection_count: int) -> float:
    """
    주사 진행 단계별 운동부하 보정 계수.
    - 1~2회차(치료 진행 중): 관절 자극 최소화를 위해 20% 감량
    - 3회차 완료: 윤활 효과 극대화로 10% 증량
    """
    if injection_count in (1, 2):
        return 0.80
    elif injection_count == 3:
        return 1.10
    return 1.00  # 0회 또는 미사용


def _injection_notes(injection_count: int) -> str:
    """주사 잔여 여부에 따른 비고 문구 반환"""
    if 0 < injection_count < 3:
        remaining = 3 - injection_count
        return f"잔여 주사 일정 확인 필요 (잔여 {remaining}회 / 총 3회 세트)"
    if injection_count == 3:
        return "히알루론산 주사 1세트 완료."
    return ""


def recommend_prescription(patient: PatientData) -> Prescription:
    """관절 건강 점수에 따른 운동부하(kg) + 충격파 처방 생성"""
    score      = calculate_joint_health_score(patient)
    bw         = patient.body_weight_kg
    inj_multi  = _injection_load_multiplier(patient.hyaluronic_injection_count)
    inj_note   = _injection_notes(patient.hyaluronic_injection_count)

    def apply_inj(base_load: float) -> float:
        return round(base_load * inj_multi, 1)

    def merge_notes(base: str) -> str:
        return f"{base}  ※ {inj_note}" if inj_note else base

    if score >= 80:
        return Prescription(
            score=score,
            grade="A (우수)",
            exercise_load_kg=apply_inj(bw * 0.60),
            exercise_description="고부하 점진적 저항운동 (스쿼트, 레그프레스 등)",
            shockwave_shots=2000,
            shockwave_intensity_bar=3.5,
            notes=merge_notes("관절 상태 양호. 근력 강화 중심 프로그램 권장.")
        )
    elif score >= 60:
        return Prescription(
            score=score,
            grade="B (양호)",
            exercise_load_kg=apply_inj(bw * 0.40),
            exercise_description="중부하 기능성 운동 (레그컬, 밴드저항 등)",
            shockwave_shots=1500,
            shockwave_intensity_bar=2.5,
            notes=merge_notes("점진적 부하 증가 가능. 주 2회 충격파 권장.")
        )
    elif score >= 40:
        return Prescription(
            score=score,
            grade="C (보통)",
            exercise_load_kg=apply_inj(bw * 0.20),
            exercise_description="저부하 관절 안정화 운동 (고유감각 훈련, 탄성밴드)",
            shockwave_shots=1000,
            shockwave_intensity_bar=2.0,
            notes=merge_notes("통증 모니터링 필수. 충격파 후 냉각 치료 병행 권장.")
        )
    elif score >= 20:
        return Prescription(
            score=score,
            grade="D (주의)",
            exercise_load_kg=apply_inj(bw * 0.10),
            exercise_description="초저부하 운동 (수중 보행, 누운 자세 하지 운동)",
            shockwave_shots=800,
            shockwave_intensity_bar=1.5,
            notes=merge_notes("관절 보호대 착용 권고. 전문의 추가 진단 권장.")
        )
    else:
        return Prescription(
            score=score,
            grade="E (위험)",
            exercise_load_kg=0.0,
            exercise_description="비부하 운동 전용 (수중치료, 누운 자세 운동만 허용)",
            shockwave_shots=500,
            shockwave_intensity_bar=1.0,
            notes=merge_notes("즉각적인 전문의 진단 강력 권고. 충격파는 의사 처방 후 시행.")
        )


# ── 리포트 출력 ───────────────────────────────────────────────────────────────
def print_report(patient: PatientData) -> None:
    score = calculate_joint_health_score(patient)
    rx    = recommend_prescription(patient)
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    bar   = "█" * (score // 5) + "░" * (20 - score // 5)

    print("=" * 62)
    print("       Body365  재활 분석 리포트")
    print(f"       분석일시: {now}")
    print("=" * 62)
    print(f"  환자명     : {patient.name}")
    print(f"  나이/성별  : {patient.age}세 / {patient.gender}")
    print(f"  근육량     : {patient.muscle_mass_kg} kg")
    print(f"  관절 간격  : {patient.joint_space_mm} mm  (X-ray)")
    print(f"  복용 약물  : {', '.join(patient.medications) if patient.medications else '없음'}")
    print(f"  충격파     : {patient.shockwave_shots}타  /  {patient.shockwave_intensity_bar} bar")
    print(f"  수기케어   : {patient.manual_care_minutes}분")
    print("-" * 62)
    print(f"  ▶ 관절 건강 점수 : {score:>3}점  [{bar}]")
    print(f"     등급 : {rx.grade}")
    print("-" * 62)
    print(f"  ▶ 추천 운동 부하    : {rx.exercise_load_kg} kg")
    print(f"     {rx.exercise_description}")
    print(f"  ▶ 다음 충격파 처방")
    print(f"     - 권장 타수 : {rx.shockwave_shots} 타")
    print(f"     - 권장 강도 : {rx.shockwave_intensity_bar} bar")
    print(f"  ▶ 비고 : {rx.notes}")
    print("=" * 62)


# ── 메인 실행 (예시 환자) ─────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = PatientData(
        name="홍길동",
        age=55,
        gender="남",
        muscle_mass_kg=26.5,
        joint_space_mm=2.8,
        medications=["세레콕시브", "글루코사민"],
        shockwave_shots=1500,
        shockwave_intensity_bar=2.5,
        manual_care_minutes=30,
        body_weight_kg=72.0,
    )
    print_report(sample)
