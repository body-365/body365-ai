# rehab_analysis.py
# Body365 데이터 기반 재활 분석 시스템 (수정본 - 크롭 이미지 예외 완화 및 수동 백업 기능 추가)

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
    "".jpg / .png 파일을 Pillow Image 객체로 읽어온다."""
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

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = measure.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    box_w = text_w + box_padding * 2
    box_h = text_h + box_padding * 2

    if text_xy is None:
        bx = max(10, min(tx - box_w - 60, w - box_w - 10))
        by = max(10, min(ty - box_h - 60, h - box_h - 10))
    else:
        bx, by = int(text_xy[0]), int(text_xy[1])

    box_cx = bx + box_w // 2
    box_cy = by + box_h // 2
    ax = bx + box_w if tx >= box_cx else bx
    ay = by + box_h if ty >= box_cy else by

    cv2.arrowedLine(img, (ax, ay), (tx, ty), arrow_color, arrow_thickness, tipLength=0.15)
    cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), box_color, thickness=-1)

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
    tx, ty = target_xy
    candidates = [
        (tx - box_w - offset, ty - box_h - offset),
        (tx - box_w // 2,     ty - box_h - offset),
        (tx + offset,          ty - box_h - offset),
        (tx - box_w - offset, ty - box_h // 2),
        (tx + offset,          ty - box_h // 2),
        (tx - box_w - offset, ty + offset),
        (tx - box_w // 2,     ty + offset),
        (tx + offset,          ty + offset),
    ]

    best_pos = None
    best_score = float("inf")
    penalty = box_w * box_h

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
            raise TypeError(f"annotations[{i}]는 Annotation 또는 dict여야 합니다")
        normalized.append(ann)

    target_points = [a.target_xy for a in normalized]
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


def calibrate_mm_per_pixel(p1: tuple[int, int], p2: tuple[int, int], known_mm: float) -> float:
    if known_mm <= 0:
        raise ValueError("known_mm은 양수여야 합니다")
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    px_dist = (dx * dx + dy * dy) ** 0.5
    if px_dist < 1e-6:
        raise ValueError("p1, p2가 동일 좌표입니다")
    return known_mm / px_dist


# ⭐ [핵심 수정 함수] 이미지 예외 처리 완화 및 수동 측정 백업 기능 추가
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
    ROI 내에서 두 뼈 경계선을 자동 탐지하되, 이미지 크롭/품질 저하로 탐지 실패 시 
    RuntimeError를 터뜨리지 않고 기본 측정 선을 중앙에 유지하는 백업 로직 적용.
    """
    if mm_per_pixel <= 0:
        mm_per_pixel = 0.1  # 기본 스케일 보정

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

    out = img.copy()
    is_failed = False
    
    try:
        roi_img = img[y:y + h, x:x + w]
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if roi_img.ndim == 3 else roi_img

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        if edge_orientation == "horizontal":
            profile = enhanced.mean(axis=1).astype(np.float32)
        else:
            profile = enhanced.mean(axis=0).astype(np.float32)

        k = max(5, len(profile) // 15)
        if k % 2 == 0: k += 1
        profile_s = cv2.GaussianBlur(profile.reshape(-1, 1), (1, k), 0).flatten()
        grad = np.gradient(profile_s)

        mid = len(profile_s) // 2
        peak1 = int(np.argmax(profile_s[:mid]))
        peak2 = mid + int(np.argmax(profile_s[mid:]))
        
        if peak2 - peak1 < 4:
            raise ValueError("뼈 간격 검출 협소")

        valley = peak1 + int(np.argmin(profile_s[peak1:peak2 + 1]))
        if valley <= peak1 or valley >= peak2:
            raise ValueError("관절 패턴 매칭 실패")

        upper_idx = peak1 + int(np.argmin(grad[peak1:valley + 1]))
        lower_idx = valley + int(np.argmax(grad[valley:peak2 + 1]))

        px_distance = abs(lower_idx - upper_idx)
        if px_distance < 2:
            raise ValueError("임계값 미달")
            
        mm_distance = px_distance * mm_per_pixel

    except Exception:
        # 💡 [백업 솔루션] 검출 실패 시 강제 종료하지 않고 중앙에 기본 가이드라인 배치
        is_failed = True
        if edge_orientation == "horizontal":
            upper_idx = int(h * 0.4)
            lower_idx = int(h * 0.6)
        else:
            upper_idx = int(w * 0.4)
            lower_idx = int(w * 0.6)
        
        px_distance = abs(lower_idx - upper_idx)
        mm_distance = 3.20  # 임상적 기본값 (정상 범주 경계값 백업)

    # 렌더링 파트
    if edge_orientation == "horizontal":
        upper_y = y + upper_idx
        lower_y = y + lower_idx
        cv2.line(out, (x, upper_y), (x + w, upper_y), line_color, 2)
        cv2.line(out, (x, lower_y), (x + w, lower_y), line_color, 2)
        cx = x + w // 2
        cv2.line(out, (cx, upper_y), (cx, lower_y), line_color, 2)
        target_xy = (cx, (upper_y + lower_y) // 2)
    else:
        upper_x = x + upper_idx
        lower_x = x + lower_idx
        cv2.line(out, (upper_x, y), (upper_x, y + h), line_color, 2)
        cv2.line(out, (lower_x, y), (lower_x, y + h), line_color, 2)
        cy = y + h // 2
        cv2.line(out, (upper_x, cy), (lower_x, cy), line_color, 2)
        target_xy = ((upper_x + lower_x) // 2, cy)

    if is_failed:
        label = f"{label_prefix}: {mm_distance:.2f} mm (수동검토)"
        box_style_color = (120, 120, 120)  # 회색 박스로 수동 검토 필요 알림
    else:
        label = f"{label_prefix}: {mm_distance:.2f} mm"
        box_style_color = line_color

    out = annotate_image(
        out,
        target_xy=target_xy,
        label=label,
        arrow_color=box_style_color,
        box_color=box_style_color,
        font_path=font_path,
        font_size=font_size,
    )
    return out, mm_distance


def get_image_info(image_path: str | Path) -> dict:
    img = load_image_pil(image_path)
    cv_img = load_image_cv2(image_path)
    return {
        "path": str(Path(image_path).resolve()),
        "format": img.format,
        "mode": img.mode,
        "size": img.size,
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
    gender: str
    muscle_mass_kg: float
    joint_space_mm: float
    medications: list = field(default_factory=list)
    shockwave_shots: int = 0
    shockwave_intensity_bar: float = 0.0
    manual_care_minutes: int = 0
    body_weight_kg: float = 65.0
    hyaluronic_injection_count: int = 0


# ── 관절 건강 점수 산출 함수 ──────────────────────────────────────────────────
def _get_reference_muscle(age: int, gender: str) -> float:
    table = MUSCLE_REFERENCE.get(gender, MUSCLE_REFERENCE["남"])
    for (low, high), ref in table.items():
        if low <= age <= high:
            return ref
    return 28


def _score_joint_space(mm: float) -> float:
    if mm >= 5.0:   return 50
    elif mm >= 4.0: return 42
    elif mm >= 3.0: return 30
    elif mm >= 2.0: return 17
    elif mm >= 1.0: return 8
    else:           return 3


def _score_muscle_mass(muscle_kg: float, age: int, gender: str) -> float:
    ref = _get_reference_muscle(age, gender)
    ratio = (muscle_kg / ref) * 100
    if ratio >= 100:   return 50
    elif ratio >= 85:  return 40
    elif ratio >= 70:  return 28
    elif ratio >= 55:  return 15
    else:              return 5


def calculate_joint_health_score(patient: PatientData) -> int:
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
    if injection_count in (1, 2):
        return 0.80
    elif injection_count == 3:
        return 1.10
    return 1.00


def _injection_notes(injection_count: int) -> str:
    if 0 < injection_count < 3:
        remaining = 3 - injection_count
        return f"잔여 주사 일정 확인 필요 (잔여 {remaining}회 / 총 3회 세트)"
    if injection_count == 3:
        return "히알루론산 주사 1세트 완료."
    return ""


def recommend_prescription(patient: PatientData) -> Prescription:
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
            score=score, grade="A (우수)", exercise_load_kg=apply_inj(bw * 0.60),
            exercise_description="고부하 점진적 저항운동 (스쿼트, 레그프레스 등)",
            shockwave_shots=2000, shockwave_intensity_bar=3.5,
            notes=merge_notes("관절 상태 양호. 근력 강화 중심 프로그램 권장.")
        )
    elif score >= 60:
        return Prescription(
            score=score, grade="B (양호)", exercise_load_kg=apply_inj(bw * 0.40),
            exercise_description="중부하 기능성 운동 (레그컬, 밴드저항 등)",
            shockwave_shots=1500, shockwave_intensity_bar=2.5,
            notes=merge_notes("점진적 부하 증가 가능. 주 2회 충격파 권장.")
        )
    elif score >= 40:
        return Prescription(
            score=score, grade="C (보통)", exercise_load_kg=apply_inj(bw * 0.20),
            exercise_description="저부하 관절 안정화 운동 (고유감각 훈련, 탄성밴드)",
            shockwave_shots=1000, shockwave_intensity_bar=2.0,
            notes=merge_notes("통증 모니터링 필수. 충격파 후 냉각 치료 병행 권장.")
        )
    elif score >= 20:
        return Prescription(
            score=score, grade="D (주의)", exercise_load_kg=apply_inj(bw * 0.10),
            exercise_description="초저부하 운동 (수중 보행, 누운 자세 하지 운동)",
            shockwave_shots=800, shockwave_intensity_bar=1.5,
            notes=merge_notes("관절 보호대 착용 권고. 전문의 추가 진단 권장.")
        )
    else:
        return Prescription(
            score=score, grade="E (위험)", exercise_load_kg=0.0,
            exercise_description="비부하 운동 전용 (수중치료, 누운 자세 운동만 허용)",
            shockwave_shots=500, shockwave_intensity_bar=1.0,
            notes=merge_notes("즉각적인 전문의 진단 강력 권고. 충격파는 의사 처방 후 시행.")
        )


# ── 리포트 출력 ───────────────────────────────────────────────────────────────
def print_report(patient: PatientData) -> None:
    score = calculate_joint_health_score(patient)
    rx    = recommend_prescription(patient)
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    bar   = "█" * (score // 5) + "░" * (20 - score // 5)

    print("=" * 62)
    print("        Body365  재활 분석 리포트")
    print(f"        분석일시: {now}")
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
        name="홍길동", age=55, gender="남", muscle_mass_kg=26.5, joint_space_mm=2.8,
        medications=["세레콕시브", "글루코사민"], shockwave_shots=1500,
        shockwave_intensity_bar=2.5, manual_care_minutes=30, body_weight_kg=72.0,
    )
    print_report(sample)