from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from face_recognition import (
    compare_faces,
    face_distance,
    face_encodings,
    face_locations,
    load_image_file,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    FaceEncoding = npt.NDArray[np.float64]


BANCO_DIR = Path("banco_imagens")
VIDEOS_DIR = Path("videos")
OUTPUT_DIR = Path("output")
VIDEOS: list[str] = ["video1.mp4", "video2.mp4"]

FRAME_SKIP = 3
SCALE_FACTOR = 0.5
TOLERANCE = 0.6
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class VideoStats:
    video_name: str
    total_faces: int = 0
    unknown: int = 0
    recognized: dict[str, int] = field(default_factory=dict)

    @property
    def total_recognized(self) -> int:
        return sum(self.recognized.values())

    def record_match(self, name: str) -> None:
        self.recognized[name] = self.recognized.get(name, 0) + 1
        self.total_faces += 1

    def record_unknown(self) -> None:
        self.unknown += 1
        self.total_faces += 1

    def print_summary(self) -> None:
        print(f"\n  Vídeo: {self.video_name}")
        print(f"  Total de faces detectadas : {self.total_faces}")
        print(f"  Reconhecidas              : {self.total_recognized}")
        print(f"  Desconhecidas             : {self.unknown}")
        for name, count in self.recognized.items():
            print(f"    → {name}: {count} detecções")


def load_known_faces(banco_dir: Path) -> tuple[list[FaceEncoding], list[str]]:
    known_encodings: list[FaceEncoding] = []
    known_names: list[str] = []

    for person_dir in sorted(banco_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = load_image_file(str(img_path))
            encodings: list[FaceEncoding] = face_encodings(img)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"  Carregado: {name} ({img_path.name})")

    return known_encodings, known_names


def _identify_face(
    encoding: FaceEncoding,
    known_encodings: list[FaceEncoding],
    known_names: list[str],
) -> str:
    matches: list[bool] = compare_faces(known_encodings, encoding, tolerance=TOLERANCE)
    if not any(matches):
        return "Desconhecido"

    distances: npt.NDArray[np.float64] = face_distance(known_encodings, encoding)
    best_idx = int(np.argmin(distances))
    return known_names[best_idx] if matches[best_idx] else "Desconhecido"


def _draw_label(
    frame: npt.NDArray[np.uint8],
    top: int,
    right: int,
    bottom: int,
    left: int,
    name: str,
) -> None:
    color = (0, 200, 0) if name != "Desconhecido" else (0, 0, 220)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.rectangle(frame, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, name, (left + 4, bottom - 8), FONT, 0.6, (255, 255, 255), 1)


def process_video(
    video_path: Path,
    known_encodings: list[FaceEncoding],
    known_names: list[str],
    output_path: Path,
) -> VideoStats:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    stats = VideoStats(video_name=video_path.name)
    frame_count = 0
    last_labels: list[tuple[int, int, int, int, str]] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small: npt.NDArray[np.uint8] = cv2.resize(rgb, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

            locations: list[tuple[int, int, int, int]] = face_locations(small)
            encodings: list[FaceEncoding] = face_encodings(small, locations)

            last_labels = []
            for (top, right, bottom, left), enc in zip(locations, encodings, strict=True):
                scale = int(1 / SCALE_FACTOR)
                top, right, bottom, left = top * scale, right * scale, bottom * scale, left * scale

                name = _identify_face(enc, known_encodings, known_names)
                last_labels.append((top, right, bottom, left, name))

                if name == "Desconhecido":
                    stats.record_unknown()
                else:
                    stats.record_match(name)

        for top, right, bottom, left, name in last_labels:
            _draw_label(frame, top, right, bottom, left, name)

        out.write(frame)

    cap.release()
    out.release()
    return stats


def main() -> None:
    if not BANCO_DIR.exists():
        print(f"Diretório '{BANCO_DIR}' não encontrado. Crie e adicione subpastas com fotos.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Carregando banco de imagens...")
    known_encodings, known_names = load_known_faces(BANCO_DIR)
    pessoas = len(set(known_names))
    print(f"{len(known_encodings)} encodings carregados para {pessoas} pessoa(s).\n")

    if not known_encodings:
        print("Banco vazio — adicione imagens em banco_imagens/<NomePessoa>/foto.jpg")
        return

    all_stats: list[VideoStats] = []

    for filename in VIDEOS:
        video_path = VIDEOS_DIR / filename
        if not video_path.exists():
            print(f"[AVISO] Vídeo não encontrado: {video_path}")
            continue

        output_path = OUTPUT_DIR / f"{video_path.stem}_processado.mp4"
        print(f"Processando: {video_path} ...")
        stats = process_video(video_path, known_encodings, known_names, output_path)
        stats.print_summary()
        print(f"  Salvo em: {output_path}")
        all_stats.append(stats)

    if all_stats:
        print("\n=== RESUMO GERAL ===")
        total_faces = sum(s.total_faces for s in all_stats)
        total_rec = sum(s.total_recognized for s in all_stats)
        total_unk = sum(s.unknown for s in all_stats)
        pct = (total_rec / total_faces * 100) if total_faces else 0.0
        print(f"Faces detectadas : {total_faces}")
        print(f"Reconhecidas     : {total_rec} ({pct:.1f}%)")
        print(f"Desconhecidas    : {total_unk}")


if __name__ == "__main__":
    main()
