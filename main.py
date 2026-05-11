from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from insightface.app import FaceAnalysis  # type: ignore[import-untyped]

# ── config ────────────────────────────────────────────────────────────────────

BANCO_DIR = Path("banco_imagens")
VIDEOS_DIR = Path("videos")
OUTPUT_DIR = Path("output")
VIDEOS: list[str] = ["video1.mp4", "video2.mp4"]

FRAME_SKIP = 3
DET_SIZE = (320, 320)
COSINE_THRESHOLD = 0.35  # above → same person
FONT = cv2.FONT_HERSHEY_SIMPLEX

Embedding = npt.NDArray[np.float32]
Frame = npt.NDArray[Any]  # cv2 MatLike-compatible

# ── stats ─────────────────────────────────────────────────────────────────────


@dataclass
class VideoStats:
    video_name: str
    total_faces: int = 0
    unknown: int = 0
    recognized: dict[str, int] = field(default_factory=dict)

    @property
    def total_recognized(self) -> int:
        return sum(self.recognized.values())

    def record(self, name: str) -> None:
        if name == "Desconhecido":
            self.unknown += 1
        else:
            self.recognized[name] = self.recognized.get(name, 0) + 1
        self.total_faces += 1

    def print_summary(self) -> None:
        pct = self.total_recognized / self.total_faces * 100 if self.total_faces else 0.0
        print(f"\n  Vídeo        : {self.video_name}")
        print(f"  Faces        : {self.total_faces}")
        print(f"  Reconhecidas : {self.total_recognized} ({pct:.1f}%)")
        print(f"  Desconhecidas: {self.unknown}")
        for name, count in self.recognized.items():
            print(f"    → {name}: {count}")


# ── face DB ───────────────────────────────────────────────────────────────────


def _cosine_sim(a: Embedding, b: Embedding) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_known_faces(
    app: Any, banco_dir: Path
) -> tuple[list[Embedding], list[str]]:
    known_embeddings: list[Embedding] = []
    known_names: list[str] = []

    for person_dir in sorted(banco_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces: list[Any] = app.get(img)
            if not faces:
                print(f"  [AVISO] Sem rosto detectado: {img_path}")
                continue
            embedding: Embedding = np.array(faces[0].embedding, dtype=np.float32)
            known_embeddings.append(embedding)
            known_names.append(name)
            print(f"  Carregado: {name} ({img_path.name})")

    return known_embeddings, known_names


def identify(
    embedding: Embedding,
    known_embeddings: list[Embedding],
    known_names: list[str],
) -> tuple[str, float]:
    if not known_embeddings:
        return "Desconhecido", 0.0
    sims = [_cosine_sim(embedding, ref) for ref in known_embeddings]
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]
    if best_sim >= COSINE_THRESHOLD:
        return known_names[best_idx], best_sim
    return "Desconhecido", best_sim


# ── drawing ───────────────────────────────────────────────────────────────────


def _draw(
    frame: Frame,
    bbox: Embedding,
    name: str,
    sim: float,
) -> None:
    x1, y1, x2, y2 = bbox.astype(int)
    color = (0, 200, 0) if name != "Desconhecido" else (0, 0, 220)
    label = f"{name} ({sim:.2f})" if name != "Desconhecido" else name
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, y2 - 28), (x2, y2), color, cv2.FILLED)
    cv2.putText(frame, label, (x1 + 4, y2 - 8), FONT, 0.55, (255, 255, 255), 1)


# ── video processing ──────────────────────────────────────────────────────────


def process_video(
    app: Any,
    video_path: Path,
    known_embeddings: list[Embedding],
    known_names: list[str],
    output_path: Path,
) -> VideoStats:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Não foi possível abrir: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    stats = VideoStats(video_name=video_path.name)
    frame_n = 0
    last_labels: list[tuple[Embedding, str, float]] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1

        if frame_n % FRAME_SKIP == 0:
            faces: list[Any] = app.get(frame)
            last_labels = []
            for face in faces:
                emb: Embedding = np.array(face.embedding, dtype=np.float32)
                name, sim = identify(emb, known_embeddings, known_names)
                bbox: Embedding = np.array(face.bbox, dtype=np.float32)
                last_labels.append((bbox, name, sim))
                stats.record(name)

        for bbox, name, sim in last_labels:
            _draw(frame, bbox, name, sim)

        out.write(frame)

    cap.release()
    out.release()
    return stats


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    if not BANCO_DIR.exists():
        print(f"'{BANCO_DIR}' não encontrado. Crie e adicione subpastas com fotos.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Inicializando modelo (1ª execução faz download ~300 MB)...")
    app: Any = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=DET_SIZE)  # type: ignore[union-attr]

    print("\nCarregando banco de imagens...")
    known_embeddings, known_names = load_known_faces(app, BANCO_DIR)
    print(f"{len(known_embeddings)} embeddings / {len(set(known_names))} pessoa(s)\n")

    if not known_embeddings:
        print("Banco vazio — adicione fotos em banco_imagens/<NomePessoa>/foto.jpg")
        return

    all_stats: list[VideoStats] = []
    for filename in VIDEOS:
        video_path = VIDEOS_DIR / filename
        if not video_path.exists():
            print(f"[AVISO] Vídeo não encontrado: {video_path}")
            continue
        output_path = OUTPUT_DIR / f"{video_path.stem}_processado.mp4"
        print(f"Processando: {video_path} ...")
        stats = process_video(app, video_path, known_embeddings, known_names, output_path)
        stats.print_summary()
        print(f"  Salvo: {output_path}")
        all_stats.append(stats)

    if all_stats:
        total = sum(s.total_faces for s in all_stats)
        rec = sum(s.total_recognized for s in all_stats)
        unk = sum(s.unknown for s in all_stats)
        pct = rec / total * 100 if total else 0.0
        print("\n=== RESUMO ===")
        print(f"Faces: {total} | Reconhecidas: {rec} ({pct:.1f}%) | Desconhecidas: {unk}")


if __name__ == "__main__":
    main()
