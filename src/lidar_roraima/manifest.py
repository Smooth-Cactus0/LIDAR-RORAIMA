from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class TileHeader:
    tile_id: str
    file_name: str
    file_path: str
    is_copc: bool
    las_version: str
    point_format: int
    is_compressed: bool
    record_length: int
    epsg: int | None
    point_count: int
    size_bytes: int
    bbox: dict[str, float]
    scale: tuple[float, float, float]
    offset: tuple[float, float, float]
    system_id: str
    generating_software: str
    creation_year: int
    creation_doy: int
    qa_flags: list[str]
    content_signature: str
    wkt_preview: str | None


def _clean_ascii(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()


def _parse_epsg_from_geo_keys(data: bytes) -> int | None:
    if not data or len(data) < 8:
        return None
    vals = struct.unpack("<" + ("H" * (len(data) // 2)), data)
    n_keys = vals[3]
    for i in range(n_keys):
        key_id, _, _, value = vals[4 + i * 4 : 8 + i * 4]
        if key_id == 3072:
            return int(value)
    return None


def _content_signature(point_count: int, bbox: dict[str, float], epsg: int | None) -> str:
    payload = f"{point_count}|{bbox['min_x']:.2f}|{bbox['max_x']:.2f}|{bbox['min_y']:.2f}|{bbox['max_y']:.2f}|{bbox['min_z']:.2f}|{bbox['max_z']:.2f}|{epsg}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def parse_las_header(path: Path) -> TileHeader:
    with path.open("rb") as f:
        header = f.read(375)
        sig = header[0:4].decode("ascii", errors="replace")
        if sig != "LASF":
            raise ValueError(f"{path.name} is not a LAS/LAZ file.")

        major = header[24]
        minor = header[25]
        version = f"{major}.{minor}"
        system_id = _clean_ascii(header[26:58])
        generating_software = _clean_ascii(header[58:90])
        creation_doy = struct.unpack_from("<H", header, 90)[0]
        creation_year = struct.unpack_from("<H", header, 92)[0]
        header_size = struct.unpack_from("<H", header, 94)[0]
        n_vlr = struct.unpack_from("<I", header, 100)[0]
        point_format_raw = header[104]
        point_format = int(point_format_raw & 0x3F)
        is_compressed = bool(point_format_raw & 0x80)
        record_length = struct.unpack_from("<H", header, 105)[0]
        legacy_point_count = struct.unpack_from("<I", header, 107)[0]
        point_count = (
            struct.unpack_from("<Q", header, 247)[0]
            if (major, minor) >= (1, 4)
            else legacy_point_count
        )

        scale = struct.unpack_from("<ddd", header, 131)
        offset = struct.unpack_from("<ddd", header, 155)
        max_x, min_x, max_y, min_y, max_z, min_z = struct.unpack_from("<dddddd", header, 179)
        bbox = {
            "min_x": float(min_x),
            "max_x": float(max_x),
            "min_y": float(min_y),
            "max_y": float(max_y),
            "min_z": float(min_z),
            "max_z": float(max_z),
        }

        epsg: int | None = None
        wkt_preview: str | None = None
        f.seek(header_size)
        for _ in range(n_vlr):
            vlr_header = f.read(54)
            if len(vlr_header) < 54:
                break
            user_id = _clean_ascii(vlr_header[2:18])
            record_id = struct.unpack_from("<H", vlr_header, 18)[0]
            rec_len = struct.unpack_from("<H", vlr_header, 20)[0]
            data = f.read(rec_len)
            if user_id == "LASF_Projection" and record_id == 34735:
                epsg = _parse_epsg_from_geo_keys(data)
            if user_id == "LASF_Projection" and record_id == 2112 and wkt_preview is None:
                wkt_preview = data.decode("utf-8", errors="ignore")[:500]

    is_copc = path.name.lower().endswith(".copc.laz")
    tile_id = path.name.replace(".copc.laz", "").replace(".laz", "")
    qa_flags: list[str] = []
    if epsg is None:
        qa_flags.append("missing_epsg")
    if not is_compressed:
        qa_flags.append("uncompressed")
    if point_count <= 0:
        qa_flags.append("empty_tile")

    signature = _content_signature(point_count=point_count, bbox=bbox, epsg=epsg)
    return TileHeader(
        tile_id=tile_id,
        file_name=path.name,
        file_path=str(path.resolve()),
        is_copc=is_copc,
        las_version=version,
        point_format=point_format,
        is_compressed=is_compressed,
        record_length=record_length,
        epsg=epsg,
        point_count=int(point_count),
        size_bytes=path.stat().st_size,
        bbox=bbox,
        scale=(float(scale[0]), float(scale[1]), float(scale[2])),
        offset=(float(offset[0]), float(offset[1]), float(offset[2])),
        system_id=system_id,
        generating_software=generating_software,
        creation_year=int(creation_year),
        creation_doy=int(creation_doy),
        qa_flags=qa_flags,
        content_signature=signature,
        wkt_preview=wkt_preview,
    )


def build_manifest(raw_data_dir: Path) -> pd.DataFrame:
    paths = sorted({path.resolve() for path in raw_data_dir.glob("*.laz")})
    rows = [parse_las_header(path) for path in paths]

    duplicates: dict[str, list[str]] = {}
    for row in rows:
        duplicates.setdefault(row.content_signature, []).append(row.file_name)

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        dup_group = duplicates[row.content_signature]
        is_duplicate = len(dup_group) > 1
        qa_flags = list(row.qa_flags)
        if is_duplicate:
            qa_flags.append("duplicate_content")
        output_rows.append(
            {
                "tile_id": row.tile_id,
                "file_name": row.file_name,
                "file_path": row.file_path,
                "is_copc": row.is_copc,
                "is_duplicate": is_duplicate,
                "las_version": row.las_version,
                "point_format": row.point_format,
                "is_compressed": row.is_compressed,
                "record_length": row.record_length,
                "epsg": row.epsg,
                "point_count": row.point_count,
                "size_bytes": row.size_bytes,
                "bbox": json.dumps(row.bbox, sort_keys=True),
                "scale": json.dumps(row.scale),
                "offset": json.dumps(row.offset),
                "system_id": row.system_id,
                "generating_software": row.generating_software,
                "creation_year": row.creation_year,
                "creation_doy": row.creation_doy,
                "qa_flags": json.dumps(sorted(set(qa_flags))),
                "content_signature": row.content_signature,
                "wkt_preview": row.wkt_preview,
            }
        )

    manifest = pd.DataFrame(output_rows).sort_values(["tile_id", "file_name"]).reset_index(drop=True)
    return manifest


def save_manifest(manifest: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output_path, index=False)
