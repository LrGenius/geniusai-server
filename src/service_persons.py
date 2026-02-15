"""
Person grouping for face embeddings: cluster faces into persons and store display names.
Persons are identified by cluster ids (person_0, person_1, ...); names are stored in a JSON file.
"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from config import DB_PATH, logger
import service_chroma as chroma_service

PERSON_NAMES_FILENAME = "person_names.json"


def _person_names_path() -> str:
    return os.path.join(DB_PATH, PERSON_NAMES_FILENAME)


def _load_person_names() -> Dict[str, str]:
    path = _person_names_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load person names from {path}: {e}")
        return {}


def _save_person_names(names: Dict[str, str]) -> None:
    path = _person_names_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(names, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Could not save person names to {path}: {e}")
        raise


def set_person_name(person_id: str, name: str) -> None:
    """Set or update the display name for a person. Empty name clears it."""
    names = _load_person_names()
    if name:
        names[person_id] = name.strip()
    else:
        names.pop(person_id, None)
    _save_person_names(names)


def get_person_name(person_id: str) -> str:
    """Return the display name for a person, or empty string."""
    return _load_person_names().get(person_id, "")


def run_clustering(distance_threshold: float = 0.55) -> Dict[str, Any]:
    """
    Cluster all face embeddings into persons and update face metadata with person_id.
    Uses AgglomerativeClustering on L2-normalized embeddings (equivalent to cosine similarity).
    distance_threshold: L2 distance below which two faces are merged (e.g. 0.5–0.6).
    Returns summary: { "person_count": N, "face_count": M, "updated": M }.
    """
    data = chroma_service.get_all_faces(include_embeddings=True)
    ids = data.get("ids", [])
    embeddings = data.get("embeddings", [])
    metadatas = data.get("metadatas", [])

    if len(ids) == 0 or embeddings is None or len(embeddings) == 0:
        logger.info("No faces to cluster.")
        return {"person_count": 0, "face_count": 0, "updated": 0}

    X = np.array(embeddings, dtype=np.float32)
    n = len(ids)

    if n == 1:
        labels = [0]
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="euclidean",
            linkage="average",
        )
        labels = clustering.fit_predict(X)

    # Map cluster index to stable person_id (person_0, person_1, ...)
    unique_labels = sorted(set(labels))
    label_to_person = {lb: f"person_{i}" for i, lb in enumerate(unique_labels)}

    new_metadatas = []
    for i, meta in enumerate(metadatas or []):
        person_id = label_to_person.get(labels[i], "person_0")
        new_meta = {
            "photo_uuid": meta.get("photo_uuid", ""),
            "thumbnail": meta.get("thumbnail", ""),
            "person_id": person_id,
        }
        new_metadatas.append(new_meta)

    chroma_service.update_face_metadatas(ids, new_metadatas)
    logger.info(f"Clustering assigned {len(unique_labels)} persons to {n} faces (threshold={distance_threshold}).")
    return {"person_count": len(unique_labels), "face_count": n, "updated": n}


def list_persons() -> List[Dict[str, Any]]:
    """
    List all persons: for each person_id, return name, face_count, photo_count, and sample thumbnail.
    """
    data = chroma_service.get_all_faces(include_embeddings=False)
    ids = data.get("ids", [])
    metadatas = data.get("metadatas", [])

    # Group by person_id
    by_person: Dict[str, Dict[str, Any]] = {}
    names = _load_person_names()

    for i, meta in enumerate(metadatas or []):
        pid = meta.get("person_id", "")
        if pid == "":
            pid = "_unassigned"
        if pid not in by_person:
            by_person[pid] = {"person_id": pid if pid != "_unassigned" else "", "face_ids": [], "photo_uuids": set(), "thumbnail": meta.get("thumbnail", "")}
        by_person[pid]["face_ids"].append(ids[i] if i < len(ids) else "")
        by_person[pid]["photo_uuids"].add(meta.get("photo_uuid", ""))

    result = []
    for pid, info in sorted(by_person.items(), key=lambda x: (x[0] == "_unassigned", x[0])):
        person_id = info["person_id"]
        result.append({
            "person_id": person_id,
            "name": names.get(person_id, "") if person_id else "",
            "face_count": len(info["face_ids"]),
            "photo_count": len(info["photo_uuids"]),
            "thumbnail": info["thumbnail"] or "",
        })
    return result


def get_photo_uuids_for_person(person_id: str) -> List[str]:
    """Return list of photo UUIDs that contain this person (unique)."""
    data = chroma_service.get_all_faces(include_embeddings=False)
    metadatas = data.get("metadatas", [])
    uuids = set()
    for meta in metadatas or []:
        if meta.get("person_id") == person_id:
            uuids.add(meta.get("photo_uuid", ""))
    return sorted(uuids)
