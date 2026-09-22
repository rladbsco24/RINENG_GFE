"""Stage existing raw caches without replacing sources, figures or exported tables."""
from __future__ import annotations

from pathlib import Path
import os
import shutil


_CACHE_DIRECTORIES = frozenset({
    'cache', 'caches', 'linear_cache', 'mechanical_cache', 'pressure_cache',
    'sensitivity_cache', 'reviewer_robustness_cache', 'force_cache',
})


def stage_existing_caches(destination_root, source_root=None):
    """Copy absent raw cache files; each producer still validates its exact key.

    A source is an existing RINENG_GFE_Standalone_Run folder, or its parent.
    None resumes the caches already present in the destination. Old FE/GFE
    records keep their old filenames and cannot satisfy the revised solver key.
    """
    destination = Path(destination_root).expanduser().resolve()
    if source_root in (None, ''):
        return {'mode': 'resume-in-place', 'root': str(destination), 'copied_files': 0}
    source = Path(source_root).expanduser().resolve()
    nested = source / 'RINENG_GFE_Standalone_Run'
    if nested.is_dir():
        source = nested
    if not source.is_dir():
        raise FileNotFoundError(f'CACHE_SOURCE_ROOT is not an existing run directory: {source}')
    if source == destination:
        return {'mode': 'resume-in-place', 'root': str(destination), 'copied_files': 0}
    copied = skipped = 0
    for directory, names, files in os.walk(source):
        current = Path(directory)
        # Never recurse into this run if the user selected an ancestor folder.
        names[:] = [name for name in names if name != '__pycache__'
                    and (current / name).resolve() != destination]
        relative = current.relative_to(source)
        if not any(part.lower() in _CACHE_DIRECTORIES or part.lower().endswith('_cache')
                   for part in relative.parts):
            continue
        for name in files:
            original = current / name
            if original.suffix.lower() not in {'.pkl', '.pickle', '.npz', '.npy', '.json'}:
                continue
            target = destination / relative / name
            if target.exists():
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            copied += 1
    result = {'mode': 'stage-raw-cache-files', 'source': str(source),
              'destination': str(destination), 'copied_files': copied,
              'existing_files_preserved': skipped,
              'reuse_rule': 'exact numerical key and solver policy; no rekeying'}
    if not copied and not skipped:
        print('No raw caches found in CACHE_SOURCE_ROOT; missing records will be computed.')
    return result
