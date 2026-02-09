"""Build utilities to prevent hangs from stale locks and stuck processes."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def cleanup_stale_build_locks(build_dir: Path, max_lock_age_seconds: int = 300) -> None:
    """Remove stale build lock files and kill processes holding them.
    
    Args:
        build_dir: Directory containing build artifacts (may contain lock files)
        max_lock_age_seconds: Consider locks older than this stale (default 5 minutes)
    """
    if not build_dir.exists():
        return
    
    lock_files = list(build_dir.glob("lock")) + list(build_dir.glob("*.lock"))
    lock_files.extend(build_dir.glob(".ninja_lock"))
    
    for lock_file in lock_files:
        if not lock_file.exists():
            continue
        
        try:
            # Check if lock file is stale
            lock_age = time.time() - lock_file.stat().st_mtime
            if lock_age > max_lock_age_seconds:
                # Be conservative: never remove locks while a live build appears active.
                # Long nvcc/ninja builds can legitimately keep the same lock for minutes.
                if _has_active_build_process(build_dir):
                    continue
                try:
                    lock_file.unlink()
                except OSError:
                    pass  # May be locked by another process
        except OSError:
            pass  # File may have been deleted


def _has_active_build_process(directory: Path) -> bool:
    """Best-effort check for an active compiler/build process touching directory."""
    if not PSUTIL_AVAILABLE:
        return False

    try:
        directory_str = str(directory.resolve())
    except Exception:
        return False

    keywords = (
        "ninja",
        "nvcc",
        "ptxas",
        "cpp_extension",
        "load_inline",
        "torch_extensions",
    )

    for proc in psutil.process_iter(["name", "cmdline", "cwd"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmdline = proc.info.get("cmdline") or []
            cmd = " ".join(cmdline).lower()
            cwd = proc.info.get("cwd") or ""

            if not cmd and not name:
                continue
            if not any(k in name or k in cmd for k in keywords):
                continue
            if directory_str in cmd:
                return True
            if cwd:
                try:
                    cwd_path = str(Path(cwd).resolve())
                except Exception:
                    cwd_path = cwd
                if cwd_path == directory_str or cwd_path.startswith(directory_str + os.sep):
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:
            continue

    return False


def _kill_processes_using_directory(directory: Path) -> None:
    """Kill processes that are using files in the given directory."""
    if not PSUTIL_AVAILABLE:
        return  # Can't check processes without psutil
    
    try:
        directory_str = str(directory.resolve())
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check if process is Python and might be compiling
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline:
                        cmdline_str = ' '.join(cmdline)
                        # Check if it's loading CUDA extensions
                        if any(keyword in cmdline_str.lower() for keyword in 
                               ['cuda_extensions', 'load', 'extension', 'compile']):
                            # Check if process has open files in this directory
                            try:
                                open_files = proc.open_files()
                                for file_info in open_files:
                                    if directory_str in file_info.path:
                                        print(f"Killing stale process {proc.info['pid']} using {directory_str}")
                                        proc.kill()
                                        break
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception:
        pass  # Fail silently if we can't check processes


def cleanup_stale_so_builds(build_dir: Path, max_age_seconds: int = 300) -> bool:
    """Remove stale build artifacts if .so is missing but build.ninja exists.
    
    This handles the case where a previous compilation failed, leaving
    build.ninja but no .so file. torch.utils.cpp_extension.load() will
    try to import the .so directly without rebuilding, causing an error.
    
    Args:
        build_dir: Directory containing build artifacts
        
    Returns:
        True if cleanup was performed, False otherwise
    """
    import shutil
    
    if not build_dir.exists():
        return False
    
    ninja_file = build_dir / "build.ninja"
    so_files = list(build_dir.glob("*.so"))
    
    # If ninja exists but no .so files, the build may have failed OR may still be running.
    # Be conservative: do not delete an actively-building directory.
    if ninja_file.exists() and not so_files:
        now = time.time()

        # If any lock looks "fresh", assume an in-progress build and do nothing.
        lock_candidates = [build_dir / "lock", build_dir / ".ninja_lock"]
        lock_candidates.extend(build_dir.glob("*.lock"))
        for lock_file in lock_candidates:
            if not lock_file.exists():
                continue
            try:
                if now - lock_file.stat().st_mtime < max_age_seconds:
                    return False
            except OSError:
                return False

        # If build.ninja itself is recent, also assume a build is in progress.
        try:
            if now - ninja_file.stat().st_mtime < max_age_seconds:
                return False
        except OSError:
            return False

        try:
            # Avoid deleting the current working directory (can break subsequent shell/ninja calls).
            try:
                cwd = Path.cwd().resolve()
                bd = build_dir.resolve()
                if cwd == bd or bd in cwd.parents:
                    os.chdir(str(build_dir.parent))
            except Exception:
                pass

            shutil.rmtree(build_dir)
            build_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            pass  # Best effort cleanup
    
    return False


def ensure_clean_build_directory(build_dir: Path, max_lock_age_seconds: int = 300) -> None:
    """Ensure build directory is clean before starting a build.
    
    This is a convenience function that:
    1. Cleans stale builds (build.ninja exists but .so missing)
    2. Cleans stale locks
    3. Removes old build artifacts if needed
    
    Args:
        build_dir: Directory containing build artifacts
        max_lock_age_seconds: Consider locks older than this stale
    """
    # First, check for stale builds (ninja exists but .so missing)
    cleanup_stale_so_builds(build_dir, max_age_seconds=max_lock_age_seconds)
    
    # Then clean stale locks
    cleanup_stale_build_locks(build_dir, max_lock_age_seconds)
    
    # Optionally clean old .o files if directory is getting large
    if build_dir.exists():
        object_files = list(build_dir.glob("*.o"))
        if len(object_files) > 50:  # Too many object files
            # Clean old object files (keep .so files)
            for obj_file in object_files:
                try:
                    if time.time() - obj_file.stat().st_mtime > 3600:  # Older than 1 hour
                        obj_file.unlink()
                except OSError:
                    pass
