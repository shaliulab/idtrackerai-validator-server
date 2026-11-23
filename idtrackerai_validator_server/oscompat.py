"""What differs when the server runs on Windows instead of Linux.

The Unix modules and calls the server uses (fcntl.flock, process groups, sh, /proc, find)
do not exist on Windows; these helpers do the same with what Windows has, and the same as
before on Linux / macOS:

- lock / unlock: an exclusive lock on an open file (fcntl.flock; msvcrt.locking of its
  first byte on Windows, a mandatory lock: lock a separate lock file, never one written
  through another handle).
- new_session(): Popen keyword arguments starting a process in its own session
  (start_new_session; CREATE_NEW_PROCESS_GROUP on Windows), so it survives the server and
  can be stopped with its children.
- kill_tree(pid, sig): the process group of such a process (os.killpg; on Windows the
  process and its descendants, with psutil).
- with_exit_code(cmd): cmd as a command line appending "[exit code N]" to its output
  (sh -c; a python wrapper on Windows).
- posix_relpath: os.path.relpath with "/" separators (experiment names are always
  written with "/", e.g. FlyHostel1/6X/2023-08-31_13-00-00).
"""
import os
import shlex
import signal
import sys
import time

IS_WINDOWS = os.name == "nt"
SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)
HAS_PROC = os.path.isdir("/proc/self")

if IS_WINDOWS:
    import msvcrt
    import subprocess
else:
    import fcntl


def lock(handle, blocking=True):
    """Exclusive lock on an open file; OSError when not blocking and held elsewhere."""
    if not IS_WINDOWS:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        return
    handle.seek(0)
    while True:
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            if not blocking:
                raise
            time.sleep(0.05)


def unlock(handle):
    if not IS_WINDOWS:
        fcntl.flock(handle, fcntl.LOCK_UN)
        return
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass


def new_session():
    """Popen keyword arguments: the process in its own session / process group."""
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_tree(pid, sig=signal.SIGTERM):
    """Signal the process group of pid (started with new_session()); on Windows terminate pid
    and its descendants. ProcessLookupError when pid is gone."""
    if not IS_WINDOWS:
        os.killpg(os.getpgid(pid), sig)
        return
    import psutil
    try:
        parent = psutil.Process(pid)
        processes = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        raise ProcessLookupError(pid)
    except psutil.Error as error:
        raise PermissionError(str(error))
    for process in processes:
        try:
            process.kill() if sig == SIGKILL else process.terminate()
        except psutil.Error:
            pass


_EXIT_CODE_WRAPPER = (
    "import subprocess, sys\n"
    "try:\n"
    "    rc = subprocess.call(sys.argv[1:])\n"
    "except OSError as error:\n"
    "    print(error, flush=True)\n"
    "    rc = 127\n"
    "print(flush=True)\n"
    "print('[exit code %d]' % rc, flush=True)\n"
)


def with_exit_code(cmd):
    """argv running cmd and then writing "\\n[exit code N]" to the same output."""
    if IS_WINDOWS:
        return [sys.executable, "-c", _EXIT_CODE_WRAPPER, *cmd]
    line = " ".join(shlex.quote(part) for part in cmd)
    return ["sh", "-c", f'{line}; rc=$?; echo; echo "[exit code $rc]"']


def posix_relpath(path, start=os.curdir):
    return os.path.relpath(path, start).replace(os.sep, "/")


def to_posix(path):
    """A relative path with "/" separators (no-op on Linux)."""
    return path.replace("\\", "/") if IS_WINDOWS else path


def env_executable(prefix, name):
    """An executable of a conda env: <prefix>/bin/<name>, on Windows <prefix>\\Scripts\\<name>.exe."""
    if IS_WINDOWS:
        return os.path.join(prefix, "Scripts", name + ".exe")
    return os.path.join(prefix, "bin", name)


def env_path(prefix, path=None):
    """PATH with a conda env's executables first (bin/; on Windows the env, Scripts and Library\\bin)."""
    path = os.environ.get("PATH", "") if path is None else path
    if IS_WINDOWS:
        folders = [prefix, os.path.join(prefix, "Scripts"), os.path.join(prefix, "Library", "bin")]
    else:
        folders = [os.path.join(prefix, "bin")]
    return os.pathsep.join(folders + [path])
