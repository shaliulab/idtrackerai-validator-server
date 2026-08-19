import subprocess
import shlex
import pathlib
import os.path
import sys
python_bin=os.path.join(sys.prefix, "bin", "python")

def main():
    here=pathlib.Path(__file__).parent.absolute()
    executable=os.path.join(here, "app.py")

    cmd=f"""
        {python_bin} {executable}  --host "{os.environ.get('CVAT_HOST', 3000)}" --port {os.environ.get('BACKEND_PORT', 5000)}
    """
    print(cmd)
 
    cmd=shlex.split(cmd)

    p=subprocess.Popen(cmd)
    p.communicate()


if __name__ == "__main__":
    main()
