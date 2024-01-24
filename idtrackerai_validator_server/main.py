import subprocess
import shlex
import pathlib
import os.path

def main():
    here=pathlib.Path(__file__).parent.absolute()
    executable=os.path.join(here, "app.py")


    cmd=shlex.split(
        f"""
        python {executable} -m flask run --host \\"0.0.0.0\\" --port 5000
        """
    )

    p=subprocess.Popen(cmd)
    p.communicate()


if __name__ == "__main__":
    main()
