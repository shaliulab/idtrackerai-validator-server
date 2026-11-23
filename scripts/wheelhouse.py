"""Make a folder of wheels installable offline from GitHub: `pip install --find-links <folder> <name>`.

The lab's packages declare each other as direct references (`flyhostel[idtrackerai] @ git+https://…`,
`ethoscopy @ file:///…/libraries/ethoscopy`, …), which pip would fetch (git, ssh keys) or cannot
resolve at all (a build machine's paths). This rewrites every `name @ url` requirement of the wheels
in the folder into `name==<version of that wheel in the folder>`, so pip takes them from the folder.
A direct reference to a package that is not in the folder is an error.

    python scripts/wheelhouse.py WHEEL_DIR

Uses `wheel unpack` / `wheel pack` (pip install wheel), which rewrite RECORD.
"""
import email.parser
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_versions(folder):
    versions = {}
    for path in folder.glob("*.whl"):
        name, version = path.name.split("-")[:2]
        versions[canonical(name)] = version
    return versions


REQ = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*(?P<extras>\[[^\]]*\])?\s*@\s*(?P<url>[^;]+?)\s*(?P<marker>;.*)?$")


def rewrite(requirement, versions):
    match = REQ.match(requirement)
    if match is None:
        return requirement
    name = match["name"]
    version = versions.get(canonical(name))
    if version is None:
        raise SystemExit("{} is required as {} but has no wheel in the folder".format(name, match["url"]))
    out = "{}{}=={}".format(name, match["extras"] or "", version)
    if match["marker"]:
        out += " " + match["marker"]
    return out


def fix_wheel(path, versions):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.check_call([sys.executable, "-m", "wheel", "unpack", "-d", tmp, str(path)],
                              stdout=subprocess.DEVNULL)
        root = next(pathlib.Path(tmp).iterdir())
        metadata = next(root.glob("*.dist-info/METADATA"))
        lines = metadata.read_text(encoding="utf-8").split("\n")
        changed = []
        for i, line in enumerate(lines):
            if line.startswith("Requires-Dist:"):
                old = line[len("Requires-Dist:"):].strip()
                new = rewrite(old, versions)
                if new != old:
                    lines[i] = "Requires-Dist: " + new
                    changed.append((old, new))
        if not changed:
            return []
        metadata.write_text("\n".join(lines), encoding="utf-8")
        out = pathlib.Path(tmp) / "out"
        out.mkdir()
        subprocess.check_call([sys.executable, "-m", "wheel", "pack", "-d", str(out), str(root)],
                              stdout=subprocess.DEVNULL)
        new = next(out.glob("*.whl"))
        path.unlink()
        shutil.move(str(new), str(path.parent / new.name))
        return changed


def main():
    folder = pathlib.Path(sys.argv[1])
    versions = wheel_versions(folder)
    for path in sorted(folder.glob("*.whl")):
        for old, new in fix_wheel(path, versions):
            print("{}: {}  ->  {}".format(path.name, old, new))
    # check: no direct reference left
    for path in folder.glob("*.whl"):
        import zipfile
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
            meta = email.parser.Parser().parsestr(z.read(name).decode("utf-8"))
            left = [r for r in meta.get_all("Requires-Dist") or [] if REQ.match(r)]
            if left:
                raise SystemExit("{} still has direct references: {}".format(path.name, left))


if __name__ == "__main__":
    main()
