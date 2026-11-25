import json, sys, cv2, numpy as np
sys.path.insert(0, "/home/vibflysleep/orca/workspaces/idtrackerai-validator-server/cowfish")
from idtrackerai_validator_server.lite_store import LiteVideoCapture
from evallib import Exp, C
import methods as M
from repo_v2_eval import RV2
from extract import basedir_of
ex = json.load(open("examples.json"))
pick = [next(o for o in ex if o["exp"].startswith("FlyHostel1_12X") and o["kinds"] == ["missing_fly"]),
        next(o for o in ex if o["leftover"]), next(o for o in ex if o["kinds"] == ["wrong_label"])]
titles = ["fly not detected (not at a notch)", "fly reappearing away from where it was lost", "flies crossing without identity"]
for o, title in zip(pick, titles):
    e = Exp(o["exp"]); f = o["f"]
    cap = LiteVideoCapture(f"/flyhostel_data/videos/{basedir_of(o['exp'])}/metadata.yaml")
    img, _ = cap.get_image(f)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    rows = e.rows(f); g = e.gt(f)
    res = {}
    for name, cls in (("v1 (current)", C.Run), ("v2", RV2)):
        run, _ = M.make_run(e, f, f, run_cls=cls); run.extend(f, e.chunk_rows(f // e.cs)); res[name] = run.results[f]
    # focus: the first error location
    k, j = o["e1"][0]
    if k == "missing_fly":
        b = next(b for b in e.boxes[f] if b["label"] == str(j)); cx, cy = (b["x0"]+b["x1"])/2, (b["y0"]+b["y1"])/2
    else:
        cx, cy = rows[j][0], rows[j][1]
    R = 160; x0, y0 = int(max(0, cx - R)), int(max(0, cy - R)); x1, y1 = x0 + 2 * R, y0 + 2 * R
    panels = []
    def crop(draw):
        im = img.copy(); draw(im); c = im[y0:y1, x0:x1]; return cv2.resize(c, (480, 480), interpolation=cv2.INTER_NEAREST)
    def put(im, t, x, y, col):
        cv2.putText(im, t, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4); cv2.putText(im, t, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    def person(im):
        for b in e.boxes[f]:
            cv2.rectangle(im, (int(b["x0"]), int(b["y0"])), (int(b["x1"]), int(b["y1"])), (0, 200, 0), 2); put(im, b["label"], b["x0"], b["y0"] - 4, (0, 255, 0))
    panels.append(("person (saved)", crop(person)))
    for name, r in res.items():
        def draw(im, r=r):
            for (x, y, _), (lab, note, conf) in zip(rows, r["rows"]):
                col = (0, 200, 255) if lab is None else (255, 200, 0)
                cv2.circle(im, (int(x), int(y)), 6, col, -1); put(im, "?" if lab is None else str(lab), x + 8, y - 8, col)
            for lab, x, y, _ in r["ghosts"]:
                cv2.rectangle(im, (int(x) - 35, int(y) - 35), (int(x) + 35, int(y) + 35), (255, 0, 255), 2); put(im, str(lab), x - 35, y - 40, (255, 0, 255))
        panels.append((name, crop(draw)))
    out = []
    for name, p in panels:
        p = p.copy(); cv2.rectangle(p, (0, 0), (480, 34), (255, 255, 255), -1); cv2.putText(p, name, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2); out.append(p)
    strip = np.hstack(out)
    head = np.full((40, strip.shape[1], 3), 255, np.uint8)
    cv2.putText(head, f"{o['exp']} frame {f}: {title} (v1 errors: {', '.join(k for k, _ in o['e1'])})", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    path = f"../examples/{o['exp']}_{f}.png"; cv2.imwrite(path, np.vstack([head, strip])); print(path)
