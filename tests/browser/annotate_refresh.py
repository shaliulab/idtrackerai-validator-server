"""Annotate tab in headless Chrome: the save of frame N landing after N+1 is open (Enter saves and opens the next
frame at once; the network slowed to 1.5 s so the save lands later).

    python tests/browser/annotate_refresh.py BASE_URL EXPERIMENT PROFILE_DIR F G

F, G: frames not saved yet, with a continuity proposal, whose next frame is in the same chunk. Expected:
"F+1 after save" shows "from saved F" (the open frame, untouched, read again: AnnotateTab.refreshOpen), and
"G+1 after save" is still "Edited" with its anchor from before (edits made on the open frame win). Run against a
test server (it saves frames F and G, writing the experiment's annotations database). WARMUP=<s>: load the page
once first (chunk data of the pre-download saved since is fetched again)."""
import json, os, sys, time
from idtrackerai_validator_server.pe_record import Chrome
BASE, EXP, profile, F, G = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
STATE = """(() => { const s = document.querySelector('section.bx-view'); const h = s?.querySelector('header')?.textContent || '';
  return s ? {frame: Number(s.dataset.frame), ready: s.dataset.ready, image: s.dataset.image, boxes: s.dataset.boxes,
  src: (h.match(/(continuity proposal|idtrackerai proposal|solver proposal|saved)/) || [])[1],
  from: (h.match(/from (saved|idtrackerai) frame ([\\d,]+)/) || []).slice(1).join(' '),
  status: document.querySelector('.sl-save')?.textContent} : null })()"""
KEY = "window.dispatchEvent(new KeyboardEvent('keydown', {key: %s, code: %s, shiftKey: %s, bubbles: true}))"
def key(c, k, shift=False):
    c.evaluate(KEY % (json.dumps(k), json.dumps(k), "true" if shift else "false"))
def wait(c, frame, t=120):
    t0 = time.time()
    while time.time() - t0 < t:
        st = c.evaluate(STATE)
        if st and st["frame"] == frame and st["ready"] == "1":
            return st
        time.sleep(0.05)
    raise TimeoutError(frame)
os.makedirs(profile, exist_ok=True)
c = Chrome(profile, viewport=(1600, 1000))
try:
    c.open_page((1600, 1000))
    c.send("Network.enable")
    slow = lambda ms: c.send("Network.emulateNetworkConditions", {"offline": False, "latency": ms, "downloadThroughput": -1, "uploadThroughput": -1})
    if os.environ.get("WARMUP"):   # a first load: kept chunk data saved elsewhere since is fetched again
        c.send("Page.navigate", {"url": f"{BASE}/?tab=annotate&experiment={EXP}&frame={F}"})
        wait(c, F); time.sleep(float(os.environ["WARMUP"]))
    # 1. untouched N+1
    c.send("Page.navigate", {"url": f"{BASE}/?tab=annotate&experiment={EXP}&frame={F}"})
    wait(c, F); time.sleep(float(os.environ.get("SETTLE", 4)))                         # the frames ahead are loaded (their proposals from before the save)
    key(c, "Tab"); key(c, "u"); time.sleep(0.4)        # F changed (a label removed): not saved as its proposal
    print("F edited         ", c.evaluate(STATE))
    slow(1500)                                         # the save lands 1.5 s later
    key(c, "Enter")                                    # save F, open F+1 at once
    print("F+1 at open      ", wait(c, F + 1))
    time.sleep(4)
    after_f = c.evaluate(STATE)
    print("F+1 after save   ", after_f)
    slow(0)
    # 2. N+1 edited before the save lands
    c.send("Page.navigate", {"url": f"{BASE}/?tab=annotate&experiment={EXP}&frame={G}"})
    wait(c, G); time.sleep(4)
    key(c, "Tab"); key(c, "u"); time.sleep(0.4)
    slow(1500)
    key(c, "Enter")
    wait(c, G + 1)
    key(c, "Tab"); key(c, "u")                         # G+1 edited before the save of G lands
    time.sleep(0.2)
    print("G+1 edited       ", c.evaluate(STATE))
    time.sleep(4)
    after_g = c.evaluate(STATE)
    print("G+1 after save   ", after_g)
finally:
    c.close()

ok_f = after_f["from"] == f"saved {F:,}"
ok_g = (after_g["status"] or "").startswith("Edited")
print(f"untouched open frame refreshed: {'PASS' if ok_f else 'FAIL'}; edited open frame kept: {'PASS' if ok_g else 'FAIL'}")
sys.exit(0 if ok_f and ok_g else 1)
