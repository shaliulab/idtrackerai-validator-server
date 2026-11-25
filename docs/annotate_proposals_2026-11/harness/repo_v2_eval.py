import sys, json
from evallib import C, S
import var_eval
class RV2(C.RunV2):
    exp = None
    def step(self, fn, rows):
        full = self.exp.full(fn)
        return C.RunV2.step(self, fn, [(r[2], r[3], int(r[7]), int(r[4]), r[5]) for r in full])
if __name__ == "__main__":
    r = var_eval.evaluate(sys.argv[1], "repo_RunV2", RV2)
    open(f"{S}/work/variants.jsonl", "a").write(json.dumps(r, default=float) + "\n"); print(r)
