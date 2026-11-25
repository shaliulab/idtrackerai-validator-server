import pandas as pd
v = pd.read_json("variants.jsonl", lines=True)
v["is12"] = v.exp.str.contains("_12X_")
v["allok_n"] = v.cold_allok * v.stretches / 100; v["stepok_n"] = v.step_ok * v.frames / 100
g = v.groupby(["is12", "variant"]).agg(n=("exp","size"), frames=("frames","sum"), step_edits=("step_edits","sum"), cold_edits=("cold_edits","sum"), stretches=("stretches","sum"), a=("allok_n","sum"), s=("stepok_n","sum"))
g["step_ok%"] = g.s / g.frames * 100; g["cold_allok%"] = g.a / g.stretches * 100
print(g.drop(columns=["a","s"]).round(1).to_string())
