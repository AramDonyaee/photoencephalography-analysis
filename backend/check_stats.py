import pandas as pd
import mixed_models

df = pd.read_excel(r"C:../../EntropyUPEsDataset.xlsx")   # same path as before
r = mixed_models.run(df, "fit", {})
print("singular:", r["singular"])
for row in r["anova"]:
    print(row)
print(r["posthoc"][0])

# variance components
prepared = mixed_models.prepare(df)
model, res, dv, fixed = mixed_models._fit(
    prepared, "Entropy ~ Electrode * Task + (1 | Subject) + (1 | Subject:Task)")
print("Subject var:", res.cov_re)
print("Subject:Task var:", res.vcomp)
print("Residual var:", res.scale)