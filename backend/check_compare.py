import pandas as pd
import mixed_models

df = pd.read_excel(r"C:../../EntropyUPEsDataset.xlsx")
r = mixed_models.run(df, "compare", {})
for row in r["compare"]:
    print(row["Model"], row["AIC"], row["BIC"], row["LogLik"], row["note"])