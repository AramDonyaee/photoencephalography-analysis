"""Excel helpers (MATLAB writematrix / writetable equivalents)."""
import os

import pandas as pd


def write_sheets(path, sheets, header=False):
    """One sheet per entry. Arrays are written without header/index like writematrix;
    DataFrames keep their header like writetable."""
    with pd.ExcelWriter(path) as writer:
        for name, data in sheets.items():
            frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            frame.to_excel(writer, sheet_name=str(name)[:31],
                           header=header or isinstance(data, pd.DataFrame), index=False)
    return os.path.basename(path)
