import pandas as pd
import os

def generate_excel_report(dataframe: pd.DataFrame, output_path: str = "outputs/telecom_audit_report.xlsx") -> str:
    """
    Exports audit dataframe to an Excel file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        dataframe.to_excel(writer, sheet_name="Site Analytics", index=False)
        
    return output_path
