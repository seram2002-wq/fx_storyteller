from io import BytesIO
from zipfile import ZipFile
import re

import pandas as pd
import requests

BIS_CBPOL_ZIP = "https://data.bis.org/static/bulk/WS_CBPOL_csv_col.zip"


def fetch_japan_policy_rate() -> pd.DataFrame:
    response = requests.get(BIS_CBPOL_ZIP, timeout=30)
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as archive:
        csv_name = archive.namelist()[0]
        with archive.open(csv_name) as f:
            data = pd.read_csv(f)

    row = data.loc[(data["FREQ"] == "D") & (data["REF_AREA"] == "JP")].iloc[0]

    date_columns = [col for col in data.columns if re.fullmatch(r"\d{4}-\d{2}-\d{2}", col)]

    result = (
        row[date_columns]
        .dropna()
        .rename_axis("date")
        .reset_index(name="policy_rate")
    )
    result["date"] = pd.to_datetime(result["date"])
    result["policy_rate"] = pd.to_numeric(result["policy_rate"])
    return result


def main() -> None:
    japan_policy_rate = fetch_japan_policy_rate()
    print(japan_policy_rate.tail())


if __name__ == "__main__":
    main()