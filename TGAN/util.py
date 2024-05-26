import os
import numpy as np
import pandas as pd
from random import randint


def load_data(
    file_path,
    break_to_smaller=False,
    break_size=60,
    leave_out_problematic_features=True,
    cutoff_data=True,
):
    DF = pd.read_excel(file_path)
    DF["datetime"] = DF["Date"] + " " + DF["Time"]
    DF["datetime"] = pd.to_datetime(DF["datetime"])
    DF.set_index("datetime", inplace=True)
    if leave_out_problematic_features and "df29" in file_path:
        DF = DF.drop(
            columns=[
                "Crave_Food",
                "Cravings",
                "With how many people",
                "Eating",
                "Eating_healthy",
            ]
        )
    DF.drop(["Date", "Time", "Duration"], axis=1, inplace=True)
    DF.sort_index(inplace=True)
    #
    DF=DF.iloc[:,:8]
    # print("Number of unique Names:", DF["Name"].nunique())
    # print("Number of columns:", len(DF.columns))
    # print("Number of rows:", len(DF))
    # print("Name of columns:", DF.columns)

    samples = []
    if break_to_smaller:
        for k in DF.Name.unique().tolist():
            temp_df = DF[DF.Name == k]
            # print(len(temp_df))
            # print(f"{len(temp_df)//break_size} samples")
            for i in range(len(temp_df) // break_size):
                samples.append(
                    temp_df.iloc[i * break_size : (i + 1) * break_size]
                    .drop(columns="Name")
                    .values
                )
            remaining = temp_df.iloc[(i + 1) * break_size :].drop(columns="Name")
            pad_length = break_size - len(remaining)
            padding_df = pd.DataFrame(
                -1, index=np.arange(pad_length), columns=remaining.columns
            )
            filled_df = pd.concat([remaining, padding_df], ignore_index=True)
            samples.append(filled_df.values)
    else:
        if cutoff_data:
            for k in DF.Name.unique().tolist():
                if len(DF[DF["Name"] == k]) >= 120:
                    temp = DF[DF["Name"] == k]
                    samples.append(temp.iloc[:120, :].drop(columns="Name").values)
        else:
            for k in DF.Name.unique().tolist():
                temp = DF[DF["Name"] == k]
                samples.append(temp.drop(columns="Name").values)

    # Print sample statistics
    print("Number of samples:", len(samples))

    samples = np.array(samples)

    print("Shape of _samples:", samples.shape)
    return samples
