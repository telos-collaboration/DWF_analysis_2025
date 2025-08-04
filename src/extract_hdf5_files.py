import pandas as pd


def extract_spectrum_channel(csv_path, directory, channel):
    df = pd.read_csv(csv_path)
    row = df[df["name"] == directory]
    if row.empty:
        raise ValueError(f"Directory {directory} not found in the CSV.")
    mean = row[channel].values[0]
    error = row[f"{channel}_err"].values[0]
    return pd.Series({"mean": mean, "error": error})


def extract_fpi(csv_path, directory):
    df = pd.read_csv(csv_path)
    row = df[df["name"] == directory]
    if row.empty:
        raise ValueError(f"Directory {directory} not found in the CSV.")
    mean = row["fpi"].values[0]
    error = row["fpi_err"].values[0]
    return pd.Series({"mean": mean, "error": error})


def extract_Z_A(csv_path, directory):
    df = pd.read_csv(csv_path)
    row = df[df["name"] == directory]
    if row.empty:
        raise ValueError(f"Directory {directory} not found in the CSV.")
    mean = row["Z_A"].values[0]
    error = row["err_Z_A"].values[0]
    return pd.Series({"mean": mean, "error": error})


def extract_w0(csv_path, directory):
    df = pd.read_csv(csv_path)
    row = df[df["directory"] == directory]
    if row.empty:
        raise ValueError(f"Directory {directory} not found in the CSV.")
    mean = row["w_0"].values[0]
    error = row["w_0_error"].values[0]
    return pd.Series({"mean": mean, "error": error})


def fill_array(plateau_fits_path, WF_measurements_path):
    data = []
    # Loop through N (first dimension) and M (second dimension)
    for N in range(1, 5):
        for M in range(1, 7):
            # Specify directories
            directory = f"cfgs/ens{N}_m{M}"
            WF_directory = f"./WF_cfgs/ens{N}_m{M}/"

            # Extract data
            spectrum_channels = {
                "PS": extract_spectrum_channel(plateau_fits_path, directory, "g0g5"),
                "V": extract_spectrum_channel(plateau_fits_path, directory, "gi"),
                "T": extract_spectrum_channel(plateau_fits_path, directory, "g0gi"),
                "AV": extract_spectrum_channel(plateau_fits_path, directory, "g5gi"),
                "AT": extract_spectrum_channel(plateau_fits_path, directory, "g0g5gi"),
                "S": extract_spectrum_channel(plateau_fits_path, directory, "id"),
            }
            fpi_data = extract_fpi(plateau_fits_path, directory)
            ZA_data = extract_Z_A(plateau_fits_path, directory)
            w0_data = extract_w0(WF_measurements_path, WF_directory)

            # Create a row of data for this combination of N and M
            row = {
                "N": N,
                "M": M,
                "m_PS": spectrum_channels["PS"]["mean"],
                "m_PS_err": spectrum_channels["PS"]["error"],
                "m_V": spectrum_channels["V"]["mean"],
                "m_V_err": spectrum_channels["V"]["error"],
                "m_T": spectrum_channels["T"]["mean"],
                "m_T_err": spectrum_channels["T"]["error"],
                "m_AV": spectrum_channels["AV"]["mean"],
                "m_AV_err": spectrum_channels["AV"]["error"],
                "m_AT": spectrum_channels["AT"]["mean"],
                "m_AT_err": spectrum_channels["AT"]["error"],
                "m_S": spectrum_channels["S"]["mean"],
                "m_S_err": spectrum_channels["S"]["error"],
                "fpi": fpi_data["mean"],
                "fpi_err": fpi_data["error"],
                "Z_A": ZA_data["mean"],
                "Z_A_err": ZA_data["error"],
                "w0": w0_data["mean"],
                "w0_err": w0_data["error"],
            }

            # Add the row to the data list
            data.append(row)
    return data
