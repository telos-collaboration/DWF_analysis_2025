from extract_hdf5_files import *

# Example usage extraction:
plateau_fits_path = "./plateau_fits_results.csv"
WF_measurements_path = "./WF_measurements.csv"
# Initialize an empty list to hold the data for all measurements
data = fill_array(plateau_fits_path, WF_measurements_path)
# Convert the list of dictionaries into a Pandas DataFrame
df = pd.DataFrame(data)

# Filter the DataFrame for N=2 and M=4
row = df[(df["N"] == 2) & (df["M"] == 4)]

# Extract the values for m_PS and m_PS_err
m_PS = row["m_PS"].values[0]
m_PS_err = row["m_PS_err"].values[0]

print(f"m_PS: {m_PS}, m_PS_err: {m_PS_err}")
