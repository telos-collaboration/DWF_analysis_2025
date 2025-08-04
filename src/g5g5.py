import xml.etree.ElementTree as ET
import glob
import os
import re
import numpy as np


def read_g5g5(directory):
    # Function to extract the numeric part from the filename
    def extract_number(filename):
        match = re.search(r"pt_ll\.(\d+)\.xml", filename)
        return int(match.group(1)) if match else None

    # Get a list of all files matching the pattern pt_ll.N.xml
    file_list = glob.glob(os.path.join(directory, "pt_ll.*.xml"))

    # Sort the file list based on the numeric part of the filenames
    file_list.sort(key=extract_number)

    # Initialize an empty list to hold the correlation data for all files
    g5g5_data = []

    # Iterate over each file in the sorted list
    for file_index, file_name in enumerate(file_list):
        # Parse the XML file
        tree = ET.parse(file_name)
        root = tree.getroot()

        # Initialize an empty list to hold the correlation data for the current file
        file_g5g5_data = []

        # Find all <elem> elements
        for elem in root.findall(".//elem"):
            # Find gamma_snk and gamma_src elements
            gamma_snk_elem = elem.find("gamma_snk")
            gamma_src_elem = elem.find("gamma_src")

            # Check if both gamma_snk and gamma_src are present and are equal to "Gamma5"
            if (
                gamma_snk_elem is not None
                and gamma_src_elem is not None
                and gamma_snk_elem.text.strip() == "Gamma5"
                and gamma_src_elem.text.strip() == "Gamma5"
            ):
                # Extract the correlation data
                corr_data = []
                for corr_elem in elem.findall(".//corr/elem"):
                    text_content = corr_elem.text.strip()
                    real_part, imag_part = text_content.strip("()").split(",")
                    corr_data.append(float(real_part))
                file_g5g5_data.append(corr_data)

        # Append the correlation data for the current file to the main list
        g5g5_data.append(file_g5g5_data)
    # print(g5g5_data)
    # Convert the list of lists into a 3D array (list of lists of lists)
    # You can use a library like numpy if you prefer an actual array structure
    g5g5_correlators_array = np.array(g5g5_data)

    # Reshape the array from (165, 1, 8) to (165, 8)
    g5g5_correlators_array = np.squeeze(g5g5_correlators_array, axis=1)

    return g5g5_correlators_array


def read_gi(directory):
    directions = ["X", "Y", "Z"]
    for dir in directions:
        # Function to extract the numeric part from the filename
        def extract_number(filename):
            match = re.search(r"pt_ll\.(\d+)\.xml", filename)
            return int(match.group(1)) if match else None

        # Get a list of all files matching the pattern pt_ll.N.xml
        file_list = glob.glob(os.path.join(directory, "pt_ll.*.xml"))

        # Sort the file list based on the numeric part of the filenames
        file_list.sort(key=extract_number)

        # Initialize an empty list to hold the correlation data for all files
        g5g5_data = []

        # Iterate over each file in the sorted list
        for file_index, file_name in enumerate(file_list):
            # Parse the XML file
            tree = ET.parse(file_name)
            root = tree.getroot()

            # Initialize an empty list to hold the correlation data for the current file
            file_g5g5_data = []

            # Find all <elem> elements
            for elem in root.findall(".//elem"):
                # Find gamma_snk and gamma_src elements
                gamma_snk_elem = elem.find("gamma_snk")
                gamma_src_elem = elem.find("gamma_src")

                # Check if both gamma_snk and gamma_src are present and are equal to "Gamma5"
                if (
                    gamma_snk_elem is not None
                    and gamma_src_elem is not None
                    and gamma_snk_elem.text.strip() == f"Gamma{dir}"
                    and gamma_src_elem.text.strip() == f"Gamma{dir}"
                ):
                    # Extract the correlation data
                    corr_data = []
                    for corr_elem in elem.findall(".//corr/elem"):
                        text_content = corr_elem.text.strip()
                        real_part, imag_part = text_content.strip("()").split(",")
                        corr_data.append(float(real_part))
                    file_g5g5_data.append(corr_data)

            # Append the correlation data for the current file to the main list
            g5g5_data.append(file_g5g5_data)

        # Convert the list of lists into a 3D array (list of lists of lists)
        # You can use a library like numpy if you prefer an actual array structure
        g5g5_correlators_array = np.array(g5g5_data)

        # Reshape the array from (165, 1, 8) to (165, 8)
        g5g5_correlators_array = np.squeeze(g5g5_correlators_array, axis=1)
        if dir == "X":
            g5g5_correlators_array_X = g5g5_correlators_array
        elif dir == "Y":
            g5g5_correlators_array_Y = g5g5_correlators_array
        elif dir == "Z":
            g5g5_correlators_array_Z = g5g5_correlators_array

    g5g5_correlators_array = (
        g5g5_correlators_array_X + g5g5_correlators_array_Y + g5g5_correlators_array_Z
    ) / 3
    return g5g5_correlators_array


def read_g0gi(directory):
    directions = ["X", "Y", "Z"]
    for dir in directions:
        # Function to extract the numeric part from the filename
        def extract_number(filename):
            match = re.search(r"pt_ll\.(\d+)\.xml", filename)
            return int(match.group(1)) if match else None

        # Get a list of all files matching the pattern pt_ll.N.xml
        file_list = glob.glob(os.path.join(directory, "pt_ll.*.xml"))

        # Sort the file list based on the numeric part of the filenames
        file_list.sort(key=extract_number)

        # Initialize an empty list to hold the correlation data for all files
        g5g5_data = []

        # Iterate over each file in the sorted list
        for file_index, file_name in enumerate(file_list):
            # Parse the XML file
            tree = ET.parse(file_name)
            root = tree.getroot()

            # Initialize an empty list to hold the correlation data for the current file
            file_g5g5_data = []

            # Find all <elem> elements
            for elem in root.findall(".//elem"):
                # Find gamma_snk and gamma_src elements
                gamma_snk_elem = elem.find("gamma_snk")
                gamma_src_elem = elem.find("gamma_src")

                # Check if both gamma_snk and gamma_src are present and are equal to "Gamma5"
                if (
                    gamma_snk_elem is not None
                    and gamma_src_elem is not None
                    and gamma_snk_elem.text.strip() == f"Sigma{dir}T"
                    and gamma_src_elem.text.strip() == f"Sigma{dir}T"
                ):
                    # Extract the correlation data
                    corr_data = []
                    for corr_elem in elem.findall(".//corr/elem"):
                        text_content = corr_elem.text.strip()
                        real_part, imag_part = text_content.strip("()").split(",")
                        corr_data.append(float(real_part))
                    file_g5g5_data.append(corr_data)

            # Append the correlation data for the current file to the main list
            g5g5_data.append(file_g5g5_data)

        # Convert the list of lists into a 3D array (list of lists of lists)
        # You can use a library like numpy if you prefer an actual array structure
        g5g5_correlators_array = np.array(g5g5_data)

        # Reshape the array from (165, 1, 8) to (165, 8)
        g5g5_correlators_array = np.squeeze(g5g5_correlators_array, axis=1)
        if dir == "X":
            g5g5_correlators_array_X = g5g5_correlators_array
        elif dir == "Y":
            g5g5_correlators_array_Y = g5g5_correlators_array
        elif dir == "Z":
            g5g5_correlators_array_Z = g5g5_correlators_array

    g5g5_correlators_array = (
        g5g5_correlators_array_X + g5g5_correlators_array_Y + g5g5_correlators_array_Z
    ) / 3
    return g5g5_correlators_array


def read_g5gi(directory):
    directions = ["X", "Y", "Z"]
    for dir in directions:
        # Function to extract the numeric part from the filename
        def extract_number(filename):
            match = re.search(r"pt_ll\.(\d+)\.xml", filename)
            return int(match.group(1)) if match else None

        # Get a list of all files matching the pattern pt_ll.N.xml
        file_list = glob.glob(os.path.join(directory, "pt_ll.*.xml"))
        # Sort the file list based on the numeric part of the filenames
        file_list.sort(key=extract_number)

        # Initialize an empty list to hold the correlation data for all files
        g5g5_data = []

        # Iterate over each file in the sorted list
        for file_index, file_name in enumerate(file_list):
            # Parse the XML file
            tree = ET.parse(file_name)
            root = tree.getroot()

            # Initialize an empty list to hold the correlation data for the current file
            file_g5g5_data = []

            # Find all <elem> elements
            for elem in root.findall(".//elem"):
                # Find gamma_snk and gamma_src elements
                gamma_snk_elem = elem.find("gamma_snk")
                gamma_src_elem = elem.find("gamma_src")

                # Check if both gamma_snk and gamma_src are present and are equal to "Gamma5"
                if (
                    gamma_snk_elem is not None
                    and gamma_src_elem is not None
                    and gamma_snk_elem.text.strip() == f"Gamma{dir}Gamma5"
                    and gamma_src_elem.text.strip() == f"Gamma{dir}Gamma5"
                ):
                    # Extract the correlation data
                    corr_data = []
                    for corr_elem in elem.findall(".//corr/elem"):
                        text_content = corr_elem.text.strip()
                        real_part, imag_part = text_content.strip("()").split(",")
                        corr_data.append(float(real_part))
                    file_g5g5_data.append(corr_data)

            # Append the correlation data for the current file to the main list
            g5g5_data.append(file_g5g5_data)

        # Convert the list of lists into a 3D array (list of lists of lists)
        # You can use a library like numpy if you prefer an actual array structure
        g5g5_correlators_array = np.array(g5g5_data)

        # Reshape the array from (165, 1, 8) to (165, 8)
        g5g5_correlators_array = np.squeeze(g5g5_correlators_array, axis=1)
        if dir == "X":
            g5g5_correlators_array_X = g5g5_correlators_array
        elif dir == "Y":
            g5g5_correlators_array_Y = g5g5_correlators_array
        elif dir == "Z":
            g5g5_correlators_array_Z = g5g5_correlators_array

    g5g5_correlators_array = (
        g5g5_correlators_array_X + g5g5_correlators_array_Y + g5g5_correlators_array_Z
    ) / 3
    return g5g5_correlators_array


def read_g0g5gi(directory):
    directions = ["X", "Y", "Z"]
    for dir in directions:
        # Function to extract the numeric part from the filename
        def extract_number(filename):
            match = re.search(r"pt_ll\.(\d+)\.xml", filename)
            return int(match.group(1)) if match else None

        # Get a list of all files matching the pattern pt_ll.N.xml
        file_list = glob.glob(os.path.join(directory, "pt_ll.*.xml"))

        # Sort the file list based on the numeric part of the filenames
        file_list.sort(key=extract_number)

        # Initialize an empty list to hold the correlation data for all files
        g5g5_data = []

        # Iterate over each file in the sorted list
        for file_index, file_name in enumerate(file_list):
            # Parse the XML file
            tree = ET.parse(file_name)
            root = tree.getroot()

            # Initialize an empty list to hold the correlation data for the current file
            file_g5g5_data = []

            # Find all <elem> elements
            for elem in root.findall(".//elem"):
                # Find gamma_snk and gamma_src elements
                gamma_snk_elem = elem.find("gamma_snk")
                gamma_src_elem = elem.find("gamma_src")

                # Check if both gamma_snk and gamma_src are present and are equal to "Gamma5"
                if (
                    gamma_snk_elem is not None
                    and gamma_src_elem is not None
                    and gamma_snk_elem.text.strip() == f"Gamma{dir}Gamma5"
                    and gamma_src_elem.text.strip() == f"SigmaT{dir}Gamma5"
                ):
                    # Extract the correlation data
                    corr_data = []
                    for corr_elem in elem.findall(".//corr/elem"):
                        text_content = corr_elem.text.strip()
                        real_part, imag_part = text_content.strip("()").split(",")
                        corr_data.append(float(real_part))
                    file_g5g5_data.append(corr_data)

            # Append the correlation data for the current file to the main list
            g5g5_data.append(file_g5g5_data)

        # Convert the list of lists into a 3D array (list of lists of lists)
        # You can use a library like numpy if you prefer an actual array structure
        g5g5_correlators_array = np.array(g5g5_data)

        # Reshape the array from (165, 1, 8) to (165, 8)
        g5g5_correlators_array = np.squeeze(g5g5_correlators_array, axis=1)
        if dir == "X":
            g5g5_correlators_array_X = g5g5_correlators_array
        elif dir == "Y":
            g5g5_correlators_array_Y = g5g5_correlators_array
        elif dir == "Z":
            g5g5_correlators_array_Z = g5g5_correlators_array

    g5g5_correlators_array = (
        g5g5_correlators_array_X + g5g5_correlators_array_Y + g5g5_correlators_array_Z
    ) / 3
    return g5g5_correlators_array


def read_g0g5(directory):
    # Function to extract the numeric part from the filename
    def extract_number(filename):
        match = re.search(r"pt_ll\.(\d+)\.xml", filename)
        return int(match.group(1)) if match else None

    # Get a list of all files matching the pattern pt_ll.N.xml
    file_list = glob.glob(os.path.join(directory, "pt_ll.*.xml"))

    # Sort the file list based on the numeric part of the filenames
    file_list.sort(key=extract_number)

    # Initialize an empty list to hold the correlation data for all files
    g5g5_data = []

    # Iterate over each file in the sorted list
    for file_index, file_name in enumerate(file_list):
        # Parse the XML file
        tree = ET.parse(file_name)
        root = tree.getroot()

        # Initialize an empty list to hold the correlation data for the current file
        file_g5g5_data = []

        # Find all <elem> elements
        for elem in root.findall(".//elem"):
            # Find gamma_snk and gamma_src elements
            gamma_snk_elem = elem.find("gamma_snk")
            gamma_src_elem = elem.find("gamma_src")

            # Check if both gamma_snk and gamma_src are present and are equal to "Gamma5"
            if (
                gamma_snk_elem is not None
                and gamma_src_elem is not None
                and gamma_snk_elem.text.strip() == "GammaTGamma5"
                and gamma_src_elem.text.strip() == "GammaTGamma5"
            ):
                # Extract the correlation data
                corr_data = []
                for corr_elem in elem.findall(".//corr/elem"):
                    text_content = corr_elem.text.strip()
                    real_part, imag_part = text_content.strip("()").split(",")
                    corr_data.append(float(real_part))
                file_g5g5_data.append(corr_data)

        # Append the correlation data for the current file to the main list
        g5g5_data.append(file_g5g5_data)

    # Convert the list of lists into a 3D array (list of lists of lists)
    # You can use a library like numpy if you prefer an actual array structure
    g5g5_correlators_array = np.array(g5g5_data)

    # Reshape the array from (165, 1, 8) to (165, 8)
    g5g5_correlators_array = np.squeeze(g5g5_correlators_array, axis=1)

    return g5g5_correlators_array


def read_g0g5_g5(directory):
    # Function to extract the numeric part from the filename
    def extract_number(filename):
        match = re.search(r"pt_ll\.(\d+)\.xml", filename)
        return int(match.group(1)) if match else None

    # Get a list of all files matching the pattern pt_ll.N.xml
    file_list = glob.glob(os.path.join(directory, "pt_ll.*.xml"))

    # Sort the file list based on the numeric part of the filenames
    file_list.sort(key=extract_number)

    # Initialize an empty list to hold the correlation data for all files
    g5g5_data = []

    # Iterate over each file in the sorted list
    for file_index, file_name in enumerate(file_list):
        # Parse the XML file
        tree = ET.parse(file_name)
        root = tree.getroot()

        # Initialize an empty list to hold the correlation data for the current file
        file_g5g5_data = []

        # Find all <elem> elements
        for elem in root.findall(".//elem"):
            # Find gamma_snk and gamma_src elements
            gamma_snk_elem = elem.find("gamma_snk")
            gamma_src_elem = elem.find("gamma_src")

            # Check if both gamma_snk and gamma_src are present and are equal to "Gamma5"
            if (
                gamma_snk_elem is not None
                and gamma_src_elem is not None
                and gamma_snk_elem.text.strip() == "GammaTGamma5"
                and gamma_src_elem.text.strip() == "Gamma5"
            ):
                # Extract the correlation data
                corr_data = []
                for corr_elem in elem.findall(".//corr/elem"):
                    text_content = corr_elem.text.strip()
                    real_part, imag_part = text_content.strip("()").split(",")
                    corr_data.append(float(real_part))
                file_g5g5_data.append(corr_data)

        # Append the correlation data for the current file to the main list
        g5g5_data.append(file_g5g5_data)

    # Convert the list of lists into a 3D array (list of lists of lists)
    # You can use a library like numpy if you prefer an actual array structure
    g5g5_correlators_array = np.array(g5g5_data)

    # Reshape the array from (165, 1, 8) to (165, 8)
    g5g5_correlators_array = np.squeeze(g5g5_correlators_array, axis=1)

    return g5g5_correlators_array


def read_id(directory):
    # Function to extract the numeric part from the filename
    def extract_number(filename):
        match = re.search(r"pt_ll\.(\d+)\.xml", filename)
        return int(match.group(1)) if match else None

    # Get a list of all files matching the pattern pt_ll.N.xml
    file_list = glob.glob(os.path.join(directory, "pt_ll.*.xml"))

    # Sort the file list based on the numeric part of the filenames
    file_list.sort(key=extract_number)

    # Initialize an empty list to hold the correlation data for all files
    g5g5_data = []

    # Iterate over each file in the sorted list
    for file_index, file_name in enumerate(file_list):
        # Parse the XML file
        tree = ET.parse(file_name)
        root = tree.getroot()

        # Initialize an empty list to hold the correlation data for the current file
        file_g5g5_data = []

        # Find all <elem> elements
        for elem in root.findall(".//elem"):
            # Find gamma_snk and gamma_src elements
            gamma_snk_elem = elem.find("gamma_snk")
            gamma_src_elem = elem.find("gamma_src")

            # Check if both gamma_snk and gamma_src are present and are equal to "Gamma5"
            if (
                gamma_snk_elem is not None
                and gamma_src_elem is not None
                and gamma_snk_elem.text.strip() == "Identity"
                and gamma_src_elem.text.strip() == "Identity"
            ):
                # Extract the correlation data
                corr_data = []
                for corr_elem in elem.findall(".//corr/elem"):
                    text_content = corr_elem.text.strip()
                    real_part, imag_part = text_content.strip("()").split(",")
                    corr_data.append(float(real_part))
                file_g5g5_data.append(corr_data)

        # Append the correlation data for the current file to the main list
        g5g5_data.append(file_g5g5_data)

    # Convert the list of lists into a 3D array (list of lists of lists)
    # You can use a library like numpy if you prefer an actual array structure
    g5g5_correlators_array = np.array(g5g5_data)

    # Reshape the array from (165, 1, 8) to (165, 8)
    g5g5_correlators_array = np.squeeze(g5g5_correlators_array, axis=1)

    return g5g5_correlators_array
