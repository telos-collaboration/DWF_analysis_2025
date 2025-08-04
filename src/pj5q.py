import xml.etree.ElementTree as ET
import glob
import os
import re
import numpy as np


def read_pj5q(directory):
    # Function to extract the numeric part from the filename
    def extract_number(filename):
        match = re.search(r'mres.(\d+)\.xml', filename)
        return int(match.group(1)) if match else None

    # Get a list of all files matching the pattern mres_*.xml in the specified directory
    file_list = glob.glob(os.path.join(directory, 'mres.*.xml'))

    # Sort the file list based on the numeric part of the filenames
    file_list.sort(key=extract_number)

    # Initialize an empty list to hold the PJ5q correlators for all files
    PJ5q_correlators = []

    # Iterate over each file in the sorted list
    for file_index, file_name in enumerate(file_list):
        # Parse the XML file
        tree = ET.parse(file_name)
        root = tree.getroot()

        # Initialize an empty list to hold the real parts of PJ5q elements for the current file
        pj5q_real_parts = []

        # Find the PJ5q element and iterate through its children
        for elem in root.findall('.//PJ5q/elem'):
            # Extract the text content of each <elem>
            text_content = elem.text.strip()
            # Remove parentheses and split the text into two parts
            real_part, imag_part = text_content.strip('()').split(',')
            # Convert the real part into a float and append it to the array
            pj5q_real_parts.append(float(real_part))

        # Append the list of real parts to the PJ5q_correlators list
        PJ5q_correlators.append(pj5q_real_parts)

    # Convert the list of lists into a 2D array (list of lists)
    # You can use a library like numpy if you prefer an actual array structure
    PJ5q_correlators_array = np.array(PJ5q_correlators)
    return PJ5q_correlators_array

