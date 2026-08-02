"""
# config_akira.py  –  copy of config.py with paths updated for Dataset_csv/ layout
#
# Changes vs config.py:
#   - data_document_path: points to Dataset_csv/KUL or Dataset_csv/DTU/128
#     depending on the dataset selected below.
#   - people_number: 16 for KUL (unchanged), 18 for DTU (unchanged)

20260802 creation for MHAnet

"""

dataset = "KUL"
time_len = 2
people_number = 16  # KUL: 16,  DTU: 18

# Set data_document_path to match the selected dataset:
#   KUL → ../../01_OriginalData/Dataset_csv/KUL
#   DTU → ../../01_OriginalData/Dataset_csv/DTU/128
data_document_path = "../../01_OriginalData/Dataset_csv/KUL"
