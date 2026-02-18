import re

state_district_codes = {
    "AN": [1, 5],    # Andaman and Nicobar Islands
    "AP": [1, 75],  # Andhra Pradesh
    "AR": [1, 20],  # Arunachal Pradesh
    "AS": [1, 50],  # Assam
    "BR": [1, 40],  # Bihar
    "CG": [1, 30],  # Chhattisgarh
    "DN": [1, 5],  # Daman and Diu
    "DD": [1, 5],  # Dadra and Nagar Haveli
    "DL": [1, 75],  # Delhi
    "GA": [1, 5],  # Goa
    "GJ": [1, 75],  # Gujarat
    "HR": [1, 30],  # Haryana
    "HP": [1, 30],  # Himachal Pradesh
    "JK": [1, 30],  # Jammu and Kashmir
    "KA": [1, 60],  # Karnataka
    "KL": [1, 30],  # Kerala
    "LA": [1, 5],  # Ladakh
    "LD": [1, 5],  # Lakshadweep
    "MP": [1, 50],  # Madhya Pradesh
    "MH": [1, 50],  # Maharashtra
    "MN": [1, 5],  # Manipur
    "ML": [1, 5],  # Meghalaya
    "MZ": [1, 5],  # Mizoram
    "NL": [1, 5],  # Nagaland
    "OD": [1, 60],  # Odisha
    "PB": [1, 30],  # Punjab
    "RJ": [1, 45],  # Rajasthan
    "SK": [1, 5],  # Sikkim
    "TN": [1, 30],  # Tamil Nadu
    "TS": [1, 40],  # Telangana
    "UP": [1, 50],  # Uttar Pradesh
    "UK": [1, 30],  # Uttarakhand
    "WB": [1, 50],  # West Bengal
}


union_teritory = {
    "CH": [1, 10],  # Chandigarh
    "DN": [1, 5],  # Daman and Diu and Dadra and Nagar Haveli
    "DD": [1, 5],  # Dadra and Nagar Haveli
    "LD": [1, 5],  # Lakshadweep
    "DL": [1, 75],  # Delhi
    "PY": [1, 5],  # Puducherry
    "LA": [1, 5],  # Ladakh
}

def is_valid_hsrp(license_plate):

    if len(license_plate) < 8 or len(license_plate) > 12:
        return False

    ordinary_pattern = r"^[A-Z]{2}\d{2}[A-Z]{2}\d{6}$"
    union_tert_plate = r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$"
    BH_Plate = r"^\d{2}BH\d{4}[A-Z]{1}$"
    
    ord_match = re.match(ordinary_pattern, license_plate)
    uni_tert_match = re.match(union_tert_plate, license_plate)
    BH_match = re.match(BH_Plate, license_plate)
    
    if ord_match:
        state_code = str(license_plate[:2])
        district_code = int(license_plate[2:4])
        if state_code not in state_district_codes:
            return False
        else:
            if district_code < state_district_codes[state_code][0] or district_code > state_district_codes[state_code][1]:
                return False
        return True
    
    elif uni_tert_match :
        state_code = str(license_plate[:2])
        if state_code not in union_teritory:
            return False
        else :
            district_code = int(license_plate[2:3])
            if license_plate[3].isdigit():
                district_code = int(license_plate[2:4])
            if district_code < union_teritory[state_code][0] or district_code > union_teritory[state_code][1]:
                return False 
        return True
    
    elif BH_match:
        year = int(license_plate[:2])
        if year < 22 or year > 26:
            return False
        return True
    
    else :
        return False

def checker(license_plates):
    count = 0
    valid = ""
    invalid = ""
    for plate in license_plates:
        if is_valid_hsrp(plate):
            if len(plate) > len(valid):
                valid = plate
            count += 1
        else:
            if len(plate) > len(invalid):
                invalid = plate
    
    if count :
        return [valid, "Valid"]
    return [invalid, "Invalid"]

