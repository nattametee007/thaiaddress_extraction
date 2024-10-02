"""
This module provides functions for parsing Thai addresses,
extracting phone numbers, emails, and performing named entity recognition (NER)
on Thai text.
"""
import os.path as op
import re
import joblib
import pandas as pd
from spacy import displacy
from pythainlp.tokenize import word_tokenize,sent_tokenize
from pythainlp import tokenize
from .utils import (
    preprocess,
    is_stopword,
    clean_location_text,
    merge_tokens,
    merge_labels
)
from Levenshtein import jaro
from pythainlp.util import Trie
import phonenumbers
import json
import logging
logging.getLogger().setLevel(logging.ERROR)

MODULE_PATH = op.dirname(__file__)
CRF_MODEL = joblib.load(op.join(MODULE_PATH, "models", "new_model_synthetic_newmm_25000_addoptional.joblib"))

with open('./thaiaddress/data/thailand-geo-json/src/geography.json', 'r') as f:
    geography = json.load(f)
df = pd.DataFrame(geography)

# Select and rename relevant columns
ADDR_DF = df[['subdistrictNameTh', 'districtNameTh', 'provinceNameTh', 'postalCode']].copy()
ADDR_DF.columns = ['subdistrict', 'district', 'province', 'zipcode']
ADDR_DF['zipcode'] = ADDR_DF['zipcode'].astype(str)


PROVINCES = list(ADDR_DF.province.unique())
DISTRICTS = list(ADDR_DF.district.unique())
SUBDISTRICTS = list(ADDR_DF.subdistrict.unique())
DISTRICTS_DICT = ADDR_DF.groupby("province")["district"].apply(list)
SUBDISTRICTS_DICT = ADDR_DF.groupby("province")["subdistrict"].apply(list)
DISTRICTS_POST_DICT = ADDR_DF.groupby("zipcode")["district"].apply(list)
SUBDISTRICTS_POST_DICT = ADDR_DF.groupby("zipcode")["subdistrict"].apply(list)
PROVINCES_POST_DICT = ADDR_DF.groupby("zipcode")["province"].apply(list)

custom_dict = SUBDISTRICTS + DISTRICTS + PROVINCES
custom_dict = Trie(custom_dict)

COLORS = {
    "NAME": "#fbd46d",
    "ADDR": "#ff847c",
    "LOC": "#87d4c5",
    "POST": "#def4f0",
    "PHONE": "#ffbffe",
    "EMAIL": "#91a6b8",
}

def extract_location(text, option="province", province=None, postal_code=None):
    """
    Extract Thai province, district, or subdistrict from a given text.

    Parameters:
    text (str): Input Thai text that contains location.
    option (str): The type of location to extract ('province', 'district', or 'subdistrict').
    province (str or None): If provided, search for districts and subdistricts within the given province.
    postal_code (str or None): If provided, search for districts and subdistricts within the given postal code.

    Returns:
    str: The extracted location that best matches the input text.
    """
    text = text.replace("\n-", " ").replace("\n", " ")

    if option == "province":
        text = text.split("จ.")[-1].split("จังหวัด")[-1]
        list_check = PROVINCES
    elif option == "district":
        text = text.split("อ.")[-1].split("อำเภอ")[-1]
        text = text.split(" เขต")[-1]
        if province:
            subdistricts_in_province = DISTRICTS_DICT.get(province, [])
            list_check = [sd for sd in DISTRICTS if sd in subdistricts_in_province]
        else:
            list_check = DISTRICTS
    elif option == "subdistrict":
        text = text.split("ต.")[-1].split("อ.")[0].split("อำเภอ")[0]
        text = text.split(" แขวง")[-1].split(" เขต")[0]

        if province:
            subdistricts_in_province = SUBDISTRICTS_DICT.get(province, [])
            list_check = [sd for sd in SUBDISTRICTS if sd in subdistricts_in_province]
        else:
            list_check = SUBDISTRICTS
    else:
        return None 

    text = text.split()
    word = [word for word in text if word in list_check]
    word = ' '.join(word) if word else ""

    best_match = ""
    highest_score = 0

    # Find the best match using Jaro distance
    for option_location in list_check:
        score = jaro(word, option_location)
        if score > highest_score:
            highest_score = score
            best_match = option_location

    return best_match if highest_score > 0.8 else "-"

def find_postal_code(subdistrict=None, district=None, province=None):
    query = {}
    if subdistrict:
        query["subdistrict"] = subdistrict
    if district:
        query["district"] = district
    if province:
        query["province"] = province
    
    result = ADDR_DF.loc[(ADDR_DF[list(query)] == pd.Series(query)).all(axis=1), "zipcode"]
    
    if not result.empty:
        return result.iloc[0]
    else:
        return "-"

def tokens_to_features(tokens, i):
   """
   Convert a list of tokens to a dictionary of features for a specific token index.

   Parameters:
   tokens (list): A list of tuples containing tokens and their labels.
   i (int): The index of the token for which features are to be extracted.

   Returns:
   dict: A dictionary containing the features for the specified token.
   """
   if len(tokens[i]) == 2:
       word, _ = tokens[i]  # unpack word and class
   else:
       word = tokens[i]

   # Features from current word
   features = {
       "bias": 1.0,
       "word.word": word,
       "word[:3]": word[:3],
       "word.isspace()": word.isspace(),
       "word.is_stopword()": is_stopword(word),
       "word.isdigit()": word.isdigit(),
   }
   if word.strip().isdigit() and len(word) == 5:
       features["word.islen5"] = True

   # Features from previous word
   if i > 0:
       prevword = tokens[i - 1][0]
       features.update(
           {
               "-1.word.prevword": prevword,
               "-1.word.isspace()": prevword.isspace(),
               "-1.word.is_stopword()": is_stopword(prevword),
               "-1.word.isdigit()": prevword.isdigit(),
           }
       )
   else:
       features["BOS"] = True  # Special "Beginning of Sequence" tag

   # Features from next word
   if i < len(tokens) - 1:
       nextword = tokens[i + 1][0]
       features.update(
           {
               "+1.word.nextword": nextword,
               "+1.word.isspace()": nextword.isspace(),
               "+1.word.is_stopword()": is_stopword(nextword),
               "+1.word.isdigit()": nextword.isdigit(),
           }
       )
   else:
       features["EOS"] = True  # Special "End of Sequence" tag

   return features

def clean_phone_numbers(phone_numbers):
    cleaned_numbers = []
    
    if phone_numbers:
        for phone_number in phone_numbers:
            cleaned_number = re.sub(r'\D', '', phone_number)  
            cleaned_numbers.append(cleaned_number)
        return cleaned_numbers
    else:
        return "-"

def extract_emails(text):
    """
    Extract email addresses from text using regular expressions.

    Parameters:
    text (str): The input text from which to extract email addresses.

    Returns:
    str or list: The first extracted email address if only one is found, or a list of 
    extracted email addresses if more than one is found, or an empty string if no email addresses are found.
    """
    emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
    if len(emails) == 1:
        return emails[0]
    elif len(emails) > 1:
        return emails
    else:
        return "-"

def extract_postal_code(text):
    """
    Extract postal codes from text using regular expressions.

    Parameters:
    text (str): The input text from which to extract postal codes.

    Returns:
    str or None: The extracted postal code if found, otherwise None.
    """
    postal_code_pattern = r'(?<!\d)\d{5}(?!\d)'  # Match 5-digit postal codes not surrounded by digits
    postal_code_matches = re.findall(postal_code_pattern, text)
 
    if postal_code_matches:
        return postal_code_matches[0]
    else:
        return "-"

def correct_location_name(misspelled_word, correct_words, threshold=0.6):
   """
   Correct a misspelled location name by finding the closest match in a list of correct words.

   Parameters:
   misspelled_word (str): The misspelled word to be corrected.
   correct_words (list): A list of correct words to compare against.
   threshold (float): The minimum similarity threshold for considering a match (default: 0.6).

   Returns:
   str: The corrected word if a match is found, otherwise the original misspelled word.
   """
   closest_match = max(correct_words, key=lambda word: jaro(misspelled_word, word))
   return closest_match if jaro(misspelled_word, closest_match) >= threshold else misspelled_word


def get_postal_code(subdistrict, province):
    result = ADDR_DF[(ADDR_DF['subdistrict'] == subdistrict) & (ADDR_DF['province'] == province)]
    if len(result) > 0:
        return result['zipcode'].values[0]
    else:
        return ""
    
def relabel_token(tokens):
    updated_tokens = []
    for token, label in tokens:
        if token == 'ห้อง':
            updated_tokens.append((token, 'ADDR'))
        else:
            updated_tokens.append((token, label))
    return updated_tokens

def find_best_subdistrict_and_district(token, subdistrict_and_district_sorted):
    best_subdistrict_and_district = None
    best_subdistrict_and_district_similarity = 0
    
    # Iterate through sorted subdistrict_and_districts to find the best match for this token
    for subdistrict_and_district_candidate in subdistrict_and_district_sorted:
        similarity = jaro(token, subdistrict_and_district_candidate)
        if similarity >= 0.8 and similarity > best_subdistrict_and_district_similarity:
            best_subdistrict_and_district_similarity = similarity
            best_subdistrict_and_district = subdistrict_and_district_candidate
    
    return best_subdistrict_and_district

def filter_only_address(address, phone_numbers, subdistrict, district, province, postal_code, email):
    # Convert phone_numbers and email to a list if they are not already
    if not isinstance(phone_numbers, list):
        phone_numbers = [phone_numbers]
    if not isinstance(email, list):
        email = [email]
    
    # Convert all other inputs to strings
    subdistrict_str = str(subdistrict)
    district_str = str(district)
    province_str = str(province)
    postal_code_str = str(postal_code)
    
    # Create a combined regex pattern to match any of the unwanted text
    patterns_to_remove = phone_numbers + [subdistrict_str, district_str, province_str, postal_code_str, "เมือง"] + email
    combined_pattern = '|'.join(map(re.escape, patterns_to_remove))
    
    # Remove unwanted patterns from the address
    cleaned_address = re.sub(combined_pattern, '', address)
    
    # Return the cleaned address
    return cleaned_address.strip()

def extract_address(text):
    if len(text) > 300:
        text_list = sent_tokenize(text, engine="crfcut")
        keywords = ["บ้าน", "บริษัท", "เลขที่", "/", "ที่อยู่", "บ้านเลขที่", "หมู่", "ซอย", "หมู่บ้าน", "ถนน", "ตำบล", "แขวง", "ที่ทำการ", "แยก", "ตรอก", "โรงแรม"]
        filtered_text_list = [sentence for sentence in text_list if any(keyword in sentence for keyword in keywords)]
        new_text = max(filtered_text_list, key=len) if filtered_text_list else ""
        if "กรุงเทพมหานคร" in text:
            new_text = new_text.replace("อำเภอ", "เขต").replace("ตำบล", "แขวง")
    else:
        new_text = preprocess(text)
    
    return new_text

def display_entities(tokens: list, labels: list):
    """
    Display tokens and labels

    References
    ----------
    Spacy, https://spacy.io/usage/visualizers
    """
    options = {"ents": list(COLORS.keys()), "colors": COLORS}

    ents = []
    text = ""
    s = 0
    for token, label in zip(tokens, labels):
        text += token
        if label != "O":
            ents.append({"start": s, "end": s + len(token), "label": label})
        s += len(token)

    text_display = {"text": text, "ents": ents, "title": None}
    displacy.render(
        text_display, style="ent", options=options, manual=True, jupyter=True
    )

def autofill_address(clean_address, subdistrict, district, province, unique_postal):
    # Create a list of non-empty input fields
    input_fields = [
        ('subdistrict', subdistrict),
        ('district', district),
        ('province', province),
        ('zipcode', unique_postal)
    ]
    non_empty_fields = [(field, value.strip()) for field, value in input_fields if value and value.strip() != '-']

    # Filter the dataframe based on non-empty input fields
    filtered_df = ADDR_DF
    for field, value in non_empty_fields:
        filtered_df = filtered_df[filtered_df[field] == value]

    # Only autofill if we have a unique match
    if filtered_df.shape[0] == 1:
        row = filtered_df.iloc[0]
        subdistrict = row['subdistrict']
        district = row['district']
        province = row['province']
        unique_postal = row['zipcode']
    else:
        # If we don't have a unique match, keep the original input values
        subdistrict = subdistrict if subdistrict and subdistrict.strip() != '-' else '-'
        district = district if district and district.strip() != '-' else '-'
        province = province if province and province.strip() != '-' else '-'
        unique_postal = unique_postal if unique_postal and unique_postal.strip() != '-' else '-'

    # Handle the case where clean_address is empty
    if not clean_address or clean_address.strip() == '':
        clean_address = '-'

    return {
        "street_address": clean_address,
        "subdistrict": subdistrict,
        "district": district,
        "province": province,
        "postcode": unique_postal
    }
    
def tokenize_and_extract_features(detected_address, tokenize_engine):
    tokens = word_tokenize(detected_address, engine=tokenize_engine, custom_dict=custom_dict)
    try:
        features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    except IndexError:
        features = []
    return tokens, features

def predict_with_crf(features):
    return CRF_MODEL.predict([features])[0]

def extract_address_and_location(preds_, tokens):
    preds_ = relabel_token(preds_)
    address = "".join([token for token, c in preds_ if c == "ADDR"]).strip()
    location = " ".join([token for token in tokens if token in custom_dict]).strip()
    address = address.replace("ห้อง", " ห้อง ")
    return address, location

def extract_and_process_postal_code(new_text):
    postal_code = extract_postal_code(new_text)
    postal_list = postal_code.split(';')
    unique_postal_list = list(set(postal_list))
    unique_postal = ';'.join(unique_postal_list)
    return unique_postal if unique_postal in PROVINCES_POST_DICT else "-"

def is_valid_address(postal_code, tokens):
    len_provinces_subdistricts_districts = len([word for word in tokens if word in PROVINCES + SUBDISTRICTS + DISTRICTS])
    if postal_code != "-":
        len_provinces_subdistricts_districts += 1
    return (len_provinces_subdistricts_districts >= 2 or 
            sum(word in ['อำเภอ', 'ตำบล', 'จังหวัด', 'เขต', 'แขวง','เเขวง'] for word in tokens) >= 2)

def extract_location_details(new_text, location, postal_code, tokens):
    province = extract_province(new_text, postal_code, location)
    district = extract_district(new_text, postal_code, location, province, tokens)
    subdistrict = extract_subdistrict(new_text, postal_code, location, province, tokens)
    
    if postal_code != "-":
        province, district, subdistrict = correct_location_names(province, district, subdistrict, postal_code)
    else:
        postal_code = find_postal_code(subdistrict, district, province)
        province, district, subdistrict = correct_location_names_without_postal(province, district, subdistrict)
    
    return subdistrict, district, province

def extract_province(new_text, postal_code, location):
    if postal_code != "-" and postal_code in PROVINCES_POST_DICT:
        return PROVINCES_POST_DICT[postal_code][0]
    elif "จังหวัด" in new_text:
        province_name_split = [part for part in new_text.split("จังหวัด") if part.strip()]
        return province_name_split[-1].split()[0] if len(province_name_split) > 1 else "-"
    else:
        return extract_location(location, option="province")

def extract_district(new_text, postal_code, location, province, tokens):
    if "อำเภอ" in new_text:
        district = new_text.split("อำเภอ")[-1].split()[0]
        return extract_location(location, option="district", province=province) if district == 'เมือง' else district
    elif "เขต" in new_text:
        return new_text.split("เขต")[-1].split()[0]
    elif postal_code != "-" and postal_code in DISTRICTS_POST_DICT:
        return extract_district_from_postal(postal_code, tokens)
    else:
        return extract_location(location, option="district", province=province)

def extract_subdistrict(new_text, postal_code, location, province, tokens):
    if "ตำบล" in new_text:
        return new_text.split("ตำบล")[-1].split()[0]
    elif "เขวง" in new_text:
        return new_text.split("เขวง")[-1].split()[0]
    elif "แขวง" in new_text:
        return new_text.split("แขวง")[-1].split()[0]
    elif postal_code != "-" and postal_code in SUBDISTRICTS_POST_DICT:
        return extract_subdistrict_from_postal(postal_code, tokens)
    else:
        return extract_location(location, option="subdistrict", province=province)

def extract_district_from_postal(postal_code, tokens):
    districts_sorted = sorted(DISTRICTS_POST_DICT[postal_code], key=len, reverse=True)
    for token in tokens:
        best_district = find_best_subdistrict_and_district(token, districts_sorted)
        if best_district:
            return best_district
    return "-"

def extract_subdistrict_from_postal(postal_code, tokens):
    subdistricts_sorted = sorted(SUBDISTRICTS_POST_DICT[postal_code], key=len, reverse=True)
    for token in tokens:
        best_subdistrict = find_best_subdistrict_and_district(token, subdistricts_sorted)
        if best_subdistrict:
            return best_subdistrict
    return "-"

def correct_location_names(province, district, subdistrict, postal_code):
    province = correct_location_name(province, PROVINCES)
    subdistrict = correct_location_name(subdistrict, SUBDISTRICTS_POST_DICT[postal_code])
    district = 'เมือง' + province if district == 'เมือง' else correct_location_name(district, DISTRICTS_POST_DICT[postal_code])
    return province, district, subdistrict

def correct_location_names_without_postal(province, district, subdistrict):
    province = correct_location_name(province, PROVINCES)
    subdistrict = correct_location_name(subdistrict, SUBDISTRICTS)
    district = 'เมือง' + province if district == 'เมือง' else correct_location_name(district, DISTRICTS)
    return province, district, subdistrict

def clean_and_filter_address(address, phone_numbers, subdistrict, district, province, postal_code,emails):
    clean_address = clean_location_text(address)
    subdistrict = clean_location_text(str(subdistrict))
    province = clean_location_text(str(province))
    return filter_only_address(clean_address, phone_numbers, subdistrict, district, province, postal_code,emails)

def display_parsed_entities(tokens, preds):
    merge, labels = merge_labels(preds)
    tokens = merge_tokens(tokens, merge)
    display_entities(tokens, labels)

def is_valid_parsed_result(subdistrict, district, province, phone_numbers, email_addresses, postal_code):
    dash_count = sum(1 for field in [subdistrict, district, province, phone_numbers, email_addresses, postal_code] if field == '-')
    return dash_count < 4

def find_indices(text, word):
    if word == '-' or not word:
        return []
    pattern = re.escape(word)
    return [(m.start(), m.end()) for m in re.finditer(pattern, text)]

def phone_numbers_extraction(text):
    numbers = phonenumbers.PhoneNumberMatcher(text, "TH")
    original_numbers = [] 
    phone_indices = []      

    for number in numbers:
        original_number = text[number.start:number.end]  
        start_index = number.start
        end_index = number.end
        
        original_numbers.append(original_number)
        phone_indices.append((start_index, end_index))

    return original_numbers, phone_indices

def process_start_stop(dict_input, phone_indices):
    fields = [key for key in dict_input.keys() if key != 'text']
    all_highlights = []
    
    for field in fields:
        word = dict_input.get(field, '')

        if isinstance(word, list):
            # Process each element in the list
            for part in word:
                if part:  # Ensure non-empty part
                    indices = find_indices(dict_input['text'], part)
                    all_highlights.extend([{'start': start, 'end': end} for start, end in indices])

        elif field == 'address':
            # Split address and find indices for each part
            address_parts = word.split()
            for part in address_parts:
                if part:  # Ensure non-empty part
                    indices = find_indices(dict_input['text'], part)
                    all_highlights.extend([{'start': start, 'end': end} for start, end in indices])

        elif field == 'phone':
            all_highlights.extend([{'start': start, 'end': end} for start, end in phone_indices])

        else:
            # Process single string fields
            if word:  # Ensure non-empty word
                indices = find_indices(dict_input['text'], word)
                all_highlights.extend([{'start': start, 'end': end} for start, end in indices])

    # Sort all highlights by the start index
    all_highlights.sort(key=lambda x: x['start'])

    # Remove duplicates and keep unique highlights
    unique_highlights = []
    for highlight in all_highlights:
        if not unique_highlights or highlight != unique_highlights[-1]:
            unique_highlights.append(highlight)

    return unique_highlights

def parse(text=None, display: bool = False, tokenize_engine: str = "newmm", fields: list = None):
    """
    Parse a given address text and extract phone numbers, emails, and address fields.

    Parameters
    ----------
    text: str, input Thai address text to be parsed
    display: bool, if True, display parsed output
    tokenize_engine: str, pythainlp tokenization engine, default is newmm-safe
    fields: list, fields to return (choose from 'address', 'phone', 'email'). Default returns all.

    Output
    ------
    result: dict, parsed output based on selected fields
    """

    if not text or text.isspace():
        return {}

    valid_fields = ['address', 'phone', 'email']

    if fields is None:
        fields = valid_fields  
    elif isinstance(fields, str):
        fields = [fields]

    fields = [field for field in fields if field in valid_fields]
    if not fields:
        return {}

    # Process the text
    detected_address = extract_address(text)
    new_text = preprocess(text)

    # Tokenize and get predictions
    tokens, features = tokenize_and_extract_features(detected_address, tokenize_engine)
    preds = predict_with_crf(features)

    original_phonenumbers, phone_indics = phone_numbers_extraction(text)
    clean_phonenumbers = clean_phone_numbers(original_phonenumbers)

    email_addresses = extract_emails(new_text)

    # Extract address components
    preds_ = list(zip(tokens, preds))
    address, location = extract_address_and_location(preds_, tokens)
    postal_code = extract_and_process_postal_code(new_text)

    result = {"original_text": text}

    if is_valid_address(postal_code, tokens):
        subdistrict, district, province = extract_location_details(new_text, location, postal_code, tokens)
        clean_address = clean_and_filter_address(address, clean_phonenumbers, subdistrict, district, province, postal_code, email_addresses)
        if display:
            display_parsed_entities(tokens, preds)

        if is_valid_parsed_result(subdistrict, district, province, clean_phonenumbers, email_addresses, postal_code):
            auto_fill_add_dict = autofill_address(clean_address, subdistrict, district, province, postal_code)
            
            if 'address' in fields:
                result['address'] = {
                    'data': auto_fill_add_dict,
                    'highlight': process_start_stop({
                        'text': text,
                        'address': ' '.join([
                            auto_fill_add_dict.get('street_address', ''),
                            auto_fill_add_dict.get('subdistrict', ''),
                            auto_fill_add_dict.get('district', ''),
                            auto_fill_add_dict.get('province', ''),
                            auto_fill_add_dict.get('postcode', '')
                        ]).strip()
                    }, phone_indics)
                }

            if 'phone' in fields and clean_phonenumbers:
                joinclean_phonenumbers =  ', '.join(clean_phonenumbers)
                result['phone'] = {
                    'data': joinclean_phonenumbers,
                    'highlight': process_start_stop({
                        'text': text,
                        'phone': joinclean_phonenumbers
                    }, phone_indics)
                }

            if 'email' in fields and email_addresses:
                result['email'] = {
                    'data': email_addresses,
                    'highlight': process_start_stop({
                        'text': text,
                        'email': email_addresses
                    }, phone_indics)
                }

    elif clean_phonenumbers != '-':
        if 'phone' in fields:
            result['phone'] = {
                'data': clean_phonenumbers,
                'highlight': process_start_stop({'text': text, 'phone': clean_phonenumbers}, phone_indics)
            }
    elif email_addresses != '-':
        if 'email' in fields:
            result['email'] = {
                'data': email_addresses,
                'highlight': process_start_stop({'text': text, 'email': email_addresses}, phone_indics)
            }

    if not any(field in result for field in ['address', 'phone', 'email']):
        return {}

    return result
