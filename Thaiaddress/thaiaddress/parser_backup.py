"""
This module provides functions for parsing Thai addresses,
extracting phone numbers, emails, and performing named entity recognition (NER)
on Thai text.
"""
import os.path as op
import re
import joblib
import pandas as pd
from fuzzywuzzy import process
from spacy import displacy
from pythainlp.tokenize import word_tokenize
from pythainlp import tokenize
from .utils import (
    preprocess,
    is_stopword,
    clean_location_text,
    isThaiWord,
)
from Levenshtein import jaro
from pythainlp.util import Trie
from pythainlp.corpus import thai_stopwords
from pythainlp.tag import pos_tag

from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
import torch

import logging
logging.getLogger().setLevel(logging.ERROR)

tokenizer = AutoTokenizer.from_pretrained("pythainlp/thainer-corpus-v2-base-model")
model = AutoModelForTokenClassification.from_pretrained("pythainlp/thainer-corpus-v2-base-model")

# read model from models path, define colors for output classes
MODULE_PATH = op.dirname(__file__)
CRF_MODEL = joblib.load(op.join(MODULE_PATH, "models", "model.joblib"))

ADDR_DF = pd.read_csv(
    op.join(MODULE_PATH, "data", "thai_address_data.csv"), dtype={"zipcode": str}
)

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

def doc2features(doc, i):
    """
    Extract features from a tokenized document for CRF model.

    Parameters:
    doc (list): A list of tuples containing tokens and their POS tags.
    i (int): The index of the token for which features are to be extracted.

    Returns:
    dict: A dictionary containing the features for the specified token.
    """
    word, postag = doc[i]

    # Features from current word
    features = {
        'word.word': word,
        'word.stopword': is_stopword(word),
        'word.isthai': isThaiWord(word),
        'word.isspace': word.isspace(),
        'postag': postag,
        'word.isdigit()': word.isdigit(),
    }
    if word.isdigit() and len(word) == 5:
        features['word.islen5'] = True

    if i > 0:
        prevword, postag1 = doc[i - 1]
        features['word.prevword'] = prevword
        features['word.previsspace'] = prevword.isspace()
        features['word.previsthai'] = isThaiWord(prevword)
        features['word.prevstopword'] = is_stopword(prevword)
        features['word.prepostag'] = postag1
        features['word.prevwordisdigit'] = prevword.isdigit()
    else:
        features['BOS'] = True  # Special "Beginning of Sequence" tag

    # Features from next word
    if i < len(doc) - 1:
        nextword, postag1 = doc[i + 1]
        features['word.nextword'] = nextword
        features['word.nextisspace'] = nextword.isspace()
        features['word.nextpostag'] = postag1
        features['word.nextisthai'] = isThaiWord(nextword)
        features['word.nextstopword'] = is_stopword(nextword)
        features['word.nextwordisdigit'] = nextword.isdigit()
    else:
        features['EOS'] = True  # Special "End of Sequence" tag

    return features

def extract_features(doc):
    """
    Extract features from a tokenized document for CRF model.

    Parameters:
    doc (list): A list of tuples containing tokens and their POS tags.

    Returns:
    list: A list of dictionaries, where each dictionary contains the features for a token.
    """
    return [doc2features(doc, i) for i in range(len(doc))]

def get_labels(doc):
    """
    Extract labels from a tokenized document.

    Parameters:
    doc (list): A list of tuples containing tokens, POS tags, and labels.

    Returns:
    list: A list of labels corresponding to the tokens in the document.
    """
    return [tag for (token, postag, tag) in doc]

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
    # preprocess the text
    text = text.replace("\n-", " ")
    text = text.replace("\n", " ")

    if option == "province":
        text = text.split("จ.")[-1].split("จังหวัด")[-1]
        list_check = PROVINCES
        text = text.split()
        word = [word for word in text if word in list_check]
        word = ' '.join(word)

    elif option == "district":
        text = text.split("อ.")[-1].split("อำเภอ")[-1]
        text = text.split(" เขต")[-1]
        list_check = DISTRICTS
        text = text.split()
        word = [word for word in text if word in list_check]
        word = ' '.join(word)

    elif option == "subdistrict":
        text = text.split("ต.")[-1].split("อ.")[0].split("อำเภอ")[0]
        text = text.split(" แขวง")[-1].split(" เขต")[0]
        list_check = SUBDISTRICTS
        text = text.split()
        word = [word for word in text if word in list_check]
        word = ' '.join(word)

    location = ""
    if postal_code is not None and SUBDISTRICTS_POST_DICT.get(postal_code) is not None:
        options_map = {
            "province": PROVINCES,
            "district": DISTRICTS_POST_DICT.get(postal_code, DISTRICTS),
            "subdistrict": SUBDISTRICTS_POST_DICT.get(postal_code, SUBDISTRICTS),
        }
    elif province is not None:
        districts = []
        for d in DISTRICTS_DICT.get(province, DISTRICTS):
            if d != "พระนครศรีอยุธยา":
                districts.append(d.replace(province, ""))
            else:
                districts.append(d)
        options_map = {
            "province": PROVINCES,
            "district": districts,
            "subdistrict": SUBDISTRICTS_DICT.get(province, SUBDISTRICTS),
        }
    else:
        options_map = {
            "province": PROVINCES,
            "district": DISTRICTS,
            "subdistrict": SUBDISTRICTS,
        }
    options = options_map.get(option)

    try:
        locs = [l for l, _ in process.extract(word, options, limit=3)]
        locs.sort(key=len, reverse=False)  # sort from short to long string
        for loc in locs:
            if loc in word:
                location = loc
        if location == "" or location == "เมือง":
            location = [l for l, _ in process.extract(word, options, limit=3)][0]
    except:
        pass
    return location


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


def extract_phone_numbers(text):
    """
    Extract phone numbers from text using regular expressions.

    Parameters:
    text (str): The input text from which to extract phone numbers.

    Returns:
    list or str: The list of extracted phone numbers, or a single phone number if only one is found, or an empty string if none are found.
    """
    phone_number_pattern = r'\b0\d{1,2}\s*\d{1,3}\s*\d{3}\s*\d{4}\b'

    matches = re.findall(phone_number_pattern, text)

    phone_numbers = [re.sub(r'\D', '', phone_number) for phone_number in matches]
    phone_numbers = [phone_number[:10] for phone_number in phone_numbers]

    if len(phone_numbers) == 1:
        return phone_numbers[0].replace("-", "")
    elif len(phone_numbers) > 1:
        return phone_numbers
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
    emails = re.findall(r"\b[\w\.-]+?@\w+?\.\w+?\b", text)
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

def fix_span_error(words, ner):
   """
   Fix span errors in the named entity recognition (NER) output.

   Parameters:
   words (list): A list of word tokens.
   ner (list): A list of NER tags corresponding to the word tokens.

   Returns:
   list: A list of tuples containing the word tokens and their corresponding NER tags, with span errors fixed.
   """
   _new_tag = []
   for i, j in zip(words, ner):
       i = tokenizer.decode(i)
       if j.startswith("B-PERSON") or j.startswith("I-PERSON") or j.startswith("B-LOCATION") or j.startswith("I-LOCATION"):
           _new_tag.append((i, j))
   return _new_tag


def get_postal_code(subdistrict, province):
    result = ADDR_DF[(ADDR_DF['subdistrict'] == subdistrict) & (ADDR_DF['province'] == province)]
    if len(result) > 0:
        return result['zipcode'].values[0]
    else:
        return ""

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


def check_phone_numbers(dash_count, phone_numbers):
    if dash_count >= 4:
        return phone_numbers if phone_numbers is not None else None
    else:
        return None

        
def parse(text=None, display=False, tokenize_engine="newmm-safe"):
    """
    Parse a given address text and extract phone numbers and emails

    Parameters
    ----------
    text: str, input Thai address text to be parsed
    display: bool, if True, display parsed output
    tokenize_engine: str, pythainlp tokenization engine, default is newmm-safe

    Output
    ------
    address: dict, parsed output
    """

    if not text or text.isspace():  # Handling None, empty string, and string with only spaces
        return   None

    text = preprocess(text)    
    tokens = word_tokenize(text, engine=tokenize_engine, custom_dict=custom_dict)
    chunks = [text[i:i+512] for i in range(0, len(text), 512)]

    try:
        features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    except IndexError as e:
        features = []
    preds = CRF_MODEL.predict([features])[0]
    # for chunk in chunks:
    #     cut = word_tokenize(chunk.replace(" ", "<_>"))
    #     inputs = tokenizer(cut, is_split_into_words=True, return_tensors="pt", max_length=512, truncation=True, padding=True)
        
    #     ids = inputs["input_ids"]
    #     mask = inputs["attention_mask"]
        
    #     outputs = model(ids, attention_mask=mask)
    #     logits = outputs[0]
    #     predictions = torch.argmax(logits, dim=2)
    #     predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
    #     ner_tag = fix_span_error(inputs['input_ids'][0], predicted_token_class)

    phone_numbers = extract_phone_numbers(text.replace('-', ''))
    email_addresses = extract_emails(text)
    
    preds_ = list(zip(tokens, preds))
    preds_ = [(value, 'LOC') if value in SUBDISTRICTS + DISTRICTS + PROVINCES else (value, tag) for value, tag in preds_]
    

    # name = "".join([token for token, c in preds_ if c == "NAME"]).strip()
    address = "".join([token for token, c in preds_ if c == "ADDR"]).strip()
    location = " ".join([token for token, c in preds_ if c == "LOC"]).strip()

    # name = "".join([token if token != '<_>' else ' ' for token, c in ner_tag if c == "B-PERSON" or c == "I-PERSON"]).strip()
    # location = " ".join([token if token != '<_>' else ' ' for token, c in ner_tag if c == "B-LOCATION" or c == "I-LOCATION"]).strip()

    if len(location.split()) <= 4:
        location = ""

    # if name == "":
    #     name ='-'
        
    postal_code = extract_postal_code(text)
    postal_list = postal_code.split(';')
    unique_postal_list = list(set(postal_list))
    unique_postal = ';'.join(unique_postal_list)

    if unique_postal is not None:
        if unique_postal not in PROVINCES_POST_DICT:
            unique_postal = "-"


    subdistrict = None
    district = None
    province = None

    len_provinces_subdistricts_districts = len([word for word in tokens if word in PROVINCES + SUBDISTRICTS + DISTRICTS])
    if unique_postal != "-":
        len_provinces_subdistricts_districts += 1
    print(len_provinces_subdistricts_districts)
    if (len_provinces_subdistricts_districts >= 3 or sum(word in ['อำเภอ', 'ตำบล', 'จังหวัด', 'เขต', 'แขวง','เเขวง'] for word in tokens) >= 2):
        if unique_postal != "-" and unique_postal in PROVINCES_POST_DICT:
            province = PROVINCES_POST_DICT[unique_postal][0]
        elif "จังหวัด" in text:
            province_name_split = [part for part in text.split("จังหวัด") if part.strip()]
            if len(province_name_split) > 1:
                province_name = province_name_split[-1].split()[0]
                province = province_name
            else:
                province = "-"
        else:
            province = extract_location(location, option="province")
            
        if "อำเภอ" in text:
            district_name = text.split("อำเภอ")[-1].split()[0]
            district = district_name
            if district == 'เมือง':
                district = extract_location(
                location, option="district", province=province)
        elif "เขต" in text:
            district_name = text.split("เขต")[-1].split()[0]
            district = district_name
        
        elif district is None and unique_postal != "-" and unique_postal in DISTRICTS_POST_DICT: # Check in case it dosen can use pattern ตำบล or ต to extract etc.  "ปภาวิน ดียิ่ง 18 ม.12 ไพศาล ประโคนชัย จังหวัดบุรีรัมย์ 31140 0942835362"
            districts_sorted = sorted(DISTRICTS_POST_DICT[unique_postal], key=len, reverse=True)
            district = "-"
            for token in tokens:
                best_district = find_best_subdistrict_and_district(token, districts_sorted)
                if best_district:
                    district = best_district
                    break

        else:
            district = '-'
            
        if "ตำบล" in text:
            subdistrict_name = text.split("ตำบล")[-1].split()[0]
            subdistrict = subdistrict_name
        elif subdistrict is None and "เขวง" in text:  # Check for "เขวง" instead of "เเขวง"
            subdistrict_name = text.split("เขวง")[-1].split()[0]
            subdistrict = subdistrict_name
        elif subdistrict is None and "แขวง" in text:  # Check for "แขวง"
            subdistrict_name = text.split("แขวง")[-1].split()[0]
            subdistrict = subdistrict_name
        
            
        elif subdistrict is None and unique_postal != "-" and unique_postal in SUBDISTRICTS_POST_DICT: # Check in case it dosen can use pattern ตำบล or ต to extract etc.  "ปภาวิน ดียิ่ง 18 ม.12 ไพศาล ประโคนชัย จังหวัดบุรีรัมย์ 31140 0942835362"
            subdistricts_sorted = sorted(SUBDISTRICTS_POST_DICT[unique_postal], key=len, reverse=True)
            subdistrict = "-"
            for token in tokens:
                best_subdistrict = find_best_subdistrict_and_district(token, subdistricts_sorted)
                if best_subdistrict:
                    subdistrict = best_subdistrict
                    break
        else:
            subdistrict = "-"

        if unique_postal != "-":
            subdistrict = correct_location_name(subdistrict,SUBDISTRICTS_POST_DICT[unique_postal])
            district = correct_location_name(district,DISTRICTS_POST_DICT[unique_postal])
            province = correct_location_name(province,PROVINCES_POST_DICT[unique_postal])
        else:
            unique_postal = get_postal_code(subdistrict,province)
            subdistrict = correct_location_name(subdistrict,SUBDISTRICTS)
            district = correct_location_name(district,DISTRICTS)
            province = correct_location_name(province,PROVINCES)
            
        
        #Remove duplicated postal code
        province = re.sub(r'[^\u0E00-\u0E7F]', '', province) #ลบกรณีแบ่งคำแล้วติดตัวเลขไปรษณีย์ เช่น แพร่12124
        subdistrict = clean_location_text(str(subdistrict))
        district = clean_location_text(str(district))
        province = clean_location_text(str(province))
        patterns = ['เขต', 'เเขวง', 'จังหวัด', 'แขวง','ตำบล','อำเภอ']
        if unique_postal == "":
            unique_postal ='-'
        patterns.extend([subdistrict, district, province, unique_postal])

        # Extend patterns with phone numbers and email addresses if they are lists
        if isinstance(phone_numbers, list):
            patterns.extend(phone_numbers)
        else:
            patterns.append(phone_numbers)

        if isinstance(email_addresses, list):
            patterns.extend(email_addresses)
        else:
            patterns.append(email_addresses)

        for pattern in patterns:
            clean_address = re.sub(pattern, '', text)
            
        dash_count = sum(1 for field in [subdistrict, district, province,phone_numbers, email_addresses,unique_postal] if field == '-')
        if dash_count >= 4:
            return 
        else:
            return {
                "text": text,
                "address": preprocess(clean_address),
                "subdistrict": subdistrict,
                "district": district,
                "province": province,
                "postal_code": unique_postal,
                "phone": phone_numbers,
                "email": email_addresses,
            }
    else:
        if phone_numbers != '-':
            return {"phone": phone_numbers}
        else:
            return None
        



