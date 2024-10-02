import re
from itertools import groupby
import numpy as np
from pythainlp.util import isthai
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize

import os
import os.path as op
import pandas as pd
import regex

MODULE_PATH = op.dirname(__file__)
ADDR_DF = pd.read_csv(
    op.join(MODULE_PATH, "data", "thai_address_data.csv"), dtype={"zipcode": str}
)

PROVINCES = list(ADDR_DF.province.unique())
DISTRICTS = list(ADDR_DF.district.unique())
SUBDISTRICTS = list(ADDR_DF.subdistrict.unique())
stopwords = list(thai_stopwords())


patterns = [
    r'ต\.', r'อ\.', r'จ\.', r'อำเภอ', r'เขต', r'แขวง', r'จังหวัด', r'ม\.', 
    r'หมู่', r'ตำบล', r'ปณ\.', r'ถนน', r'ถ\.', r'บ\.', r'ซอย', r'ซ\.', 
    r'(?<!@)\.(?!\w)', r'กทม.', r'กรุงเทพ', r'กทม',r'รหัสไปรษณีย์',r'ไปรษณีย์'
]

pattern_district_subdistrict = [
    (r"แขวง/เขต (\S+)", r"แขวง\1 เขต\1"),
    (r"แขวง/เขต(\S+)", r"แขวง/เขต \1"),
    (r"เขต/แขวง (\S+)", r"เขต\1 แขวง\1"),
    (r"เขต/แขวง(\S+)", r"เขต/แขวง \1"),
    (r"อำเภอ/ตำบล (\S+)", r"อำเภอ\1 ตำบล\1"),
    (r"อำเภอ/ตำบล(\S+)", r"อำเภอ/ตำบล \1"),
    (r"ตำบล/อำเภอ (\S+)", r"ตำบล\1 อำเภอ\1"),
    (r"ตำบล/อำเภอ(\S+)", r"ตำบล/อำเภอ \1")
]



def add_province_prefix_to_text(text, provinces):
    for province in provinces:
        if 'จังหวัด' + province not in text and 'จ.' + province not in text:
            text = text.replace(province, 'จังหวัด' + province + " ")
    return text


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_spaces_from_phone_numbers(text):
    phone_number_pattern = r'\b0\d{1,2}[ _.,-]*\d{1,3}[ _.,-]*\d{3}[ _.,-]*\d{3,4}\b'
    matches = re.findall(phone_number_pattern, text)
    for match in matches:
        phone_number = re.sub(r'[ _.,-]', '', match)
        if len(phone_number) > 10:
            phone_number = phone_number[:10]
        text = text.replace(match, phone_number)

    return text



def remove_emoji(text):
    """
    Remove emojis from a given text
    """
    regrex_pattern = re.compile(
        pattern="["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )
    return regrex_pattern.sub(r"", text)


# Define a function to handle the replacement เขต/แขวง
def replace_patterns(text):
    for pattern, replacement in pattern_district_subdistrict:
        match = re.search(pattern, text)
        if match:
            text = re.sub(pattern, replacement, text)
            break
    return text



def thai_to_number(text):
    thai_numbers = {
        '๐': '0', '๑': '1', '๒': '2', '๓': '3', '๔': '4',
        '๕': '5', '๖': '6', '๗': '7', '๘': '8', '๙': '9'
    }
    output = ''
    for char in text:
        if char in thai_numbers:
            output += thai_numbers[char]
        else:
            output += char
    return output

def preprocess(text: str) -> str:
    """
    Generalized function to preprocess an input
    """
    text = text.strip()
    text = text.replace("จัด ", "")
    text = text.replace("ส่ง ", "")
    text = text.replace("จัดส่ง", "")
    text = text.replace("ชือ.", "")
    text = text.replace("ชื่อ ", "")
    text = text.replace("ผู้รับ", "")
    text = text.replace("ส่งที่ ", " ")
    text = text.replace("ที่อยู่ ", " ")
    text = text.replace("ที้อยุ่จัดส่ง", " ")
    text = text.replace("ที่อยู่จ้า ", " ")
    text = text.replace("ส่งของที่ ", " ")
    text = re.sub(r'ที่อยู่จัดส่ง', '', text)
    text = re.sub(r'ที่อยู่', '', text)
    text = re.sub(r'จัดส่ง', '', text)
    text = re.sub(r'ที่อยู่จัด', '', text)
    text = re.sub(r'จัด', '', text)
    text = re.sub(r'คุณ', '', text)
    text = thai_to_number(text)
    text = text.replace("ส่งมาที่", " ")
    text = text.replace("เบอร์", " ")
    text = text.replace("เบอ", " ")
    text = text.replace("โทร", " ")
    text = text.replace("เบอร์โทรศัพท์", " ")
    text = text.replace("โทรศัพท์", " ")
    text = text.replace("มือถือ", " ")
    # text = text.replace("\n-", " ")
    text = text.replace("\n", " ")
    text = text.replace(": ", " ")
    # text = text.replace("(", " ")
    # text = text.replace(")", " ")
    text = text.replace('"', " ")
    text = text.replace(',', " ")
    # text = text.replace('-', "")
    text = re.sub(r"แขวง/เขต(\S+)", r"แขวง/เขต \1", text)
    text = replace_patterns(text)
    # text = add_province_prefix_to_text(text, PROVINCES)
    text = remove_urls(text)
    text = text.replace('แขวง.', ' แขวง')
    text = text.replace('เขต.', ' เขต')
    text = text.replace('ตำบล.', ' ตำบล')
    text = text.replace('อำเภอ.', ' อำเภอ')
    text = text.replace('จังหวัด.', ' จังหวัด')
    text = text.replace('จ.', ' จังหวัด')
    text = text.replace('อ.', ' อำเภอ')
    text = text.replace('ต.', ' ตำบล')
    text = text.replace('ต,', ' ตำบล')
    text = text.replace('อ,', ' อำเภอ')
    text = text.replace('จ,', ' จังหวัด')
    text = text.replace('ต_', ' ตำบล')
    text = text.replace('อ_', ' อำเภอ')
    text = text.replace('จ_', ' จังหวัด')
    text = text.replace('ต-', ' ตำบล')
    text = text.replace('อ-', ' อำเภอ')
    text = text.replace('จ-', ' จังหวัด')
    text = text.replace('ตำบล_', ' ตำบล')
    text = text.replace('อำเภอ_', ' อำเภอ')
    text = text.replace('จังหวัด_', ' จังหวัด')
    text = text.replace(' อเมือง', ' อ.เมือง')
    text = text.replace('*', '').replace('\\', '')
    text = text.replace('เขต ','เขต')
    text = text.replace('แขวง ','แขวง')
    text = text.replace("ต่างจังหวัด","")
    text = text.replace('จังหวัด ','จังหวัด')
    text = text.replace('ตำบล ','ตำบล')
    text = text.replace('อำเภอ ','อำเภอ')
    text = text.replace('กทม', ' กรุงเทพมหานคร ')
    text = text.replace("กทม.", "กรุงเทพมหานคร ")
    text = text.replace("กท.", "กรุงเทพมหานคร ")
    text = text.replace("กรุงเทพฯ", "กรุงเทพมหานคร ")
    text = text.replace("กรุงเทพ", "กรุงเทพมหานคร")
    text = text.replace("ที่อยู่", "")
    text = text.replace("นางสาว","")
    text = text.replace("นาย","")
    text = text.replace("นาง","")
    text = text.replace("ตำบล/แขวง","ตำบล")
    text = text.replace("อำเภอ/เขต","อำเภอ")
    
    # text = text.replace("บลท", "บ้านเลขที่ ")
    # text = text.replace("หมู่", "หมู่ ")
    # text = text.replace("ซ.", "ซอย ")
    # text = text.replace("ซอย", "ซอย ")
    # text = text.replace("ถ.", "ถนน ")
    # text = text.replace("ถนน", "ถนน ")
    # text = text.replace("หมุ่", "หมู่ ")
    # text = text.replace('ม.', "หมู่ ")
    # text = re.sub(r'(?<!\S)ม(?=\d)', 'หมู่', text)
    # text = re.sub(r'(?<!\S)บ(?!\S)', 'บ้าน', text)

    # Remove spaces from phone numbers in the text
    text = remove_spaces_from_phone_numbers(text)
    
    # Remove extra spaces caused by removing postal codes
    text = re.sub(r"\s{2,}", " ", text).strip()
    
    # Remove emojis from the text
    text = remove_emoji(text)
    
    # Join non-empty text fragments back into a single space-separated string
    text = " ".join([t for t in text.strip().split(" ") if t.strip() != ""])
    
    # Add a space before specific patterns in the text
    for pattern in patterns:
        text = re.sub(rf'({pattern})', r' \1', text)
    
    # Add a space before sequences of digits longer than 10 characters
    text = re.sub(r'(\d{10,})', r' \1', text)
    # Add a space between digits and Thai characters
    text = re.sub(r'(\d+)([ก-๙]+)', r'\1 \2', text)
    # Remove certain characters
    text = re.sub(r'~', '', text)
    # Remove parentheses
    text = re.sub(r'[\(\)]', '', text)
    # Remove the zero-width space character
    text = text.replace("\u200b", "")

    # Remove dots that are not part of a number or email
    text = re.sub(r'(?<!\S)\.(?!\S|\d{2,}@)', '', text)
    
    return text



def clean_location_text(text: str,phone_numbers=None,postal_numbers=None) -> str:
    """
    Clean location before using fuzzy string match
    """
    text = text.replace("แขวง", " ")
    text = text.replace("เขต", " ")
    text = text.replace("อำเภอ", " ")
    text = text.replace("ตำบล", " ")
    text = text.replace("ต.", " ")
    text = text.replace("ตฺ", "ต.")
    text = text.replace("อ.", " ")
    text = text.replace("จ.", " ")
    text = text.replace("คอหงส์", "คอหงษ์")
    text = text.replace("จังหวัด", "")
    text = text.replace("อำเภอ", "")
    text = text.replace("ตำบล", "")
    text = text.replace("เขต", "")
    text = text.replace("เเขวง", "")
    text = text.replace("จังห", "")
    text = re.sub(str(phone_numbers), '', text)
    text = re.sub(str(postal_numbers), '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    

    return text


def get_digit(text: str) -> str:
    """
    Get digit output from a given text
    """
    return "".join([c for c in text if c.isdigit()])


def is_stopword(word: str) -> bool:  # เช็คว่าเป็นคำฟุ่มเฟือย
    """
    Check if a word is stop word or not using PyThaiNLP

    Reference
    ----------
    Pythainlp, https://github.com/PyThaiNLP/pythainlp
    """
    return word in thai_stopwords()


def range_intersect(r1: range, r2: range):
    """
    Check if range is intersected

    References
    ----------
    Stack Overflow, https://stackoverflow.com/questions/6821156/how-to-find-range-overlap-in-python
    """
    return range(max(r1.start, r2.start), min(r1.stop, r2.stop)) or None


def merge_labels(preds: list):
    """
    Get merged labels and merge tuple to merge tokens
    """
    preds = list(np.ravel(preds))
    merge, labels = [], []
    s = 0
    for label, g in groupby(preds):
        g = list(g)
        labels.append(label)
        if len(g) > 1:
            merge.append((s, s + len(g)))
        s += len(g)
    return merge, labels


def merge_tokens(tokens: list, merge: list) -> list:
    """
    Merge tokens with an input merge

    References
    ----------
    Stack Overflow, https://stackoverflow.com/questions/43550219/merge-elements-in-list-based-on-given-indices
    """
    for t in merge[::-1]:
        merged = "".join(tokens[t[0] : t[1]])  # merging values within a range
        tokens[t[0] : t[1]] = [merged]  # slice replacement
    return tokens

def isThai(chr):
    
 cVal = ord(chr)
 if(cVal >= 3584 and cVal <= 3711):
  return True
 return False

def isThaiWord(word):
 t=True
 for i in word:
  l=isThai(i)
  if l!=True and i!='.':
   t=False
   break
 return t

def is_stopword(word):
    return word in stopwords
def is_s(word):
    if word == " " or word =="\t" or word=="":
        return True
    else:
        return False

def lennum(word,num):
    if len(word)==num:
        return True
    return False
