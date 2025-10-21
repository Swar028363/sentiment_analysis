import re

def has_html(s):
    return bool(re.findall(r"<.*?>",s))

def get_words(s):
    return re.findall(r"\w+", s)

def get_charecters(s):
    return re.findall(r"\w", s)

def count_words(s):
    return len(re.findall(r"\w+", s))

def count_charecters(s):
    return len(re.findall(r"\w", s))