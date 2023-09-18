import re

class DataCleaner():
    
    def __init__(self):
        pass
        
    def process(self, content):
        content = re.compile('<[^<>]+>').sub('', content)
        content = re.compile('[0-9]{1,2}\\/[0-9]{1,2}\\/[0-9]{4}').sub(' datemask ', content)
        content = re.compile('[0-9]+').sub('numbermask', content)
        content = re.compile('(http|https)://[^\s]*').sub('httpaddrmask', content)
        content = re.compile('[^\s]+@[^\s]+').sub('emailaddrmask', content)
        content = re.compile('[$]+').sub('dollarmask', content)
        return content