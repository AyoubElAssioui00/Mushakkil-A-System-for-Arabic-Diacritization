import pickle



#Load the data

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]
    
#Load Letters, Classes, Diacritics

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
#Transformations
#-------------------------------------------------------------------------------
'''transliteration'''

ARABIC_LETTERS = [
  u'ء', u'آ', u'أ', u'ؤ', u'إ',
  u'ئ', u'ا', u'ب', u'ة', u'ت',
  u'ث', u'ج', u'ح', u'خ', u'د',
  u'ذ', u'ر', u'ز', u'س', u'ش',
  u'ص', u'ض', u'ط', u'ظ', u'ع',
  u'غ', u'ـ', u'ف', u'ق', u'ك',
  u'ل', u'م', u'ن', u'ه', u'و',
  u'ى', u'ي', u'ً', u'ٌ', u'ٍ',
  u'َ', u'ُ', u'ِ', u'ّ', u'ْ',
]

SYMBOL_LETTERS = [
  '\'', '|', '>', '&', '<',
  '}', 'A', 'b', 'p', 't',
  'v', 'j', 'H', 'x', 'd',
  '*', 'r', 'z', 's', '$',
  'S', 'D', 'T', 'Z', 'E',
  'g', '_', 'f', 'q', 'k',
  'l', 'm', 'n', 'h', 'w',
  'Y', 'y', 'F', 'N', 'K',
  'a', 'u', 'i', '~', 'o'
]

def transliteration(data, domain, range):
    d = { u:v for u, v in zip(domain, range) }

    new_lines = list()
    for line in data:
      new_line = ''
      for ch in line.strip():
        if ch in d.keys():
          new_line += d[ch]
        else:
          new_line += ch
      new_lines.append(new_line)

    return new_lines

'''Clean the data'''


def clean_text(text,arabic_letters,diacritics_list):
  L=[]
  for i in range(len(text)):
    L+=[''.join([ch for ch in text[i] if ch in arabic_letters or ch in diacritics_list or ch==' ' or ch=="."])]
  return L

'''Prepare for training'''

def split_sentence(sentence,arabic_letters):
  data=[]
  k=len(sentence)
  for i in range(k):
    if sentence[i] not in arabic_letters+' ':
      continue
    if sentence[i]==' ':
       data+=[(' ',"")]
    else:
       j=i+1
       while j<k-1 and sentence[j] not in arabic_letters+' ':
          j+=1
       if j==i+1:
          data+=[(sentence[i],"")]
       else:
          data+=[(sentence[i],sentence[i+1:j])]        
  return data

def split_train(text,arabic_letters):
   
   return [split_sentence(text[i],arabic_letters) for i in range(len(text))]

'''Prepare for testing'''

def extract_observation(text,arabic_letters):
   data=split_train(text,arabic_letters)
   test_observations=states=[]
   for i in data:
      test_observations+=[[a[0] for a in i]]
                     
   return test_observations