import pandas as pd
data = pd.read_csv('C:/Users/Nikita/OneDrive/Desktop/Course_Materials/NLP/Assignments/Assignment2/offensive-language-detection/data/olid-subset-diagnostic-tests.csv')
import nltk
import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb
data_text = data['text']
import spacy
nlp = spacy.load('en_core_web_sm')
pdata_text = list(nlp.pipe(data_text))
negation_text = Perturb.perturb(pdata_text, Perturb.add_negation, keep_original=False)