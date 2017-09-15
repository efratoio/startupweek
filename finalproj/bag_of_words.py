
# coding: utf-8

# In[44]:


#UTF-8
import json
import pickle
from os import path
import os,re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model

dict_file_path = path.join("..","Frames-dataset","word_dict")
frames_file_path = path.join("..","Frames-dataset","frames.json")
chat_file_path = path.join("..","Frames-dataset","chats_dict")
chatvec_file_path = path.join("..","Frames-dataset","chats_vecs")
chat_text_file_path = path.join("..","Frames-dataset","chats_text")


# In[28]:


def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word

def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset): return word
    else: return "UUUNKKK" # unknown token



# In[17]:



def create_dicts():
    chats = []
    with open(frames_file_path,'r') as f:
        chats = json.load(f)

    word_dict = {}    
    for chat in chats:
        chat_word_dict={}
        for turn in chat["turns"][:8]:
            for word in turn["text"]:
                word = canonicalize_word(word)
                if word in word_dict:
                    word_dict[word]+=1
                else:
                    word_dict[word]=1

                if word in chat_word_dict:
                    chat_word_dict[word]+=1
                else:
                    chat_word_dict[word]=1
        with open(path.join(chat_file_path,chat["id"]+".dict"),"w") as f:
            pickle.dump(chat_word_dict,f)
    
    with open(dict_file_path,"w") as f:
        pickle.dump(word_dict,f)
        


# In[18]:


def create_vectors():
    with open(dict_file_path,'r') as f:
        word_dict  = pickle.load(f)
    lst = []
    for key in word_dict:
        lst.append(key)
    
    for file_name in os.listdir(chat_file_path):
        vec = []
        ind = file_name.split('.')[0]
        chat_dict={}
        with open(path.join(chat_file_path,file_name),'r') as f:
            chat_dict = pickle.load(f)
        for word in lst:
            if word in chat_dict:
                vec.append(chat_dict[word])
            else:
                vec.append(0)
        vec = np.array(vec)
        with open(path.join(chatvec_file_path,ind),'w') as f:
            pickle.dump(vec,f)
        
       


# In[85]:


def extract_text(prnct):
    global chat_text_file_path
    chat_text_file_path = path.join("..","Frames-dataset","chats_text1")
    chats = []
    with open(frames_file_path,'r') as f:
        chats = json.load(f)

    for chat in chats:
        text=[]
        booked="False"
        for i,turn in enumerate(chat["turns"]):
            if i <8:
                for word in turn["text"].split(" "):
                    word = canonicalize_word(word)
                    text.append(word)       
            for arg in turn['labels']['acts']:
                for d in arg['args']:
                    if d['key'] == 'action' and d['val'] == 'book':
                        booked="True"
        if len(text)>0:
            if not path.exists(chat_text_file_path+str(4)):
                os.makedirs(chat_text_file_path+str(4))
            with open(path.join(chat_text_file_path+str(prnct),chat["id"]+"."+booked),"w") as f:
                f.write(" ".join(text).strip())

prc = 1
extract_text(prc)
    


# In[89]:


chat_text_file_path= path.join("..","Frames-dataset","chats_text1")+str(prc)

filenames = [os.path.join(chat_text_file_path,f) for f in np.array(os.listdir(chat_text_file_path))]
    # filenames=[]
    # for file_name in filenames1:
    #     file_name = os.path.join(chat_text_file_path,file_name)
    #     with open(file_name,"r") as file:
    #         chat = json.load(file)
    #         if len(chat["turns"])>=4:
    #             filenames.append(file_name)



# filenames_with_path = [os.path.join(chat_text_file_path, fn) for fn in filenames]

# tragedies and comedies are coded with 'TR' or 'CO',
# e.g., PCorneille_TR-V-1647-Heraclius0001.txt
booked = []

for fn in filenames:
    booked.append(True if fn[-4:]=="True" else False)

booked = np.array(booked)

# print(len(booked))
# print(booked.sum()/float(len(booked)))
# .strip() removes the trailing newline '\n' from each line in the file
print(filenames)    
vectorizer = CountVectorizer(input='filename', min_df=15, max_df=.95, stop_words='english', max_features=3000)

dtm = vectorizer.fit_transform(filenames)

dtm = dtm.toarray()

vocab = np.array(vectorizer.get_feature_names())


X_train, X_test, y_train, y_test = train_test_split(
    dtm, booked, test_size=0.3, random_state=42)

logreg = linear_model.LogisticRegression(
            multi_class='multinomial', max_iter=128, solver='lbfgs', C=1000000, verbose=1)

logreg.fit(X_train, y_train)

prediction =  logreg.predict(X_test)
print(y_test == prediction)
print((y_test == prediction).sum()/ float(len(y_test)))


