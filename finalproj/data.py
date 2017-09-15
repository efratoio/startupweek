
# coding: utf-8

import semantic
import os
import json
import numpy as np
from nltk import tokenize
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
def build_label_dict():
    d = {}
    label_path = r"../Frames-dataset/labels.txt"
    label_file = open(label_path, encoding='utf-8')
    for line in label_file:
        k, v = line.split(",")
        d[k] = True if v.strip()=="True" else False
    return d




#returns chat dictionary that include the label of each chat, and turns where each turn has sentences list and elapsed 
#time + the speaker id

def gen_chat_data():
    chat_path = r"../Frames-dataset/chats"
    chats = {}
    d = build_label_dict()
    for filename in os.listdir(chat_path):
        chat_file = open(os.path.join(chat_path, filename), encoding='utf-8')
        chat = json.load(chat_file)
        
        turns = []
        if 'turns' in chat:
            tsp = chat['turns'][0]['timestamp']
       
            for turn in chat['turns']:
               
                ts = turn['timestamp'] - tsp
                tsp = turn['timestamp']
                turns.append({"ti":ts,"text":turn["text"],"author":turn["author"]})

        chats[filename[:-5]] = {}
        chats[filename[:-5]]["turns"] = turns
        chats[filename[:-5]]["label"] = d[filename[:-5]]
        chats[filename[:-5]]["chat"] = chat
    return chats


def load_config(config_path):
    config_file = open(config_path, encoding='utf-8')
    return json.load(config_file)

def create_data(config,chats):
    texts = []
    labels=[]
    chats_txt=[]
    vecs=[]
    turns=[]
    g = None
    if config["SEMANTIC"]:
        g = semantic.load_graph()
    if config["VANILA"]:
        for idx in chats.keys():

            labels.append(chats[idx]["label"])
            if config["PARTIAL"]:
                text = "\n".join([x["text"] for x in chats[idx]["turns"][:config["LENGTH"]]])
                
            else:
                text = "\n".join([x["text"] for x in chats[idx]["turns"]])
            texts.append(text)
            sentences = tokenize.sent_tokenize(text)
            turns.append(sentences)

        tokenizer = Tokenizer(num_words=config["MAX_NB_WORDS"])
        tokenizer.fit_on_texts(texts)

        data = np.zeros((len(texts), config["MAX_SENTS"], config["MAX_SENT_LENGTH"]), dtype='int32')
        for i, sentences in enumerate(turns):
            if config["PARTIAL"] and i==config["LENGTH"]:
                break;
            for j, sent in enumerate(sentences):
                if j< config["MAX_SENTS"]:
                    wordTokens = text_to_word_sequence(sent)
                    k=0
                    for _, word in enumerate(wordTokens):
                        if k<config["MAX_SENT_LENGTH"] and tokenizer.word_index[word]<config["MAX_NB_WORDS"]:
                            data[i,j,k] = tokenizer.word_index[word]
                            k=k+1   
        return data,labels,vecs,tokenizer

    else:
        for idx in chats.keys():
            sem_vec = None
            if config["SEMANTIC"]:
                sem_vec = semantic.prepare_frames_vector(chats[idx]["chat"],g)

            labels.append(chats[idx]["label"])
            if config["PARTIAL"]:
                text = "\n".join([x["text"] for x in chats[idx]["turns"][:config["LENGTH"]]])
                
            else:
                text = "\n".join([x["text"] for x in chats[idx]["turns"]])
            turns=[]
            vec=[]
            for i,turn in enumerate(chats[idx]["turns"]):
                v = []
                if config["PROPS"]:
                    v.extend([len(turn["text"]),turn["ti"],0 if turn["author"].lower()=="wizard" else 1])
                if config["SEMANTIC"]:
                    v.extend(sem_vec[i])


                vec.append(v)

                texts.append(turn["text"])
                sentences = tokenize.sent_tokenize(turn["text"])
                turns.append(sentences)
            vecs.append(vec)
            chats_txt.append(turns)
        
        tokenizer = Tokenizer(num_words=config["MAX_NB_WORDS"])
        tokenizer.fit_on_texts(texts)

        data = np.zeros((len(chats_txt), config["MAX_TURNS"],config["MAX_SENTS"], config["MAX_SENT_LENGTH"]), dtype='int32')
        for m, turns in enumerate(chats_txt):
            for i, sentences in enumerate(turns):
                if config["PARTIAL"] and i==config["LENGTH"]:
                    break;
                for j, sent in enumerate(sentences):
                    if j< config["MAX_SENTS"]:
                        wordTokens = text_to_word_sequence(sent)
                        k=0
                        for _, word in enumerate(wordTokens):
                            if k<config["MAX_SENT_LENGTH"] and tokenizer.word_index[word]<config["MAX_NB_WORDS"]:
                                data[m,i,j,k] = tokenizer.word_index[word]
                                k=k+1   
        return data,labels,vecs,tokenizer




def prepare_datasets(config,chats):
    if config["PARTIAL"]:
        chats = {k: v for k, v in chats.items() if len(v["turns"])>config["LENGTH"]}
 

    data,labels,vecs,tokenizer = create_data(config,chats)

    aux_data=[]
    if config["PROPS"] or config["SEMANTIC"]:
        aux_data = np.zeros((data.shape[0], config["MAX_TURNS"],config["VEC_SIZE"]), dtype='float32')
        for i,vec in enumerate(vecs):
            if config["PARTIAL"]:
                for j,v in enumerate(vec[:config["LENGTH"]]):
                    aux_data[i,j,:]= np.array(v)
            else:
                for j,v in enumerate(vec):
                    aux_data[i,j,:]= np.array(v)
        print(vecs[:4])
        print(aux_data[:4])
        norm_hlp = aux_data.reshape((data.shape[0]*config["MAX_TURNS"],config["VEC_SIZE"]))
        norm_hlp = norm_hlp.max(axis=0).astype(np.float)
        print(norm_hlp)
        norm_hlp[(norm_hlp==0)] = 1
        aux_data = aux_data.astype(np.float)/norm_hlp

    word_index = tokenizer.word_index
    print('Total %s unique tokens.' % len(word_index))

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(config["VALIDATION_SPLIT"] * data.shape[0])
    nb_test_samples = int(config["TEST_SPLIT"] * data.shape[0])

    if "RNN" in config.keys():
        x_train = aux_data[:-(nb_validation_samples+nb_test_samples)]
        y_train = labels[:-(nb_validation_samples+nb_test_samples)]
        x_val = aux_data[-(nb_validation_samples+nb_test_samples):-nb_test_samples]
        y_val = labels[-(nb_validation_samples+nb_test_samples):-nb_test_samples]
        x_test = aux_data[-nb_test_samples:]
        y_test = labels[-nb_test_samples:]
        return word_index,x_train, y_train,x_val, y_val,x_test,y_test       
    if config["PROPS"]:
        x_train = (data[:-(nb_validation_samples+nb_test_samples)],aux_data[:-(nb_validation_samples+nb_test_samples)])
        y_train = labels[:-(nb_validation_samples+nb_test_samples)]
        x_val = (data[-(nb_validation_samples+nb_test_samples):-nb_test_samples],aux_data[-(nb_validation_samples+nb_test_samples):-nb_test_samples])
        y_val = labels[-(nb_validation_samples+nb_test_samples):-nb_test_samples]
        x_test = (data[-nb_test_samples:],aux_data[-nb_test_samples:])
        y_test = labels[-nb_test_samples:]
        return word_index,x_train, y_train,x_val, y_val,x_test,y_test
    
    else:
        x_train = data[:-(nb_validation_samples+nb_test_samples)]
        y_train = labels[:-(nb_validation_samples+nb_test_samples)]
        x_val = data[-(nb_validation_samples+nb_test_samples):-nb_test_samples]
        y_val = labels[-(nb_validation_samples+nb_test_samples):-nb_test_samples]
        x_test = data[-nb_test_samples:]
        y_test = labels[-nb_test_samples:]
    
        return word_index,x_train, y_train,x_val, y_val,x_test,y_test
