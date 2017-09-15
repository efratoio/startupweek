
# coding: utf-8

# In[11]:


import rdflib
import os, sys


def load_graph():
    g = rdflib.Graph()
    g.parse("../semantics_extraction/onto.nt", format="nt")
    return g

def check_in_ontology(g, a, b):
    q = "select * where {<http://o.org/"+a+"> ?p <http://o.org/"+b+">}"
    x1 = g.query(q)

    
    return x1


def get_type(g, a):
    q = "select ?t where {<http://o.org/"+a+"> <http://o.org/type> ?t}"
    x1 = g.query(q)
    for x in x1:
        print(a,x)


def normalize_word(w):
    # TODO: stemming
    return w.replace(",","").replace(".","").strip()
    
def check_text(g, text):
    text = text.split()
    for t1 in text:
        t1 = normalize_word(t1)
        for t2 in text:
            t2 = normalize_word(t2)
            if t1 != t2:
                check_in_ontology(g, t1, t2)
                
def geographic_difference(g, s, t):
    # Currently - equal weight for every connection
    s = s.lower().replace(" ", "_").split(",")[0].strip()
    t = t.lower().replace(" ", "_").split(",")[0].strip()
    if s == t:
        return 0
    return 3 - min(len(check_in_ontology(g, s, t)), 3)

def month_to_num(m):
    if m == "jan":
        return 1
    if m == "feb":
        return 2
    if m == "mar":
        return 3
    if m == "apr":
        return 4
    if m == "may":
        return 5
    if m == "jun":
        return 6
    if m == "jul":
        return 7
    if m == "aug":
        return 8
    if m == "sep":
        return 9
    if m == "oct":
        return 10
    if m == "nov":
        return 11
    if m == "dec":
        return 12
    return 0
    


def date_difference(s, t):
    if s is None or t is None:
        return 0
    s_splitted = s.lower().split()
    t_splitted = t.lower().split()
    t_d = s_d = 0
    s_m = month_to_num(s_splitted[0][0:3])
    if len(s_splitted) > 1:
        s_d = s_splitted[1]
    t_m = month_to_num(t_splitted[0][0:3])
    if len(t_splitted) > 1:
        t_d = t_splitted[1]
    
    #print(s_m, s_d, t_m, t_d)
    ret = 0
    try:
        ret = abs((s_m*30+int(s_d)) - (t_m*30+int(t_d)))
    except:
        pass
    return ret

# In[13]:


import json

def load_json(frame_path):
    chat_file = open(frame_path, encoding='utf-8')
    chat = json.load(chat_file)
    return chat

def prepare_frames_vector(chat,g):
    final_vector = []
    dst_city = or_city = str_date = end_date = ""
    or_city_diff = dst_city_diff = str_date_diff = end_date_diff = 0
    for t in chat["turns"]:
        if "text" in t:
            prev_or_city = or_city
            prev_dst_city = dst_city
            prev_str_date = str_date
            prev_end_date = end_date
            
            for f in t["labels"]["frames"]:
                #print(f["info"])
                info = f["info"]
                if "or_city" in info:
                    or_city = info["or_city"][0]["val"]
                if "dst_city" in info:
                    dst_city = info["dst_city"][0]["val"]
                if "str_date" in info:
                    str_date = info["str_date"][0]["val"]
                if "end_date" in info:
                    end_date = info["end_date"][0]["val"]

            if len(prev_or_city) > 1:
                or_city_diff = geographic_difference(g, prev_or_city, or_city)
            if len(prev_dst_city) > 1:
                dst_city_diff = geographic_difference(g, prev_dst_city, dst_city)
                
            if prev_str_date is not None and len(prev_str_date) > 1:
                str_date_diff = date_difference(str_date, prev_str_date)
            if prev_end_date is not None and len(prev_end_date) > 1:
                end_date_diff = date_difference(end_date, prev_end_date)

            # print()
            # print(t["author"])
            # print(t["text"])                
            # print(or_city_diff, dst_city_diff, str_date_diff, end_date_diff)
            
            final_vector.append([ or_city_diff+dst_city_diff, str_date_diff+end_date_diff])
    
    return final_vector
                    

