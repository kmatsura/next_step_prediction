import os
import re
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from ginza import *
import spacy
from dotenv import load_dotenv
from pprint import pprint


def sub_number_and_symbol(sentence):
    """
    記号及び英数字の除去
    """
    # 半角全角英数字除去
    sentence = re.sub(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+', " ", sentence)
    # 記号もろもろ除去
    sentence = re.sub(
        r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□☸♥°♬(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", sentence)
    return sentence


def words_division(sentences_list, padding=True):
    """
    ginzaを使って、形態素解析。単語ID用の辞書も作る。
    """
    nlp = spacy.load("ja_ginza")  # GiNZAモデルの読み込み
    word2index = {}
    if padding == True:  # パディング文字列の追加
        word2index["<pad>"] = 0
    divided_list = []
    max_len = 0  # 系列の最大長さを調べる用
    for sentences in sentences_list:
        # print(sentences)
        sentences = sub_number_and_symbol(sentences)  # 記号英数字の除去
        try:
            docs = nlp(sentences)
        except:
            print("error")
            print(sentences)
        tmp = []
        for sent in docs.sents:
            for token in sent:
                tmp.append(token.lemma_)
                if token.lemma_ in word2index:
                    continue
                else:
                    word2index[token.lemma_] = len(word2index)
        divided_list.append(tmp)
        if max_len < len(tmp):
            max_len = len(tmp)
    return divided_list, word2index, max_len


def main(is_test = False):
    load_dotenv(verbose=True)
    BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATADIR = os.path.join(BASEDIR, os.environ.get("DATADIR"))
    FILENAME = "raw_data.csv"
    assert os.path.exists(os.path.join(DATADIR, FILENAME)
                          ), "input file not found"
    OUTPUTFILENAME = "raw_data_preprocessed.pkl"
    datasets = pd.read_csv(os.path.join(DATADIR, FILENAME))
    if is_test == True:
        datasets = datasets.iloc[:100, :]  # test用
    datasets = datasets.dropna()
    datasets = datasets.sort_values(['title', 'pos'], ascending=[True, True])  # datasetsのソート
    # 手順の単語分割
    datasets["memo_divided"], word2index, max_len = words_division(list(datasets["memo"]))
    # 入出力データの作成
    input_data = []
    output_data = []
    for i in range(len(datasets)-1):
        if datasets.iloc[i]["title"] == datasets.iloc[i+1]["title"]:
            try: 
                input_data.append([word2index[w] for w in datasets.iloc[i]["memo_divided"]])
                output_data.append([word2index[w] for w in datasets.iloc[i+1]["memo_divided"]])
            except:
                print("error")
                print(datasets.iloc[i]["memo_divided"])
                print(datasets.iloc[i+1]["memo_divided"])
        else:
            continue
    
    # データの保存
    with open(os.path.join(DATADIR, OUTPUTFILENAME), 'wb') as f:
        pickle.dump((word2index, max_len, input_data, output_data), f)

if __name__ == "__main__":
    main()
