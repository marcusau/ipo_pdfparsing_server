#/usr/bin/env python
# -*- coding: utf-8 -*-
import os, pathlib, sys,click,logging,string,csv,json,tempfile,re,filetype

sys.path.append(os.getcwd())

parent_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(parent_path))

master_path = parent_path.parent
sys.path.append(str(master_path))

project_path = master_path.parent
sys.path.append(str(project_path))


from itertools import chain
from typing import Union,Dict,List,Tuple,Optional
from collections import OrderedDict,Counter

import time
from datetime import date,datetime,timedelta
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


import requests
import fitz

import cleantext
import xml.etree.ElementTree as ET
from rapidfuzz import  fuzz
from rapidfuzz import process as fuzz_process
from intervaltree import Interval, IntervalTree
from pyflashtext import KeywordProcessor

from ML.eng_preprocessor import preprocess_pdf as eng_preprocess
from Config.setting import config

import hug

temp_dir=tempfile.TemporaryDirectory()


###---------------------------------------------------------------------------------------------------------------------------------------------

def download_asa(stockcode:Union[str,int],eng_asa_url:str,chi_asa_url:str):
    eng_pdf_path,chi_pdf_path=os.path.join(temp_dir.name,f"{str(stockcode)}_eng.pdf"),  os.path.join(temp_dir.name, f"{str(stockcode)}_chi.pdf")


    eng_asa_response = requests.get(config.external_API.pdf+eng_asa_url)
    eng_kind = filetype.guess(eng_asa_response.content)

    chi_asa_response = requests.get(config.external_API.pdf + chi_asa_url)
    chi_kind = filetype.guess(chi_asa_response.content)

    if eng_kind is None:
        return {"status":"error", 'content':f"cannot guess file type of {config.external_API.pdf+eng_asa_url} "}
    elif chi_kind is None:
        return  {"status":"error", 'content':f"cannot guess file type of {config.external_API.pdf +chi_asa_url} "}
    elif eng_kind.extension is not 'pdf':
        return  {"status":"error", 'content':f"this is not pdf file in  {config.external_API.pdf + eng_asa_url} "}
    elif chi_kind.extension is not 'pdf':
        return  {"status":"error", 'content':f"this is not pdf file in {config.external_API.pdf + eng_asa_url} "}

    else:

        with open(eng_pdf_path, 'wb') as f:
            f.write(eng_asa_response.content)

        print(f'finish stockcode: {stockcode} , eng  : asa url :{eng_asa_url} and save pdf to : {eng_pdf_path}')


        with open(chi_pdf_path, 'wb') as f:
            f.write(chi_asa_response.content)

        print(f'finish stockcode: {stockcode} , chi : asa url :{chi_asa_url} and save pdf to : {chi_pdf_path},\n')

        return  {"status":"sucess", 'content':{'eng':eng_pdf_path,'chi':chi_pdf_path}}
###----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

search_word_path=config.search_word_file
word4search={}
with open(search_word_path,'r',encoding='utf-8') as f:
    for line in f.readlines() :
        t_, lang, t_words=line.strip().split('\t')
        if t_ not in word4search:
            word4search[t_]={'chi':[], 'eng':[]}
        word4search[t_][lang].append(t_words)


#########-----------------------------------------------------------------------------------------------------------------

def search_toc(toc_pages,tag:str='parties',lang:str='eng'):
    fuzz_toc = {t: fuzz_process.extractOne(t, choices=word4search[tag][lang], scorer=fuzz.WRatio)[1] for t in    toc_pages}
    tag_toc = max(fuzz_toc, key=fuzz_toc.get)
    tag_pages=toc_pages[tag_toc]
    tag_pages=list(filter(lambda x:x>=0,tag_pages))
    return tag_pages,tag_toc

#########-----------------------------------------------------------------------------------------------------------------

def fetch_toc_pages(fitz_toc):
    raw_toc = OrderedDict()
    for i, item in enumerate(fitz_toc):
        lvl, title, pno, ddict = item

        row_text = str(title, )

        row_text = ' '.join([w.capitalize() for w in row_text.split(' ')])
        row_text = cleantext.replace_emails(row_text,'')
        row_text = cleantext.replace_urls(row_text,'')
        row_text = cleantext.normalize_whitespace(row_text)

        toc_pagenum = int(pno)
        raw_toc[i] = {'content': row_text, 'start': toc_pagenum, }

    toc = {}
    for k, v in raw_toc.items():
        if k < max(raw_toc.keys()):
            start = v['start'] - 1
            end = int(raw_toc[k + 1].get('start')) - 1
            toc[v['content']] = list(range(start, end + 1))  # {'start': start, 'end': end}
    return toc

#########-----------------------------------------------------------------------------------------------------------------


def locate_pages(pdf_file_path,tag='parties',lang=lang):

    doc = fitz.open(pdf_file_path)
    fitz_toc = doc.getToC(simple = False)
    if len(fitz_toc) == 0:
        print("No Table of Contents available")
        return None,None
    else:
        toc_pages=fetch_toc_pages(fitz_toc)
        pi_pages,tag_toc =search_toc(toc_pages, tag= tag, lang=lang)
        if not pi_pages:
            print(f'cannot find the pages of "directors and parties involved for stocks:{pdf_file_path}')
            return None,None
        else:
            return pi_pages, tag_toc

#########-----------------------------------------------------------------------------------------------------------------



def words2indices(origin, vocab):
    if isinstance(origin[0], list):
        result = [[vocab[w] for w in sent] for sent in origin]
    else:
        result = [vocab[w] for w in origin]
    return result

def indices2words(origin, vocab):
    if isinstance(origin[0], list):
        result = [[vocab.id2word(w) for w in sent] for sent in origin]
    else:
        result = [vocab.id2word(w) for w in origin]
    return result


class Vocab:
    def __init__(self, word2id, id2word):
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'
        self.START = '<START>'
        self.END = '<END>'
        self.__word2id = word2id
        self.__id2word = id2word

    def get_word2id(self):
        return self.__word2id

    def get_id2word(self):
        return self.__id2word

    def __getitem__(self, item):
        if self.UNK in self.__word2id:
            return self.__word2id.get(item, self.__word2id[self.UNK])
        return self.__word2id[item]

    def __len__(self):
        return len(self.__word2id)

    def id2word(self, idx):
        return self.__id2word[idx]

    @staticmethod
    def build(data, max_dict_size, freq_cutoff, is_tags):

        word_counts = Counter(chain(*data))
        valid_words = [w for w, d in word_counts.items() if d >= freq_cutoff]
        valid_words = sorted(valid_words, key=lambda x: word_counts[x], reverse=True)
        valid_words = valid_words[: max_dict_size]
        valid_words += ['<PAD>']
        word2id = {w: idx for idx, w in enumerate(valid_words)}
        if not is_tags:
            word2id['<UNK>'] = len(word2id)
            valid_words += ['<UNK>']
        return Vocab(word2id=word2id, id2word=valid_words)

    def save(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({'word2id': self.__word2id, 'id2word': self.__id2word}, f, ensure_ascii=False)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            entry = json.load(f)
        return Vocab(word2id=entry['word2id'], id2word=entry['id2word'])


def pad(data, padded_token, device):
    lengths = [len(sent) for sent in data]
    max_len = lengths[0]
    padded_data = []
    for s in data:
        padded_data.append(s + [padded_token] * (max_len - len(s)))
    return torch.tensor(padded_data, device=device), lengths


class BiLSTMCRF(nn.Module):
    def __init__(self, sent_vocab, tag_vocab, dropout_rate=0.5, embed_size=256, hidden_size=256):
        super(BiLSTMCRF, self).__init__()
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.sent_vocab = sent_vocab
        self.tag_vocab = tag_vocab
        self.embedding = nn.Embedding(len(sent_vocab), embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=True)
        self.hidden2emit_score = nn.Linear(hidden_size * 2, len(self.tag_vocab))
        self.transition = nn.Parameter(torch.randn(len(self.tag_vocab), len(self.tag_vocab)))  # shape: (K, K)

    def forward(self, sentences, tags, sen_lengths):
        mask = (sentences != self.sent_vocab[self.sent_vocab.PAD]).to(self.device)  # shape: (b, len)
        sentences = sentences.transpose(0, 1)  # shape: (len, b)
        sentences = self.embedding(sentences)  # shape: (len, b, e)
        emit_score = self.encode(sentences, sen_lengths)  # shape: (b, len, K)
        loss = self.cal_loss(tags, mask, emit_score)  # shape: (b,)
        return loss

    def encode(self, sentences, sent_lengths):
        padded_sentences = pack_padded_sequence(sentences, sent_lengths)
        hidden_states, _ = self.encoder(padded_sentences)
        hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True)  # shape: (b, len, 2h)
        emit_score = self.hidden2emit_score(hidden_states)  # shape: (b, len, K)
        emit_score = self.dropout(emit_score)  # shape: (b, len, K)
        return emit_score

    def cal_loss(self, tags, mask, emit_score):
        batch_size, sent_len = tags.shape
        # calculate score for the tags
        score = torch.gather(emit_score, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)  # shape: (b, len)
        score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)  # shape: (b,)
        # calculate the scaling factor
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, sent_len):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  # shape: (uf, 1, K)
            emit_and_transition = emit_score[: n_unfinished, i].unsqueeze(dim=1) + self.transition  # shape: (uf, K, K)
            log_sum = d_uf.transpose(1, 2) + emit_and_transition  # shape: (uf, K, K)
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)  # shape: (uf, 1, K)
            log_sum = log_sum - max_v  # shape: (uf, K, K)
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)  # shape: (uf, 1, K)
            d = torch.cat((d_uf, d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)  # shape: (b, K)
        max_d = d.max(dim=-1)[0]  # shape: (b,)
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)  # shape: (b,)
        llk = total_score - d  # shape: (b,)
        loss = -llk  # shape: (b,)
        return loss

    def predict(self, sentences, sen_lengths):
        batch_size = sentences.shape[0]
        mask = (sentences != self.sent_vocab[self.sent_vocab.PAD])  # shape: (b, len)
        sentences = sentences.transpose(0, 1)  # shape: (len, b)
        sentences = self.embedding(sentences)  # shape: (len, b, e)
        emit_score = self.encode(sentences, sen_lengths)  # shape: (b, len, K)
        tags = [[[i] for i in range(len(self.tag_vocab))]] * batch_size  # list, shape: (b, K, 1)
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, sen_lengths[0]):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  # shape: (uf, 1, K)
            emit_and_transition = self.transition + emit_score[: n_unfinished, i].unsqueeze(dim=1)  # shape: (uf, K, K)
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition  # shape: (uf, K, K)
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()  # list, shape: (nf, K)
            tags[: n_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
            d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_unfinished:]), dim=0)  # shape: (b, 1, K)
        d = d.squeeze(dim=1)  # shape: (b, K)
        _, max_idx = torch.max(d, dim=1)  # shape: (b,)
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]
        return tags

    def save(self, filepath):
        params = { 'sent_vocab': self.sent_vocab,  'tag_vocab': self.tag_vocab, 'args': dict(dropout_rate=self.dropout_rate, embed_size=self.embed_size, hidden_size=self.hidden_size),
            'state_dict': self.state_dict() }
        torch.save(params, filepath)

    @staticmethod
    def load(filepath, device_to_load):
        params = torch.load(filepath, map_location=lambda storage, loc: storage)
        model = BiLSTMCRF(params['sent_vocab'], params['tag_vocab'], **params['args'])
        model.load_state_dict(params['state_dict'])
        model.to(device_to_load)
        return model

    @property
    def device(self):
        return self.embedding.weight.device

def get_chunk_type(tag_name):

   # tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type

def get_chunks(seq):

    default = "O"#tags["O"]
   # idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None

    for i, tok in enumerate(seq):
        if tok == default:
            if chunk_type is not None:
                chunk_type, chunk_start = None, None
            else:
                pass
        else:
            if re.search('\-',tok):
                tok_class, tok_chunk_type = tok.split('-')[0],tok.split('-')[1]
                if tok_class == 'S':
                    chunk = (tok_chunk_type, i, i+1)
                    chunks.append(chunk)
                    chunk_type, chunk_start = None, None
                if tok_class == 'B':
                    chunk_start = i
                    chunk_type = tok_chunk_type
                if tok_class == 'I':
                    if chunk_type is not None:
                        if chunk_type == tok_chunk_type:
                            pass
                        else:
                            chunk_type, chunk_start = None, None
                    else:
                        pass
                if tok_class == 'E':
                    if chunk_type is not None:
                        if chunk_type == tok_chunk_type:
                            chunk = (chunk_type, chunk_start, i+1)
                            chunks.append(chunk)
                            chunk_type, chunk_start = None, None
                        else:
                            chunk_type, chunk_start = None, None
                    else:
                        pass
    return chunks


###--------------------------------------------------


ner_eng_model_file=config.ner_models.eng.model
ner_eng_sen_vocabfile=config.ner_models.eng.sent_vocab
ner_eng_tag_vocabfile=config.ner_models.eng.tag_vocab

eng_sent_vocab = Vocab.load(ner_eng_sen_vocabfile)
eng_tag_vocab = Vocab.load(ner_eng_tag_vocabfile)

device = torch.device('cpu' )
ner_eng_model = BiLSTMCRF.load(filepath=ner_eng_model_file,device_to_load=device)
#
print('start testing...')
ner_eng_model.eval()
print('using device', device)
eng_max_len=config.ner_models.eng.max_len

def scan_ner(text:str):
    with torch.no_grad():
        text=text.replace('. .','')
        text=text.replace('*','')
        ori_tokens = [w for w in text.split(' ') if w !='']

        tokens = ['<START>'] + ori_tokens + ['<END>']

        tokens_idx = words2indices([tokens], eng_sent_vocab)[0]

        lengths = len(tokens_idx)

        padded_data = tokens_idx + [eng_sent_vocab[eng_sent_vocab.PAD]] * (eng_max_len - len(tokens_idx))
        padded_tokens_idx, tokens_idx_len = torch.tensor([padded_data], device=device), [lengths]
        pred_tag_idx = ner_eng_model.predict(padded_tokens_idx, tokens_idx_len)[0][1:-1]
        pred_tags = [eng_tag_vocab.id2word(p) for p in pred_tag_idx]
        return ori_tokens,pred_tags

# ###---------------------------------------------------------------------------------------------------------------------------------------------


flags = fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE  | fitz.TEXT_PRESERVE_SPANS


def fetch_pdf(pdf_path,tag='sponsors'):
    lang='eng' if 'eng' in pdf_path else 'chi'
    doc = fitz.open(pdf_path)

    assert tag.lower() in ['underwriting','sponsors','sponsor','underwriter','underwriters']

    tag_search='underwriting' if tag.lower() in ['underwriting','underwriter','underwriters'] else 'parties'
    tag_pages,tag_toc=locate_pages(pdf_path,tag=tag_search,lang=lang)
   # print(f"tag:{tag}, toc page :{tag_pages}")
    if  tag_pages is None:
        return {'status':'error','content':'cannot locate table of content of pdf, please check filetype and make sure the file is IPO prospectus'}
    else:
        pi_texts = []
        tag_pages_search=tag_pages[:1] if  tag.lower() in ['underwriting','underwriter','underwriters'] else tag_pages
        for pi_page in tag_pages_search:
            pi_page_xhtml = doc[pi_page].get_textpage(flags).extractXHTML()
            for tt in ET.fromstring(pi_page_xhtml).itertext():
                tt = cleantext.normalize_whitespace(tt.strip())
                tt = cleantext.replace_emails(tt.strip())
                tt = cleantext.replace_urls(tt.strip())
                tt = cleantext.fix_bad_unicode(tt)
                tt = re.sub(tag_toc, '', tt, flags=re.I)
                tt = re.sub('\— \d+ \—|– \d+ \–|\− \d+ \−', '', tt)
                tt = re.sub('�', '-', tt)
                if tt:  # and not re.search('\– \w+ \–|\— \d+ \—|– \d+ \–', tt):
                    pi_texts.append(tt.strip())
        pi_texts = '\n'.join(pi_texts)
        pi_texts = re.sub(' \s\s|\t|\s\s\s', "", pi_texts)
        process_text =eng_preprocess(pi_texts) if lang=='eng' else ''

        ner_results={}
        for text in process_text .split('\n'):
            tokens,ner_tags=scan_ner(text=text)
         #   print(tokens,ner_tags)
            ners={' '.join(tokens[i[1]:i[2]]):i[0] for i in get_chunks(ner_tags)  } if len(get_chunks(ner_tags))>0  else None
            if ners :
                ner_results.update(ners)

        if  tag.lower() in ['underwriting','underwriter','underwriters'] :
            underwriters_results=[w for w,n in ner_results.items() if n=='ORG']
            return {'status':'success','content':underwriters_results}

        else:
            keyword_processor = KeywordProcessor(case_sensitive=True)
            for k in ner_results.keys():
                keyword_processor.add_keyword(k)

            keywords_found = keyword_processor.extract_keywords(process_text, span_info=True)
            word_tree=IntervalTree(Interval(k[1],k[2] , (k[0],ner_results.get(k[0])))  for k in keywords_found)
            word_tree=sorted(word_tree)

            org_tree=IntervalTree(i for i in word_tree if i.data[1]=='ORG')
            title_tree=[i for i in word_tree if i.data[1]=='TITLE']

            content_region_begin=[t.end for t in title_tree]
            content_region_end=[t.begin for t in title_tree][1:]+[len(process_text)]
            content_title=[t.data[0] for t in title_tree]


            party_info={}
            for title, t_r_start,t_r_end in zip(content_title,content_region_begin,content_region_end):
                title_content=[w.data[0] for w in org_tree[t_r_start:t_r_end] ]
                if title_content and title not in party_info:
                    party_info[title]=title_content

            print(f'party info: {party_info},  ')
            sponsor_info=[party_info[key] for key in party_info if  'sponsor' in key.lower() and 'legal' not in key.lower() and 'tax' not in key.lower() ]
            if sponsor_info:

                return  {'status':'success','content':[i.strip().strip(',') for i in sponsor_info[0]]}
            else:
                return {'status': 'error', 'content': sponsor_info}
####-------------------------



def get_secfirm_list(sec_firm_api:str)->Dict:
    resp=requests.get(sec_firm_api)
    if resp.status_code is not 200:

        return None
    else:
        return {row.get('cSponsorNameEng'):{'chi':row.get('cSponsorNameChi'),'id':row.get('cSponsorID')} for row in json.loads(resp.content)['content']}

##--------------------------------------------------------------------

def main(code,eng_asa_url,chi_asa_url):

    webteam_return = {"sponsors": {"chi": [], "eng": [], },    "underwriters": {"chi": [], "eng": [], },   }


    download_results=download_asa(code,eng_asa_url,chi_asa_url)

    if download_results['status']=='error':

        return  download_results
    else:

        eng_pdf_path=download_results['content']['eng']
        sponsor_info=fetch_pdf(pdf_path=eng_pdf_path, tag='sponsor')
       # print(f"sponsor info:{sponsor_info}")
        underwriter_info =fetch_pdf(pdf_path=eng_pdf_path, tag='underwriting')

        if sponsor_info['status']=='error' or underwriter_info['status']=='error':
            return sponsor_info
        else:
            webteam_return['sponsors']['eng']=[{'found':i} for i in sponsor_info['content']]
            webteam_return['underwriters']['eng']=[{'found':i} for i in underwriter_info['content']]

            secfirm_list=get_secfirm_list(sec_firm_api=config.external_API.sec_firm)

            for tag in ['sponsors','underwriters']:
                for eng_item in webteam_return[tag]['eng']:
                    sec_eng_name=eng_item['found']
                    fuzz_search_result=fuzz_process.extractOne(sec_eng_name,secfirm_list.keys(), scorer=fuzz.WRatio, score_cutoff=90)
                    match_case= fuzz_search_result[0] if fuzz_search_result else None
                    if match_case:
                        secfirm_name=secfirm_list.get(match_case,None)

                        sec_id=    secfirm_name.get('id',None)

                        sec_chi_name =secfirm_name.get('chi',None)
                    #    print(f"sec_eng_name:{sec_eng_name}, match_case:{match_case}, secfirm_name:{secfirm_name} ,  sec_id:{sec_id}, sec_chi_name: {sec_chi_name}")
                        eng_item['id']=sec_id
                        webteam_return[tag]['chi']+=[{'found':sec_chi_name,'id':sec_id}]
                    else:
                        eng_item['id'] = None
                        webteam_return[tag]['chi'] += [{'found': None, 'id': None}]

            return {'status':"success",'content': webteam_return}


@hug.post(config.API.route.main)
def ipo_pdf_parser(code,chi,eng,type=None):
    results=main(code,  eng,chi)

    if results['status']!='success':
        return results
    else:
        result=results['content']
        return {"status": "success" ,'content':result.get(type,None)   if type else result}



if __name__ == "__main__":

    hug.API(__name__).http.serve(port=int(config.API.port))