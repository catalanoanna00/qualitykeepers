# Create your tests here

# from quants.custom.chain import CustomChain
# from quants.custom.prompts import CUSTOM_COMBINE_PROMPT
# from quants.custom.validation import validate_answers
# from quants.custom.llm import CustomOpenAI
# from quants.custom.postgres_faiss import PostgresFAISS
# from quants.custom.helper_models.document import DocumentHelper
# import models

from django.test import TestCase
import json
import os
import django
from dotenv import load_dotenv, find_dotenv

from django.db import models

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings


load_dotenv(find_dotenv())

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from bot.models import SourceDocument
from bot.models import Question
from quants import qa

embedding = OpenAIEmbeddings()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# FUNCTIONS ******************************************************************************************

'''
    # return my_file, question, id_

    # ARGS: squad_dev SQuAD dev set in json format (optional)
    
    # RETURN: my_file: a string of the document used to answer the question
            # question: a string of the question to be answered
            # id_: the question id of the question (from the original json)
'''
def process_json(): 
# add parameter to process json from a specific document 
# for now, we are hard-coding the document that we are pulling from 

    file_name = '/Users/family/Desktop/quads/quants-main/dev-v2.0.json'

    with open(file_name, "r") as fid:
        data = json.load(fid)

    data_list = []
    

    for k in range(len(data['data'])):

        for i in range(len(data['data'][k]['paragraphs'])):
            temp_dict = {'context':'',
                    'question':[],
                    'id':[]}
        
            temp_dict['context'] = data['data'][k]['paragraphs'][i]['context']

            for j in range(len(data['data'][k]['paragraphs'][i]['qas'])):
                temp_dict['question'].append(data['data'][k]['paragraphs'][i]['qas'][j]['question'])
                temp_dict['id'].append(data['data'][k]['paragraphs'][i]['qas'][j]['id'])

            data_list.append(temp_dict)

    return data_list

'''
    Maps to /documents
    Return a a Django QuerySet of all SourceDocuments to be used for answering the question
'''       
def upload_source_doc(my_file):
    data = [my_file]
    sources = "name"
    
    text_splitter = CharacterTextSplitter(chunk_size=1500, separator='\n')
    docs = []
    for i, d in enumerate(data):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        # metadatas.extend([{'source': sources[i]}] * len(splits))

    embeddings = embedding.embed_documents(docs)

    SourceDocument.objects.all().delete() # is this what is causing only one doc to be uploaded at a time 

    for i in range(len(embeddings)):
        SourceDocument.objects.create(
            name=['source'],
            content=my_file,
            embedding=embeddings[i]
            )
    queryset = SourceDocument.objects.all()
    return queryset

'''
Return the answer to a question with given context in json format 
context: 
question:
answer: 
'''
def get_answers_json(input_files):
    data_set = {}

    for f in input_files:
        query = upload_source_doc(f['context'])
        questions = f['question']
        id = f['id']

        for i in range(len(questions)):
            val_ans, orig_ans, snips = qa.process_query(query, questions[i])
            data_set[id[i]] = val_ans
            
    with open('/Users/family/Desktop/quads/quants-main/quads-test.json', "w") as json_file:
        json.dump(data_set, json_file)
    #return ans




# TESTING ****************************************************************************************

input_files = process_json()
get_answers_json(input_files)
    
print('End of Execution')


# OTHER FUNCTIONS ********************************************************************************

# def make_testing_json(validated_answer, id_):
#     return testing_json
    # ARGS: validated_answer is the answer to question in a string
         #  id_ is the question id of the question being answered
    # RETURN: testing_json is the file to feed into the SQuAD evaluator in the form id : answer
    
# def put_it_together(squad_dev):
#     query, question, id_ = process_json(squad_dev)
    
#     validated_answer = process_query(query, question)
#     testing_json = make_testing_json(validated_answer, id_)
#     return testing_json 

# Feed testing_json and squad_dev to 

# '''
#     Creates Django Query sets for each question in list 
# '''   

# def upload_questions(q_list):
#     for question in q_list: 
#         queryset = Question.objects.create(
#             text = question
#         )
#         queryset = SourceDocument.objects.all()
