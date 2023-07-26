# quants qa imports
from quants.custom.chain import CustomChain
from quants.custom.prompts import CUSTOM_COMBINE_PROMPT
from quants.custom.validation import validate_answers
from quants.custom.llm import CustomOpenAI
from quants.custom.postgres_faiss import PostgresFAISS
from quants.custom.helper_models.document import DocumentHelper

# nemo guardrails imports
from nemoguardrails.actions.fact_checking import check_facts
from nemoguardrails.actions.hallucination.hallucination import check_hallucination
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.config import Model
from langchain.llms import OpenAI
import asyncio
import os 
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def process_query(query, question):
    """
    Given documents and a question, use an LLM to generate an answer using
    context only from the given documents.

    Args:
        query: a Django QuerySet of all SourceDocuments to be used for 
            answering the question
        question: a string of the question to be answered

    Returns:
        a string of the validated answer, a string of the answer before
        validation, and a list of Snippets of the source documents used in
        answering the question
    """
    # generate store on the fly
    store = PostgresFAISS.from_query(query)
    # instantiate the custom chain
    chain = CustomChain.from_llm(
        llm=CustomOpenAI(temperature=0, model_name='gpt-3.5-turbo'), 
        combine_prompt=CUSTOM_COMBINE_PROMPT,
        vectorstore=store
    )
    # run the chain
    results = chain({'question': question})
    validated_answer = validate_answers(results)
    # post processing of the results
    original_answer = results.get(chain.answer_key).split(':::')
    sources = results.get(chain.sources_answer_key)
    raw_snippets = results.get(chain.snippets_key)

    snippets = DocumentHelper.parse_snippets(sources, raw_snippets)

    print('SNIPPETS ', raw_snippets)
    print('ORIG ANSWER ', original_answer)
    print('ORIG VAL ANSWER ', validated_answer)

    # another layer of validated answer



    # run guardrails on original answer
    # fact checking 
    llm = 'openai'
    model_name = 'gpt-3.5-turbo'
    model_config = Model(type="main", engine=llm, model=model_name)

    rails_config = RailsConfig(models=[model_config])
    task_manager = LLMTaskManager(rails_config)
    base_llm = OpenAI(temperature=0.9,
                    model_name=model_name 
                    )
    
    # relevant chunks = raw snippets 
    # last bot message = original answer
    context = {'relevant_chunks' : raw_snippets, 'last_bot_message' : original_answer}
    factually_correct = asyncio.run(check_facts(task_manager, context, base_llm)) 

    # hallucination checking 
    llm = 'openai'
    model_name = 'text-davinci-003'
    model_config = Model(type="main", engine=llm, model=model_name)

    rails_config = RailsConfig(models=[model_config])
    task_manager = LLMTaskManager(rails_config)
    base_llm = OpenAI(temperature=0.9,
                    model_name=model_name 
                    )

    # last bot message = original answer
    # last bost prompt = question
    context = {'last_bot_message' : original_answer, '_last_bot_prompt': question}
    contains_hallucinations = asyncio.run(check_hallucination(task_manager, context, base_llm, True))

    if validated_answer == "I don't know":
        if factually_correct:
            validated_answer = original_answer[0]
    else: 
        if contains_hallucinations and not factually_correct:
            validated_answer = "I don't know"
    
    
    print('FACTS ', factually_correct)
    print('HALLUC ', contains_hallucinations)

    return validated_answer, original_answer, snippets

def extract_keywords(snips):
    pass