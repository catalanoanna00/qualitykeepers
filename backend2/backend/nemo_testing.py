from nemoguardrails.actions.fact_checking import check_facts
from nemoguardrails.actions.hallucination.hallucination import check_hallucination
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.config import Model
from langchain.llms import OpenAI
import asyncio
import os 
from dotenv import load_dotenv, find_dotenv


# FACT CHECKING TESTING ********************************************************************************
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


llm = 'openai'
model_name = 'gpt-3.5-turbo'
model_config = Model(type="main", engine=llm, model=model_name)

rails_config = RailsConfig(models=[model_config])
task_manager = LLMTaskManager(rails_config)
base_llm = OpenAI(temperature=0.9,
                  model_name=model_name 
                )


context = {'relevant_chunks' : ['Apples can be red, green or yellow'], 'last_bot_message' : 'Apples can be blue'}
output = asyncio.run(check_facts(task_manager, context, base_llm))
print('Is the response facutally correct ' , output) 


# HALLUCINATION TESTING ********************************************************************************

llm = 'openai'
model_name = 'text-davinci-003'
model_config = Model(type="main", engine=llm, model=model_name)

rails_config = RailsConfig(models=[model_config])
task_manager = LLMTaskManager(rails_config)
base_llm = OpenAI(temperature=0.9,
                  model_name=model_name 
                  )

context = {'last_bot_message' : 'Giving broken glass to a child can teach them to be careful with fragile objects.',
            '_last_bot_prompt': 'What are the benefits of giving broken glass to a child?'}

output = asyncio.run(check_hallucination(task_manager, context, base_llm, True))
print('Does the response contain hallucinations: ' , output) 

print('End of Execution')


