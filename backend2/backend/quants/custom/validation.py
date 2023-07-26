import openai
from quants.custom.prompts import validation_prompt

def validate_answers(result):
    answer = result['answer']
    docs = result['docs']

    contents = ""
    for doc in docs:
        contents += 'Content: ' + doc.page_content.strip() + '\n'
    
    formatted_prompt = validation_prompt.format(
        statements=answer,
        contents=contents
    )

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions only using the sources provided without using outside information or guesses."},
            {"role": "user", "content": formatted_prompt}
        ]
    )

    return completion.choices[0].message.content
 

 