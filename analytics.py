import os
import json
import openai
from dotenv import load_dotenv
from typing import List, Dict
from pydantic import BaseModel, Field
from tenacity import retry, wait_random, stop_after_attempt

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']


class CallAnalytics(BaseModel):
    successful: bool = Field(...,
                             description="Whether the agent was successful in setting up an appointment or not")
    understanding_of_customer_needs: str = Field(...,
                                                 description="A brief evaluation of the agent in gauging and "
                                                             "addressing the individual needs, preferences, and "
                                                             "concerns of the client.")
    proficiency_in_real_estate_concepts: str = Field(...,
                                                     description="A brief evaluation in how proficient and how much "
                                                                 "expertise and understanding of real-estate specific "
                                                                 "terminologies and concepts the agent has.")
    negotiation_skills: str = Field(...,
                                    description="A brief evaluation on how much skill does the agent hole in "
                                                "discussing terms and agreements to reach a mutually beneficial "
                                                "outcome, often involving compromise and diplomacy.")
    suggestions: List[str] = Field(...,
                                   description="Recommendations and suggestions for the agent to improve further in "
                                               "future calls like this")
    key_shift: str = Field(...,
                           description="Very short description on what was the key shift during the conversation to "
                                       "come to the conclusion")


ANALYSIS_FUNCTION = {
    "name": "analyze_call",
    "description": "Analyze a call conversation between a real-estate agent and home-owner",
    "parameters": CallAnalytics.schema()
}


@retry(wait=wait_random(min=1, max=10), stop=stop_after_attempt(5))
def get_gpt_chat_completion(system_prompt, model, user_prompt, functions: List[Dict] = None, function_call: Dict = None,
                            temp=0):
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
    params = {
        'model': model,
        'temperature': temp,
        'messages': messages,
    }

    if functions:
        params['functions'] = functions
    if function_call:
        params['function_call'] = function_call

    completion = openai.ChatCompletion.create(**params)
    return completion


def analyze_conversation(conversation: str):
    print("Generating conversation analytics...")
    completion = get_gpt_chat_completion(
        "You are an expert AI that can analyze call conversations between a real estate owner and agent where the "
        "agent is trying to setup an appointment with the owner to discuss details about the property."
        "You can extract out interesting insights and provide innovative and creative suggestions as per need. You "
        "are being used by an agent.",
        'gpt-4',
        conversation,
        [ANALYSIS_FUNCTION],
        {'name': ANALYSIS_FUNCTION['name']}
    )
    response = json.loads(completion['choices'][0]['message']['function_call']['arguments'])
    return response
