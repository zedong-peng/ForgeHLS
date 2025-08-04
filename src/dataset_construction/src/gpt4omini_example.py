from openai import OpenAI

# Get the OPENAI_API_KEY from the environment variables
import os

def query_gpt4omini(prompt):
    # openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_api_key = ""
    if openai_api_key:
        print("OpenAI API Key has setup.")
    else:
        print("Failed to setup OpenAI API Key.")
    openai_api_base = ""

    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    llm_response = client.chat.completions.create(
        messages=messages,
        # model="gpt-4",
        # model="o1-preview",
        # model="o1",
        # model="o1-mini",
        # model="chatgpt-4o-latest",
        # model = "deepseek-v3-241226",
        model = "deepseek-r1-250120",
        # max_tokens=256,
        temperature=0.1,
        stream=False,
        n=1
    )
    # print("-" * 20, "prompt", "-" * 20)
    # print(prompt)
    # print("-" * 20, "llm_response", "-" * 20)
    # print(llm_response)
    try:
        print("Response received successfully.")
        print(f"llm_response: {llm_response.choices[0].message.content}")
    except:
        print("Failed to print response.")
        print(f"llm_response: {llm_response}")
    # print("-" * 20, "llm_outputs", "-" * 20)
    # print(llm_outputs)
    # print("-" * 20, "End", "-" * 20)
    return llm_response.choices[0].message.content

if __name__ == "__main__":
    query_gpt4omini("hello")