import gradio as gr
import os
from groq import Groq
from pygments.lexer import default

api_key=os.getenv("GROQ_API")
client = Groq(api_key=api_key)

chatHistory = [
    {"role": "system",
     "content": "You are a helpful recipe maker."},
]

def conversation(user_message):
    global chatHistory

    chatHistory.append({"role":"user", "content": user_message})

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=chatHistory,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    response_content =""
    for chunk in completion:
        response_content += chunk["content"]

    chatHistory.append({"role":"assistant", "content": response_content})

def generate_recipe(query, dietary_pref, metric):

    global chatHistory

    if not query.strip():
        return "Please tell me about what you would like in a recipe."

    chatHistory.append({"role":"user", "content": query})

    messages = [
        {"role": "system",
         "content": f'''
        You are a helpful AI chef and assistant. Your name is Remy. Your goal is to engage in conversation with the user whilst being helpful.
        --if the user is chatting, respond naturally and include food puns in your response. Do not mention that the user didn't ask for a recipe.
        --if the user asks about cooking tips and tricks, respond with tips and tricks. Do not mention that the user didn't ask for a recipe.
        --if a user EXPLICITLY asks for a recipe then generate one. 
        --if a user asks for a substitution or a question about a recipe, respond by referring to the recipe most recently generated {chatHistory[len(chatHistory)-2]["content"]}.
        --include emojis in your responses
        
        **If a user asks for a recipes, follow these rules**
        1) The user may have dietary restrictions. If they do, you must abide by these. Do not provide recipes that would go against these restrictions.
        2) If the user doesn't provide dietary restrictions, you have no limitations. Do not mention anything about not having dietary restrictions.
        3) Provide any measurements in the measurement system specified. 
        4) Provide nutritional information per serving at the bottom.
        5) Be polite and talk to them!
        '''},

        {"role": "user",
         "content": f"Generate a recipe for: {query}. Follow the dietary restrictions below: {dietary_pref}. Provide measurements in the {metric} system."},
    ]

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    response_content =""

    for chunk in completion:
        response_content += chunk.choices[0].delta.content or ""
    chatHistory.append({"role":"assistant", "content": response_content})
    return chatHistory

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Chat"):
            with gr.Row():
                dietary_restrictions = gr.Textbox(placeholder="Enter restrictions here", label="Dietary restrictions")
                metric = gr.Radio(["Metric", "Imperial"], label="Measurement")
            chatbot = gr.Chatbot(type="messages")
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Ask for a recipe here...",
                    lines=1
                )
                send_button = gr.Button("LET ME COOK")
            send_button.click(
                fn = generate_recipe,
                inputs = [user_input, dietary_restrictions, metric],
                outputs = chatbot,
                queue=True
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=user_input,
            )



demo.launch()