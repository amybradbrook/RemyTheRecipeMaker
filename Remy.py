import gradio as gr
import os
from groq import Groq
from pygments.lexer import default

api_key=os.getenv("GROQ_API")
client = Groq(api_key=api_key)

chatHistory = [
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

def userQuery(query):
    global chatHistory
    chatHistory.append({"role":"user", "content": query})
    return chatHistory

def generate_recipe(query, dietary_pref, metric):

    global chatHistory

    if metric==None:
        metric="Imperial"

    if dietary_pref==None:
        dietary_pref="None specified"

    messages = [
        {"role": "system",
         "content": f'''
        You are a helpful AI chef and assistant. Your name is Remy. Your goal is to provide the user with cooking tips, tricks, recipes, and to modify recipes as needed. 
        --if the user asks about cooking tips and tricks, respond with tips and tricks. Do not mention that the user didn't ask for a recipe.
        --if a user *EXPLICITLY* asks for a recipe, then generate one. 
        --Refer to previous recipes, if any {chatHistory[len(chatHistory)-2]["content"]}, unless the user asks for a new one
        --if a user asks for a substitution or a question about a recipe, respond by referring to the recipe most recently generated {chatHistory[len(chatHistory)-2]["content"]}.
        --include emojis in your responses, but not too many
        
        **If a user asks for a recipes, follow these rules**
        1) The user may have dietary restrictions. If they do, you must abide by these. Do not provide recipes that would go against these restrictions.
        2) If the user doesn't provide dietary restrictions, you have no limitations. Do not mention anything about not having dietary restrictions.
        3) Provide any measurements in the measurement system specified. 
        4) Provide nutritional information per serving at the bottom.
        5) Be polite and talk to them!
        6) Provide a cooking time estimate
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

CSS="""
     .letmecookButton{
        font-size: 20px !important;
        background-color: #98C264;
     }
     
     .Title{
        font-size: 25px !important;
        text-align: "center" !important;
        display: block;
     }
     
     .info{
        text-align:"center" !important;
        display: block !important;
     }
     """
with gr.Blocks( fill_width=True, css=CSS) as demo:
    with gr.Row():
        gr.Image("Food.png", show_label=False, show_download_button=False, show_fullscreen_button=False, elem_classes="top_image")
    with gr.Sidebar():
        gr.Markdown("# 🍳 Recipe Preferences", elem_classes="Title")
        gr.Markdown("Customize your recipes with the options below!", elem_classes="info")
        with gr.Column(scale=4):
            dietary_restrictions=gr.Textbox(placeholder="Enter restrictions here", label="Dietary restrictions")
            metric = gr.Radio(["Metric", "Imperial"], label="Measurement")

    chatbot = gr.Chatbot(type="messages", show_copy_button=True, label="Remy", avatar_images= [None, "LETMECOOK.png"])
    with gr.Row(equal_height=True, height=50):
        user_input = gr.Textbox(
            placeholder="Ask for a recipe here...",
            lines=1,
            scale=1,
            show_label=False,
            elem_classes="userText",
        )
        send_button = gr.Button("Cook", elem_classes="letmecookButton", scale=0)
    send_button.click(
        fn = userQuery,
        inputs = user_input,
        outputs = chatbot
    ).then(
        fn = generate_recipe,
        inputs = [user_input, dietary_restrictions, metric],
        outputs = chatbot,
        queue=True,
    ).then(
        fn=lambda: "",
        inputs=None,
        outputs=user_input,
    )

demo.launch()