import subprocess
import os
def llm_forward(img_path, command):

    img_path = os.path.abspath(img_path)
    ollama_command = "ollama run granite3.2-vision:2b"
    img_path = ' \"' + img_path + '\" '

    sent_command = ollama_command + img_path + command

    result = subprocess.run(sent_command, capture_output=True, text=True, shell=True)

    return result.stdout

prompt = "Describe this image in about 50 words in a humorous way."

print(llm_forward("./static/uploads/test.jpg", prompt))