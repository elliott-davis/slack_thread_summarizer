from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os
import openai
import time


app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

model = T5ForConditionalGeneration.from_pretrained("philSchmid/flan-t5-base-samsum")
tokenizer = T5Tokenizer.from_pretrained("philSchmid/flan-t5-base-samsum")

name_db = {}
model_name = "WizardLM/WizardLM-70B-V1.0"
openai.api_key = os.getenv("VLLM_KEY")
openai.api_base = "https://vllm.libra.decc.vmware.com/api/v1/"
llm = ChatOpenAI(
    model=model_name,
    openai_api_base="https://vllm.libra.decc.vmware.com/api/v1/",
    openai_api_key=os.getenv("VLLM_KEY"),
    streaming=True,
)
# Cache the user ID's and real names from the thread so we don't blow past our rate limit
def get_real_name(client, id: str) -> str:
    if id in name_db:
        return name_db[id]
    real_name = client.users_info(user=id)["user"]["real_name"]
    name_db[id] = real_name
    return real_name

@app.event("app_mention")
def summarize_thread(client, event, say):
    start = time.time()

    thread_ts = event.get("thread_ts", None)
    if thread_ts is None:
        say("Sorry, I can only summarize threads")
        return
    
    channel_id = event.get("channel", None)
    result = client.conversations_replies(channel=channel_id, ts=thread_ts)
    
    if not result["ok"]:
        say("failed to get thread text. Try again later", thread_ts=thread_ts)
        return
    
    # Combine the content of the thread into a single document formatted like:
    # Elliott Davis: And he said whaaaaa
    # Matt Dresser: Nooooo
    conversation = '\n'.join(get_real_name(
        client, message["user"]) + ": " + message["text"] for message in result["messages"][:-1])
    

    if ("text" in event and "llm" in event["text"].lower()):
        chain = load_summarize_chain(llm, chain_type="stuff")
        end = time.time()
        say(text="Summary: \"{}\" generated in {}".format(chain.run([Document(page_content=conversation)]), (end - start)), thread_ts=thread_ts)
    else:
        # Tokenize the document
        inputs = tokenizer.encode("summarize: " + conversation, return_tensors="pt", max_length=1024, truncation=True)

        # Generate the summary
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

        # Decode the summary tokens back to text
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        end = time.time()
        say(text="Summary: \"{}\" generated in {}".format(summary, (end - start)), thread_ts=thread_ts)

if __name__ == "__main__":
    # Used in dev mode since work blocks ngrok
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
    # Uncomment for prod usage
    # app.start(port=int(os.environ.get("PORT", 3000)))
