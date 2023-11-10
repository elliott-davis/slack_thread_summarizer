from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

model = T5ForConditionalGeneration.from_pretrained("philSchmid/flan-t5-base-samsum")
tokenizer = T5Tokenizer.from_pretrained("philSchmid/flan-t5-base-samsum")

name_db = {}

# Cache the user ID's and real names from the thread so we don't blow past our rate limit
def get_real_name(client, id: str) -> str:
    if id in name_db:
        return name_db[id]
    real_name = client.users_info(user=id)["user"]["real_name"]
    name_db[id] = real_name
    return real_name

@app.event("app_mention")
def summarize_thread(client, event, say):
    
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
    
    # Tokenize the document
    inputs = tokenizer.encode("summarize: " + conversation, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the summary tokens back to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    say(text=summary, thread_ts=thread_ts)

if __name__ == "__main__":
    # Used in dev mode since work blocks ngrok
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
    # Uncomment for prod usage
    # app.start(port=int(os.environ.get("PORT", 3000)))
