from haystack.nodes import TransformersSummarizer
from haystack import Document
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

summarizer = TransformersSummarizer(
    model_name_or_path="philschmid/flan-t5-base-samsum")

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
    conversation = Document('\n'.join(get_real_name(
        client, message["user"]) + ": " + message["text"] for message in result["messages"][:-1]))
    summary = summarizer.predict(documents=[conversation])

    say(text=summary[0].meta["summary"], thread_ts=thread_ts)

if __name__ == "__main__":
    # Used in dev mode since work blocks ngrok
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
    # Uncomment for prod usage
    # app.start(port=int(os.environ.get("PORT", 3000)))
