# Thread Summarizer

A slack bot built using bolt to summarize slack threaded conversations.

## Setup

Set the environment variables for:
* `SLACK_BOT_TOKEN` (starts with `xoxb`)
* `SLACK_SIGNING_SECRET`
* `SLACK_APP_TOKEN` (starts with `xapp`)

### Permissions

You need to configure your bot with the following permissions:
* app_mentions:read
* channels:history
* channels:join
* channels:read
* chat:write
* commands
* groups:history
* im:history
* mpim:history
* users:read

### Events

The only event this bot listens for is `app_mention`