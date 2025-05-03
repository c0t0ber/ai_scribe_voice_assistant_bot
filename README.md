# AI Voice Assistant

A Telegram bot that processes audio files and voice messages using ElevenLabs for transcription. This project provides an automated solution for converting voice messages and audio files into text using AI technology.

## Features

- Accepts audio files and voice messages
- Transcribes audio using ElevenLabs
- Docker support for easy deployment
- Environment-based configuration

## Installation

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai_scribe_voice_assistant_bot.git
   cd ai_scribe_voice_assistant_bot
   ```

2. Create a `.env` file with your configuration:
   ```bash
   cp .env.example .env
   # Edit the .env file with your API keys and settings
   ```

3. Build and start the Docker container:
   ```bash
   docker-compose up -d
   ```

### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai_scribe_voice_assistant_bot.git
   cd ai_scribe_voice_assistant_bot
   ```

2. Create and activate a virtual environment using UV:
   ```bash
   just init-venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   just install-deps
   ```

4. Create a `.env` file with your configuration:
   ```bash
   cp .env.example .env
   # Edit the .env file with your API keys and settings
   ```

## Usage

### Running with Docker

```bash
docker compose up -d
```

### Running manually

```bash
just run
```

## Bot Commands

- `/start` - Start the bot and display help information
- `/help` - Show help message
- `/model <model_name>` - Change the LLM model
- `/prompt <prompt_text>` - Change the prompt template
