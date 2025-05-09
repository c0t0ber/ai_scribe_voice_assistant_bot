import asyncio
import logging
from pathlib import Path
from typing import Optional

from elevenlabs import SpeechToTextChunkResponseModel
from elevenlabs.client import ElevenLabs
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic_settings import BaseSettings, SettingsConfigDict
from telegram import Bot, InputFile, Message, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """
Ты эксперт по обработке речевых транскрипций. Проанализируй следующую транскрипцию и преобразуй ее в структурированный обзор, который передает суть и настроение обсуждения, оставаясь при этом понятным. Работай по этим правилам:

1.  Выдели основную тему и главные мысли.
2.  Собери связанные идеи в логичные блоки с понятными подзаголовками.
3.  Важные моменты, цифры или данные можешь выделить (например, полужирным).
4.  Для перечислений используй списки с маркерами.
5.  Формулируй мысли четко и кратко, но не слишком официально.
6.  Важные или характерные цитаты оставляй в кавычках.
7.  Структурируй информацию так, как она идет в обсуждении, или по важности.
8.  В конце основного текста добавь короткое резюме или вывод.
9.  После заключения создай отдельный раздел "Ключевые факты". Тут перечисли упомянутые факты строго и по делу, без эмоций и оценок рассказчика. Каждый факт — отдельный пункт списка.
10. Весь ответ должен быть оформлен как валидный Markdown документ.
11. Ответь на том же языке, что и транскрипция.

Транскрипция:
{text}
"""

MAX_FILE_SIZE_MB = 20  # Maximum allowed file size in megabytes
MAX_TELEGRAM_TEXT_LENGTH = 4096  # Telegram's maximum message length


class Settings(BaseSettings):  # type: ignore[explicit-any]
    """Application settings."""

    TELEGRAM_BOT_TOKEN: str
    ELEVENLABS_API_KEY: str
    OPENAI_API_KEY: str
    LLM_MODEL: str
    DEFAULT_PROMPT: str = DEFAULT_PROMPT
    LLM_BASE_URL: str = ""
    TEMP_DIR: Path = Path("temp")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class VoiceAssistantBot:
    """Telegram bot for processing audio files and voice messages."""

    def __init__(self) -> None:
        """Initialize the bot and its dependencies."""
        self.settings = Settings()
        self.settings.TEMP_DIR.mkdir(exist_ok=True)

        self.elevenlabs_client = ElevenLabs(api_key=self.settings.ELEVENLABS_API_KEY, timeout=60*5)
        self.openai_client = OpenAI(api_key=self.settings.OPENAI_API_KEY, base_url=self.settings.LLM_BASE_URL or None)

    def run(self) -> None:
        logger.info("Starting bot initialization")

        application = Application.builder().token(self.settings.TELEGRAM_BOT_TOKEN).build()

        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("model", self.model_command))
        application.add_handler(CommandHandler("prompt", self.prompt_command))
        application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self.handle_audio))

        logger.info("Bot is starting...")
        application.run_polling()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None:
            logger.error("Received update with no message")
            return

        if context.user_data is None:
            context.user_data = {}

        model = context.user_data.get("model", self.settings.LLM_MODEL)
        prompt = context.user_data.get("prompt", self.settings.DEFAULT_PROMPT)
        base_url = self.settings.LLM_BASE_URL

        await update.message.reply_text(
            "Hi! I'm your voice assistant bot. Send me an audio file or voice message, "
            "and I'll transcribe it and structure it for you!\n\n"
            "Available commands:\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/model <model_name> - Change the LLM model (e.g., /model gpt-4-turbo)\n"
            "/prompt <prompt_text> - Change the prompt template (e.g., /prompt Summarize this text:\n\n{text})\n\n"
            f"Current settings:\n"
            f"- LLM Model: {model}\n"
            f"- LLM Base URL: {base_url}\n"
            f"- Default Prompt: {prompt}"
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        await self.start(update, context)  # Reuse the start message

    async def model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /model command to change the LLM model."""
        if update.message is None:
            logger.error("Received update with no message")
            return

        if not context.args:
            await update.message.reply_text("Please specify a model. Example: /model gpt-4o-mini")
            return

        if context.user_data is None:
            context.user_data = {}

        model = context.args[0]
        context.user_data["model"] = model
        await update.message.reply_text(f"Model changed to: {model}")

    async def prompt_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /prompt command to change the prompt."""
        if update.message is None:
            logger.error("Received update with no message")
            return

        if not context.args:
            await update.message.reply_text("Please specify a prompt. Example: /prompt Summarize this text:\n\n{text}")
            return

        if context.user_data is None:
            context.user_data = {}

        prompt = " ".join(context.args)
        context.user_data["prompt"] = prompt
        await update.message.reply_text(f"Prompt changed to: {prompt}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(Exception),
    )
    async def transcribe_audio(self, audio_path: Path) -> str:
        logger.info("Starting transcription with ElevenLabs")

        def _read_file_and_transcribe() -> SpeechToTextChunkResponseModel:
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
                transcription = self.elevenlabs_client.speech_to_text.convert(
                    file=audio_data,
                    model_id="scribe_v1",
                    tag_audio_events=True,
                    #language_code="ru", # чтобы не ограничиваться только русским языком
                    diarize=True,
                    # это баг в elevenlabs, пока работает только так
                    additional_formats='[{"format": "txt"}]',  # type: ignore
                )
                return transcription.additional_formats[0].content  # type: ignore

        result = await asyncio.get_event_loop().run_in_executor(None, _read_file_and_transcribe)

        logger.info("Successfully transcribed audio")
        return result.text or ""

    async def download_file(self, file_id: str, bot: Bot) -> Path:
        logger.info(f"Starting download of file {file_id}")
        file = await bot.get_file(file_id)

        local_path = self.settings.TEMP_DIR / f"{file_id}"
        logger.info(f"Downloading file to {local_path}")

        try:
            await file.download_to_drive(local_path)
            logger.info(f"Successfully downloaded file to {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(Exception),
    )
    async def structure_text(self, text: str, context: ContextTypes.DEFAULT_TYPE) -> str:
        logger.info("Starting text structuring with OpenAI")

        model = self.settings.LLM_MODEL
        prompt = self.settings.DEFAULT_PROMPT

        if context.user_data is not None:
            model = context.user_data.get("model", self.settings.LLM_MODEL)
            prompt = context.user_data.get("prompt", self.settings.DEFAULT_PROMPT)

        def _call_openai() -> ChatCompletion:
            return self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt.format(text=text)}],
            )

        response = await asyncio.get_event_loop().run_in_executor(None, _call_openai)

        structured_text = response.choices[0].message.content
        logger.info("Successfully structured text with OpenAI")
        return structured_text or ""

    async def send_text_or_file(self, update: Update, text: str, file_name: str, file_extension: str) -> None:
        """Send text or file depending on its length."""
        if update.message is None:
            logger.error("Received update with no message")
            return

        if len(text) > MAX_TELEGRAM_TEXT_LENGTH:
            filename = f"{file_name}.{file_extension}"
            input_file = InputFile(
                obj=text,
                filename=filename,
            )
            await update.message.reply_document(document=input_file)
        else:
            await update.message.reply_text(text)

            filename = f"{file_name}.{file_extension}"
            input_file = InputFile(
                obj=text,
                filename=filename,
            )
            await update.message.reply_document(document=input_file)

    async def handle_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming audio files and voice messages."""
        if update.effective_user is None or update.message is None:
            logger.error("Received update with no user or message")
            return

        logger.info(f"Received audio message from user {update.effective_user.id}")
        local_path: Optional[Path] = None
        status_message: Optional[Message] = None

        try:
            file_id = ""
            if update.message.voice:
                file_id = update.message.voice.file_id
                logger.info("Processing voice message")
            elif update.message.audio:
                file_id = update.message.audio.file_id
                logger.info("Processing audio file")
            else:
                logger.warning("Received message without audio content")
                await update.message.reply_text("Please send an audio file or voice message.")
                return

            file_info = await context.bot.get_file(file_id)
            if file_info.file_size and file_info.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                await update.message.reply_text(f"File size exceeds the limit of {MAX_FILE_SIZE_MB} MB.")
                return

            status_message = await update.message.reply_text("📥 Downloading your audio file...")

            local_path = await self.download_file(file_id, context.bot)

            if status_message:
                await status_message.edit_text("🔍 Transcribing audio using ElevenLabs...")

            transcription = await self.transcribe_audio(local_path)
            await self.send_text_or_file(update, transcription, f"transcription_{file_id}", "txt")
            logger.info("Sent raw transcription to user")

            if status_message:
                await status_message.edit_text("🧠 Structuring text using AI model...")

            structured_text = await self.structure_text(transcription, context)
            await self.send_text_or_file(update, structured_text, f"structured_text_{file_id}", "md")
            logger.info("Sent structured text to user")

            if status_message:
                await status_message.edit_text("✅ Processing complete!")

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            error_message = "Sorry, I couldn't process that audio."

            if status_message:
                await status_message.edit_text(f"❌ {error_message}")
            else:
                if update.message:
                    await update.message.reply_text(error_message)
        finally:
            if local_path:
                asyncio.create_task(self.cleanup_temp_file(local_path))

    async def cleanup_temp_file(self, file_path: Path) -> None:
        """Clean up temporary files asynchronously."""
        try:
            if file_path and file_path.exists():
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: file_path.unlink() if file_path.exists() else None
                )
                logger.info(f"Deleted temporary file {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")


def main() -> None:
    """Start the bot."""
    bot = VoiceAssistantBot()
    bot.run()


if __name__ == "__main__":
    main()
