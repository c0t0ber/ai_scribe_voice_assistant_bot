import logging

from dotenv import load_dotenv

from ai_voice_assistant.bot import main

load_dotenv()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main()
