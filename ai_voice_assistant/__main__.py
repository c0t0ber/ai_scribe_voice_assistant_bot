import logging

from dotenv import load_dotenv

from ai_voice_assistant.bot import main

load_dotenv()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    main()
