import os
import logging
import json
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from groq import Groq

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
HISTORY_FILE = "history.json"
BOT_INFO_FILE = "bot_info.txt"

# Initialize Groq client safely
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None

if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
else:
    logger.warning("GROQ_API_KEY not found. AI features will be unavailable.")

# Global application instance for persistence between requests
application = None

def load_bot_info():
    """Reads bot information from the text file."""
    if os.path.exists(BOT_INFO_FILE):
        with open(BOT_INFO_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "You are a helpful Telegram bot."

def load_history():
    """Loads conversation history from the JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error("Error decoding history.json. Starting fresh.")
    return {}

def save_history(history):
    """Saves conversation history to the JSON file."""
    # Note: On Vercel/Render, this might be ephemeral depending on the plan!
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        logger.error(f"Could not save history to disk: {e}")

def format_potato_task(name, description, points, deadline):
    """Formats a task into a PotatoTask string."""
    prompt = (
        f"Write this task description properly and make descriptive and it sould like a game side quest "
        f"not more than 4 sentences. the task is {name} {description} {points} {deadline} "
        f"do not use * * or ** ** formatting in responses. Just give the description, no title anymore."
    )
    formatted_description = prompt_groq(prompt)
    return (
        f"New Task Alert!\n"
        f"Name: {name}\n"
        f"Description: {description}\n"
        f"Points: {points} PotatoPoints\n"
        f"Deadline: {deadline}\n\n"
        f"Quest Description: \n{formatted_description}"
    )

def prompt_groq(prompt, system_prompt=None, history=None):
    """Stand-alone function to get a response from Groq."""
    if not groq_client:
        logger.error("Groq client not initialized. Cannot process request.")
        return "I'm sorry, my AI features are currently disabled because the API key is missing."

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    if history:
        messages.extend(history[-10:])
    
    messages.append({"role": "user", "content": prompt})
    
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b", 
            messages=messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling Groq API: {e}")
        return "Sorry, I'm having trouble thinking right now."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /start command."""
    user_id = str(update.effective_user.id)
    history = load_history()
    history[user_id] = [] # Reset history on start
    save_history(history)
    await update.message.reply_text("Hello! I'm your Telegram bot. I'm now running in Polling mode for better reliability.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /help command."""
    help_text = (
        "Available commands:\n"
        "/start - Start/Reset the bot\n"
        "/help - Show this help message\n"
        "/newtask [name] | [description] | [points] | [deadline] - Create a PotatoTask\n\n"
        "You can also just talk to me!"
    )
    await update.message.reply_text(help_text)

async def new_task(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for creating a new PotatoTask via command."""
    try:
        args_str = " ".join(context.args)
        if not args_str:
            await update.message.reply_text("Please provide task details: /newtask Name | Description | Points | Deadline")
            return
        
        parts = [p.strip() for p in args_str.split('|')]
        if len(parts) < 4:
            await update.message.reply_text("Incorrect format. Use: Name | Description | Points | Deadline")
            return
            
        task_text = format_potato_task(parts[0], parts[1], parts[2], parts[3])
        await update.message.reply_text(task_text)
    except Exception as e:
        logger.error(f"Error in new_task: {e}")
        await update.message.reply_text("Sorry, I couldn't format that task.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processes user messages using prompt_groq and history."""
    if not update.message or not update.message.text:
        return

    user_id = str(update.effective_user.id)
    user_text = update.message.text
    
    history_data = load_history()
    bot_info = load_bot_info()
    
    if user_id not in history_data:
        history_data[user_id] = []
        
    system_prompt = f"Knowledge base: {bot_info}\nYou can also format tasks as PotatoTasks if requested."
    response_text = prompt_groq(user_text, system_prompt=system_prompt, history=history_data[user_id])
    
    history_data[user_id].append({"role": "user", "content": user_text})
    history_data[user_id].append({"role": "assistant", "content": response_text})
    save_history(history_data)
    
    await update.message.reply_text(response_text)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan to initialize the Telegram application."""
    global application
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN not found. Bot will not start.")
        yield
        return

    application = ApplicationBuilder().token(bot_token).build()
    
    # Add handlers
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('newtask', new_task))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    async with application:
        await application.initialize()
        await application.start()
        
        # Default to Polling on Render/local
        use_polling = os.getenv("USE_POLLING", "True").lower() == "true"
        
        if use_polling:
            logger.info("Starting bot in POLLING mode...")
            # Run polling in the background
            asyncio.create_task(application.updater.start_polling())
        else:
            logger.info("Bot application initialized in Webhook mode (Warning: Not primary).")
            
        yield
        
        if use_polling and application.updater.running:
            await application.updater.stop()
        await application.stop()
        await application.shutdown()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def index():
    """Health check endpoint for Render."""
    mode = "Polling" if os.getenv("USE_POLLING", "True").lower() == "true" else "Webhook"
    return {
        "status": "Bot is running", 
        "mode": mode,
        "ai_enabled": groq_client is not None
    }

@app.get("/health")
async def health_check():
    """Dedicated health check endpoint."""
    return {"status": "ok"}

# Webhook endpoint kept for legacy/compatibility but not used in Polling mode
@app.post("/webhook")
async def webhook(request: Request):
    """Handle incoming Telegram updates via webhook (if enabled)."""
    global application
    if application is None:
        return Response(status_code=500, content="Application not initialized")
        
    try:
        data = await request.json()
        update = Update.de_json(data, application.bot)
        await application.process_update(update)
        return Response(status_code=200, content="OK")
    except Exception as e:
        logger.error(f"Error processing update: {e}")
        return Response(status_code=500, content=str(e))

if __name__ == '__main__':
    # Local development settings
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

