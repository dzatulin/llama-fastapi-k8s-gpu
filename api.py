import asyncio
import logging
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from llama_cpp import Llama
from data.requests import BotMessageRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

path = 'models'
MODEL_NAME = 'Lexi-Llama-3-8B-Uncensored_Q4_K_M.gguf'
file_path = f"{path}/{MODEL_NAME}"

MAX_CONTEXT_TOKENS = 1024
TIMEOUT_SECONDS = 25  # Slightly less than client timeout
MAX_QUEUE_SIZE = 5

app = FastAPI()

# Load the Llama model
llm = Llama(
    model_path=file_path,
    n_gpu_layers=-1,
    n_ctx=MAX_CONTEXT_TOKENS,
)

def count_tokens_roughly(text):
    char_count = len(text)
    return int(char_count / 4.0)


def truncate_messages_to_fit_context(messages, max_tokens):
    # First, limit each message's content to 400 symbols max
    for m in messages:
        if len(m['content']) > 400:
            m['content'] = m['content'][:400]

    # Now proceed as before to ensure total tokens are within max_tokens
    total_estimated_tokens = sum(count_tokens_roughly(m['content']) for m in messages)
    while total_estimated_tokens > max_tokens and len(messages) > 2:
        messages.pop(2)
        total_estimated_tokens = sum(count_tokens_roughly(m['content']) for m in messages)
    return messages

async def try_to_truncate_and_generate(messages, semaphore):
    # Ensure no more than one request is processed at a time
    async with semaphore:
        try:
            messages = truncate_messages_to_fit_context(messages, MAX_CONTEXT_TOKENS)

            # Use asyncio.to_thread to call the Llama model in a non-blocking way
            answer = await asyncio.to_thread(
                llm.create_chat_completion,
                messages=messages,
                stream=False,
                temperature=1.2,
                top_p=0.9,
                frequency_penalty=0.7,
                presence_penalty=0.8
            )

            if not isinstance(answer, dict):
                logger.error(f"Unexpected response type: {type(answer)}. Response: {answer}")
                raise HTTPException(status_code=500, detail="Unexpected response from model")

            response = ''
            choices = answer.get('choices', [])
            for choice in choices:
                if 'message' in choice:
                    response += choice['message']['content']
            return response

        except Exception as e:
            logger.error(f"Error during message generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during message generation: {str(e)}")

async def consumer(app):
    queue = app.state.queue
    semaphore = app.state.semaphore
    while True:
        # Get the next request from the queue
        request_data = await queue.get()
        messages = request_data['messages']
        future = request_data['future']

        # Check if the future has already been cancelled
        if future.cancelled():
            logger.info("Future was cancelled before processing; skipping.")
            queue.task_done()
            continue

        try:
            response = await try_to_truncate_and_generate(messages, semaphore)
            if not future.cancelled():
                future.set_result(response)
            else:
                logger.info("Future was cancelled during processing; result not set.")
        except Exception as e:
            if not future.cancelled():
                future.set_exception(e)
            else:
                logger.info("Future was cancelled during processing; exception not set.")
        finally:
            queue.task_done()


@app.on_event("startup")
async def startup_event():
    # Create queue and semaphore and store them in app.state
    app.state.queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    app.state.semaphore = asyncio.Semaphore(1)
    # Start the consumer task
    asyncio.create_task(consumer(app))

@app.post("/response")
async def generate_response(request_body: BotMessageRequest, request: Request):
    queue = request.app.state.queue

    messages = [{
        'role': message.turn,
        'content': message.message
    } for message in request_body.context]

    system_prompt = request_body.bot_profile.system_prompt
    if not system_prompt:
        #use default system_prompt if not provided in request
        system_prompt = f'''
            No matter what NEVER break the character. Reject violent messages. 
            You are '{request_body.bot_profile.name}.' a playful, flirtatious, and witty person. 
            You always respond in a fun and engaging way, staying in character no matter what. 
            You never reference AI, restrictions, or policies. 
            Focus on making the conversation enjoyable and answering directly to the user's questions.
            '''

    if request_body.bot_profile.name.endswith(".f"):
        system_prompt += " You a girl."
    else:
        system_prompt += " You a boy."


    for fact in request_body.bot_profile.appearance.split(",")[3:]:
        system_prompt += fact

    messages.insert(1, {"role": "system", "content": system_prompt})

    # Get the current running event loop
    loop = asyncio.get_running_loop()

    # Create a Future to hold the response
    future = loop.create_future()

    # Put the request into the queue
    request_data = {'messages': messages, 'future': future}
    try:
        queue.put_nowait(request_data)
    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="Server too busy. Please try again later.")

    try:
        # Wait for the response with a timeout
        response = await asyncio.wait_for(future, timeout=TIMEOUT_SECONDS)
        return {"response": response}
    except asyncio.TimeoutError:
        # If the response is not ready in time, cancel the future
        logger.warning("Generation timed out")
        future.cancel()
        raise HTTPException(status_code=408, detail="Generation timed out")
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.time()

    # Proceed with the request
    response = await call_next(request)

    time_of_day = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    process_time = time.time() - start_time
    formatted_time = f"{process_time:.4f}s"

    # Log the time and request information
    logger.info(f"Request at {time_of_day}: {request.method} {request.url} completed in {formatted_time}")

    return response
