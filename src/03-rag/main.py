
import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


class ChatService:

    def __init__(self):
        api_key = os.environ["OPENROUTER_API_KEY"]

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        # Conversation history
        self.messages = []

    async def ask(self, question: str) -> str:

        # Add user question to history
        self.messages.append({
            "role": "user",
            "content": question
        })

        full_answer = []
        usage = None

        try:
            # Request a streaming response
            stream = await self.client.chat.completions.create(
                model="nvidia/nemotron-3.5-lightning:free",
                messages=self.messages,
                stream=True,
                stream_options={"include_usage": True}
            )

            print("\nAssistant: ", end="", flush=True)

            # Receive response chunks
            async for chunk in stream:

                # Capture token usage when available
                if chunk.usage is not None:
                    usage = chunk.usage

                # Some final chunks may contain usage but no choices
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content

                if content:
                    print(content, end="", flush=True)
                    full_answer.append(content)

            print("\n")

            # Combine all streamed chunks
            answer = "".join(full_answer)

            # Save assistant response to history
            self.messages.append({
                "role": "assistant",
                "content": answer
            })

            # Display token usage
            if usage is not None:
                print("Token Usage:")
                print(f"Input tokens:  {usage.prompt_tokens}")
                print(f"Output tokens: {usage.completion_tokens}")
                print(f"Total tokens:  {usage.total_tokens}")
            else:
                print("Token usage was not returned by the provider.")

            return answer

        except Exception:
            # Remove the unanswered user message if the request fails
            self.messages.pop()
            raise


async def main():
    chat_service = ChatService()

    print("Chat started! Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() == "exit":
            print("Chat ended.")
            break

        if not query:
            continue

        try:
            await chat_service.ask(query)
            print()

        except Exception as e:
            print(f"\nError calling LLM: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
