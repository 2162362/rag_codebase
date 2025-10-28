import os
import main  # This imports your main.py file
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

main.cocoindex.init()

try:
    azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_endpoint_name = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_KEY")

    azure_client = AzureOpenAI(
        api_version="2024-12-01-preview",  # A common, recent API version

    )

    if not all([azure_client.api_key, azure_deployment_name]):
        raise ValueError("Azure OpenAI environment variables are not fully set.")

except (ValueError, TypeError) as e:
    print(f"Error: {e}")
    print("Please make sure AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, and "
          "AZURE_OPENAI_DEPLOYMENT_NAME are set in your .env file.")
    exit(1)


def ask_codebase(query: str) -> str:
    """
    Asks a natural language question to the codebase.
    1. Retrieves relevant code chunks.
    2. Augments a prompt with that code.
    3. Generates a natural language answer using Azure OpenAI.
    """

    print(f"-> Retrieving context for: '{query}'")

    # 1. RETRIEVAL: Use the search function from main.py
    try:
        query_output = main.search(query)
    except Exception as e:
        return f"Error during search: {e}. Is your database running and indexed?"

    if not query_output.results:
        return "I couldn't find any relevant code snippets to answer that question."

    # 2. AUGMENTATION: Build the context string
    context_parts = []

    for result in query_output.results:
        context_parts.append(
            f"---\n"
            f"File: {result['filename']} (Lines {result['start']['line']} to {result['end']['line']})\n"
            f"```\n{result['code']}\n```"
        )

    formatted_context = "\n".join(context_parts)

    # Define the prompts for the LLM
    system_prompt = (
        "You are an expert AI assistant who answers questions about a software codebase. "
        "You will be given a user's question and a set of relevant code snippets. "
        "Your answer must be based *only* on the provided code snippets. "
        "If the answer is not in the provided snippets, say 'I cannot answer this question based on the provided code.' "
        "Do not make up information. Be concise and clear."
    )

    user_message = (
        f"Here is the relevant code context:\n"
        f"{formatted_context}\n\n"
        f"Question: {query}"
    )


    print("-> Generating answer...")
    try:
        response = azure_client.chat.completions.create(
            model=azure_deployment_name,  # Your deployment name (e.g., "gpt-4o")
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,  # Low temperature for factual, grounded answers
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error calling Azure OpenAI: {e}"


def start_qa_loop() -> None:
    """
    Runs a simple loop to ask questions.
    """
    print("\n--- Codebase Q&A (powered by Azure OpenAI) ---")
    print("Type your question and press Enter. Type 'quit' or 'exit' to end.")

    while True:
        try:
            query = input("\nQuestion: ")
            if query.lower() in ["quit", "exit"]:
                break
            if not query:
                continue

            answer = ask_codebase(query)
            print(f"\nAnswer:\n{answer}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    start_qa_loop()