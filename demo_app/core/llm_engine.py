import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

# Initialize the Groq client
# It will automatically pick up GROQ_API_KEY from the environment
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    print("Warning: GROQ_API_KEY not found in environment.")
    client = None
else:
    client = Groq(api_key=api_key)

def generate_response(personality_prompt: str, retrieved_context: list[str], user_input: str) -> str:
    """
    Generates a response using the Groq API (LLaMA 3 or Mixtral),
    augmented with the retrieved memories from the Endee vector DB.
    """
    if not client:
        return "I'm sorry, my LLM engine isn't configured properly (Missing API Key)."

    # Format the retrieved context into a single string
    context_text = "\n".join(f"- {ctx}" for ctx in retrieved_context)
    
    # Construct the system message: personality + memory context
    system_content = (
        f"{personality_prompt}\n\n"
        f"You have access to the following past memories with the user:\n"
        f"{context_text if context_text else 'No past memories yet.'}\n\n"
        f"Use these memories to inform your response if they are relevant."
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input}
    ]

    try:
        # We process the completion using a fast Groq model
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant", # Or 'mixtral-8x7b-32768'
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return "I'm having trouble thinking right now."
