# Setup and Usage

1. Create and activate a virtual environment:

   **python3 -m venv .venv**

   **source .venv/bin/activate**

2. Install dependencies:

    **pip install -r requirements.txt**

3. Set the OpenRouter API key:

   The project uses an LLM agent via OpenRouter, so an API key is required.
   
   **export OPENROUTER_API_KEY="YOUR_API_KEY_HERE"**
   
   **export LLM_MODEL="openai/gpt-4o-mini"**

4. Run the program with manual instruction input:
   
   **python -m main**

   You will be prompted to enter a single instruction in the terminal, for example:

   "Drop Cabin and Ticket and train random forest"

5. Run the program using an instruction file. Create a file called instruction.txt and write the instruction inside that file. Then run:
   
   **python -m main --infile instructions.txt**

6. Save output to a file inside the outputs folder:

   **python -m main --infile instructions.txt --outfile outputs/output.txt**

   The output will be printed to the console and also saved to output.txt.
