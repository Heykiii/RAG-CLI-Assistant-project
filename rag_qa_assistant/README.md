# Minimalist QA Assistant

This is a simple command-line tool that answers questions using your own documents. It uses LangChain, OpenAI, and FAISS under the hood. Just put your `.md` or `.txt` files in the `docs/` folder, and ask questions in plain language.

## How it Works

- Reads your Markdown and text files.
- Breaks them into chunks for efficient search.
- Builds a fast FAISS index for quick lookups.
- Uses OpenAI to generate answers based on the most relevant chunks from your docs.

## File Structure

```
rag_qa_assistant/
├── qa.py           # Main program
├── ingest.py       # (optional) Script to build the vector store
├── vector_store/   # Stores the FAISS index
├── docs/           # Put your notes and docs here
└── README.md       # This file
```

## Quick Start

1. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```
   Or, if you prefer, install them one by one:

   ```
   pip install langchain==0.1.14 langchain-community>=0.0.9 langchain-openai>=0.0.8 openai==1.30.1 faiss-cpu==1.11.0 tiktoken==0.7.0 python-dotenv==1.0.1
   ```

2. **Add your OpenAI API key**

   Create a file called `.env` (in the project folder):

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Put your text or Markdown files in the `docs/` folder.**

4. **Ask a question**

   ```
   python qa.py --question "What is this project about?"
   ```

   You can use a different folder for documents:

   ```
   python qa.py --question "Summarize the documentation" --docs my_notes/
   ```

## Notes

- On the first run, it builds a FAISS index from your documents (might take a little time).
- Next time, it will use the existing index for faster answers.
- Only `.md` and `.txt` files are used.
- If your docs folder is empty or missing, you’ll get an error.

## Troubleshooting

- Check that your `.env` file exists and has your OpenAI API key.
- Only put `.md` and `.txt` files in your docs folder.
- Make sure all the required packages are installed (versions matter).

## Customization

- Edit `qa.py` to change how documents are split, which model is used, etc.

## License

MIT License.
