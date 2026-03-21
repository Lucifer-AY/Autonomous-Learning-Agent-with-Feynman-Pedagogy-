
"""Research Utilities and Tools.

This module provides search and content processing utilities for the research agent,
including web search capabilities and content summarization tools.
"""

import os

from pathlib import Path
from datetime import datetime
from typing_extensions import Annotated, List, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model 
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from tavily import TavilyClient

from deep_research_from_scratch.state_research import Summary
from deep_research_from_scratch.prompts import summarize_webpage_prompt

# Load environment variables from .env for local development and script execution.
load_dotenv(override=True)

# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    # Use %#d for Windows compatibility, fallback to %-d for others
    try:
        return datetime.now().strftime("%a %b %#d, %Y")
    except ValueError:
        return datetime.now().strftime("%a %b %-d, %Y")

def get_current_dir() -> Path:
    """Get the current directory of the module.

    This function is compatible with Jupyter notebooks and regular Python scripts.

    Returns:
        Path object representing the current directory
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:  # __file__ is not defined
        return Path.cwd()

# ===== CONFIGURATION =====

def get_local_llm_provider() -> str:
    """Get configured local LLM provider."""
    return os.getenv("LOCAL_LLM_PROVIDER", "ollama").strip().lower()


def get_configured_local_model_name() -> str:
    """Resolve local model name from env vars.

    Priority:
    1. LOCAL_LLM_MODEL (explicit model id)
    2. LOCAL_LLM_MODEL_PATH (extract owner/model from local path)
    3. OLLAMA_MODEL (backward-compatible fallback)
    """
    model_name = os.getenv("LOCAL_LLM_MODEL", "").strip()
    if model_name:
        return model_name

    model_path = os.getenv("LOCAL_LLM_MODEL_PATH", "").strip()
    if model_path:
        normalized = model_path.replace("\\", "/").rstrip("/")
        parts = [part for part in normalized.split("/") if part]
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        return parts[-1]

    return os.getenv("OLLAMA_MODEL", "mistral")


def init_local_chat_model(**kwargs):
    """Create a chat model using local provider settings.

    Supported providers:
    - ollama: Uses OLLAMA endpoint and model id
    - lmstudio: Uses LM Studio OpenAI-compatible API endpoint
    """
    provider = get_local_llm_provider()
    model_name = get_configured_local_model_name()

    if provider == "lmstudio":
        return init_chat_model(
            model=model_name,
            model_provider="openai",
            base_url=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
            api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
            **kwargs,
        )

    if provider == "ollama":
        return init_chat_model(f"ollama:{model_name}", **kwargs)

    raise ValueError(
        f"Unsupported LOCAL_LLM_PROVIDER '{provider}'. Use 'ollama' or 'lmstudio'."
    )


def get_structured_output_model(schema, fallback_to_gemini=True):
    """Get a model that supports structured output.
    
    Tries local model first. If it doesn't support structured output,
    falls back to Gemini if GOOGLE_API_KEY is available.
    
    Args:
        schema: Pydantic model for structured output
        fallback_to_gemini: Whether to fallback to Gemini (default: True)
        
    Returns:
        A model with structured output support bound
    """
    try:
        # Try local model with structured output
        local_model = init_local_chat_model()
        return local_model.with_structured_output(schema)
    except (ValueError, Exception) as e:
        # If structured output fails and fallback enabled, use Gemini
        if fallback_to_gemini and (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            try:
                gemini_model = init_chat_model("google_genai:models/gemini-flash-latest")
                return gemini_model.with_structured_output(schema)
            except Exception as gemini_error:
                raise RuntimeError(
                    f"Both local model and Gemini structured output failed. "
                    f"Local error: {str(e)}. Gemini error: {str(gemini_error)}"
                ) from gemini_error
        else:
            raise RuntimeError(
                f"Local model does not support structured output and no Gemini API key available. "
                f"Set GOOGLE_API_KEY or GEMINI_API_KEY for fallback. Error: {str(e)}"
            ) from e


summarization_model = init_local_chat_model()

# Lazy initialization of TavilyClient to avoid errors when API key is not set
_tavily_client = None

def get_tavily_client():
    """Get or lazily initialize TavilyClient on first use."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient()
    return _tavily_client

# ===== SEARCH FUNCTIONS =====

def tavily_search_multiple(
    search_queries: List[str], 
    max_results: int = 3, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
) -> List[dict]:
    """Perform search using Tavily API for multiple queries.

    Args:
        search_queries: List of search queries to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        List of search result dictionaries
    """

    # Execute searches sequentially. Note: yon can use AsyncTavilyClient to parallelize this step.
    client = get_tavily_client()
    search_docs = []
    for query in search_queries:
        result = client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        search_docs.append(result)

    return search_docs

def summarize_webpage_content(webpage_content: str) -> str:
    """Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw webpage content to summarize

    Returns:
        Formatted summary with key excerpts
    """
    try:
        # Set up structured output model for summarization
        structured_model = summarization_model.with_structured_output(Summary)

        # Generate summary
        summary = structured_model.invoke([
            HumanMessage(content=summarize_webpage_prompt.format(
                webpage_content=webpage_content, 
                date=get_today_str()
            ))
        ])

        # Format summary with clear structure
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except Exception as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content

def deduplicate_search_results(search_results: List[dict]) -> dict:
    """Deduplicate search results by URL to avoid processing duplicate content.

    Args:
        search_results: List of search result dictionaries

    Returns:
        Dictionary mapping URLs to unique results
    """
    unique_results = {}

    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result

    return unique_results

def process_search_results(unique_results: dict) -> dict:
    """Process search results by summarizing content where available.

    Args:
        unique_results: Dictionary of unique search results

    Returns:
        Dictionary of processed results with summaries
    """
    summarized_results = {}

    for url, result in unique_results.items():
        # Use existing content if no raw content for summarization
        if not result.get("raw_content"):
            content = result['content']
        else:
            # Summarize raw content for better processing
            content = summarize_webpage_content(result['raw_content'])

        summarized_results[url] = {
            'title': result['title'],
            'content': content
        }

    return summarized_results

def format_search_output(summarized_results: dict) -> str:
    """Format search results into a well-structured string output.

    Args:
        summarized_results: Dictionary of processed search results

    Returns:
        Formatted string of search results with clear source separation
    """
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"

    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output

# ===== RESEARCH TOOLS =====

@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Fetch results from Tavily search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')

    Returns:
        Formatted string of search results with summaries
    """
    # Execute search for single query
    search_results = tavily_search_multiple(
        [query],  # Convert single query to list for the internal function
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    # Deduplicate results by URL to avoid processing duplicate content
    unique_results = deduplicate_search_results(search_results)

    # Process results with summarization
    summarized_results = process_search_results(unique_results)

    # Format output for consumption
    return format_search_output(summarized_results)

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"
