"""Combined learning + deep research workflow.

Flow:
Start
-> clarify_with_user
-> check_data_exists
    - if data exists -> load_report
    - if no data -> write_research_brief
-> prepare_mcp_research
-> mcp_research_subgraph
-> supervisor_subgraph
-> final_report_generation
-> parallel(save_report_to_file, generate_structure)
-> create_content
-> administer_quiz
-> evaluate_submission
-> simplified_teaching (if failed)
-> administer_quiz
"""

import os
import re
import uuid
import operator

from pathlib import Path
from datetime import datetime
from typing import List, Annotated, Literal, Optional, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    filter_messages,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field

from deep_research_from_scratch.prompts import (
    final_report_generation_prompt,
    compress_research_system_prompt,
    compress_research_human_message,
)
from deep_research_from_scratch.research_agent_scope import clarify_with_user as scope_clarify_with_user, write_research_brief
from deep_research_from_scratch.multi_agent_supervisor import supervisor_agent
from deep_research_from_scratch.utils import (
    get_today_str,
    think_tool,
    get_current_dir,
    init_local_chat_model,
    get_structured_output_model,
)


class Checkpoint(TypedDict):
    id: str
    name: str
    objective: str
    study_material: str
    quiz_questions: list[str]
    user_answers: list[str]
    score: int
    passed: bool
    feedback: str
    simplified_material: str


class LearningProjectState(MessagesState):
    """State for the learning project workflow, extending MessagesState for automatic message handling."""
    report: str
    research_brief: Optional[str]
    user_request: str

    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_notes: Annotated[list[str], operator.add]
    notes: Annotated[list[str], operator.add]
    final_report: str

    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    compressed_research: str
    research_topic: str

    checkpoints: list[Checkpoint]
    current_checkpoint_index: int


class LearningProjectInput(MessagesState):
    """Input state for learning project, extends MessagesState for automatic message initialization."""
    pass


class CheckpointItem(BaseModel):
    name: str = Field(description="Name of the checkpoint")
    objective: str = Field(description="Objective of the checkpoint")


class CheckpointResponse(BaseModel):
    checkpoints: List[CheckpointItem]


class CheckpointContent(BaseModel):
    study_material: str = Field(description="Brief study material (approx 100 words)")
    quiz_questions: List[str] = Field(description="Exactly 3 assessment questions")


class EvaluationResult(BaseModel):
    score: int = Field(description="Score out of 100")
    feedback: str = Field(description="Constructive feedback for the student")
    passed: bool = Field(description="True if score >= 70, False if failed")


class SimplifiedContent(BaseModel):
    simplified_material: str = Field(description="Simple explanation using Feynman Technique (short, plain language, no jargon)")


writer_model = init_local_chat_model()
compress_model = init_local_chat_model()
mcp_llm_model = init_local_chat_model()


mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            str(get_current_dir() / "files"),
        ],
        "transport": "stdio",
    }
}

_client = None

_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "been",
    "being",
    "between",
    "could",
    "does",
    "from",
    "have",
    "into",
    "more",
    "that",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "topic",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
}


def get_mcp_client():
    global _client
    if _client is None:
        _client = MultiServerMCPClient(mcp_config)
    return _client


def get_today_str_local() -> str:
    try:
        return datetime.now().strftime("%a %b %#d, %Y")
    except ValueError:
        return datetime.now().strftime("%a %b %-d, %Y")


def _extract_topic_text(state: LearningProjectState) -> str:
    research_brief = state.get("research_brief")
    if isinstance(research_brief, str) and research_brief.strip():
        return research_brief

    user_request = state.get("user_request")
    if isinstance(user_request, str) and user_request.strip():
        return user_request

    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content

    return ""


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) >= 4 and w not in _STOPWORDS}


def _find_relevant_markdown_file(files_dir: Path, topic: str) -> Optional[Path]:
    md_files = list(files_dir.glob("*.md"))
    if not md_files:
        return None

    topic_tokens = _keywords(topic)
    if not topic_tokens:
        return max(md_files, key=lambda f: f.stat().st_mtime)

    best_file: Optional[Path] = None
    best_overlap_count = 0
    best_overlap_ratio = 0.0

    for file_path in md_files:
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            continue

        doc_tokens = _keywords(content)
        if not doc_tokens:
            continue

        overlap_count = len(topic_tokens.intersection(doc_tokens))
        overlap_ratio = overlap_count / max(1, len(topic_tokens))

        # Sufficiency gate: minimum content size and keyword coverage.
        content_word_count = len(content.split())
        is_sufficient = content_word_count >= 250 and (
            overlap_count >= 3 or overlap_ratio >= 0.3
        )
        if not is_sufficient:
            continue

        if (overlap_count > best_overlap_count) or (
            overlap_count == best_overlap_count and overlap_ratio > best_overlap_ratio
        ):
            best_file = file_path
            best_overlap_count = overlap_count
            best_overlap_ratio = overlap_ratio

    return best_file


def check_data_exists(state: LearningProjectState) -> Command[Literal["load_report", "write_research_brief"]]:
    """Route to an existing report when sufficient, otherwise continue with research."""
    print("--- Checking for Existing Data ---")

    files_dir = Path(__file__).parent / "files"
    if not files_dir.exists():
        print("Files directory not found. Moving to research...")
        return Command(goto="write_research_brief")

    topic = _extract_topic_text(state)
    matched_file = _find_relevant_markdown_file(files_dir, topic)
    if matched_file is None:
        print("No relevant/sufficient markdown file found. Moving to research...")
        return Command(goto="write_research_brief")

    print(f"Found relevant file: {matched_file.name}. Skipping research.")
    return Command(goto="load_report")


def clarify_with_user(state: LearningProjectState) -> Command[Literal["check_data_exists", "__end__"]]:
    """Run scope clarification and route to data check when clarification is sufficient."""
    result = scope_clarify_with_user(state)
    if result.goto == "write_research_brief":
        return Command(goto="check_data_exists", update=result.update)
    return result


def load_report(state: LearningProjectState):
    print("--- Loading Report from Files Directory ---")

    files_dir = Path(__file__).parent / "files"
    if not files_dir.exists():
        raise FileNotFoundError(f"Files directory not found: {files_dir}")

    topic = _extract_topic_text(state)
    selected_file = _find_relevant_markdown_file(files_dir, topic)

    if selected_file is None:
        raise FileNotFoundError(
            f"No topic-relevant and sufficient markdown file found in: {files_dir}"
        )

    report_content = selected_file.read_text(encoding="utf-8")

    return {"report": report_content}


def prepare_mcp_research(state: LearningProjectState):
    brief = state.get("research_brief", "")
    return {
        "user_request": brief,
        "research_topic": brief,
        "researcher_messages": [HumanMessage(content=brief)],
    }


async def mcp_llm_call(state: LearningProjectState):
    client = get_mcp_client()
    mcp_tools = await client.get_tools()
    tools = mcp_tools + [think_tool]
    model_with_tools = mcp_llm_model.bind_tools(tools)

    return {
        "researcher_messages": [
            model_with_tools.invoke(state["researcher_messages"])
        ]
    }


async def mcp_tool_node(state: LearningProjectState):
    tool_calls = state["researcher_messages"][-1].tool_calls

    async def execute_tools():
        client = get_mcp_client()
        mcp_tools = await client.get_tools()
        tools = mcp_tools + [think_tool]
        tools_by_name = {tool.name: tool for tool in tools}

        observations = []
        for tool_call in tool_calls:
            tool = tools_by_name[tool_call["name"]]
            if tool_call["name"] == "think_tool":
                observation = tool.invoke(tool_call["args"])
            else:
                observation = await tool.ainvoke(tool_call["args"])
            observations.append(observation)

        return [
            ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
            for observation, tool_call in zip(observations, tool_calls)
        ]

    messages = await execute_tools()
    return {"researcher_messages": messages}


def mcp_compress_research(state: LearningProjectState):
    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = [
        HumanMessage(content=system_message),
        *state.get("researcher_messages", []),
        HumanMessage(content=compress_research_human_message),
    ]
    response = compress_model.invoke(messages)

    raw_notes = [
        str(m.content)
        for m in filter_messages(
            state["researcher_messages"],
            include_types=["tool", "ai"],
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)],
        "notes": [str(response.content)],
    }


def mcp_should_continue(state: LearningProjectState) -> Literal["mcp_tool_node", "mcp_compress_research"]:
    last_message = state["researcher_messages"][-1]
    if last_message.tool_calls:
        return "mcp_tool_node"
    return "mcp_compress_research"


async def final_report_generation(state: LearningProjectState):
    notes = state.get("notes", [])
    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str_local(),
    )

    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])

    return {
        "final_report": final_report.content,
        "report": final_report.content,
        "messages": ["Here is the final report: " + final_report.content],
    }


async def save_report_to_file(state: LearningProjectState):
    final_report = state.get("final_report", "")

    module_dir = os.path.dirname(os.path.abspath(__file__))
    files_dir = os.path.join(module_dir, "files")
    os.makedirs(files_dir, exist_ok=True)

    report_id = str(uuid.uuid4())
    filename = f"report_{report_id}.md"
    filepath = os.path.join(files_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(final_report)

    return {"messages": [f"Report saved to: {filepath}"]}


def generate_structure(state: LearningProjectState):
    print("--- Generating Structure ---")
    report = state["report"]

    structure_gen = get_structured_output_model(CheckpointResponse)
    response = structure_gen.invoke(f"Extract learning checkpoints from this report: {report}")

    clean_checkpoints = []
    for item in response.checkpoints:
        data = item.model_dump()
        data["id"] = str(uuid.uuid4())
        data["study_material"] = ""
        data["quiz_questions"] = []
        data["user_answers"] = []
        data["score"] = 0
        data["passed"] = False
        data["feedback"] = ""
        data["simplified_material"] = ""
        clean_checkpoints.append(data)

    return {"checkpoints": clean_checkpoints, "current_checkpoint_index": 0}


def create_content(state: LearningProjectState):
    print("--- Creating Content (Batch) ---")
    report = state["report"]
    user_req = state.get("user_request", state.get("research_brief", ""))
    checkpoints = state["checkpoints"]

    content_gen = get_structured_output_model(CheckpointContent)

    prompts = []
    for cp in checkpoints:
        prompt = f"""You are creating educational content for a learning checkpoint.

Report Context: {report}
User Goal: {user_req}

Checkpoint Details:
- Name: {cp['name']}
- Objective: {cp['objective']}

REQUIREMENTS:
1. Create a clear, concise study material (approximately 100 words) that explains the key concepts of this checkpoint
2. Create EXACTLY 3 assessment questions that test understanding of the study material

IMPORTANT: Your response MUST include both fields:
- study_material: The explanation text
- quiz_questions: A list of exactly 3 questions (as strings)

Example format:
- study_material: "Python is a high-level programming language..."
- quiz_questions: ["What is X?", "Explain Y?", "How does Z work?"]

Now create the content:"""
        prompts.append(prompt)

    results = content_gen.batch(prompts)

    updated_checkpoints = []
    for cp, res in zip(checkpoints, results):
        cp["study_material"] = res.study_material
        cp["quiz_questions"] = res.quiz_questions
        updated_checkpoints.append(cp)

    return {"checkpoints": updated_checkpoints}


def administer_quiz(state: LearningProjectState):
    idx = state.get("current_checkpoint_index", 0)
    checkpoints = state["checkpoints"]

    if idx >= len(checkpoints):
        return {}

    current_cp = checkpoints[idx]
    material = current_cp["simplified_material"] if current_cp["simplified_material"] else current_cp["study_material"]

    user_view = {
        "title": current_cp["name"],
        "material": material,
        "questions": current_cp["quiz_questions"],
    }

    user_answers = interrupt(user_view)

    current_cp["user_answers"] = user_answers
    checkpoints[idx] = current_cp
    return {"checkpoints": checkpoints}


def evaluate_submission(state: LearningProjectState):
    idx = state["current_checkpoint_index"]
    checkpoints = state["checkpoints"]
    current_cp = checkpoints[idx]

    evaluator_gen = get_structured_output_model(EvaluationResult)
    prompt = f"""
    Topic: {current_cp['name']}
    Questions: {current_cp['quiz_questions']}
    Answers: {current_cp['user_answers']}
    Rubric: Pass mark is 70.
    """
    result = evaluator_gen.invoke(prompt)

    current_cp["score"] = result.score
    current_cp["passed"] = result.passed
    current_cp["feedback"] = result.feedback
    checkpoints[idx] = current_cp

    next_idx = idx + 1 if result.passed else idx
    return {"checkpoints": checkpoints, "current_checkpoint_index": next_idx}


def simplified_teaching(state: LearningProjectState):
    idx = state["current_checkpoint_index"]
    checkpoints = state["checkpoints"]
    current_cp = checkpoints[idx]

    simplified_gen = get_structured_output_model(SimplifiedContent)
    prompt = f"""The student struggled with this topic. Use the FEYNMAN TECHNIQUE to create a MUCH SIMPLER explanation.

Topic: {current_cp['name']}
Original Explanation: {current_cp['study_material']}

Questions Asked:
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(current_cp['quiz_questions'])])}

Student's Answers:
{chr(10).join([f"{i+1}. {a}" for i, a in enumerate(current_cp['user_answers'])])}

Feedback on Their Answers: {current_cp['feedback']}

FEYNMAN TECHNIQUE RULES:
1. Use ONLY simple, everyday language - avoid all technical jargon
2. Explain as if talking to a 10-year-old
3. Use analogies and real-world examples
4. Break complex ideas into simple parts
5. Be very short and concise
6. Focus on the core concept the student got wrong

Create a simplified explanation that helps the student understand the concept:"""

    result = simplified_gen.invoke(prompt)
    current_cp["simplified_material"] = result.simplified_material
    checkpoints[idx] = current_cp

    return {"checkpoints": checkpoints}


def decide_next_step(state: LearningProjectState) -> Literal["administer_quiz", "simplified_teaching", "__end__"]:
    idx = state["current_checkpoint_index"]
    checkpoints = state["checkpoints"]

    if idx >= len(checkpoints):
        return END

    current_cp = checkpoints[idx]
    if "passed" in current_cp and current_cp["passed"] is False:
        return "simplified_teaching"
    return "administer_quiz"


mcp_builder = StateGraph(LearningProjectState)
mcp_builder.add_node("mcp_llm_call", mcp_llm_call)
mcp_builder.add_node("mcp_tool_node", mcp_tool_node)
mcp_builder.add_node("mcp_compress_research", mcp_compress_research)
mcp_builder.add_edge(START, "mcp_llm_call")
mcp_builder.add_conditional_edges(
    "mcp_llm_call",
    mcp_should_continue,
    {
        "mcp_tool_node": "mcp_tool_node",
        "mcp_compress_research": "mcp_compress_research",
    },
)
mcp_builder.add_edge("mcp_tool_node", "mcp_llm_call")
mcp_builder.add_edge("mcp_compress_research", END)
mcp_research_subgraph = mcp_builder.compile()


learning_project_builder = StateGraph(LearningProjectState, input=LearningProjectInput)

learning_project_builder.add_node("clarify_with_user", clarify_with_user)
learning_project_builder.add_node("check_data_exists", check_data_exists)
learning_project_builder.add_node("load_report", load_report)
learning_project_builder.add_node("write_research_brief", write_research_brief)
learning_project_builder.add_node("prepare_mcp_research", prepare_mcp_research)
learning_project_builder.add_node("mcp_research_subgraph", mcp_research_subgraph)
learning_project_builder.add_node("supervisor_subgraph", supervisor_agent)
learning_project_builder.add_node("final_report_generation", final_report_generation)
learning_project_builder.add_node("save_report_to_file", save_report_to_file)
learning_project_builder.add_node("generate_structure", generate_structure)
learning_project_builder.add_node("create_content", create_content)
learning_project_builder.add_node("administer_quiz", administer_quiz)
learning_project_builder.add_node("evaluate_submission", evaluate_submission)
learning_project_builder.add_node("simplified_teaching", simplified_teaching)

# New flow: START -> clarify_with_user -> check_data_exists
learning_project_builder.add_edge(START, "clarify_with_user")

# If data exists and is sufficient, skip research and go directly to learning-content generation.
learning_project_builder.add_edge("load_report", "generate_structure")

# Continue with research workflow
learning_project_builder.add_edge("write_research_brief", "prepare_mcp_research")
learning_project_builder.add_edge("prepare_mcp_research", "mcp_research_subgraph")
learning_project_builder.add_edge("mcp_research_subgraph", "supervisor_subgraph")
learning_project_builder.add_edge("supervisor_subgraph", "final_report_generation")

learning_project_builder.add_edge("final_report_generation", "save_report_to_file")
learning_project_builder.add_edge("final_report_generation", "generate_structure")

learning_project_builder.add_edge("save_report_to_file", END)
learning_project_builder.add_edge("generate_structure", "create_content")
learning_project_builder.add_edge("create_content", "administer_quiz")
learning_project_builder.add_edge("administer_quiz", "evaluate_submission")
learning_project_builder.add_edge("simplified_teaching", "administer_quiz")

learning_project_builder.add_conditional_edges(
    "evaluate_submission",
    decide_next_step,
    {
        "administer_quiz": "administer_quiz",
        "simplified_teaching": "simplified_teaching",
        END: END,
    },
)

agent = learning_project_builder.compile()

learning_project = agent
learning_project_agent = learning_project
