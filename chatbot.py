# ============================================================================
# 1. IMPORTS AND INITIALIZATION
# ============================================================================

import operator
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langsmith import traceable
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()


# ============================================================================
# 2. LLM CONFIGURATION
# ============================================================================

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0
)


# ============================================================================
# 3. DATA MODELS
# ============================================================================

class EvaluationSchema(BaseModel):
    """Schema for structured output from evaluation LLM calls."""
    feedback: str = Field(description='Feedback for the essay and paragraphs.')
    score: int = Field(description='Score the essay and paragraphs out of 10.')

struct_llm = llm.with_structured_output(EvaluationSchema)


class ChatBotState(TypedDict, total=False):
    essay: str                             
    language_feedback: str                 
    analysis_feedback: str                 
    clarity_feedback: str                  
    final_feedback: str                    
    individual_score: Annotated[List[int], operator.add]  
    avg_score: float


# ============================================================================
# 4. EVALUATION FUNCTIONS
# ============================================================================

@traceable(name='language_evaluation', metadata={'dimension': 'language'})
def eval_language(state: ChatBotState):
    prompt = f'Evaluate the language of this essay: {state["essay"]}\nProvide feedback and give it a score out of 10.'
    response = struct_llm.invoke(prompt)
    return {'language_feedback': response.feedback, 'individual_score': [response.score]}


@traceable(name='analysis_evaluation', metadata={'dimension': 'analysis'})
def eval_analysis(state: ChatBotState):
    prompt = f'Evaluate the depth and analysis of this essay: {state["essay"]}\nProvide feedback and give it a score out of 10.'
    response = struct_llm.invoke(prompt)
    return {'analysis_feedback': response.feedback, 'individual_score': [response.score]}


@traceable(name='clarity_evaluation', metadata={'dimension': 'clarity'})
def eval_clarity(state: ChatBotState):
    prompt = f'Evaluate the clarity and readability of this essay: {state["essay"]}\nProvide feedback and give it a score out of 10.'
    response = struct_llm.invoke(prompt)
    return {'clarity_feedback': response.feedback, 'individual_score': [response.score]}


@traceable(name='feedback_evaluation', metadata={'dimension': 'feedback'})
def final_feedback(state: ChatBotState):
    prompt = (
        'Based on the below feedbacks, create a comprehensive final feedback.\n'
        f'Language Feedback: {state.get("language_feedback", "")}\n'
        f'Analysis Feedback: {state.get("analysis_feedback", "")}\n'
        f'Clarity Feedback: {state.get("clarity_feedback", "")}'
    )

    response = llm.invoke(prompt)
    scores = state.get('individual_score', [])
    avg_scores = sum(scores) / len(scores) if scores else 0
    return {'final_feedback': response.content, 'avg_score': avg_scores}


# ============================================================================
# 5. WORKFLOW GRAPH CONSTRUCTION
# ============================================================================

graph = StateGraph(ChatBotState)

# Add evaluation nodes
graph.add_node('eval_language', eval_language)
graph.add_node('eval_analysis', eval_analysis)
graph.add_node('eval_clarity', eval_clarity)
graph.add_node('final_feedback', final_feedback)

# Define workflow edges
# Start → All three evaluation nodes (parallel execution)
graph.add_edge(START, 'eval_language')
graph.add_edge(START, 'eval_analysis')
graph.add_edge(START, 'eval_clarity')

# All evaluation nodes → Final feedback synthesis
graph.add_edge('eval_language', 'final_feedback')
graph.add_edge('eval_analysis', 'final_feedback')
graph.add_edge('eval_clarity', 'final_feedback')

# Final feedback → End
graph.add_edge('final_feedback', END)
workflow = graph.compile()


# ============================================================================
# 6. SAMPLE ESSAY
# ============================================================================

essay2 = '''Curiosity is the fundamental spark that drives human progress. It is the restless desire to understand the "why" and "how" behind the world around us. From a young age, this trait manifests as simple questions, but as we mature, it evolves into the rigorous inquiry that fuels scientific breakthroughs, artistic masterpieces, and technological innovation. Without this innate pull toward the unknown, society would remain stagnant, tethered to the comfort of existing knowledge.
However, curiosity is more than just a tool for professional or academic advancement; it is a vital component of personal growth. By remaining open to new ideas and diverse perspectives, individuals foster empathy and intellectual flexibility. Engaging with the unfamiliar challenges our biases and expands our mental horizons, making life a continuous journey of learning rather than a destination of fixed beliefs'''


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    result = workflow.invoke(
        {'essay': essay2},
        config={
            'tags': ['langgraph', 'langsmith', 'essayevaluator'],
            'metadata': {
                'essay_length': len(essay2),
                'model': 'Groq',
                'dimensions': ['language', 'analysis', 'clarity', 'feedback'],
            },
        },
    )

    # ========================================================================
    # 8. OUTPUT RESULTS
    # ========================================================================
    
    print("\n=== Evaluation Results ===")
    # print("Language feedback:\n", result.get("language_feedback", ""), "\n")
    # print("Analysis feedback:\n", result.get("analysis_feedback", ""), "\n")
    # print("Clarity feedback:\n", result.get("clarity_feedback", ""), "\n")
    print("Final feedback:\n", result.get("final_feedback", ""), "\n")
    print("Individual scores:", result.get("individual_score", []))
    print("Average score:", result.get("avg_score", 0.0))
