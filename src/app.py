import gradio as gr
from rag import answer_question

TITLE = "KI-MINTFIT — STEM Misconception Advisor"
DESCRIPTION = "Ask about a STEM misconception you encountered in primary-school teacher education."
PLACEHOLDER = "e.g. A teacher says plants get their food from the soil. What misconception is this?"


def format_sources(sources):
    lines = []
    for s in sources:
        line = f"• {s['citation']}"
        if s.get("url"):
            line += f"\n  {s['url']}"
        lines.append(line)
    return "\n\n".join(lines)


def format_retrieved(retrieved_cards):
    lines = []
    for item in retrieved_cards:
        score = item["score"]
        topic = item["card"]["topic"]
        rank = item["rank"]
        indicator = "🟢" if score >= 0.5 else "🟡" if score >= 0.3 else "🔴"
        lines.append(f"{indicator} Rank {rank} | {topic}\n   score: {score:.4f}")
    return "\n\n".join(lines)


def respond(query):
    if not query.strip():
        return "", "", "", "", "Please enter a question."

    answer, supporting_info, retrieved_cards = answer_question(query, k=3)
    answer = answer.strip() 
    
    if supporting_info is None:
        return answer, "", "", "", "No relevant cards found."

    diagnostic = supporting_info["diagnostic_question"]
    exercise = supporting_info["suggested_exercise"]
    sci_sources = format_sources(supporting_info["sources"])
    retrieved_text = format_retrieved(retrieved_cards)
    
    return answer, diagnostic, exercise, sci_sources, retrieved_text


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"## {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        query_box = gr.Textbox(
            label="Your question",
            placeholder=PLACEHOLDER,
            lines=2,
            scale=5
        )
        ask_btn = gr.Button("Ask", variant="primary", scale=1)

    with gr.Row():
        with gr.Column(scale=3):
            answer_box = gr.Textbox(label="Answer", lines=6, interactive=False)
            diagnostic_box = gr.Textbox(label="Diagnostic question", lines=3, interactive=False)
            exercise_box = gr.Textbox(label="Suggested exercise", lines=4, interactive=False)
            sources_box = gr.Textbox(label="Scientific sources", lines=5, interactive=False)

        with gr.Column(scale=1):
            retrieved_box = gr.Textbox(label="Retrieved cards", lines=10, interactive=False)

    ask_btn.click(
        fn=respond,
        inputs=query_box,
        outputs=[answer_box, diagnostic_box, exercise_box, sources_box, retrieved_box]
    )
    query_box.submit(
        fn=respond,
        inputs=query_box,
        outputs=[answer_box, diagnostic_box, exercise_box, sources_box, retrieved_box]
    )

if __name__ == "__main__":
    demo.launch()