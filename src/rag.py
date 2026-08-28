import ollama
from vector_search import retrieve, init

LLM_MODEL = "llama3.2"

#Load embedding model, FAISS index, metadata and knowledge base cards
model, index, metadata, cards_by_id = init()


#Function to build grounded prompt from user query and retrieved cards
def build_prompt(query, retrieved_cards):
    context_parts = []

    #Read required information from each retrieved card and add it to context
    for item in retrieved_cards:
        card = item['card']
        context_parts.append(
            f"Card ID: {card['id']}\n"
            f"Topic: {card['topic']}\n"
            f"Misconception: {card['misconception']}\n"
            f"Correct explanation: {card['correct_explanation']}\n"
        )

    #Join retrieved cards and create the final context for LLM
    context = "\n\n".join(context_parts)

    #Create grounded prompt and limit LLM answer to knowledge base context
    return f"""You are an educational assistant supporting primary-school teachers \
with mathematics and science subject knowledge.
Answer the user's question using only the provided knowledge-base context.
If the context does not contain enough information, say so clearly.
Keep the answer clear and concise.

User question: {query}

Knowledge-base context:
{context}

Use only the most relevant card. 
If a card is not related to the question, ignore it.

Structure your answer as:
Misconception: [name it]
Correct explanation: [from the context]

Answer:"""


#Function to send grounded prompt to local Llama model through Ollama
def call_llm(prompt):
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


#Function to get supporting information directly from top-ranked knowledge base card
def build_supporting_info(retrieved_cards):
    if not retrieved_cards:
        return None

    #Use the top-ranked card as the main supporting card
    main_card = retrieved_cards[0]["card"]

    #Get structured information directly from KB instead of generating them with LLM
    supporting_info = {
        "topic": main_card["topic"],
        "diagnostic_question": main_card["diagnostic_question"]["question"],
        "suggested_exercise": main_card["suggested_exercise"]["description"],
        "sources": main_card["sources"]
    }

    return supporting_info


#Function to integrate retrieval, abstention, grounded prompt and LLM answer
def answer_question(query, k=3):
    #Retrieve top-k related cards from vector search
    retrieved_cards = retrieve(query, model, index, metadata, cards_by_id, k)

    #Abstention if there is no retrieved card
    if not retrieved_cards:
        return "I'm sorry, I couldn't find relevant information in my knowledge base.", None, retrieved_cards

    #Abstention if the top retrieved score is too low
    if retrieved_cards[0]["score"] < 0.35:
        return "I'm sorry, I don't have reliable information about this topic in my knowledge base.", None, retrieved_cards

    #Build grounded prompt from retrieved cards and generate answer with LLM
    prompt = build_prompt(query, retrieved_cards)
    answer = call_llm(prompt)

    #Get diagnostic question, exercise and sources directly from Rank-1 KB card
    supporting_info = build_supporting_info(retrieved_cards)

    return answer, supporting_info, retrieved_cards


#Test complete RAG pipeline--------------------------------------------------------------------------
if __name__ == "__main__":
    query = (
        # "Plants get their food from the soil through their roots"
        "Warum schwimmen manche Objekte und andere sinken?"
    )

    answer, supporting_info, retrieved_cards = answer_question(query)

    print("\nANSWER:\n")
    print(answer)

    #Print supporting information only if system does not abstain
    if supporting_info is not None:
        print("\nTOPIC:")
        print(supporting_info["topic"])

        print("\nDIAGNOSTIC QUESTION:")
        print(supporting_info["diagnostic_question"])

        print("\nSUGGESTED EXERCISE:")
        print(supporting_info["suggested_exercise"])

        print("\nSCIENTIFIC SOURCES:")
        for source in supporting_info["sources"]:
            print("-", source["citation"])
            if source["url"]:
                print("  ", source["url"])

    print("\nRETRIEVED SOURCES:")
    for item in retrieved_cards:
        print(f"- Rank {item['rank']} | {item['card']['topic']} | score={item['score']:.4f}")
#------------------------------------------------------------------------------