import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

KB_PATH=Path("data/knowledge_base.json")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

#Function to load knowledge ase json file
def load_knowledge_base(path:Path):
    with open(path,'r',encoding="utf-8") as file:
        return json.load(file)

knowledge_base=load_knowledge_base(KB_PATH)
model=SentenceTransformer(MODEL_NAME)
#Test--------------------------------------------------------------------------
#Test the loaded model, dimension of eatch embedding vectore, and maximum length accepted with model for text that will be embedded vector
# print(MODEL_NAME)
# print(model.get_sentence_embedding_dimension())   #384
# print(model.get_max_seq_length())   #128
# print(f"legth of knowledge base is: {len(knowledge_base)} ")   #12 cards
# print(knowledge_base[0]['topic'])   #"Comparing fractions"
#------------------------------------------------------------------------------

#Function for convert cards' selected keys values (will be embedded) to a sequence of text
def build_retrieval_text(card):
    parts=[card['topic'],card['misconception'],*card['misconception_variants']]
    return " ".join(parts)

retrieval_texts=[build_retrieval_text(card) for card in knowledge_base]
#Test--------------------------------------------------------------------------
# print(f"retrieval_text is {len(retrieval_texts)}")   #12- There is a retreival text for each card  
# # assert(len(knowledge_base)==len(retrieval_texts))    #12=12
# for card, text in zip (knowledge_base,retrieval_texts):
#     assert text.strip()  #Not empty text and remove first and end space
#     print(card["id"], "->", len(text), "characters")   #math_fraction_comparison_01 -> 429 characters
#------------------------------------------------------------------------------

#Create the embeddings vectors for each text 
#Convert to numpy array instead of tensors oftarch or tensorflow to easy use with other libraries
#Normalize the vectors by divied each vector to absolute value of it to convert all vectores value to 1 , for using 
# in dot/inner product that can use to calculate cosine similaty in semantic search nd RAG systems 
embeddings=model.encode(retrieval_texts,convert_to_numpy=True,normalize_embeddings=True) 
#Test--------------------------------------------------------------------------
# print(embeddings.shape)#(12,384)
# print(embeddings[1],"\n\n")#[ 0.02960433  0.04716035  0.01192581  0.02539203  ...]
# print(embeddings.shape[0])#12
# print(embeddings.shape[1]) #384                 
# assert(embeddings.shape[0]==len(knowledge_base))            #12=12
# assert(embeddings.shape[1]==model.get_embedding_dimension())#384=384
#------------------------------------------------------------------------------

#Create artifacts directory and Save the embeddings and metadata into it 
OUTPUT_DIR=Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)

EMBEDDINGS_PATH=OUTPUT_DIR/"card_embeddings.npy"
METADATA_PATH=OUTPUT_DIR/"card_metadata.json"

np.save(EMBEDDINGS_PATH,embeddings)  #Save embeddings into card_embeddings.npy file

#Read three keys of each card from knowledge base and Create a dictionery for each card and 
# Save a list of metadata dictioneries into metadata.json file
#This metadata demonstrates each row assocciated with which card 
metadata=[
    {
    "id":card["id"],
    "subject":card["subject"],
    "topic":card["topic"]
    }
for card in knowledge_base]

with open(METADATA_PATH,"w",encoding="utf-8") as file:# save metadata in json file
    json.dump(metadata, file, ensure_ascii=False, indent=2)
#Test--------------------------------------------------------------------------
# assert embeddings.shape == (len(knowledge_base),model.get_sentence_embedding_dimension())
# assert EMBEDDINGS_PATH.exists()
# assert METADATA_PATH.exists()
# print("\nPhase 3 artifacts saved successfully.")
# print("Embeddings:", EMBEDDINGS_PATH) #Embeddings: artifacts\card_embeddings.npy
# print("Metadata:", METADATA_PATH) #Metadata: artifacts\card_metadata.json
# print("Embedding matrix shape:", embeddings.shape) #(12, 384)
#------------------------------------------------------------------------------




