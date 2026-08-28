import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

EMBEDDINGS_PATH = Path("artifacts/card_embeddings.npy")
METADATA_PATH = Path("artifacts/card_metadata.json")
KB_PATH=Path("data/knowledge_base.json")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
def init():
   embeddings=np.load(EMBEDDINGS_PATH) #Load embeddings
   with open(METADATA_PATH,'r',encoding='utf-8') as file: #Load metadata file
      metadata=json.load(file)
   #Test--------------------------------------------------------------------------
   # print(embeddings.shape) #(12, 384)
   # print(len(metadata))    #12
   #------------------------------------------------------------------------------

   with open(KB_PATH,'r',encoding='utf-8') as file: # Load knowledge Base
      knowledge_base=json.load(file)

   cards_by_id={card['id']:card for card in knowledge_base} #Creat a dictionery with key=card['id'] and value= total card info

   #Craete Faiss vectore store
   dimension=embeddings.shape[1] #384
   index=faiss.IndexFlatIP(dimension) # built a database with dimension (384) columns in ram for fast search
   index.add(embeddings) #Bind the embedding vectors to the db indexs
   #Test--------------------------------------------------------------------------
   # print("Vectors in index: ",index.ntotal) #12
   #------------------------------------------------------------------------------

   model=SentenceTransformer(MODEL_NAME) # Run the model
   return model, index, metadata, cards_by_id

#Fuction to do search
def search_cards(query,model,index,metadata,k=3):
   query_embedding=model.encode([query],convert_to_numpy=True,normalize_embeddings=True) # Creat the query embedding
   scores, indices =index.search(query_embedding,k) #Semantic search by calculating cosine similarity
   results = []
   for rank, (idx, score) in enumerate(zip(indices[0], scores[0]),start=1):# Look up in metadata
        card = metadata[idx]
        results.append({
            "rank": rank,
            "id": card["id"],
            "subject": card["subject"],
            "topic": card["topic"],
            "score": float(score)
        })

   return results

#Function to Create a list of 3 retrieved card with full info of card and their retrieve rank and score
def attach_full_card(search_results,cards_by_id):
   retrieved =[]
   for result in search_results:
      card=cards_by_id[result['id']]
      retrieved.append({
            "card":card,
            "rank":result['rank'],
            "score":result['score']
      })
   return retrieved 

#function to integrate the search and return the full ino of retrieved cards
def retrieve(query,model,index,metadata,cards_by_id,k=3):
   search_results=search_cards(query,model,index,metadata,k)
   full_retrieve=attach_full_card(search_results,cards_by_id)
   return full_retrieve   


#Test inline query--------------------------------------------------------------------------
if __name__ == "__main__":
   model, index, metadata, cards_by_id = init()
   retrieved_cards = retrieve(
   query="Why can a heavy ship float?",
   model=model,
   index=index,
   metadata=metadata,
   cards_by_id=cards_by_id,
   k=3
   )

   for item in retrieved_cards:
      print("\nRank:", item["rank"])
      print("Score:", item["score"])
      print("ID:", item["card"]["id"])
      print("Topic:", item["card"]["topic"])
      print("Correct explanation:")
      print(item["card"]["correct_explanation"][:200])
#------------------------------------------------------------------------------


#Test search function--------------------------------------------------------------------------
# query = "Why can a heavy ship float while a small metal screw sinks?"
# results = search_cards(query=query,model=model,index=index,metadata=metadata,k=3)
# for result in results:
#     print(result)
#------------------------------------------------------------------------------


"""
#Test inline query--------------------------------------------------------------------------

query = "Why can a heavy ship float while a small metal screw sinks?"  #User query
query_embedding=model.encode([query],convert_to_numpy=True,normalize_embeddings=True) # Creat the query embedding
#Test--------------------------------------------------------------------------
# print(query_embedding)
# print(query_embedding.shape) #(1, 384)
#------------------------------------------------------------------------------

k=3  #Retrive 3 top related cards
scores, indeies =index.search(query_embedding,k) #Semantic search by calculating cosine similarity
#Test--------------------------------------------------------------------------
# print(indeces) #[[ 5 10  8]]
# print(scores)  #[[0.7395704  0.32414705 0.27095243]]
for rank, (idx,score) in enumerate (zip(indices[0],scores[0]),start=1):
   card=metadata[idx]
   print(f"rank {rank} -> {card['id']} | {card['topic']} | score :{score:.4f}")
   
   rank 1 -> sci_floating_sinking_01 | Floating, sinking and density : 0.7396    
   rank 2 -> sci_evaporation_01 | Evaporation and conservation of matter : 0.3241
   rank 3 -> sci_force_motion_01 | Force and motion : 0.2710
#------------------------------------------------------------------------------
   """