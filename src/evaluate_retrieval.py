import json
from pathlib import Path
from vector_search import init, search_cards

EVAL_PATH=Path("data/evaluation_queries.json")


#Function to load evaluation queries json file
def load_evaluation_set(path:Path):
    with open(path,'r',encoding="utf-8") as file:
        return json.load(file)


#Function to evaluate retrieval results
def evaluate():
    model,index,metadata,cards_by_id=init()
    evaluation_set=load_evaluation_set(EVAL_PATH)

    recall_at_1_hits=0
    recall_at_3_hits=0
    reciprocal_ranks=[]

    detailed_results=[]

    #Search all 12 cards to find the real rank of expected card for calculating MRR
    k_all=len(metadata)

    for item in evaluation_set:
        query=item["query"]
        expected_id=item["expected_id"]

        #Retrieve and rank all cards for each evaluation query
        results=search_cards(
            query=query,
            model=model,
            index=index,
            metadata=metadata,
            k=k_all
        )

        retrieved_ids=[result["id"] for result in results]

        #Calculate Recall@1 hits
        if retrieved_ids[0]==expected_id:
            recall_at_1_hits+=1

        #Calculate Recall@3 hits
        if expected_id in retrieved_ids[:3]:
            recall_at_3_hits+=1

        #Find expected card rank and calculate Reciprocal Rank
        if expected_id in retrieved_ids:
            rank=retrieved_ids.index(expected_id)+1
            reciprocal_rank=1.0/rank
        else:
            rank=None
            reciprocal_rank=0.0

        reciprocal_ranks.append(reciprocal_rank)

        #Save detailed results for qualitative analysis
        detailed_results.append({
            "query":query,
            "expected_id":expected_id,
            "predicted_top1":retrieved_ids[0],
            "expected_rank":rank,
            "top3":[
                {
                    "rank":result["rank"],
                    "id":result["id"],
                    "topic":result["topic"],
                    "score":result["score"]
                }
                for result in results[:3]
            ]
        })

    #Calculate final evaluation metrics
    n=len(evaluation_set)

    recall_at_1=recall_at_1_hits/n
    recall_at_3=recall_at_3_hits/n
    mrr=sum(reciprocal_ranks)/n

    #Print final evaluation results
    print("\n"+"="*70)
    print("KI-MINTFIT RETRIEVAL EVALUATION")
    print("="*70)

    print(f"\nQueries:   {n}")
    print(f"Recall@1:  {recall_at_1:.3f} ({recall_at_1_hits}/{n})")
    print(f"Recall@3:  {recall_at_3:.3f} ({recall_at_3_hits}/{n})")
    print(f"MRR:       {mrr:.3f}")

    #Print detailed results for each evaluation query
    print("\nDETAILED RESULTS")
    print("-"*70)

    for i,result in enumerate(detailed_results,start=1):
        status="PASS" if result["expected_rank"]==1 else "MISS"

        print(f"\n{i}. [{status}] {result['query']}")
        print(f"Expected: {result['expected_id']}")
        print(f"Expected rank: {result['expected_rank']}")

        for retrieved in result["top3"]:
            print(
                f"  Rank {retrieved['rank']} | "
                f"{retrieved['id']} | "
                f"{retrieved['topic']} | "
                f"score={retrieved['score']:.4f}"
            )


#Test evaluation--------------------------------------------------------------------------
if __name__=="__main__":
    evaluate()
#------------------------------------------------------------------------------