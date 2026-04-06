# TASK 8 — Compare Two Embedding Models

from langchain_openai import OpenAIEmbeddings


def compare_embedding_models(sentence: str) -> dict:
    """Embeds a sentence with two models and compares their dimensions."""
    

    model_a = OpenAIEmbeddings(model="text-embedding-3-small")
    model_b = OpenAIEmbeddings(model="text-embedding-3-large")
   
    vec_a = model_a.embed_query(sentence)
    vec_b = model_b.embed_query(sentence)
    
    
    dims_a = len(vec_a)
    dims_b = len(vec_b)
   
    result = {
        "sentence": sentence,
        "model_a": {
            "model": "text-embedding-3-small",
            "dims": dims_a,
            "first_3": vec_a[:3]
        },
        "model_b": {
            "model": "text-embedding-3-large",
            "dims": dims_b,
            "first_3": vec_b[:3]
        },
        "dim_ratio": dims_b / dims_a if dims_a != 0 else 0
    }
    
    return result


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────
if __name__ == "__main__":
    
    sentence = "Vector databases power semantic search."
    
    output = compare_embedding_models(sentence)
    
    print("\n=== Embedding Comparison Result ===\n")
    
    print("Sentence:", output["sentence"])
    
    print("\nModel A:")
    print("  Model:", output["model_a"]["model"])
    print("  Dimensions:", output["model_a"]["dims"])
    print("  First 3 values:", output["model_a"]["first_3"])
    
    print("\nModel B:")
    print("  Model:", output["model_b"]["model"])
    print("  Dimensions:", output["model_b"]["dims"])
    print("  First 3 values:", output["model_b"]["first_3"])
    
    print("\nDimension Ratio (B/A):", output["dim_ratio"])



    #--------------------------------


