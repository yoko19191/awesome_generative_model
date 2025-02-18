import time
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dotenv import load_dotenv, find_dotenv
import os 
import sys

_ = load_dotenv(find_dotenv())

class EmbeddingBenchmark:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.models = {
            "text-embedding-3-small": {"dim": 1536, "max_tokens": 8191},
            "text-embedding-3-large": {"dim": 3072, "max_tokens": 8191},
            "text-embedding-ada-002": {"dim": 1536, "max_tokens": 8191},
        }
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        """获取文本的 embedding"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding for model {model}: {e}")
            return None

    def benchmark_performance(self, 
                            texts: List[str],
                            model_names: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """评估模型性能"""
        if model_names is None:
            model_names = list(self.models.keys())
            
        results = {}
        
        for model_name in model_names:
            print(f"Testing model: {model_name}")
            start_time = time.time()
            
            # 生成 embeddings
            embeddings = []
            for text in texts:
                embedding = self.get_embedding(text, model_name)
                if embedding:
                    embeddings.append(embedding)
            
            end_time = time.time()
            
            if embeddings:
                # 计算性能指标
                results[model_name] = {
                    "inference_time": end_time - start_time,
                    "avg_time_per_text": (end_time - start_time) / len(texts),
                    "dimension": len(embeddings[0]),
                    "memory_usage": sum(sys.getsizeof(emb) for emb in embeddings),
                }
                
                # 计算 embeddings 之间的相似度矩阵
                similarity_matrix = cosine_similarity(embeddings)
                results[model_name]["avg_similarity"] = np.mean(similarity_matrix)
                results[model_name]["std_similarity"] = np.std(similarity_matrix)
            
        return results

    def evaluate_similarity_task(self,
                               pairs: List[tuple],
                               labels: List[int],
                               model_name: str,
                               threshold: float = 0.5) -> Dict[str, float]:
        """评估相似度任务性能
        
        Args:
            pairs: 文本对列表 [(text1, text2), ...]
            labels: 标签列表 [1, 0, ...] (1表示相似，0表示不相似)
            model_name: 模型名称
            threshold: 相似度阈值
        """
        predictions = []
        
        for text1, text2 in pairs:
            emb1 = self.get_embedding(text1, model_name)
            emb2 = self.get_embedding(text2, model_name)
            
            if emb1 is not None and emb2 is not None:
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                predictions.append(1 if similarity > threshold else 0)
            else:
                predictions.append(0)
        
        # 计算评估指标
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

def example_usage():
    """使用示例"""
    # 初始化
    benchmark = EmbeddingBenchmark()
    
    # 基础性能测试
    test_texts = [
        "This is a test sentence.",
        "Another test sentence for embedding.",
        "Testing embedding models performance."
    ]
    
    performance_results = benchmark.benchmark_performance(test_texts)
    print("Performance Results:", performance_results)
    
    # 相似度任务评估
    similarity_pairs = [
        ("The cat sits on the mat.", "A cat is sitting on the mat."),
        ("The weather is nice today.", "I love playing basketball."),
    ]
    similarity_labels = [1, 0]  # 1表示相似，0表示不相似
    
    similarity_results = benchmark.evaluate_similarity_task(
        similarity_pairs,
        similarity_labels,
        "text-embedding-3-small"
    )
    print("Similarity Task Results:", similarity_results)

if __name__ == "__main__":
    example_usage()
