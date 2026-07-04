from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache

class IntentDetector:
    def __init__(self):
        # مدل فارسی سریع - jina-embeddings-v2-small-en یا paraphrase-multilingual-MiniLM
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # نمونه‌های نماینده هر intent
        self.intent_examples = {
            "tax": ["مالیات من چقدر است", "درآمد مالیاتی", "tax declaration"],
            "military": ["وضعیت سربازی", "معافیت نظام وظیفه", "خدمت سربازی"],
            "traffic": ["خلافی ماشین", "جریمه رانندگی", "تخلف پلاک"],
            "insurance": ["بیمه تامین اجتماعی", "سابقه بیمه", "بازنشستگی"],
            "subsidy": ["یارانه من", "کمک معیشت", "سوبسید"],
            "judicial": ["سوء پیشینه", "پرونده قضایی", "دادگاه"],
            "profile": ["پروفایل من", "اطلاعات شخصی", "مشخصات"]
        }
        
        # محاسبه و کش embeddings
        self.intent_embeddings = {}
        for intent, examples in self.intent_examples.items():
            embeddings = self.model.encode(examples)
            self.intent_embeddings[intent] = np.mean(embeddings, axis=0)
    
    @lru_cache(maxsize=1000)
    def detect_service_intent(self, message):
        message_embedding = self.model.encode(message)
        
        similarities = {}
        for intent, intent_emb in self.intent_embeddings.items():
            similarity = np.dot(message_embedding, intent_emb) / (
                np.linalg.norm(message_embedding) * np.linalg.norm(intent_emb)
            )
            similarities[intent] = similarity
        
        best_intent = max(similarities, key=similarities.get)
        best_score = similarities[best_intent]
        
        # threshold برای reject کردن پیام‌های نامرتبط
        if best_score < 0.4:
            return None
        
        return best_intent

# استفاده
detector = IntentDetector()
result = detector.detect_service_intent("خلافی ماشینم رو چک کن")  # -> "traffic"
