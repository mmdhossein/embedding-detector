"""
Persian Service Intent Classification — ML Pipeline
=====================================================
Replaces keyword-based `detect_service_intent()` with a trained sklearn model.

Approach:
  1. Synthetic Persian dataset (200+ samples per intent, 7 intents)
  2. TF-IDF character n-grams (captures Persian morphology without a stemmer)
  3. Linear SVM (fast, strong baseline for text)
  4. Evaluation + export to pickle for production use
"""

import random
import json
import re
import pickle
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ──────────────────────────────────────────────
# 1.  SYNTHETIC DATASET
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Templates per intent — each entry is a base sentence.
# We permute keywords and add natural noise to create many variants.
RAW_TEMPLATES = {
    "tax": [
        "مالیات من چقدر شده",
        "میخوام مالیات سال {year} رو ببینم",
        "پرونده مالیاتی من رو چک کنید",
        "آیا بدهی مالیاتی دارم؟",
        "مالیات بر درآمد من چقدره",
        "اقساط مالیاتی خودم رو پرداخت کنم",
        "فرم اظهارنامه مالیاتی رو از کجا دانلود کنم",
        "شماره مالیاتی من رو میخوام",
        "آخرین وضعیت مالیات من چی هست",
        "کد اقتصادی من چیه",
        "مالیات عملکرد سال {year} رو بررسی کنید",
        "دفتر کل مالیاتی رو ببینم",
        "یادآوری پرداخت مالیات",
        "مدت زمان تسویه مالیاتی",
        "مالیات بر ارزش افزوده",
        "فیش پرداخت مالیات رو میخوام",
        "مشکلات اظهارنامه مالیاتی",
        "درآمد مشمول مالیات",
        "معافیت مالیاتی برای حقوق‌بگیران",
        "سقف معافیت مالیاتی چقدره",
        "مالیات آیا تسویه شده",
        "شماره پرونده مالیاتی من رو پیدا کنید",
        "چطور مالیات خودم رو کاهش بدم",
        "تمدید مهلت اظهارنامه مالیاتی",
        " جریمه تاخیر مالیاتی چقدر میشه",
        "مالیات املاک و مستغلات من رو ببینم",
        "هزینه مالیاتی قابل قبول",
        "پرداخت مالیات آنلاین",
        "مشاوره مالیاتی رایگان",
    ],
    "military": [
        "وضعیت سربازی من چیه",
        "آیا معافیت تحصیلی دارم؟",
        "کارت پایان خدمتم رو گم کردم",
        "نظام وظیفه — وضعیت پیگیری",
        "معافیت پزشکی سربازی چطوره",
        "تاریخ اعزام به خدمت من کیه",
        "پرونده سربازی من رو ببینید",
        "خدمت سربازی رو تا کی باید انجام بدم",
        "آیا من مشمول هستم یا معاف؟",
        "مدت خدمت سربازی چقدره",
        "بیمه سربازی رو چک کنید",
        "نمره پزشکی سربازی من چیه",
        "کارت سبز سربازی رو چطور بگیرم",
        "ویژگی‌های معافیت کفالت",
        "آیا معافیت همه‌کری دارم",
        "خدمت وظیفه عمومی رو تمدید کنم",
        "سابقه خدمت من رو استعلام کنید",
        "فرم معافیت سربازی",
        "آیا غیبت سربازی دارم؟",
        "تمدید غیبت سربازی چطوره",
        "معافیت ایثارگری و سربازی",
        "دانشجویان و معافیت تحصیلی",
        "ثبت‌نام سربازی آنلاین",
        "پوشش بیمه دوران سربازی",
        "سربازی در خارج از کشور",
        "وضعیت کارت پایان خدمت",
        "تاخیر در اعزام سربازی",
        "مجوز خروج از کشور برای مشمولان",
        "پرونده سربازی من دردسر داره",
        "شماره پرونده سربازی رو پیدا کنید",
    ],
    "traffic": [
        "خلافی خودرو من رو ببینم",
        "جریمه پلاک {plate} چقدره",
        "نمره منفی گواهینامه من چیه",
        "آیا تخلف رانندگی ثبت شده؟",
        "استعلام خلافی پلاک خودرو",
        "پرداخت جریمه رانندگی آنلاین",
        "تخلفات سرعت و جریمه",
        "خودرو من چقدر جریمه داره",
        "چطور جریمه خودم رو تخفیف بگیرم",
        "آیا جریمه معوقه دارم؟",
        "استعلام مدارک خودرو",
        "تخلفات ثبت‌شده خودرو من",
        "پلاک ماشین من رو چک کنید",
        "تعداد تخلفات رانندگی من",
        "پیگیری پرونده تخلف رانندگی",
        "اعتراض به جریمه رانندگی",
        "کد رهگیری جریمه خودرو",
        "وضعیت گواهینامه رانندگی من",
        "تخلف عبور از چراغ قرمز",
        "جریمه عدم استفاده از کلاه ایمنی",
        "آیا ماشین من توقیف شده؟",
        "استعلام بیمه شخص ثالث خودرو",
        "فیش جریمه رو دانلود کنم",
        "نحوه پرداخت اقساط جریمه",
        "تخلفات حامل بار و جریمه",
        "آیا پلاکم سیاه شده؟",
        "جریمه عدم تنظیم یا عدم ظاهر شدن",
        "پرداخت خلافی با گوشی",
        "تخلف سرعت در اتوبان",
        "استعلام وضعیت پلاک ملی",
    ],
    "insurance": [
        "سابقه بیمه تامین اجتماعی من رو ببینید",
        "بیمه بازنشستگی من چقدره",
        "آیا بیمه من فعال هست؟",
        "مدت سابقه بیمه تامین اجتماعی",
        "بازنشستگی زنان و بیمه",
        "مبلغ مستمری بازنشستگی چقدره",
        "فرم درخواست بیمه تامین اجتماعی",
        "شماره بیمه تامین اجتماعی من رو بگیرم",
        "آیا مقرری بیمه بیکاری میگیرم؟",
        "استعلام لیست بیمه شده‌ها",
        "بیمه تامین اجتماعی حقوق بازنشستگان",
        "پرداخت حق بیمه به تامین اجتماعی",
        "بیمه اجباري كارفرما",
        "تمدید بیمه تامین اجتماعی",
        "مشکلات بیمه تامین اجتماعی",
        "چطور سابقه بیمه خودم رو ببینم",
        "کارت بیمه تامین اجتماعی رو چطور بگیرم",
        "وضعیت درخواست بازنشستگی من",
        "فرهنگ بیمه تامین اجتماعی",
        "بیمه سلامت و تامین اجتماعی",
        "آیا کارفرمای من بیمه داده؟",
        "استعلام مقرری بیمه بیکاری",
        "معافیت از حق بیمه",
        "بیمه خوداشتغالی",
        "پرداخت آنلاین حق بیمه",
        "نرخ حق بیمه کارفرما",
        "بیمه رو چطور افزایش بدم",
        "تفاوت بیمه تامین اجتماعی و بیمه سلامت",
        "مشاوره بازنشستگی رایگان",
        "پرسش و پاسخ بیمه تامین اجتماعی",
    ],
    "subsidy": [
        "یارانه نقدی من رو ببینید",
        "آیا یارانه من قطع شده؟",
        "وضعیت یارانه من چیه",
        "ثبت‌نام یارانه جدید",
        "کمک معیشت چی هست و چطور ثبت‌نام کنم",
        "مبلغ یارانه من چقدره",
        "یارانه نان رو چک کنید",
        "آیا واجد شرایط کمک معیشت هستم؟",
        "استعلام یارانه خانوار",
        "ثبت‌نام مجدد یارانه",
        "سوبسید بنزین چطور تقسیم میشه",
        "کارت یارانه رو چطور بگیرم",
        "اعتراض به قطع یارانه",
        "تعداد افراد تحت پوشش یارانه من",
        "یارانه مسکن چی شده؟",
        "وضعیت درخواست کمک معیشت",
        "پرداخت یارانه نقدی عقب افتاده",
        "یارانه حامل انرژی",
        "آیا یارانه اضافه میگیرم؟",
        "شرایط دریافت یارانه مسکن",
        "ثبت‌نام یارانه با شماره ملی",
        "یارانه学子 تا کی ادامه داره",
        "استعلام سهمیه بنزین",
        "چرا یارانه من کم شده؟",
        "کمک معیشت ویژه کرونا",
        "یارانه آب و برق",
        "پرداخت یارانه به حساب بانکی",
        "ثبت‌نام کمک معیشت غیرحضوری",
        "شرایط قطع یارانه چی هست",
        "یارانه نقدی سال {year}",
    ],
    "judicial": [
        "وضعیت پرونده قضایی من رو ببینید",
        "آیا سوء پیشینه دارم؟",
        "شماره پرونده کیفری من رو پیدا کنید",
        "دادگاه من کیه و کی هست",
        "استعلام سوء پیشینه کیفری",
        "پیگیری پرونده حقوقی",
        "وضعیت واخواهی پرونده من",
        "آیا حکم دادگاه صادر شده؟",
        "دادخواست شکایت چطور ثبت کنم",
        "گواهی عدم سوء پیشینه رو بگیرم",
        "آیا پرونده باز دارم؟",
        "دادگاه بدوی و تجدیدنظر",
        "وکیل برای پرونده کیفری",
        "هزینه دادرسی چقدره",
        "نوبت دادگاه رو چک کنم",
        "استعلام آرای قضایی",
        "مدارک لازم برای شکایت کیفری",
        "وضعیت اجرای حکم",
        "پرونده کلاهبرداری من",
        "چطور از دادگاه اطلاع بگیرم",
        "آیا به زندان میفتم؟",
        "پرونده چک برگشتی من",
        "پیامک دادگاه رو دریافت کنم",
        "شکایت اینترنتی از متهم",
        "نوع جرم و مجازات",
        "آیا حکم قطعی شده؟",
        "استعلام وضعیت زندانی",
        "مشاوره حقوقی رایگان",
        "آیا نام من در لیست متهمان هست؟",
        "نامه اعتراض به رای دادگاه",
    ],
    "profile": [
        "اطلاعات شخصی من رو ببینید",
        "پروفایل کاربری من رو نمایش بده",
        "مشخصات من رو آپدیت کن",
        "شماره تلفن و آدرسم رو ویرایش کنم",
        "تغییر رمز عبور حساب کاربری",
        "اطلاعات من ناقص هست",
        "چطور مشخصات خودم رو تکمیل کنم؟",
        "آیا پروفایل من تایید شده؟",
        "عکس پروفایل رو عوض کنم",
        "اطلاعات هویتی من رو ببینم",
        "تاریخ تولد من رو اصلاح کنم",
        "کد پستی رو در پروفایل آپدیت کنم",
        "مشخصات شغلی من رو ویرایش کنم",
        "استعلام شماره ملی از پروفایل",
        "آیا اطلاعات من درست ثبت شده؟",
        "تغییر شماره موبایل در پروفایل",
        "اطلاعات بانکی من رو ببینم",
        "حذف حساب کاربری",
        "سطح دسترسی حساب من چی هست",
        "آیا حساب من فعال هست؟",
        "مشخصات آدرس محل سکونت",
        "تغییر ایمیل پروفایل",
        "اطلاعات تحصیلی من رو آپدیت کنم",
        "مشخصات خانوادگی در پروفایل",
        "استعلام وضعیت احراز هویت",
        "چطور حساب کاربری جدید بسازم",
        "اطلاعات من رو برای نظام وظیفه بفرستید",
        "مشخصات من رو برای بیمه بروزرسانی کن",
        "مشخصات فردی و شماره شناسنامه",
    ],
}

# Noise words to sprinkle into messages for realism
NOISE = [
    "لطفا", "ببینید", "ممنون", "سلام", "خسته نباشید", "با سلام",
    "کمک کنید", "چطور", "میشه", "لطفا بگید", "من میخوام",
    "برای من", "همین الان", "فوری", "فک کنم", "نمی‌دونم",
    "گفتم که", "بعدش", "الان", "دیروز", "فردا", "یه",
    "خیلی", "کمی", "یه ذره", "دقیقا", "احتمالا", "شاید",
    "همین", "اون", "اینا", "اونوقت", "ولی", "اما",
]

PLATE_NUMBERS = [
    "11ای123", "22ب456", "33ج789", "44د001",
    "12الف234", "78د567", "45ب890",
]

YEARS = ["1401", "1402", "1403", "1404", "1405"]


def _fill_placeholders(text: str) -> str:
    """Replace {year} and {plate} with random real-looking values."""
    text = text.replace("{year}", random.choice(YEARS))
    text = text.replace("{plate}", random.choice(PLATE_NUMBERS))
    return text


def _add_noise(text: str, prob=0.5) -> str:
    """Optionally prepend/append a noise token to make data more realistic."""
    if random.random() < prob:
        noise_tok = random.choice(NOISE)
        if random.random() < 0.5:
            text = noise_tok + " " + text
        else:
            text = text + " " + noise_tok
    return text.strip()


def generate_dataset(samples_per_intent: int = 250) -> list[dict]:
    """Build a synthetic dataset from templates + noise + placeholders."""
    dataset = []
    for intent, templates in RAW_TEMPLATES.items():
        for _ in range(samples_per_intent):
            base = random.choice(templates)
            base = _fill_placeholders(base)
            message = _add_noise(base, prob=0.55)
            # Randomly shuffle word order with low probability (very informal)
            if random.random() < 0.08:
                words = message.split()
                if len(words) > 3:
                    i, j = sorted(random.sample(range(len(words)), 2))
                    words[i], words[j] = words[j], words[i]
                    message = " ".join(words)
            dataset.append({"message": message, "intent": intent})
    random.shuffle(dataset)
    return dataset


# ──────────────────────────────────────────────
# 2.  TEXT PREPROCESSING (light — TF-IDF handles most)
# ──────────────────────────────────────────────
_persian_punct_re = re.compile(r"[؟!?،؛«»\(\)\.\:\-\_]")


def preprocess(text: str) -> str:
    """Minimal preprocessing: lower-case + remove Persian punctuation."""
    text = text.lower()
    text = _persian_punct_re.sub(" ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────
# 3.  MODEL TRAINING
# ──────────────────────────────────────────────
def train_model(dataset, test_size=0.2, cv_folds=5):
    """Train a TF-IDF → LinearSVC pipeline and return results."""
    messages = [preprocess(d["message"]) for d in dataset]
    labels = [d["intent"] for d in dataset]

    X_train, X_test, y_train, y_test = train_test_split(
        messages, labels, test_size=test_size, random_state=SEED, stratify=labels,
    )

    # TF-IDF with character n-grams (captures Persian roots & suffixes)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",        # character n-grams within word boundaries
        ngram_range=(3, 6),        # 3- to 6-char subsequences
        max_features=30000,
        sublinear_tf=True,         # log-normalize term frequencies
        min_df=2,
        norm="l2",
    )

    clf = LinearSVC(
        C=1.0,
        max_iter=10000,
        class_weight="balanced",
        random_state=SEED,
    )

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])

    # Cross-validation on training set
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring="accuracy")
    print(f"Cross-validation accuracy  ({cv_folds}-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train on full training set
    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTest-set accuracy: {test_acc:.4f}")
    print("\n" + classification_report(y_test, y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(labels)))
    print("Confusion Matrix (rows=true, cols=predicted):")
    intent_names = sorted(set(labels))
    print("     " + "  ".join(f"{n:>9s}" for n in intent_names))
    for row_label, row in zip(intent_names, cm):
        print(f"{row_label:>9s} " + "  ".join(f"{v:>9d}" for v in row))

    return pipeline, test_acc


# ──────────────────────────────────────────────
# 4.  PREDICTION HELPERS
# ──────────────────────────────────────────────
def predict_intent(pipeline, message: str) -> str | None:
    """Predict intent for a single Persian message. Returns None if confidence too low."""
    processed = preprocess(message)
    intent = pipeline.predict([processed])[0]

    # Optional: use decision_function for confidence gating
    decision_scores = pipeline.decision_function([processed])[0]
    classes = pipeline.classes_
    max_score = decision_scores.max()
    # Threshold — can be tuned; LinearSVC margins are typically wide
    if max_score < 0.1:
        return None
    return intent


def predict_intent_with_confidence(pipeline, message: str) -> dict:
    """Return intent + per-class confidence scores (via softmax on decision values)."""
    processed = preprocess(message)
    decision = pipeline.decision_function([processed])[0]
    classes = pipeline.classes_

    # Convert raw SVM margins to probabilities via softmax
    exp_scores = np.exp(decision - decision.max())  # numerical stability
    probabilities = exp_scores / exp_scores.sum()

    ranked = sorted(zip(classes, probabilities), key=lambda x: -x[1])
    top_intent, top_prob = ranked[0]

    return {
        "message": message,
        "intent": top_intent if top_prob > 0.25 else None,
        "confidence": float(top_prob),
        "all_scores": {intent: round(float(prob), 4) for intent, prob in ranked},
    }


# ──────────────────────────────────────────────
# 5.  MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    output_dir = Path("./download")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate dataset ──
    print("=" * 60)
    print("Generating synthetic Persian intent dataset …")
    dataset = generate_dataset(samples_per_intent=300)
    print(f"Dataset size: {len(dataset)} samples across {len(RAW_TEMPLATES)} intents")
    intent_counts = {}
    for d in dataset:
        intent_counts[d["intent"]] = intent_counts.get(d["intent"], 0) + 1
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent:>12s}: {count:>4d}")

    # Save dataset as JSON
    ds_path = output_dir / "intent_dataset.json"
    with open(ds_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\nDataset saved → {ds_path}")

    # ── Train model ──
    print("\n" + "=" * 60)
    print("Training TF-IDF + LinearSVC pipeline …")
    pipeline, test_acc = train_model(dataset)

    # ── Save model ──
    model_path = output_dir / "intent_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved → {model_path}")

    # ── Demo predictions ──
    print("\n" + "=" * 60)
    print("Demo predictions on unseen messages:")
    test_messages = [
        "سلام، میخوام ببینم مالیات ماشینم چقدر شده",
        "کارت پایان خدمت سربازی من رو گم کردم",
        "لطفا خلافی پلاک من رو چک کنید ممنون",
        "سابقه بیمه تامین اجتماعی خودم رو ببینم",
        "یارانه نقدی من قطع شده، چرا؟",
        "آیا سوء پیشینه کیفری دارم؟",
        "اطلاعات شخصی من رو میخوام ویرایش کنم",
        "نمی‌دونم سربازیام معاف شده یا نه، کمکم کنید",
        "جریمه عبور از چراغ قرمز رو پرداخت کنم",
        "مستمریم بازنشستگی من چقدره و کی واریز میشه",
        "کمک معیشت رو ثبت‌نام کردم ولی هنوز واریز نشده",
        "پرونده دادگاه کیفری من رو پیگیری کنید",
        "پروفایل کاربری من تایید نشده، لطفا بررسی کنید",
        # Ambiguous / OOD
        "هوا امروز خوبه",
        "فوتبال عصرانه بگیم؟",
    ]

    for msg in test_messages:
        result = predict_intent_with_confidence(pipeline, msg)
        intent_str = result["intent"] or "UNKNOWN"
        conf_str = f'{result["confidence"]:.2%}'
        top3 = list(result["all_scores"].items())[:3]
        top3_str = ", ".join(f"{k}={v:.0%}" for k, v in top3)
        print(f"\n  Message : {msg}")
        print(f"  Intent  : {intent_str}  (conf {conf_str})")
        print(f"  Top-3   : {top3_str}")

    # ── Comparison with keyword baseline ──
    print("\n" + "=" * 60)
    print("Comparison: ML model vs. keyword baseline")

    # Re-define the keyword function here for comparison
    def detect_service_intent_keyword(message):
        message_lower = message.lower()
        tax_keywords = ["مالیات", "مالیاتی", "درآمد", "tax"]
        military_keywords = ["سربازی", "نظام وظیفه", "خدمت", "معافیت", "وظیفه"]
        traffic_keywords = ["خلافی", "جریمه", "رانندگی", "خودرو", "ماشین", "پلاک", "تخلف"]
        insurance_keywords = ["بیمه", "تامین اجتماعی", "تأمین اجتماعی", "سابقه بیمه", "بازنشستگی"]
        subsidy_keywords = ["یارانه", "سوبسید", "کمک معیشت"]
        judicial_keywords = ["قضایی", "سوء پیشینه", "دادگاه", "پرونده", "کیفری"]
        profile_keywords = ["پروفایل", "اطلاعات من", "مشخصات", "اطلاعات شخصی"]

        if any(k in message_lower for k in tax_keywords):
            return "tax"
        if any(k in message_lower for k in military_keywords):
            return "military"
        if any(k in message_lower for k in traffic_keywords):
            return "traffic"
        if any(k in message_lower for k in insurance_keywords):
            return "insurance"
        if any(k in message_lower for k in subsidy_keywords):
            return "subsidy"
        if any(k in message_lower for k in judicial_keywords):
            return "judicial"
        if any(k in message_lower for k in profile_keywords):
            return "profile"
        return None

    # Run both on the full dataset
    ml_correct, kw_correct, total = 0, 0, len(dataset)
    mismatches = []
    for d in dataset:
        true_label = d["intent"]
        ml_pred = predict_intent(pipeline, d["message"]) or "NONE"
        kw_pred = detect_service_intent_keyword(d["message"]) or "NONE"
        if ml_pred == true_label:
            ml_correct += 1
        if kw_pred == true_label:
            kw_correct += 1
        if ml_pred != kw_pred:
            mismatches.append((d["message"], true_label, kw_pred, ml_pred))

    print(f"\n  Keyword baseline accuracy: {kw_correct/total:.2%}  ({kw_correct}/{total})")
    print(f"  ML model accuracy:        {ml_correct/total:.2%}  ({ml_correct}/{total})")
    print(f"\n  Mismatch examples (keyword → ML):")
    random.shuffle(mismatches)
    for msg, true_l, kw_l, ml_l in mismatches[:15]:
        if kw_l == true_l and ml_l != true_l:
            marker = " ← ML missed"
        elif ml_l == true_l and kw_l != true_l:
            marker = " ← ML fixed"
        else:
            marker = " ← both wrong"
        print(f"    TRUE={true_l:>10s}  KW={kw_l:>10s}  ML={ml_l:>10s}  | \"{msg[:50]}\"{marker}")

    print("\n" + "=" * 60)
    print("Done! Files saved to:", output_dir)
    print("  - intent_dataset.json   (training data)")
    print("  - intent_classifier.pkl (trained sklearn pipeline)")
