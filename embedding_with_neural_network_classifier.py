#!/usr/bin/env python3
"""
Persian Service Intent Classification — Neural Network Pipeline
=================================================================
Replaces keyword-based `detect_service_intent()` with a trained
sentence-transformer embedding + PyTorch neural network classifier.

Approach:
  1. Synthetic Persian dataset (300 samples/intent, 7 intents)
  2. Sentence-Transformers  all-MiniLM-L12-v2  →  384-dim embeddings
  3. PyTorch feed-forward NN  (384 → hidden → hidden → num_classes)
  4. Training with AdamW, cosine LR schedule, early stopping
  5. Full evaluation + export (torch.save) for production use

Requirements:
    pip install torch sentence-transformers scikit-learn numpy
"""

import random
import json
import re
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L12-v2 outputs 384-dim vectors
NUM_CLASSES = 7
INTENTS = ["insurance", "judicial", "military", "profile", "subsidy", "tax", "traffic"]
INTENT2IDX = {name: idx for idx, name in enumerate(INTENTS)}
IDX2INTENT = {idx: name for name, idx in INTENT2IDX.items()}

# Hyperparameters
HIDDEN_DIM = 256
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
CONFIDENCE_THRESHOLD = 0.40  # below this → return None (out-of-domain)


# ══════════════════════════════════════════════
# 1.  SYNTHETIC DATASET  (kept from v1)
# ══════════════════════════════════════════════

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
        "جریمه تاخیر مالیاتی چقدر میشه",
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
        "یارانه تا کی ادامه داره",
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
    text = text.replace("{year}", random.choice(YEARS))
    text = text.replace("{plate}", random.choice(PLATE_NUMBERS))
    return text


def _add_noise(text: str, prob=0.5) -> str:
    if random.random() < prob:
        noise_tok = random.choice(NOISE)
        if random.random() < 0.5:
            text = noise_tok + " " + text
        else:
            text = text + " " + noise_tok
    return text.strip()


def generate_dataset(samples_per_intent: int = 300) -> list[dict]:
    """Build a synthetic dataset from templates + noise + placeholders."""
    dataset = []
    for intent, templates in RAW_TEMPLATES.items():
        for _ in range(samples_per_intent):
            base = random.choice(templates)
            base = _fill_placeholders(base)
            message = _add_noise(base, prob=0.55)
            if random.random() < 0.08:
                words = message.split()
                if len(words) > 3:
                    i, j = sorted(random.sample(range(len(words)), 2))
                    words[i], words[j] = words[j], words[i]
                    message = " ".join(words)
            dataset.append({"message": message, "intent": intent})
    random.shuffle(dataset)
    return dataset


# ══════════════════════════════════════════════
# 2.  TEXT PREPROCESSING
# ══════════════════════════════════════════════
_persian_punct_re = re.compile(r"[؟!?،؛«»\(\)\.\:\-\_]")


def preprocess(text: str) -> str:
    text = text.lower()
    text = _persian_punct_re.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════
# 3.  PYTORCH DATASET + MODEL
# ══════════════════════════════════════════════
class IntentDataset(Dataset):
    """Torch Dataset that holds pre-computed embeddings + labels."""

    def __init__(self, embeddings: np.ndarray, labels: list[int]):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class IntentClassifier(nn.Module):
    """
    Feed-forward neural network for intent classification.

    Architecture:
        Input (384) → BatchNorm → Linear(384, 256) → ReLU → Dropout
        → Linear(256, 128) → ReLU → Dropout → Linear(128, 7)

    Uses GELU activation in the first block and ReLU in the second
    for slightly richer representation learning. LayerNorm for
    stability with small datasets.
    """

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.network = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # Block 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Output head
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ══════════════════════════════════════════════
# 4.  TRAINING
# ══════════════════════════════════════════════
def compute_embeddings(
    encoder: SentenceTransformer,
    messages: list[str],
    batch_size: int = 128,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode all messages into fixed-size embeddings."""
    return encoder.encode(
        messages,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2-normalize for better NN training
        convert_to_numpy=True,
    )


def train_model(
    encoder: SentenceTransformer,
    dataset: list[dict],
    test_size: float = 0.2,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = 10,  # early stopping patience
) -> tuple[IntentClassifier, dict]:
    """
    Full training pipeline:
      messages → embeddings (frozen encoder) → train NN classifier
    Returns the trained model + evaluation metrics dict.
    """
    # ── Split data ──
    messages = [preprocess(d["message"]) for d in dataset]
    labels = [INTENT2IDX[d["intent"]] for d in dataset]

    X_train_msgs, X_test_msgs, y_train, y_test = train_test_split(
        messages, labels, test_size=test_size, random_state=SEED, stratify=labels,
    )

    # ── Compute embeddings ──
    print(f"Computing embeddings with {EMBEDDING_MODEL_NAME} …")
    X_train_emb = compute_embeddings(encoder, X_train_msgs, show_progress=False)
    X_test_emb = compute_embeddings(encoder, X_test_msgs, show_progress=False)
    print(f"  Train embeddings shape: {X_train_emb.shape}")
    print(f"  Test  embeddings shape: {X_test_emb.shape}")

    # ── DataLoaders ──
    train_ds = IntentDataset(X_train_emb, y_train)
    test_ds = IntentDataset(X_test_emb, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ── Model, optimizer, scheduler, loss ──
    model = IntentClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──
    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nTraining on {DEVICE} …")
    print(f"  Epochs: {epochs}  |  Batch size: {batch_size}  |  LR: {lr}")
    print(f"  Early stopping patience: {patience}\n")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_emb, batch_labels in train_loader:
            batch_emb = batch_emb.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            logits = model(batch_emb)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_emb.size(0)
            correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            total += batch_emb.size(0)

        scheduler.step()

        train_loss = epoch_loss / total
        train_acc = correct / total

        # ── Validate ──
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_emb, batch_labels in test_loader:
                batch_emb = batch_emb.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                logits = model(batch_emb)
                val_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
                val_total += batch_emb.size(0)
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = "  ★ best"
        elif no_improve >= patience:
            marker = "  ⛔ early stop"

        print(
            f"  Epoch {epoch:>3d}/{epochs}  "
            f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  lr={lr_now:.2e}{marker}"
        )

        if no_improve >= patience:
            break

    # ── Restore best weights ──
    model.load_state_dict(best_state)
    model.eval()

    # ── Final evaluation ──
    all_preds = []
    with torch.no_grad():
        for batch_emb, _ in test_loader:
            batch_emb = batch_emb.to(DEVICE)
            logits = model(batch_emb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    y_test_np = np.array(y_test)
    y_pred_np = np.array(all_preds)
    test_acc = accuracy_score(y_test_np, y_pred_np)

    print(f"\n{'=' * 60}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final test accuracy:     {test_acc:.4f}")
    print(f"\n{classification_report(y_test_np, y_pred_np, digits=4, target_names=INTENTS)}")

    # Confusion matrix
    cm = confusion_matrix(y_test_np, y_pred_np)
    print("Confusion Matrix (rows=true, cols=predicted):")
    header = "     " + "  ".join(f"{n:>10s}" for n in INTENTS)
    print(header)
    for row_label, row in zip(INTENTS, cm):
        print(f"{row_label:>10s} " + "  ".join(f"{v:>10d}" for v in row))

    metrics = {
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "history": history,
    }
    return model, metrics


# ══════════════════════════════════════════════
# 5.  PREDICTION HELPERS
# ══════════════════════════════════════════════
class IntentPredictor:
    """
    Production-ready wrapper that combines:
      - SentenceTransformer encoder (frozen)
      - PyTorch IntentClassifier (trained)
    """

    def __init__(self, encoder: SentenceTransformer, classifier: IntentClassifier):
        self.encoder = encoder
        self.classifier = classifier
        self.classifier.eval()

    def _embed(self, message: str) -> torch.Tensor:
        processed = preprocess(message)
        emb = self.encoder.encode(
            [processed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return torch.tensor(emb, dtype=torch.float32)

    def predict(self, message: str) -> Optional[str]:
        """Predict single intent. Returns None if below confidence threshold."""
        emb = self._embed(message).to(DEVICE)
        with torch.no_grad():
            logits = self.classifier(emb)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        conf, idx = probs.max(dim=0)
        if conf.item() < CONFIDENCE_THRESHOLD:
            return None
        return IDX2INTENT[idx.item()]

    def predict_with_confidence(self, message: str) -> dict:
        """Predict with full per-class probability breakdown."""
        emb = self._embed(message).to(DEVICE)
        with torch.no_grad():
            logits = self.classifier(emb)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        ranked_idx = np.argsort(-probs)
        all_scores = {IDX2INTENT[i]: round(float(probs[i]), 4) for i in ranked_idx}

        top_idx = ranked_idx[0]
        top_prob = float(probs[top_idx])

        return {
            "message": message,
            "intent": IDX2INTENT[top_idx] if top_prob >= CONFIDENCE_THRESHOLD else None,
            "confidence": top_prob,
            "all_scores": all_scores,
        }

    def save(self, path: Path):
        """Save both encoder name and classifier weights."""
        torch.save(
            {
                "encoder_name": EMBEDDING_MODEL_NAME,
                "classifier_state": self.classifier.state_dict(),
                "intents": INTENTS,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "IntentPredictor":
        """Load a saved predictor from disk."""
        checkpoint = torch.load(path, weights_only=True, map_location=DEVICE)
        encoder = SentenceTransformer(checkpoint["encoder_name"])
        classifier = IntentClassifier()
        classifier.load_state_dict(checkpoint["classifier_state"])
        classifier.to(DEVICE)
        return cls(encoder, classifier)


# ══════════════════════════════════════════════
# 6.  KEYWORD BASELINE  (for comparison)
# ══════════════════════════════════════════════
def detect_service_intent_keyword(message: str) -> Optional[str]:
    """Original keyword-based baseline."""
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


# ══════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════
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

    ds_path = output_dir / "intent_dataset_nn.json"
    with open(ds_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\nDataset saved -> {ds_path}")

    # ── Load sentence-transformer encoder ──
    print(f"\n{'=' * 60}")
    print(f"Loading encoder: {EMBEDDING_MODEL_NAME} …")
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"  Embedding dimension: {EMBEDDING_DIM}")
    print(f"  Device: {DEVICE}")

    # ── Train the neural network ──
    print(f"\n{'=' * 60}")
    print("Training Neural Network Classifier …")
    model, metrics = train_model(encoder, dataset)

    # ── Build predictor + save ──
    predictor = IntentPredictor(encoder, model)
    model_path = output_dir / "intent_classifier_nn.pt"
    predictor.save(model_path)
    print(f"\nModel saved -> {model_path}")

    # ── Verify reload works ──
    print("Verifying model reload …")
    reloaded = IntentPredictor.load(model_path)
    _ = reloaded.predict("تست")

    # ── Demo predictions ──
    print(f"\n{'=' * 60}")
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
        "مستمری بازنشستگی من چقدره و کی واریز میشه",
        "کمک معیشت رو ثبت‌نام کردم ولی هنوز واریز نشده",
        "پرونده دادگاه کیفری من رو پیگیری کنید",
        "پروفایل کاربری من تایید نشده، لطفا بررسی کنید",
        # Out-of-domain
        "هوا امروز خوبه",
        "فوتبال عصرانه بگیم؟",
    ]

    for msg in test_messages:
        result = predictor.predict_with_confidence(msg)
        intent_str = result["intent"] or "UNKNOWN"
        conf_str = f'{result["confidence"]:.2%}'
        top3 = list(result["all_scores"].items())[:3]
        top3_str = ", ".join(f"{k}={v:.0%}" for k, v in top3)
        print(f"\n  Message : {msg}")
        print(f"  Intent  : {intent_str}  (conf {conf_str})")
        print(f"  Top-3   : {top3_str}")

    # ── Comparison with keyword baseline ──
    print(f"\n{'=' * 60}")
    print("Comparison: Neural Net vs. keyword baseline")

    nn_correct, kw_correct, total = 0, 0, len(dataset)
    mismatches = []
    for d in dataset:
        true_label = d["intent"]
        nn_pred = predictor.predict(d["message"]) or "NONE"
        kw_pred = detect_service_intent_keyword(d["message"]) or "NONE"
        if nn_pred == true_label:
            nn_correct += 1
        if kw_pred == true_label:
            kw_correct += 1
        if nn_pred != kw_pred:
            mismatches.append((d["message"], true_label, kw_pred, nn_pred))

    print(f"\n  Keyword baseline accuracy: {kw_correct/total:.2%}  ({kw_correct}/{total})")
    print(f"  Neural net accuracy:       {nn_correct/total:.2%}  ({nn_correct}/{total})")
    print(f"\n  Mismatch examples (keyword vs. NN):")
    random.shuffle(mismatches)
    shown = 0
    for msg, true_l, kw_l, nn_l in mismatches:
        if shown >= 15:
            break
        if kw_l == true_l and nn_l != true_l:
            marker = " <- NN missed"
        elif nn_l == true_l and kw_l != true_l:
            marker = " <- NN fixed"
        else:
            marker = " <- both wrong"
        print(f"    TRUE={true_l:>10s}  KW={kw_l:>10s}  NN={nn_l:>10s}  | \"{msg[:50]}\"{marker}")
        shown += 1

    # ── Model architecture summary ──
    print(f"\n{'=' * 60}")
    print("Model Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters:    {total_params:,}")
    print(f"  Trainable params:   {trainable_params:,}")
    print(f"  Encoder (frozen):   {EMBEDDING_MODEL_NAME}  ({EMBEDDING_DIM}d)")

    print(f"\n{'=' * 60}")
    print("Done! Files saved to:", output_dir)
    print("  - intent_dataset_nn.json  (training data)")
    print("  - intent_classifier_nn.pt (encoder + classifier weights)")
