def detect_service_intent(message): # todo Make this Embedding model
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