# 🔌 راهنمای تنظیم API سفارشی

## 🎯 پیکربندی برای سرویس‌دهندگان مختلف

---

## ✅ گام 1: ویرایش `self_development_spec.yaml`

### برای API سفارشی شما:

```yaml
llm:
  mode: "online"
  
  custom_api:
    enabled: true
    base_url: "https://your-api-server.com/v1"  # 🔧 آدرس سرور شما
    api_key_env: "CUSTOM_API_KEY"
    model: "claude-sonnet-4-20250514"  # Sonnet 4.5
    
    timeout: 300
    retry: 3
    
    # Headers اضافی (اگر نیاز باشد)
    custom_headers:
      "X-API-Version": "2024-10"
      # "X-Custom-Header": "value"
  
  online:
    use_cache: true  # برای کاهش هزینه
    max_tokens: 3000
    temperature: 0.7
  
  # Fallback غیرفعال
  fallback_online: false
  fallback_to_mcp: false

# کنترل هزینه
cost_control:
  enabled: true
  max_cost_per_task: 0.10  # دلار
  max_total_cost: 1.00     # دلار - برای 18 task
  warn_threshold: 0.50
  
  max_input_tokens: 2000
  max_output_tokens: 3000
```

---

## 🌐 پیکربندی برای سرویس‌دهندگان محبوب

### 1️⃣ OpenRouter (پیشنهادی)

```yaml
llm:
  custom_api:
    enabled: true
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: "OPENROUTER_API_KEY"
    model: "anthropic/claude-sonnet-4-20250514"
    
    custom_headers:
      "HTTP-Referer": "https://your-site.com"
      "X-Title": "Auto-Dev-LLM"
```

**محاسبه هزینه:**
```
Sonnet 4.5 در OpenRouter:
Input:  $3.00 / 1M tokens
Output: $15.00 / 1M tokens

کل پروژه: $0.76
```

### 2️⃣ Together.ai

```yaml
llm:
  custom_api:
    enabled: true
    base_url: "https://api.together.xyz/v1"
    api_key_env: "TOGETHER_API_KEY"
    model: "anthropic/claude-sonnet-4-20250514"
```

### 3️⃣ Groq (سریع و ارزان)

```yaml
llm:
  custom_api:
    enabled: true
    base_url: "https://api.groq.com/openai/v1"
    api_key_env: "GROQ_API_KEY"
    model: "llama-3.1-70b-versatile"  # مدل رایگان
```

**محاسبه هزینه:** رایگان! 🎉

### 4️⃣ Anthropic مستقیم

```yaml
llm:
  custom_api:
    enabled: true
    base_url: "https://api.anthropic.com/v1"
    api_key_env: "ANTHROPIC_API_KEY"
    model: "claude-sonnet-4-20250514"
    
    # برای Anthropic باید endpoint را تغییر دهید
    endpoints:
      chat: "/messages"
```

### 5️⃣ Azure OpenAI

```yaml
llm:
  custom_api:
    enabled: true
    base_url: "https://YOUR-RESOURCE.openai.azure.com"
    api_key_env: "AZURE_OPENAI_KEY"
    model: "gpt-4"
    
    custom_headers:
      "api-key": "${AZURE_OPENAI_KEY}"
```

### 6️⃣ سرور محلی (LM Studio, Ollama)

```yaml
llm:
  custom_api:
    enabled: true
    base_url: "http://localhost:1234/v1"  # LM Studio
    # یا: "http://localhost:11434/v1"  # Ollama
    api_key_env: "DUMMY_KEY"  # هر چیزی
    model: "llama-3.1-8b"
```

**محاسبه هزینه:** رایگان! 🎉

---

## ⚙️ گام 2: تنظیم متغیرهای محیطی

### ایجاد فایل `.env`:

```bash
# برای API سفارشی
CUSTOM_API_KEY=your-api-key-here

# یا برای OpenRouter
OPENROUTER_API_KEY=sk-or-v1-xxxxx

# یا برای Groq
GROQ_API_KEY=gsk_xxxxx

# یا برای Together
TOGETHER_API_KEY=xxxxx
```

### بارگذاری در کد:

```bash
# Linux/Mac
export CUSTOM_API_KEY="your-key"

# Windows
set CUSTOM_API_KEY=your-key

# یا با dotenv
pip install python-dotenv
```

---

## 🧪 گام 3: تست اتصال

### اسکریپت تست سریع:

```python
# test_custom_api.py
import asyncio
import os
from src.llm.llama_wrapper import LLMWrapper, LLMRequest

async def test_connection():
    config = {
        'mode': 'custom',
        'custom_api': {
            'enabled': True,
            'base_url': 'https://your-api-server.com/v1',
            'api_key_env': 'CUSTOM_API_KEY',
            'model': 'claude-sonnet-4-20250514'
        },
        'online': {
            'use_cache': True
        },
        'cost_control': {
            'max_total_cost': 0.10  # فقط 10 سنت برای تست
        }
    }
    
    wrapper = LLMWrapper(config)
    
    # تست ساده
    request = LLMRequest(
        prompt="بگو سلام",
        max_tokens=50
    )
    
    response = await wrapper.generate(request)
    
    if response.success:
        print("✅ اتصال موفق!")
        print(f"📝 پاسخ: {response.content}")
        print(f"💰 هزینه: ${response.cost:.4f}")
        print(f"🎯 Tokens: {response.tokens_used}")
    else:
        print(f"❌ خطا: {response.error}")

if __name__ == "__main__":
    asyncio.run(test_connection())
```

**اجرا:**
```bash
python test_custom_api.py
```

---

## 💰 محاسبه هزینه دقیق برای سرور شما

### فرمول محاسبه:

```python
def calculate_project_cost(
    input_price_per_1m: float,   # قیمت input شما
    output_price_per_1m: float,  # قیمت output شما
    num_tasks: int = 18
):
    # هر task
    input_tokens = 1500
    output_tokens = 2500
    
    # محاسبه
    input_cost = (input_tokens * num_tasks * input_price_per_1m) / 1_000_000
    output_cost = (output_tokens * num_tasks * output_price_per_1m) / 1_000_000
    
    total = input_cost + output_cost
    
    return {
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total,
        'per_task': total / num_tasks
    }

# مثال با Sonnet 4.5
result = calculate_project_cost(
    input_price_per_1m=3.00,
    output_price_per_1m=15.00
)

print(f"کل هزینه: ${result['total_cost']:.2f}")
print(f"هزینه هر task: ${result['per_task']:.3f}")
```

---

## 📊 جدول مقایسه سرویس‌دهندگان

| Provider | Model | Input | Output | Total (18 tasks) |
|----------|-------|-------|--------|------------------|
| **OpenRouter** | Sonnet 4.5 | $3.00 | $15.00 | **$0.76** ⭐ |
| **Anthropic** | Sonnet 4.5 | $3.00 | $15.00 | **$0.76** |
| **OpenAI** | GPT-4o | $2.50 | $10.00 | **$0.59** |
| **Groq** | Llama 3.1 70B | FREE | FREE | **$0.00** 🎉 |
| **Together** | Llama 3.1 70B | $0.60 | $0.60 | **$0.13** |
| **Local** | Any | FREE | FREE | **$0.00** 🎉 |

---

## 🔧 عیب‌یابی

### مشکل 1: "Connection refused"

```bash
# بررسی آدرس
curl https://your-api-server.com/v1/models

# بررسی با header
curl https://your-api-server.com/v1/chat/completions \
  -H "Authorization: Bearer $CUSTOM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"hi"}],"max_tokens":10}'
```

### مشکل 2: "Invalid API key"

```bash
# بررسی متغیر محیطی
echo $CUSTOM_API_KEY

# تست با Python
python -c "import os; print(os.getenv('CUSTOM_API_KEY'))"
```

### مشکل 3: "Model not found"

```bash
# لیست مدل‌های موجود
curl https://your-api-server.com/v1/models \
  -H "Authorization: Bearer $CUSTOM_API_KEY"
```

### مشکل 4: "Rate limit exceeded"

در `self_development_spec.yaml`:
```yaml
scheduler:
  max_concurrent_tasks: 1  # کاهش به 1
  check_interval: 60  # صبر بیشتر

llm:
  custom_api:
    retry: 5  # تلاش بیشتر
```

---

## 🎯 بهینه‌سازی هزینه

### 1. استفاده از Cache

```yaml
llm:
  online:
    use_cache: true  # صرفه‌جویی 30-40%
```

### 2. کاهش Max Tokens

```yaml
cost_control:
  max_output_tokens: 2500  # به جای 3000
```

### 3. اجرای تدریجی

```bash
# اول فقط 1 feature
python main.py --batch --features git-automation

# سپس بقیه
python main.py --batch --features version-control rollback-recovery
```

### 4. استفاده از مدل ارزان‌تر برای تست

```yaml
llm:
  custom_api:
    model: "gpt-3.5-turbo"  # برای تست
    # model: "claude-sonnet-4-20250514"  # برای production
```

---

## 📈 مانیتورینگ هزینه Real-Time

```python
# در هنگام اجرا
from src.llm.llama_wrapper import LLMWrapper

# دریافت خلاصه هزینه
summary = wrapper.get_cost_summary()
print(f"💰 هزینه تا کنون: ${summary['total_cost']}")
print(f"📊 درصد استفاده: {summary['percentage']}%")
print(f"💳 باقیمانده: ${summary['remaining']}")
```

---

## ✅ Checklist نهایی

- [ ] `base_url` سرور خود را در spec وارد کردید
- [ ] `api_key` را در `.env` یا متغیر محیطی تنظیم کردید
- [ ] `model` صحیح را انتخاب کردید
- [ ] تست اتصال موفق بود (`test_custom_api.py`)
- [ ] محدودیت هزینه (`max_total_cost`) تنظیم شد
- [ ] آماده اجرا هستید! 🚀

---

## 🎉 اجرای نهایی

```bash
# با تنظیمات سفارشی شما
python bootstrap_self_dev.py
```

**سیستم از سرور API شما استفاده خواهد کرد!** ✨