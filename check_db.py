# save as test_groq.py, run: python test_groq.py
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

for name in ["GROQ_API_KEY_EU", "GROQ_API_KEY_US", "GROQ_API_KEY_JUDGE"]:
    key = os.getenv(name)
    try:
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "say ok"}],
            model="groq/compound"
        )
        print(f"{name}: ✅ OK")
    except Exception as e:
        print(f"{name}: ❌ {e}")


'''# save as check_models.py, run: python check_models.py
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY_JUDGE"))
models = client.models.list()
for m in models.data:
    print(m.id)'''