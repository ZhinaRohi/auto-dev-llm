"""
LLM Wrapper - رابط یکپارچه برای تمام LLMها (MCP, Offline, Online)
"""

import asyncio
import aiohttp
import time
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
import json


class LLMProvider(Enum):
    """ارائه‌دهندگان LLM"""
    MCP = "mcp"
    OFFLINE = "offline"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMRequest:
    """درخواست به LLM"""
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    context: Optional[List[Dict[str, str]]] = None


@dataclass
class LLMResponse:
    """پاسخ از LLM"""
    content: str
    model: str
    provider: LLMProvider
    tokens_used: int
    duration: float
    success: bool
    error: Optional[str] = None


class MCPClient:
    """کلاینت MCP Server"""
    
    def __init__(self, api_url: str, timeout: int = 300, retry: int = 3):
        self.api_url = api_url
        self.timeout = timeout
        self.retry = retry
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """ارسال درخواست به MCP"""
        start_time = time.time()
        
        payload = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "system_prompt": request.system_prompt
        }
        
        for attempt in range(self.retry):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.api_url}/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            duration = time.time() - start_time
                            
                            return LLMResponse(
                                content=data.get('content', ''),
                                model=data.get('model', 'mcp-model'),
                                provider=LLMProvider.MCP,
                                tokens_used=data.get('tokens', 0),
                                duration=duration,
                                success=True
                            )
                        else:
                            error_text = await response.text()
                            raise Exception(f"MCP error: {response.status} - {error_text}")
            
            except Exception as e:
                if attempt == self.retry - 1:
                    duration = time.time() - start_time
                    return LLMResponse(
                        content='',
                        model='mcp-failed',
                        provider=LLMProvider.MCP,
                        tokens_used=0,
                        duration=duration,
                        success=False,
                        error=str(e)
                    )
                await asyncio.sleep(2 ** attempt)  # Exponential backoff


class OfflineLLM:
    """LLM آفلاین (LLaMA/StarCoder)"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """بارگذاری مدل"""
        try:
            # این قسمت باید با کتابخانه واقعی پیاده‌سازی شود
            # مثلاً llama-cpp-python یا transformers
            print(f"⏳ در حال بارگذاری مدل از: {self.model_path}")
            
            # TODO: پیاده‌سازی واقعی با llama.cpp یا ctransformers
            # from llama_cpp import Llama
            # self.model = Llama(model_path=self.model_path)
            
            print("✅ مدل بارگذاری شد")
        except Exception as e:
            print(f"❌ خطا در بارگذاری مدل: {e}")
            self.model = None
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """تولید کد با مدل آفلاین"""
        if not self.model:
            return LLMResponse(
                content='',
                model='offline-not-loaded',
                provider=LLMProvider.OFFLINE,
                tokens_used=0,
                duration=0,
                success=False,
                error="مدل بارگذاری نشده است"
            )
        
        start_time = time.time()
        
        try:
            # TODO: پیاده‌سازی واقعی
            # output = self.model(
            #     request.prompt,
            #     max_tokens=request.max_tokens,
            #     temperature=request.temperature
            # )
            
            # شبیه‌سازی برای تست
            await asyncio.sleep(1)
            output = "# کد تولید شده با LLM آفلاین\npass"
            
            duration = time.time() - start_time
            
            return LLMResponse(
                content=output,
                model='llama-3.1-7b',
                provider=LLMProvider.OFFLINE,
                tokens_used=len(output.split()),
                duration=duration,
                success=True
            )
        
        except Exception as e:
            duration = time.time() - start_time
            return LLMResponse(
                content='',
                model='llama-failed',
                provider=LLMProvider.OFFLINE,
                tokens_used=0,
                duration=duration,
                success=False,
                error=str(e)
            )


class OnlineLLM:
    """LLM آنلاین (OpenAI/Anthropic)"""
    
    def __init__(self, provider: str, api_key: str, model: str):
        self.provider = provider
        self.api_key = api_key
        self.model = model
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """تولید کد با API آنلاین"""
        start_time = time.time()
        
        if self.provider == "openai":
            return await self._generate_openai(request, start_time)
        elif self.provider == "anthropic":
            return await self._generate_anthropic(request, start_time)
        else:
            return LLMResponse(
                content='',
                model='unknown',
                provider=LLMProvider.OPENAI,
                tokens_used=0,
                duration=0,
                success=False,
                error=f"ارائه‌دهنده نامعتبر: {self.provider}"
            )
    
    async def _generate_openai(self, request: LLMRequest, start_time: float) -> LLMResponse:
        """تولید با OpenAI API"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                messages = []
                if request.system_prompt:
                    messages.append({"role": "system", "content": request.system_prompt})
                
                if request.context:
                    messages.extend(request.context)
                
                messages.append({"role": "user", "content": request.prompt})
                
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                }
                
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        tokens = data['usage']['total_tokens']
                        duration = time.time() - start_time
                        
                        return LLMResponse(
                            content=content,
                            model=self.model,
                            provider=LLMProvider.OPENAI,
                            tokens_used=tokens,
                            duration=duration,
                            success=True
                        )
                    else:
                        error_text = await response.text()
                        raise Exception(f"OpenAI error: {response.status} - {error_text}")
        
        except Exception as e:
            duration = time.time() - start_time
            return LLMResponse(
                content='',
                model=self.model,
                provider=LLMProvider.OPENAI,
                tokens_used=0,
                duration=duration,
                success=False,
                error=str(e)
            )
    
    async def _generate_anthropic(self, request: LLMRequest, start_time: float) -> LLMResponse:
        """تولید با Anthropic API"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                }
                
                messages = []
                if request.context:
                    messages.extend(request.context)
                
                messages.append({"role": "user", "content": request.prompt})
                
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                }
                
                if request.system_prompt:
                    payload["system"] = request.system_prompt
                
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['content'][0]['text']
                        tokens = data['usage']['input_tokens'] + data['usage']['output_tokens']
                        duration = time.time() - start_time
                        
                        return LLMResponse(
                            content=content,
                            model=self.model,
                            provider=LLMProvider.ANTHROPIC,
                            tokens_used=tokens,
                            duration=duration,
                            success=True
                        )
                    else:
                        error_text = await response.text()
                        raise Exception(f"Anthropic error: {response.status} - {error_text}")
        
        except Exception as e:
            duration = time.time() - start_time
            return LLMResponse(
                content='',
                model=self.model,
                provider=LLMProvider.ANTHROPIC,
                tokens_used=0,
                duration=duration,
                success=False,
                error=str(e)
            )


class LLMWrapper:
    """رابط یکپارچه برای تمام LLMها"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = config.get('mode', 'mcp')
        
        # آماده‌سازی کلاینت‌ها
        self.mcp_client = None
        self.offline_llm = None
        self.online_llm = None
        
        self._setup_clients()
    
    def _setup_clients(self):
        """راه‌اندازی کلاینت‌ها"""
        # MCP
        if self.mode == 'mcp' or self.config.get('fallback_online'):
            mcp_config = self.config.get('mcp', {})
            self.mcp_client = MCPClient(
                api_url=mcp_config.get('api_url', 'http://localhost:5005'),
                timeout=mcp_config.get('timeout', 300),
                retry=mcp_config.get('retry', 3)
            )
        
        # Offline
        if self.mode == 'offline':
            offline_config = self.config.get('offline_model', {})
            self.offline_llm = OfflineLLM(
                model_path=offline_config.get('path', './models/model.gguf')
            )
        
        # Online (Fallback)
        if self.config.get('fallback_online'):
            online_config = self.config.get('online', {})
            import os
            api_key = os.getenv(online_config.get('api_key_env', 'OPENAI_API_KEY'))
            
            self.online_llm = OnlineLLM(
                provider=online_config.get('provider', 'openai'),
                api_key=api_key or '',
                model=online_config.get('model', 'gpt-4')
            )
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """تولید کد با استفاده از LLM"""
        
        # تلاش با MCP
        if self.mode == 'mcp' and self.mcp_client:
            response = await self.mcp_client.generate(request)
            if response.success:
                return response
            
            print(f"⚠️  MCP ناموفق بود: {response.error}")
        
        # تلاش با Offline
        if self.mode == 'offline' and self.offline_llm:
            response = await self.offline_llm.generate(request)
            if response.success:
                return response
            
            print(f"⚠️  Offline LLM ناموفق بود: {response.error}")
        
        # Fallback به Online
        if self.config.get('fallback_online') and self.online_llm:
            print("🔄 Fallback به API آنلاین...")
            response = await self.online_llm.generate(request)
            return response
        
        # همه روش‌ها ناموفق بودند
        return LLMResponse(
            content='',
            model='none',
            provider=LLMProvider.MCP,
            tokens_used=0,
            duration=0,
            success=False,
            error="هیچ LLM موفقی در دسترس نیست"
        )
    
    async def generate_code(
        self,
        task_description: str,
        file_path: str,
        context: Optional[str] = None
    ) -> LLMResponse:
        """تولید کد برای یک task خاص"""
        
        system_prompt = """شما یک برنامه‌نویس ماهر هستید که کد با کیفیت بالا تولید می‌کنید.
قوانین:
1. کد باید کامل، قابل اجرا و بدون خطا باشد
2. از type hints استفاده کنید
3. docstring برای توابع و کلاس‌ها بنویسید
4. error handling مناسب داشته باشید
5. کد باید تمیز و خوانا باشد (PEP 8)
6. فقط کد را برگردانید، بدون توضیحات اضافی"""
        
        prompt = f"""Task: {task_description}
File: {file_path}

{"Context:\n" + context if context else ""}

لطفاً کد کامل این فایل را تولید کنید:"""
        
        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=4096,
            temperature=0.3
        )
        
        return await self.generate(request)
    
    async def generate_tests(
        self,
        code: str,
        file_path: str
    ) -> LLMResponse:
        """تولید تست برای کد"""
        
        system_prompt = """شما یک تست‌نویس متخصص هستید.
قوانین:
1. تست‌های جامع و کامل بنویسید
2. از pytest استفاده کنید
3. موارد مرزی را پوشش دهید
4. تست‌ها باید قابل اجرا باشند
5. docstring برای تست‌ها بنویسید"""
        
        prompt = f"""کد زیر را تست کنید:

```python
{code}
```

File path: {file_path}

لطفاً تست‌های pytest کامل تولید کنید:"""
        
        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.3
        )
        
        return await self.generate(request)
    
    async def review_code(self, code: str) -> LLMResponse:
        """بررسی و بهبود کد"""
        
        system_prompt = """شما یک code reviewer متخصص هستید.
مشکلات احتمالی را شناسایی و بهبود پیشنهاد دهید."""
        
        prompt = f"""کد زیر را بررسی کنید:

```python
{code}
```

مشکلات و پیشنهادات بهبود را لیست کنید:"""
        
        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=1024,
            temperature=0.5
        )
        
        return await self.generate(request)


# تست سریع
if __name__ == "__main__":
    async def test_llm():
        config = {
            'mode': 'mcp',
            'mcp': {
                'api_url': 'http://localhost:5005',
                'timeout': 300,
                'retry': 3
            },
            'fallback_online': True,
            'online': {
                'provider': 'openai',
                'api_key_env': 'OPENAI_API_KEY',
                'model': 'gpt-4'
            }
        }
        
        wrapper = LLMWrapper(config)
        
        # تولید کد ساده
        response = await wrapper.generate_code(
            task_description="ایجاد تابع محاسبه فیبوناچی",
            file_path="fibonacci.py"
        )
        
        if response.success:
            print(f"✅ کد تولید شد ({response.provider.value}):")
            print(response.content[:500])
            print(f"\n⏱️  مدت زمان: {response.duration:.2f}s")
            print(f"🎯 Tokens: {response.tokens_used}")
        else:
            print(f"❌ خطا: {response.error}")
    
    asyncio.run(test_llm())