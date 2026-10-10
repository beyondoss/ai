---
description: Learn about Groq&#x27;s Flex Processing service tier, optimized for high-throughput workloads with fast inference and higher rate limits.
title: Flex Processing - GroqDocs
image: https://console.groq.com/og_cloudv5.jpg
---

# Flex Processing

Flex Processing is a service tier optimized for high-throughput workloads that prioritizes fast inference and can handle occasional request failures. This tier offers significantly higher rate limits while maintaining the same pricing as on-demand processing.

## [Availability](#availability)

Flex processing is available for all [models](https://console.groq.com/docs/models) to paid customers only with 10x higher rate limits compared to on-demand processing. Pricing matches the on-demand tier.

## [How flex behaves](#how-flex-behaves)

* Requests run at higher rate limits while capacity is available.
* If flex capacity is unavailable, requests will fail quickly with status `498` and error `capacity_exceeded`. Add jittered backoff and retries to smooth spikes.

## [Example Usage](#example-usage)

curlJavaScriptPythonJSON

shell

```
import os
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def main():
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}"
            },
            json={
                "service_tier": "flex",
                "model": "llama-3.3-70b-versatile",
                "messages": [{
                    "role": "user",
                    "content": "whats 2 + 2"
                }]
            }
        )
        print(response.json())
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
```
