"""Maintenance Assistant chatbot backed by a model deployed in Azure AI Foundry.

Uses the Azure OpenAI-compatible endpoint exposed by the Foundry deployment.
All secrets come from environment variables (set as App Service settings) so
nothing sensitive is ever committed:

    AZURE_OPENAI_ENDPOINT     e.g. https://<resource>.openai.azure.com/
    AZURE_OPENAI_API_KEY      the deployment key
    AZURE_OPENAI_DEPLOYMENT   deployment name, e.g. gpt-4o-mini
    AZURE_OPENAI_API_VERSION  optional, defaults to 2024-10-21

If the variables are absent the assistant degrades gracefully so the rest of
the app still runs locally before Foundry is provisioned.
"""
import os
from typing import Dict, List, Optional

SYSTEM_PROMPT = (
    "You are the Maintenance Assistant for a Smart Predictive Maintenance "
    "Platform used by industrial reliability engineers. You help with machine "
    "troubleshooting, preventive-maintenance guidance, and interpreting sensor "
    "readings (air/process temperature, rotational speed, torque, tool wear). "
    "When the user shares a failure-prediction result, explain in plain, "
    "practical language what it likely means and recommend concrete next steps. "
    "Be concise, safety-conscious, and admit uncertainty. If a question is "
    "unrelated to maintenance or reliability engineering, politely steer back."
)


def is_configured() -> bool:
    return bool(
        os.getenv("AZURE_OPENAI_ENDPOINT")
        and os.getenv("AZURE_OPENAI_API_KEY")
        and os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )


def _client():
    from openai import AzureOpenAI

    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


def chat(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    prediction_context: Optional[str] = None,
) -> str:
    """Return the assistant's reply to `message`.

    `history` is an optional list of {"role": "user"|"assistant", "content": str}.
    `prediction_context` optionally injects the latest prediction so the
    assistant can explain it.
    """
    if not is_configured():
        return (
            "The Maintenance Assistant is not configured yet. Once the Azure AI "
            "Foundry model is deployed and the AZURE_OPENAI_* environment "
            "variables are set, I'll be able to answer your questions."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if prediction_context:
        messages.append(
            {
                "role": "system",
                "content": f"Latest failure-prediction result for context: {prediction_context}",
            }
        )
    for turn in (history or [])[-10:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    client = _client()
    resp = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=messages,
        temperature=0.3,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()
