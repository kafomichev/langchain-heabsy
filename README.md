# langchain-heabsy

[Heabsy](https://heabsy.com/platform) chat model for [LangChain](https://docs.langchain.com).
Heabsy is an OpenAI-compatible inference API for open models operated by FEYA, s.r.o.
Models in the EEA tier run on dedicated GPUs in EEA data centres with zero data retention;
models routed through third parties are labelled in the [catalog](https://heabsy.com/models).

```bash
pip install langchain-heabsy
export HEABSY_API_KEY=...   # create a key at https://platform.heabsy.com
```

```python
from langchain_heabsy import ChatHeabsy

llm = ChatHeabsy(model="qwen38")
print(llm.invoke("Reply with the single word: ready").content)
```

`ChatHeabsy` is `langchain_openai.ChatOpenAI` with the Heabsy base URL
(`https://api.heabsy.com/v1`), `stream_usage=True` and reasoning text kept in
`AIMessage.additional_kwargs["reasoning_content"]` (non-streaming calls).
Setup and troubleshooting: https://heabsy.com/integrations/langchain

Heabsy is not affiliated with or endorsed by LangChain. The LangChain name is used
only to describe compatibility.
