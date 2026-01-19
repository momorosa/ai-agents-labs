from google.genai import types
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

def get_fee_for_payment_method(method: str) -> dict:
    """Looks up the transaction fee percentage for a given payment method.

    Use this tool when a user mentions a payment method and you need to calculate
    the fee charged on the original amount (before currency conversion).

    Args:
        method: The name of the payment method, e.g. "platinum credit card".

    Returns:
        Success: {"status": "success", "fee_percentage": 0.02}
        Error: {"status": "error", "error_message": "..."}
    """
    fee_database = {
        "platinum credit card": 0.02,
        "gold debit card": 0.035,
        "bank transfer": 0.01,
    }
    
    fee = fee_database.get(method.lower())
    if fee is not None:
        return {"status": "success", "fee_percentage": fee}
    return {
        "status": "error",
        "error_message": f"Payment method '{method}' not found.",
    }

def get_exchange_rate(base_currency: str, target_currency: str) -> dict:
    """Looks up and returns the exchange rate between two currencies.

    Use this tool when you need an exchange rate to convert between currencies.

    Args:
        base_currency: ISO code you convert from (e.g., "USD")
        target_currency: ISO code you convert to (e.g., "EUR")

    Returns:
        Success: {"status": "success", "rate": 0.93}
        Error: {"status": "error", "error_message": "..."}
    """
    rate_database = {
        "usd": {"eur": 0.93, "jpy": 134.5, "inr": 82.58},
    }
    
    base = base_currency.lower()
    target = target_currency.lower()
    
    rate = rate_database.get(base, {}).get(target)
    if rate is not None:
        return {"status": "success", "rate": rate}
    return {
        "status": "error",
        "error_message": f"Unsupported currency pair: {base_currency}/{target_currency}",
    }

# Expose an agent for the webUI to discover
# agent = Agent(
#     name="helpful_assistant",
#     model=Gemini(
#         model="gemini-2.5-flash-lite",
#         retry_options=retry_config,
#     ),
#     description="A simple agent for answering questions with search.",
#     instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
#     tools=[google_search],
# )

agent = Agent(
    name="currency_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
    ),
    instruction="""You are a smart currency conversion assistant.

    For currency conversion requests:
    1. Use `get_fee_for_payment_method()` to find transaction fees
    2. Use `get_exchange_rate()` to get currency conversion rates
    3. Check the "status" field in each tool's response for errors
    4. Calculate the final amount after fees based on the output from `get_fee_for_payment_method` and `get_exchange_rate` methods and provide a clear breakdown.
    5. First, state the final converted amount.
        Then, explain how you got that result by showing the intermediate amounts. Your explanation must include: the fee percentage and its
        value in the original currency, the amount remaining after the fee, and the exchange rate used for the final conversion.

    If any tool returns status "error", explain the issue to the user clearly.
    """,
    tools=[get_fee_for_payment_method, get_exchange_rate],
)

root_agent = agent