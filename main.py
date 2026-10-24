"""E7: final banking assistant. One Hugging Face model does three jobs:
tool selection, intent classification (retrieval few-shot), and the customer reply."""

import json
import zlib

from jsonschema import ValidationError, validate

from src.baseline import INTENTS, parse_intent
from src.llm import generate
from src.retrieval import build_prompt

INTENT_SCHEMA = {
    "type": "object",
    "properties": {"intent": {"type": "string", "enum": INTENTS}},
    "required": ["intent"],
    "additionalProperties": False,
}


def classify_intent(message: str) -> dict:
    """Classify with the LLM using the 5 most similar labelled examples, then validate."""
    output = generate(build_prompt(message), max_new_tokens=32)
    result = {"intent": parse_intent(output)}
    try:
        validate(result, INTENT_SCHEMA)
    except ValidationError:
        return {"error": f"Could not map the message to a known intent (model said {output!r})."}
    return result


# Simulated backend. Actions complete and get a reference; questions return information;
# everything else (disputes, declined or failed payments) needs a human agent.
ACTIONS = {
    "lost_or_stolen_card": "Card blocked. A replacement card is on its way.",
    "compromised_card": "Card blocked. A replacement card is on its way.",
    "lost_or_stolen_phone": "App access frozen on the lost phone.",
    "cancel_transfer": "Transfer cancelled. The money returns to the account within 1 working day.",
    "change_pin": "PIN change started. Set the new PIN at any ATM.",
    "pin_blocked": "PIN unblocked.",
    "passcode_forgotten": "Passcode reset link sent to the registered email.",
    "activate_my_card": "Card activated.",
    "order_physical_card": "Physical card ordered.",
    "get_physical_card": "Physical card ordered.",
    "getting_spare_card": "Spare card ordered.",
    "card_about_to_expire": "Replacement card ordered. It arrives before the current card expires.",
    "getting_virtual_card": "Virtual card created. It is visible in the app.",
    "get_disposable_virtual_card": "Disposable virtual card created. It is visible in the app.",
    "request_refund": "Refund request opened.",
    "edit_personal_details": "Personal details update opened in the app.",
    "terminate_account": "Account closure request opened.",
}

INFO = {
    "card_delivery_estimate": "Standard delivery takes 3 to 5 working days. Express delivery takes 1 to 2 working days.",
    "card_arrival": "Standard delivery takes 3 to 5 working days. Track the card in the app under Cards.",
    "transfer_timing": "Transfers in the same currency arrive within 1 working day. International transfers take 2 to 5 working days.",
    "exchange_rate": "The app uses the interbank exchange rate on weekdays.",
    "exchange_charge": "Exchange is free up to 1000 per month. Above that a 0.5% fee applies.",
    "exchange_via_app": "Exchange currency in the app under Accounts, Exchange.",
    "top_up_limits": "The top-up limit is 5000 per day and 20000 per month.",
    "topping_up_by_card": "Top up with a debit card in the app under Add money.",
    "top_up_by_card_charge": "Top-ups by debit card are free. Credit card top-ups carry a 1% fee.",
    "top_up_by_bank_transfer_charge": "Top-ups by bank transfer are free.",
    "top_up_by_cash_or_cheque": "Cash and cheque deposits are not supported. Use a bank transfer or card.",
    "automatic_top_up": "Set up auto top-up in the app under Add money, Auto top-up.",
    "transfer_fee_charged": "Transfers in the same currency are free. International transfers cost 0.5%.",
    "card_payment_fee_charged": "Card payments are free. Payments in a foreign currency on weekends carry a 1% fee.",
    "cash_withdrawal_charge": "ATM withdrawals are free up to 200 per month. Above that a 2% fee applies.",
    "atm_support": "The card works at any ATM that shows the Visa or Mastercard logo.",
    "card_acceptance": "The card works anywhere Visa or Mastercard is accepted.",
    "supported_cards_and_currencies": "Visa and Mastercard are supported. Accounts can hold 25 currencies.",
    "visa_or_mastercard": "The card is a Mastercard. Visa is available on request.",
    "fiat_currency_support": "Accounts can hold 25 fiat currencies, including GBP, EUR, and USD.",
    "country_support": "The service is available to residents of the EEA, the UK, the US, and Australia.",
    "age_limit": "The minimum age is 18.",
    "apple_pay_or_google_pay": "The card works with Apple Pay and Google Pay. Add it from the wallet app.",
    "card_linking": "Link the card in the app under Cards, Add card.",
    "disposable_card_limits": "One disposable virtual card can be created per day.",
    "receiving_money": "Share the account number and sort code or IBAN shown in the app under Account details.",
    "transfer_into_account": "Send money to the account number and sort code or IBAN shown in the app.",
    "verify_my_identity": "Verify identity in the app with a photo ID and a selfie.",
    "why_verify_identity": "Identity verification is a legal requirement to prevent fraud and money laundering.",
    "verify_source_of_funds": "Upload a bank statement or payslip in the app under Verification.",
    "pending_card_payment": "Pending card payments settle within 3 working days.",
    "pending_cash_withdrawal": "Pending ATM withdrawals settle within 3 working days.",
    "pending_top_up": "Pending top-ups arrive within 1 working day.",
    "pending_transfer": "Pending transfers arrive within 1 working day.",
}


def handle_intent(intent: str) -> dict:
    if intent not in INTENTS:
        return {"error": f"Unknown intent '{intent}'. Call classify_intent first."}
    if intent in ACTIONS:
        return {"intent": intent, "status": "done", "result": ACTIONS[intent], "reference": f"REF-{zlib.crc32(intent.encode()) % 10000:04d}"}
    if intent in INFO:
        return {"intent": intent, "status": "information", "result": INFO[intent]}
    return {"intent": intent, "status": "unsupported", "result": "This request needs a human agent. A support ticket has not been opened."}


TOOL_FUNCS = {"classify_intent": classify_intent, "handle_intent": handle_intent}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "classify_intent",
            "description": "Classify the customer message into one of the 77 BANKING77 intents.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "handle_intent",
            "description": "Perform the banking action, or return the information, for the intent returned by classify_intent.",
            "parameters": {
                "type": "object",
                "properties": {"intent": {"type": "string"}},
                "required": ["intent"],
            },
        },
    },
]

SYSTEM = """You are a customer support assistant for an online bank.
Reply to the customer in two or three short sentences based only on the tool results.
If the tool results contain a reference number, mention it. If the status is "unsupported", tell the customer to contact a human agent.
Do not claim any action that the tool results do not show."""


def call_tool(messages: list[dict], name: str, **arguments) -> dict:
    """Run one tool and record the call and its result in the chat history (Qwen tool-call format)."""
    output = TOOL_FUNCS[name](**arguments)
    print(f"  [tool] {name}({arguments}) -> {output}")
    messages.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]})
    messages.append({"role": "tool", "content": json.dumps(output)})
    return output


def handle(message: str) -> str:
    # The code fixes the tool order. The 1.5B model did not do this reliably on its own (see docs/PROGRESS.md, step 9).
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}]
    intent = call_tool(messages, "classify_intent", message=message)
    if "error" in intent:
        return "Sorry, I could not understand your request. Could you say it in a different way?"
    call_tool(messages, "handle_intent", intent=intent["intent"])
    # Sampling is used only here, for the free-text reply (E3 conclusion).
    return generate(messages, max_new_tokens=80, tools=TOOLS, do_sample=True, temperature=0.7).strip()


if __name__ == "__main__":
    print("Banking assistant ready. Press Ctrl-D to exit.")
    while True:
        try:
            message = input("\nCustomer: ").strip()
        except EOFError:
            break
        if message:
            print("Assistant:", handle(message))
