"""20 hand-written function-calling cases (provenance: authored for this project, 2026-10-30,
released under Apache-2.0 alongside the harness; style modelled on BFCL categories
simple / multiple / parallel / irrelevance; no text copied from BFCL).

gold_calls: list of {name, arguments: {param: [allowed values...]}}; "__optional__" in the list
means the parameter may be omitted. Empty gold_calls => the model must NOT call a tool.
"""


def fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


S = lambda d, **kw: {"type": "string", "description": d, **kw}  # noqa: E731
I = lambda d: {"type": "integer", "description": d}  # noqa: E731
N = lambda d: {"type": "number", "description": d}  # noqa: E731
B = lambda d: {"type": "boolean", "description": d}  # noqa: E731

WEATHER = fn("get_current_weather", "Get the current weather for a city.",
             {"city": S("City name, e.g. 'Paris'"), "unit": S("Temperature unit", enum=["celsius", "fahrenheit"])},
             ["city"])
FX = fn("convert_currency", "Convert an amount between currencies using today's rate.",
        {"amount": N("Amount to convert"), "from_currency": S("ISO 4217 code"), "to_currency": S("ISO 4217 code")},
        ["amount", "from_currency", "to_currency"])
EMAIL = fn("send_email", "Send an email.",
           {"to": S("Recipient email address"), "subject": S("Subject line"), "body": S("Plain-text body")},
           ["to", "subject", "body"])
FLIGHTS = fn("search_flights", "Search one-way flights.",
             {"origin": S("IATA airport code"), "destination": S("IATA airport code"),
              "date": S("Departure date YYYY-MM-DD"), "passengers": I("Number of passengers")},
             ["origin", "destination", "date"])
TIMER = fn("set_timer", "Start a countdown timer.", {"minutes": I("Duration in minutes"), "label": S("Optional label")},
           ["minutes"])
STOCK = fn("get_stock_price", "Get the latest stock price for a ticker symbol.", {"ticker": S("Ticker, e.g. AAPL")},
           ["ticker"])
TRANSLATE = fn("translate_text", "Translate text to a target language.",
               {"text": S("Text to translate"), "target_language": S("ISO 639-1 code, e.g. 'de'")},
               ["text", "target_language"])
CAL = fn("create_calendar_event", "Create a calendar event.",
         {"title": S("Event title"), "date": S("YYYY-MM-DD"), "start_time": S("HH:MM 24h"),
          "duration_minutes": I("Length in minutes")}, ["title", "date", "start_time"])
SQL = fn("run_sql_query", "Run a read-only SQL query against the analytics warehouse.",
         {"query": S("SQL SELECT statement")}, ["query"])
FILES = fn("list_files", "List files in a directory.", {"path": S("Directory path"), "recursive": B("Recurse")},
           ["path"])
BMI = fn("calculate_bmi", "Calculate body-mass index.", {"weight_kg": N("Weight in kg"), "height_m": N("Height in metres")},
         ["weight_kg", "height_m"])
LIGHT = fn("set_light_state", "Turn a smart light on/off and set brightness.",
           {"room": S("Room name"), "on": B("Power state"), "brightness": I("0-100")}, ["room", "on"])
TICKET = fn("create_support_ticket", "Open a customer-support ticket.",
            {"customer_id": S("Customer id"), "priority": S("Priority", enum=["low", "medium", "high"]),
             "summary": S("Short summary")}, ["customer_id", "priority", "summary"])
TZ = fn("get_time_in_timezone", "Get the current local time for an IANA timezone.", {"timezone": S("e.g. Asia/Tokyo")},
        ["timezone"])

O = "__optional__"
CASES = [
    # ---------------- simple (single tool available, single call)
    dict(id="tool_simple_01", category="simple", tools=[WEATHER], user="What's the weather like in Lisbon right now? Use celsius.",
         gold_calls=[{"name": "get_current_weather", "arguments": {"city": ["Lisbon", "Lisbon, Portugal"], "unit": ["celsius"]}}]),
    dict(id="tool_simple_02", category="simple", tools=[FX], user="How much is 250 US dollars in euros?",
         gold_calls=[{"name": "convert_currency", "arguments": {"amount": [250], "from_currency": ["USD"], "to_currency": ["EUR"]}}]),
    dict(id="tool_simple_03", category="simple", tools=[TIMER], user="Set a 25 minute timer labelled pomodoro.",
         gold_calls=[{"name": "set_timer", "arguments": {"minutes": [25], "label": ["pomodoro", "Pomodoro"]}}]),
    dict(id="tool_simple_04", category="simple", tools=[STOCK], user="What's NVIDIA trading at? (ticker NVDA)",
         gold_calls=[{"name": "get_stock_price", "arguments": {"ticker": ["NVDA"]}}]),
    dict(id="tool_simple_05", category="simple", tools=[BMI], user="I weigh 72.5 kg and I'm 1.80 m tall. What's my BMI?",
         gold_calls=[{"name": "calculate_bmi", "arguments": {"weight_kg": [72.5], "height_m": [1.8]}}]),
    dict(id="tool_simple_06", category="simple", tools=[FILES], user="List everything under /var/log recursively.",
         gold_calls=[{"name": "list_files", "arguments": {"path": ["/var/log", "/var/log/"], "recursive": [True]}}]),
    dict(id="tool_simple_07", category="simple", tools=[TZ], user="What time is it in Tokyo right now? The IANA zone is Asia/Tokyo.",
         gold_calls=[{"name": "get_time_in_timezone", "arguments": {"timezone": ["Asia/Tokyo"]}}]),
    dict(id="tool_simple_08", category="simple", tools=[LIGHT], user="Turn on the kitchen light at 40% brightness.",
         gold_calls=[{"name": "set_light_state", "arguments": {"room": ["kitchen"], "on": [True], "brightness": [40]}}]),
    # ---------------- multiple (several tools available, pick one)
    dict(id="tool_multiple_01", category="multiple", tools=[WEATHER, FLIGHTS, FX, TIMER],
         user="Find me flights from SFO to JFK on 2026-12-03 for 2 passengers.",
         gold_calls=[{"name": "search_flights", "arguments": {"origin": ["SFO"], "destination": ["JFK"], "date": ["2026-12-03"], "passengers": [2]}}]),
    dict(id="tool_multiple_02", category="multiple", tools=[EMAIL, CAL, TICKET],
         user="Email bob@example.com with subject 'Lunch' and body 'Are we still on for noon tomorrow?'",
         gold_calls=[{"name": "send_email", "arguments": {"to": ["bob@example.com"], "subject": ["Lunch"], "body": ["Are we still on for noon tomorrow?"]}}]),
    dict(id="tool_multiple_03", category="multiple", tools=[TRANSLATE, STOCK, WEATHER],
         user="Translate 'Where is the train station?' into German.",
         gold_calls=[{"name": "translate_text", "arguments": {"text": ["Where is the train station?"], "target_language": ["de"]}}]),
    dict(id="tool_multiple_04", category="multiple", tools=[CAL, TIMER, EMAIL],
         user="Put 'Quarterly review' on my calendar for 2026-11-14 at 15:30, lasting 90 minutes.",
         gold_calls=[{"name": "create_calendar_event", "arguments": {"title": ["Quarterly review"], "date": ["2026-11-14"], "start_time": ["15:30"], "duration_minutes": [90]}}]),
    dict(id="tool_multiple_05", category="multiple", tools=[SQL, FILES, STOCK],
         user="Run this query on the warehouse: SELECT COUNT(*) FROM orders",
         gold_calls=[{"name": "run_sql_query", "arguments": {"query": ["SELECT COUNT(*) FROM orders", "SELECT COUNT(*) FROM orders;"]}}]),
    dict(id="tool_multiple_06", category="multiple", tools=[TICKET, EMAIL, LIGHT],
         system="You are a support agent for Acme Corp.",
         user="Customer C-1042 says their router keeps rebooting. Open a high priority ticket with summary 'Router reboot loop'.",
         gold_calls=[{"name": "create_support_ticket", "arguments": {"customer_id": ["C-1042"], "priority": ["high"], "summary": ["Router reboot loop"]}}]),
    # ---------------- parallel (two calls expected)
    dict(id="tool_parallel_01", category="parallel", tools=[WEATHER],
         user="What's the current weather in Oslo and in Cairo? Use celsius for both.",
         gold_calls=[{"name": "get_current_weather", "arguments": {"city": ["Oslo"], "unit": ["celsius"]}},
                     {"name": "get_current_weather", "arguments": {"city": ["Cairo"], "unit": ["celsius"]}}]),
    dict(id="tool_parallel_02", category="parallel", tools=[STOCK, FX],
         user="Get me the stock prices of AAPL and MSFT.",
         gold_calls=[{"name": "get_stock_price", "arguments": {"ticker": ["AAPL"]}},
                     {"name": "get_stock_price", "arguments": {"ticker": ["MSFT"]}}]),
    dict(id="tool_parallel_03", category="parallel", tools=[TIMER, LIGHT],
         user="Turn off the bedroom light and set a 10 minute timer.",
         gold_calls=[{"name": "set_light_state", "arguments": {"room": ["bedroom"], "on": [False], "brightness": [O, 0]}},
                     {"name": "set_timer", "arguments": {"minutes": [10], "label": [O]}}]),
    # ---------------- irrelevance (tools present but none applicable)
    dict(id="tool_irrelevance_01", category="irrelevance", tools=[WEATHER, FX],
         user="Who wrote the novel 'Pride and Prejudice'?", gold_calls=[]),
    dict(id="tool_irrelevance_02", category="irrelevance", tools=[STOCK, TIMER],
         user="What is 17 multiplied by 23?", gold_calls=[]),
    dict(id="tool_irrelevance_03", category="irrelevance", tools=[FLIGHTS, CAL],
         user="Write a two-line poem about autumn leaves.", gold_calls=[]),
]
