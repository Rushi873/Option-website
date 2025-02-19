import yfinance as yf
import requests
import google.generativeai as genai
from bs4 import BeautifulSoup

# Configure Gemini API
genai.configure(api_key="AIzaSyDd_UVZ_1OeLahVrJ0A-hbazQcr1FOpgPE")

# Define stock symbol and competitors
stock_symbol = "ITC.NS"  # NSE format (Use '.BO' for BSE)
competitors = ["HINDUNILVR.NS", "BRITANNIA.NS", "GODREJCP.NS"]  # FMCG sector

# Fetch real-time stock data
stock = yf.Ticker(stock_symbol)
stock_data = stock.history(period="1d")

# Extract technical data
current_price = stock_data["Close"].iloc[-1]
volume = stock_data["Volume"].iloc[-1]
moving_avg_50 = stock.history(period="50d")["Close"].mean()
moving_avg_200 = stock.history(period="200d")["Close"].mean()

# Get fundamental data
info = stock.info
market_cap = info.get("marketCap", "N/A")
pe_ratio = info.get("trailingPE", "N/A")
eps = info.get("trailingEps", "N/A")

# Fetch competitor data for sector comparison
competitor_data = {}
for competitor in competitors:
    comp_stock = yf.Ticker(competitor)
    comp_price = comp_stock.history(period="1d")["Close"].iloc[-1]
    competitor_data[competitor] = comp_price

# Function to scrape news from Google Finance
def get_google_finance_news(query):
    url = f"https://www.google.com/finance/quote/{query}:NSE"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    news_items = soup.find_all("div", class_="Yfwt5")  # Class for news headlines
    headlines = [item.text for item in news_items[:3]]  # Extract top 3 news
    return headlines if headlines else ["No major updates at the moment."]

# Fetch latest news
latest_news = get_google_finance_news(stock_symbol)

# Construct the prompt
prompt = f"""
Analyze {stock_symbol} in the Indian stock market, providing real-time insights with sector comparisons and upcoming events that may impact stock prices. Provide real-time data and analysis fetch from the API.

1️⃣ **Technical Analysis:**
   - **Current Price:** ₹{current_price}
   - **Trading Volume:** {volume}
   - **50-day MA:** ₹{moving_avg_50:.2f}
   - **200-day MA:** ₹{moving_avg_200:.2f}
   - **Sector Comparison:** {stock_symbol} compared with {competitors[0]}, {competitors[1]}, {competitors[2]} based on price trends and moving averages.
   - **Support & Resistance Levels:** Recent highs and lows influencing price action.
   - **Trend Analysis:** Analysis of bullish or bearish signals based on key indicators.
   - **Pattern Formation:** Observation of head & shoulders, double tops/bottoms, or flags.

2️⃣ **Fundamental Analysis:**  
   - **Market Capitalization:** ₹{market_cap}
   - **P/E Ratio:** {pe_ratio}
   - **EPS:** ₹{eps}
   - **Revenue & Profit Margins:** Analysis of revenue growth and operating margins.
   - **Sector Benchmarking:** How {stock_symbol} performs against sector averages.
   - **Debt-to-Equity Ratio:** Financial stability insights.

3️⃣ **Latest News & Events Impacting {stock_symbol}:**  
   - **Recent News:**  
     1. {latest_news[0]}
     2. {latest_news[1] if len(latest_news) > 1 else ''}
     3. {latest_news[2] if len(latest_news) > 2 else ''}
   - **Upcoming Events & Influencing Factors:**  
     - Scheduled earnings reports, policy changes, or global economic factors that could impact the stock price.
     - Government regulations, industry shifts, or interest rate announcements.

4️⃣ **Trading Outlook:**  
   - Overall trend, risk factors, and opportunities based on technical and fundamental indicators.
   - Sector positioning and whether {stock_symbol} is a strong buy, hold, or sell.
"""

# Fetch AI response from Gemini API
model = genai.GenerativeModel("gemini-1.5-pro")
response = model.generate_content(prompt)

# Print final stock analysis
print(response.text)
