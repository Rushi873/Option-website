# Option Strategy Builder Website

A comprehensive web application for visualizing and analyzing options trading strategies with AI-powered market insights. This project provides traders with an intuitive platform to build, analyze, and optimize options strategies while leveraging real-time market data and artificial intelligence.

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/13933d56-3a3d-468e-bb14-7637d45f075d" width="300"/></td>
    <td><img src="https://github.com/user-attachments/assets/f4d25e55-e8f4-4c16-99bb-c4392f8f8272" width="300"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/ed151032-1fc5-425b-9110-63a91ae840dd" width="300"/></td>
    <td><img src="https://github.com/user-attachments/assets/b1d1ccbf-dc05-47bd-ae16-530a78a740ad" width="300"/></td>
  </tr>
</table>


## 📋 Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Features](#features)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Database Design](#database-design)
- [AI Integration](#ai-integration)
- [User Interface](#user-interface)
- [Technical Analysis](#technical-analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)

## 🎯 Overview

The Option Strategy Builder is a sophisticated financial tool designed to simplify options trading analysis. It combines real-time market data, advanced calculations, and AI-powered insights to provide traders with comprehensive strategy visualization and analysis capabilities.

### **Breif about the financial market**
- **Equity Market**: Platform for buying and selling publicly held company shares
- **Derivative Market**: Trading of financial instruments including:
  - Option Contracts
  - Future Contracts
  - Forward Contracts
  - Swap Contracts

## 💡 Motivation

Traditional options trading platforms often lack comprehensive visualization tools and real-time AI-powered insights. This project addresses several key pain points:

- **Visualization Gap**: Need for easy-to-understand payoff scenario visualization
- **Manual Calculations**: Automation of complex metrics, Greeks, and trading charges
- **Fragmented Analysis**: Centralized platform for contextual market analysis
- **Limited Insights**: Real-time AI-powered analysis of fundamental, technical, and market trends

## ✨ Features

### **Core Functionality**
- **Interactive Option Chain**: Real-time NSE option chain data with click-to-add functionality
- **Strategy Builder**: Intuitive interface for building complex multi-leg options strategies
- **Payoff Visualization**: Dynamic charts showing profit/loss scenarios across different price points
- **Greeks Calculation**: Comprehensive calculation of option Greeks for individual legs and entire strategies
- **Real-time Data**: Auto-refresh capabilities with configurable intervals (default: 3 seconds)

### **Advanced Analytics**
- **AI-Powered Insights**: Google Gemini API integration for market analysis
- **Technical Analysis**: Moving averages, support/resistance levels, momentum indicators
- **Fundamental Analysis**: P/E ratios, market cap, earnings data
- **News Integration**: Real-time news feeds related to underlying assets
- **Strategy Metrics**: Maximum profit/loss, risk-reward ratios, breakeven points

### **User Experience**
- **ATM Highlighting**: Automatic highlighting of at-the-money strikes
- **Visual Price Changes**: Real-time price change highlighting
- **Responsive Design**: Optimized for various screen sizes
- **Interactive Charts**: Plotly.js powered visualizations

## 🔧 How It Works

### **Simple 4-Step Process**

1. **Select**: Choose your desired asset and expiry date - option chain loads automatically
2. **Click**: Click on Call/Put LTP values in the chain to add legs to your strategy
3. **Modify**: Adjust lot sizes and toggle between buy/sell positions in the strategy table
4. **Update**: Click "Update & Calculate" to generate payoff charts and analysis

### **Example: Understanding Options (Kid-Friendly)**
Think of options like a toy car deal:
- **Pay $1 today** for the option to buy a toy car for $10 anytime this month
- **If price rises to $15**: You buy for $10, saving $5
- **If price stays at $10 or drops**: You don't buy, losing only the $1 premium

## 🏗️ Architecture

### **Frontend**
- **Technologies**: HTML5, CSS3, JavaScript
- **Visualization**: Plotly.js for interactive charts

### **Backend**
- **Framework**: FastAPI (Python)
- **API Design**: RESTful architecture
- **Request Handling**: Asynchronous processing
- **Data Orchestration**: Centralized logic management

### **Data Flow**
```
User Interface → FastAPI Backend → Database/External APIs → AI Processing → Response
```

## 🗄️ Database Design

### **MySQL Database Structure**

#### **Asset Table**
- Stores master list of 234 supported assets
- Includes lot sizes for each asset
- Enables quick asset lookup and validation

#### **Option Chain Table**
- Real-time option data storage
- Fields: Open Interest (OI), Implied Volatility (IV), Last Traded Price (LTP)
- Dynamic updates via background tasks
- Asset-specific data management

#### **Expiry Table**
- Available expiry dates for each asset
- Populates expiry dropdown menus
- Maintains data consistency

## 🤖 AI Integration

### **Google Gemini API Integration**
The platform leverages Google Gemini API to provide intelligent market analysis by processing:

#### **Data Sources**
- **Technical Data**: Moving averages, price action, volume analysis
- **Fundamental Data**: Financial ratios, earnings, market capitalization
- **News Data**: Recent headlines and market sentiment
- **Options Data**: Greeks, volatility, open interest patterns

#### **AI-Generated Insights**
- Market trend analysis and predictions
- Greeks interpretation and strategy recommendations
- Risk assessment and position sizing suggestions
- Market sentiment analysis from news data

## 🖥️ User Interface

### **Interactive Features**

#### **Option Chain Display**
- Real-time NSE option chain data
- Color-coded price changes
- ATM strike highlighting
- Sortable columns for easy analysis

#### **Strategy Builder Table**
- Drag-and-drop functionality
- Real-time lot size adjustments
- Buy/Sell toggle switches
- Remove/modify positions

#### **Analysis Panels**
- **Technical Analysis**: Moving averages, trend analysis, support/resistance
- **Fundamental Analysis**: Key financial metrics and ratios
- **News Feed**: Latest market-relevant news
- **Strategy Metrics**: Comprehensive strategy analysis

### **Sample Technical Analysis Output**
```
Current Price: ₹1825.15
Trading Volume: 2,217,947
50-day MA: ₹1895.42
200-day MA: ₹1781.88

Trend Analysis: Bearish trend indicated by 50-day MA below 200-day MA
Support Levels: 1780-1800
Resistance Levels: 1850-1870, 1895 (50-day MA)
```

## 📊 Data Sources

### **Real-time Market Data**
- **jugaad-data**: Live NSE option chain data
- **RapidAPI (Yahoo Finance)**: Stock/index technical data
- **feedparser/BeautifulSoup**: Google News RSS feeds

### **Update Mechanisms**
- **Background Tasks**: Automatic data refresh for selected assets
- **Configurable Intervals**: User-defined refresh rates
- **Error Handling**: Robust data validation and fallback mechanisms

## 🛠️ Installation

### **Prerequisites**
```bash
Python 3.8+
MySQL 8.0+
Node.js (for frontend dependencies)
```

### **Backend Setup**
```bash
# Clone the repository
git clone https://github.com/Rushi873/Option-website
cd option-strategy-builder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your database and API credentials

# Run database migrations
python manage.py migrate

# Start the FastAPI server
uvicorn main:app --reload
```

### **Frontend Setup**
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## 📈 Usage

### **Building Your First Strategy**

1. **Asset Selection**
   - Choose from 234 supported assets
   - Select appropriate expiry date

2. **Strategy Construction**
   - Click on Call/Put prices in the option chain
   - Adjust lot sizes as needed
   - Toggle between buy/sell positions

3. **Analysis**
   - Review payoff charts
   - Analyze Greeks and risk metrics
   - Consider AI-generated insights

4. **Optimization**
   - Modify positions based on analysis
   - Test different scenarios
   - Implement risk management rules

### **Strategy Metrics Explained**

- **Max Profit**: Maximum potential profit from the strategy
- **Max Loss**: Maximum potential loss (risk)
- **Risk-Reward Ratio**: Ratio of potential profit to potential loss
- **Breakeven Points**: Stock prices where strategy breaks even
- **Net Premium**: Total premium paid/received for the strategy

## 🔮 Future Enhancements

### **Planned Features**

#### **Enhanced AI Capabilities**
- LLM-powered strategy recommendations based on Greeks analysis
- Personalized trading advice based on user preferences
- Market regime detection and strategy adaptation

#### **Strategy Library**
- Pre-built famous options strategies (Iron Condor, Butterfly, Straddle, etc.)
- User strategy saving and sharing capabilities
- Strategy performance backtesting

#### **UI/UX Improvements**
- Modern, intuitive interface design
- Mobile app development
- Advanced charting capabilities
- Real-time collaboration features

#### **Advanced Analytics**
- Portfolio-level risk management
- Multi-asset strategy analysis
- Historical performance tracking
- Custom indicator development

### **Technical Roadmap**
- WebSocket integration for real-time data streaming
- Advanced caching mechanisms for improved performance
- Multi-broker integration for order execution
- Cloud-native architecture migration

## 🤝 Contributing

We welcome contributions to improve the Option Strategy Builder! Here's how you can help:

### **Development Guidelines**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### **Areas for Contribution**
- Bug fixes and performance improvements
- New strategy implementations
- UI/UX enhancements
- Documentation improvements
- Test coverage expansion

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Rushi Glasswala**  

## 🙏 Acknowledgments

- NSE for providing market data access
- Google Gemini API for AI capabilities
- Open source community for various libraries and tools

---

**Note**: This project is for educational and research purposes. Please ensure compliance with relevant financial regulations and consult with financial advisors before making trading decisions.
