"""
Response Generator Component
Generate contextual responses based on intent, entities, and user data
"""

import random
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generate contextual responses for financial queries"""
    
    def __init__(self):
        """Initialize response generator with templates and financial knowledge"""
        self.response_templates = self._initialize_templates()
        self.financial_knowledge = self._initialize_knowledge_base()
        self.conversational_responses = self._initialize_conversational_responses()
    
    def _initialize_templates(self) -> Dict[str, List[str]]:
        """Initialize response templates for different intents"""
        return {
            'account_balance': [
                "Based on your accounts, you currently have ${total_balance:.2f} available.",
                "Your current balance across all accounts is ${total_balance:.2f}.",
                "You have ${total_balance:.2f} in your accounts right now.",
                "Looking at your accounts, your total balance is ${total_balance:.2f}."
            ],
            'spending_analysis': [
                "In the last {days} days, you've spent ${total_spending:.2f}.",
                "Your spending over the past {days} days totals ${total_spending:.2f}.",
                "You've had ${total_spending:.2f} in expenses over the last {days} days."
            ],
            'future_balance': [
                "In {days} days, your projected balance will be ${projected_balance:.2f}.",
                "Based on your spending patterns, you'll have approximately ${projected_balance:.2f} in {days} days.",
                "Your estimated balance after {days} days: ${projected_balance:.2f}."
            ],
            'investment_advice': [
                "Here's some investment guidance: {advice}",
                "For investing, consider this: {advice}",
                "Investment tip: {advice}"
            ],
            'financial_education': [
                "Let me explain {topic}: {explanation}",
                "Here's what you need to know about {topic}: {explanation}",
                "{topic} explained: {explanation}"
            ],
            'budget_planning': [
                "For budgeting, I recommend: {advice}",
                "Here's a budgeting suggestion: {advice}",
                "Budget planning tip: {advice}"
            ],
            'error': [
                "I'm sorry, I couldn't find that information in your account data.",
                "I don't have access to that specific information right now.",
                "I'm unable to retrieve that data at the moment."
            ]
        }
    
    def _initialize_knowledge_base(self) -> Dict[str, Dict[str, str]]:
        """Initialize financial education knowledge base"""
        return {
            'compound_interest': {
                'explanation': 'Compound interest is interest calculated on both the initial principal and the accumulated interest from previous periods. It\'s often called "interest on interest" and can significantly accelerate wealth building over time.',
                'example': 'If you invest $1,000 at 7% annual compound interest, after 10 years you\'d have about $1,967 - earning $967 in interest!',
                'tips': ['Start investing early to maximize compound growth', 'Be consistent with contributions', 'Choose investments with good long-term returns']
            },
            '401k': {
                'explanation': 'A 401(k) is an employer-sponsored retirement savings plan that allows you to contribute pre-tax dollars, reducing your current taxable income. Many employers offer matching contributions.',
                'example': 'If you earn $50,000 and contribute 10% ($5,000) to your 401(k), your taxable income becomes $45,000.',
                'tips': ['Contribute at least enough to get full employer match', 'Increase contributions with salary raises', 'Consider Roth 401(k) for tax-free withdrawals in retirement']
            },
            'stocks': {
                'explanation': 'Stocks represent ownership shares in a company. When you buy stocks, you become a partial owner and can benefit from the company\'s growth through price appreciation and dividends.',
                'example': 'If you buy 100 shares of a company at $50 each, you\'ve invested $5,000. If the stock price rises to $60, your investment is worth $6,000.',
                'tips': ['Diversify across different companies and sectors', 'Think long-term', 'Don\'t try to time the market']
            },
            'bonds': {
                'explanation': 'Bonds are debt securities where you lend money to governments or corporations in exchange for regular interest payments and return of principal at maturity.',
                'example': 'A $1,000 bond with 5% annual interest pays $50 per year and returns your $1,000 at maturity.',
                'tips': ['Bonds are generally safer than stocks', 'Government bonds are typically safer than corporate bonds', 'Consider bond funds for diversification']
            },
            'emergency_fund': {
                'explanation': 'An emergency fund is money set aside for unexpected expenses like job loss, medical bills, or major repairs. It provides financial security and peace of mind.',
                'example': 'If your monthly expenses are $3,000, aim for an emergency fund of $9,000-$18,000 (3-6 months of expenses).',
                'tips': ['Keep it in a high-yield savings account', 'Aim for 3-6 months of living expenses', 'Build it gradually - even $500 is a good start']
            },
            'budgeting': {
                'explanation': 'Budgeting is planning how to spend your money by tracking income and expenses. It helps ensure you\'re living within your means and working toward financial goals.',
                'example': 'The 50/30/20 rule: 50% for needs, 30% for wants, 20% for savings and debt repayment.',
                'tips': ['Track all expenses for a month first', 'Use budgeting apps or spreadsheets', 'Review and adjust monthly']
            }
        }
    
    def _initialize_conversational_responses(self) -> Dict[str, List[str]]:
        """Initialize conversational and empathetic responses"""
        return {
            'greeting': [
                "Hello! I'm here to help with your financial questions.",
                "Hi there! What can I help you with regarding your finances today?",
                "Welcome! I'm ready to assist with your financial planning and questions."
            ],
            'encouragement': [
                "You're taking great steps toward better financial health!",
                "It's smart that you're being proactive about your finances.",
                "Great question! Financial literacy is so important."
            ],
            'low_balance_concern': [
                "I notice your balance is getting low. Consider reviewing your recent spending.",
                "Your account balance is lower than usual. Would you like to see your recent transactions?",
                "It might be good to review your budget given your current balance."
            ],
            'positive_progress': [
                "You're making excellent progress toward your financial goals!",
                "Your financial discipline is really paying off!",
                "Keep up the great work with your savings!"
            ]
        }
    
    async def generate_response(self, intent: str, entities: Dict[str, List], 
                              user_data: Dict[str, Any], original_message: str) -> str:
        """Generate contextual response based on intent, entities, and user data"""
        try:
            if intent == 'account_balance':
                return await self._handle_balance_query(entities, user_data)
            
            elif intent == 'spending_analysis':
                return await self._handle_spending_query(entities, user_data)
            
            elif intent == 'future_balance':
                return await self._handle_future_balance_query(entities, user_data)
            
            elif intent == 'investment_advice':
                return await self._handle_investment_query(entities, user_data, original_message)
            
            elif intent == 'financial_education':
                return await self._handle_education_query(entities, user_data, original_message)
            
            elif intent == 'budget_planning':
                return await self._handle_budget_query(entities, user_data)
            
            else:
                return await self._handle_unknown_query(original_message)
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return random.choice(self.response_templates['error'])
    
    async def _handle_balance_query(self, entities: Dict, user_data: Dict) -> str:
        """Handle account balance queries"""
        try:
            # Import here to avoid circular imports
            from components.data_manager import DataManager
            data_manager = DataManager({'mock_data': True})
            
            # Check if specific account type was mentioned
            account_types = [ent['value'] for ent in entities.get('account_type', [])]
            account_type = account_types[0] if account_types else None
            
            balance_data = await data_manager.get_account_balance(
                user_data.get('user_id', 'default'), 
                account_type
            )
            
            total_balance = balance_data['total_balance']
            account_count = balance_data['account_count']
            
            template = random.choice(self.response_templates['account_balance'])
            response = template.format(total_balance=total_balance)
            
            # Add contextual information
            if account_type:
                response += f" This includes your {account_type} account(s)."
            elif account_count > 1:
                response += f" This total includes {account_count} accounts."
            
            # Add encouragement or concern based on balance
            if total_balance < 1000:
                response += " " + random.choice(self.conversational_responses['low_balance_concern'])
            elif total_balance > 10000:
                response += " " + random.choice(self.conversational_responses['positive_progress'])
            
            return response
        
        except Exception as e:
            logger.error(f"Error handling balance query: {e}")
            return "I'm having trouble accessing your account information right now."
    
    async def _handle_spending_query(self, entities: Dict, user_data: Dict) -> str:
        """Handle spending analysis queries"""
        try:
            from components.data_manager import DataManager
            data_manager = DataManager({'mock_data': True})
            
            # Extract time period and category from entities
            categories = [ent['value'] for ent in entities.get('category', [])]
            category = categories[0] if categories else None
            
            # Extract time period - default to 30 days
            time_periods = entities.get('time_period', [])
            days = 30
            if time_periods:
                from components.entity_extractor import EntityExtractor
                extractor = EntityExtractor()
                normalized_days = extractor.normalize_time_period(time_periods[0]['value'])
                if normalized_days:
                    days = normalized_days
            
            spending_data = await data_manager.get_spending_analysis(
                user_data.get('user_id', 'default'),
                category=category,
                days=days
            )
            
            total_spending = spending_data['total_spending']
            spending_by_category = spending_data['spending_by_category']
            
            template = random.choice(self.response_templates['spending_analysis'])
            response = template.format(total_spending=total_spending, days=days)
            
            # Add category-specific information
            if category and category in spending_by_category:
                cat_data = spending_by_category[category]
                response += f" Specifically on {category}, you spent ${cat_data['total']:.2f} ({cat_data['percentage']:.1f}% of total spending)."
            
            # Add top spending categories
            if spending_by_category:
                top_categories = sorted(spending_by_category.items(), 
                                      key=lambda x: x[1]['total'], reverse=True)[:3]
                
                response += " Your top spending categories were: "
                response += ", ".join([f"{cat} (${data['total']:.2f})" 
                                     for cat, data in top_categories])
                response += "."
            
            return response
        
        except Exception as e:
            logger.error(f"Error handling spending query: {e}")
            return "I'm having trouble analyzing your spending data right now."
    
    async def _handle_future_balance_query(self, entities: Dict, user_data: Dict) -> str:
        """Handle future balance projection queries"""
        try:
            from components.data_manager import DataManager
            data_manager = DataManager({'mock_data': True})
            
            # Extract time period
            time_periods = entities.get('time_period', [])
            days = 30  # default
            
            if time_periods:
                from components.entity_extractor import EntityExtractor
                extractor = EntityExtractor()
                normalized_days = extractor.normalize_time_period(time_periods[0]['value'])
                if normalized_days:
                    days = normalized_days
            
            projection_data = await data_manager.project_future_balance(
                user_data.get('user_id', 'default'), days
            )
            
            current_balance = projection_data['current_balance']
            projected_balance = projection_data['projected_balance']
            daily_net = projection_data['daily_net_change']
            
            template = random.choice(self.response_templates['future_balance'])
            response = template.format(projected_balance=projected_balance, days=days)
            
            # Add context about the projection
            if daily_net > 0:
                response += f" Based on your current income and spending patterns, you're saving about ${daily_net:.2f} per day."
            elif daily_net < 0:
                response += f" Based on your current patterns, you're spending about ${abs(daily_net):.2f} more than you earn per day."
            else:
                response += " You're currently breaking even with your income and expenses."
            
            # Add advice based on projection
            if projected_balance < current_balance * 0.5:
                response += " Consider reviewing your spending to improve this projection."
            elif projected_balance > current_balance * 1.2:
                response += " Great job! Your financial discipline is really paying off."
            
            return response
        
        except Exception as e:
            logger.error(f"Error handling future balance query: {e}")
            return "I'm having trouble projecting your future balance right now."
    
    async def _handle_investment_query(self, entities: Dict, user_data: Dict, original_message: str) -> str:
        """Handle investment advice queries"""
        investment_types = [ent['value'] for ent in entities.get('investment_type', [])]
        
        # General investment advice based on query content
        message_lower = original_message.lower()
        
        if 'stocks' in message_lower or 'stock market' in message_lower:
            advice = "Start with broad market index funds like VTI or SPY for diversification. Consider dollar-cost averaging by investing regularly regardless of market conditions. Only invest money you won't need for at least 5-10 years."
        
        elif 'bonds' in message_lower:
            advice = "Bonds provide stability and income. Government bonds are safest, while corporate bonds offer higher yields with more risk. Consider bond index funds for diversification. Bonds typically perform well when interest rates fall."
        
        elif 'mutual funds' in message_lower or 'etf' in message_lower:
            advice = "ETFs and mutual funds offer instant diversification. Look for low expense ratios (under 0.5%). Index funds that track the S&P 500 or total market are great starting points. ETFs are more tax-efficient than mutual funds."
        
        elif 'retirement' in message_lower or '401k' in message_lower:
            advice = "Maximize any employer 401(k) match first - it's free money! Consider a mix of stocks and bonds based on your age. A common rule: subtract your age from 110 to get your stock percentage (e.g., 30 years old = 80% stocks, 20% bonds)."
        
        elif 'beginner' in message_lower or 'start' in message_lower:
            advice = "Start with your emergency fund (3-6 months expenses), then maximize employer 401(k) match. For taxable investing, consider target-date funds or a simple three-fund portfolio: US stocks, international stocks, and bonds."
        
        else:
            advice = "Diversify your investments across different asset classes. Start with low-cost index funds, invest regularly, and think long-term. Don't try to time the market - time in the market beats timing the market."
        
        template = random.choice(self.response_templates['investment_advice'])
        response = template.format(advice=advice)
        
        # Add user-specific context if available
        if user_data.get('investments'):
            response += f" I see you already have some investments. "
        
        return response
    
    async def _handle_education_query(self, entities: Dict, user_data: Dict, original_message: str) -> str:
        """Handle financial education queries"""
        message_lower = original_message.lower()
        
        # Identify the topic
        topic = None
        explanation = ""
        
        for knowledge_topic, content in self.financial_knowledge.items():
            if knowledge_topic.replace('_', ' ') in message_lower or knowledge_topic in message_lower:
                topic = knowledge_topic.replace('_', ' ').title()
                explanation = content['explanation']
                
                # Add example if available
                if 'example' in content:
                    explanation += f"\n\nExample: {content['example']}"
                
                # Add tips if available
                if 'tips' in content:
                    explanation += f"\n\nTips:\n• " + "\n• ".join(content['tips'])
                
                break
        
        # Handle specific common questions
        if not topic:
            if 'compound interest' in message_lower:
                topic = 'Compound Interest'
                explanation = self.financial_knowledge['compound_interest']['explanation']
            elif 'invest' in message_lower and 'how' in message_lower:
                topic = 'Getting Started with Investing'
                explanation = "Start by building an emergency fund, then maximize employer 401(k) matching. For additional investing, consider low-cost index funds through brokerages like Vanguard, Fidelity, or Charles Schwab."
            elif 'budget' in message_lower:
                topic = 'Budgeting'
                explanation = self.financial_knowledge['budgeting']['explanation']
            elif 'save money' in message_lower:
                topic = 'Money Saving Tips'
                explanation = "Track your expenses, create a budget, automate savings, cook at home more, review subscriptions, negotiate bills, and use the 24-hour rule before making non-essential purchases."
            else:
                topic = 'Financial Planning'
                explanation = "Good financial planning involves budgeting, building an emergency fund, paying off high-interest debt, saving for retirement, and investing for long-term goals. Start with one area and build from there."
        
        template = random.choice(self.response_templates['financial_education'])
        return template.format(topic=topic, explanation=explanation)
    
    async def _handle_budget_query(self, entities: Dict, user_data: Dict) -> str:
        """Handle budget planning queries"""
        try:
            from components.data_manager import DataManager
            data_manager = DataManager({'mock_data': True})
            
            # Get user's spending data
            spending_data = await data_manager.get_spending_analysis(
                user_data.get('user_id', 'default'), days=30
            )
            
            monthly_income = user_data.get('monthly_income', 0)
            monthly_expenses = spending_data['total_spending']
            
            # Generate personalized budget advice
            if monthly_income > 0:
                savings_rate = ((monthly_income - monthly_expenses) / monthly_income) * 100
                
                if savings_rate >= 20:
                    advice = f"Excellent! You're saving {savings_rate:.1f}% of your income. Consider increasing investments or setting new financial goals."
                elif savings_rate >= 10:
                    advice = f"Good job! You're saving {savings_rate:.1f}% of your income. Try to gradually increase to 20% if possible."
                elif savings_rate >= 0:
                    advice = f"You're saving {savings_rate:.1f}% of your income. Aim for at least 10-20%. Look for areas to cut expenses or increase income."
                else:
                    advice = f"You're spending more than you earn. Priority #1: reduce expenses or increase income. Start by tracking every expense for a month."
                
                # Add specific recommendations based on spending categories
                spending_by_category = spending_data.get('spending_by_category', {})
                if spending_by_category:
                    top_category = max(spending_by_category.items(), key=lambda x: x[1]['total'])
                    advice += f" Your biggest expense category is {top_category[0]} (${top_category[1]['total']:.2f}/month)."
            else:
                advice = "Start by tracking your income and all expenses for a month. Use the 50/30/20 rule: 50% needs, 30% wants, 20% savings. Build an emergency fund first, then focus on other goals."
            
            template = random.choice(self.response_templates['budget_planning'])
            return template.format(advice=advice)
        
        except Exception as e:
            logger.error(f"Error handling budget query: {e}")
            return "For budgeting, I recommend starting with the 50/30/20 rule: 50% for needs, 30% for wants, and 20% for savings and debt repayment. Track your expenses for a month to see where your money goes."
    
    async def _handle_unknown_query(self, original_message: str) -> str:
        """Handle unknown or unclear queries"""
        message_lower = original_message.lower()
        
        # Try to provide helpful responses for common financial terms
        if any(word in message_lower for word in ['help', 'what can you do', 'capabilities']):
            return """I can help you with:
• Account balances and spending analysis
• Financial education (compound interest, investing, budgeting)
• Investment advice and portfolio guidance
• Budget planning and savings strategies
• Future balance projections

Try asking: "What's my current balance?" or "Explain compound interest" or "How should I start investing?"
"""
        
        elif any(word in message_lower for word in ['thank', 'thanks']):
            return random.choice([
                "You're welcome! I'm here whenever you need financial guidance.",
                "Happy to help! Feel free to ask more questions anytime.",
                "Glad I could assist! Good luck with your financial goals."
            ])
        
        elif any(word in message_lower for word in ['hello', 'hi', 'hey']):
            return random.choice(self.conversational_responses['greeting'])
        
        else:
            return """I'm not sure I understand that request. I can help with:
• Account information ("What's my balance?")
• Spending analysis ("How much did I spend on groceries?")
• Financial education ("What is compound interest?")
• Investment advice ("How should I invest?")
• Budget planning ("Help me create a budget")

What would you like to know about your finances?"""
    
    def _format_currency(self, amount: float) -> str:
        """Format currency values consistently"""
        return f"${amount:,.2f}"
    
    def _format_percentage(self, percentage: float) -> str:
        """Format percentage values consistently"""
        return f"{percentage:.1f}%"
    
    def add_response_template(self, intent: str, template: str):
        """Add a new response template for an intent"""
        if intent not in self.response_templates:
            self.response_templates[intent] = []
        
        self.response_templates[intent].append(template)
        logger.info(f"Added response template for intent: {intent}")
    
    def add_knowledge_entry(self, topic: str, explanation: str, example: str = None, tips: List[str] = None):
        """Add a new entry to the financial knowledge base"""
        self.financial_knowledge[topic] = {
            'explanation': explanation
        }
        
        if example:
            self.financial_knowledge[topic]['example'] = example
        
        if tips:
            self.financial_knowledge[topic]['tips'] = tips
        
        logger.info(f"Added knowledge entry for topic: {topic}")