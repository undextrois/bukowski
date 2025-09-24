"""
Data Manager Component
Handles user data retrieval, API integration, and mock data
"""

import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import random

logger = logging.getLogger(__name__)

class DataManager:
    """Manage user financial data with API integration and mock data"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data manager"""
        self.config = config
        self.use_mock_data = config.get('mock_data', True)
        self.api_base_url = config.get('user_data_api')
        self.api_timeout = config.get('api_timeout', 10)
        
        # Cache for user data
        self.user_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize mock data
        self._initialize_mock_data()
    
    def _initialize_mock_data(self):
        """Initialize realistic mock financial data"""
        self.mock_users = {
            'default': {
                'user_id': 'default',
                'name': 'John Doe',
                'accounts': [
                    {
                        'account_id': 'acc_001',
                        'account_type': 'checking',
                        'account_name': 'Primary Checking',
                        'balance': 2500.75,
                        'currency': 'USD'
                    },
                    {
                        'account_id': 'acc_002',
                        'account_type': 'savings',
                        'account_name': 'Emergency Savings',
                        'balance': 8500.00,
                        'currency': 'USD'
                    },
                    {
                        'account_id': 'acc_003',
                        'account_type': 'credit',
                        'account_name': 'Main Credit Card',
                        'balance': -1200.50,
                        'currency': 'USD',
                        'credit_limit': 5000.00
                    },
                    {
                        'account_id': 'acc_004',
                        'account_type': 'investment',
                        'account_name': '401k Retirement',
                        'balance': 45000.00,
                        'currency': 'USD'
                    }
                ],
                'transactions': self._generate_mock_transactions(),
                'monthly_income': 4500.00,
                'monthly_expenses': {
                    'rent': 1200.00,
                    'groceries': 400.00,
                    'utilities': 150.00,
                    'transportation': 300.00,
                    'entertainment': 200.00,
                    'insurance': 250.00,
                    'other': 300.00
                },
                'financial_goals': [
                    {
                        'goal_id': 'goal_001',
                        'name': 'Emergency Fund',
                        'target_amount': 10000.00,
                        'current_amount': 8500.00,
                        'target_date': '2025-06-01'
                    },
                    {
                        'goal_id': 'goal_002',
                        'name': 'House Down Payment',
                        'target_amount': 50000.00,
                        'current_amount': 12000.00,
                        'target_date': '2026-12-01'
                    }
                ],
                'investments': [
                    {
                        'symbol': 'VTI',
                        'name': 'Vanguard Total Stock Market ETF',
                        'shares': 50,
                        'current_price': 220.00,
                        'total_value': 11000.00
                    },
                    {
                        'symbol': 'BND',
                        'name': 'Vanguard Total Bond Market ETF',
                        'shares': 100,
                        'current_price': 82.50,
                        'total_value': 8250.00
                    }
                ]
            }
        }
    
    def _generate_mock_transactions(self) -> List[Dict[str, Any]]:
        """Generate realistic mock transaction history"""
        transactions = []
        categories = ['groceries', 'gas', 'restaurants', 'utilities', 'entertainment', 'shopping', 'salary']
        
        # Generate transactions for the last 60 days
        for i in range(60):
            date = datetime.now() - timedelta(days=i)
            
            # Generate 1-3 transactions per day
            num_transactions = random.randint(1, 3)
            
            for _ in range(num_transactions):
                category = random.choice(categories)
                
                # Salary transactions
                if category == 'salary' and date.day in [1, 15]:  # Bi-monthly salary
                    amount = 2250.00
                    description = 'Salary Deposit'
                else:
                    # Expense transactions
                    if category == 'groceries':
                        amount = -random.uniform(20, 120)
                        description = random.choice(['Supermarket', 'Grocery Store', 'Whole Foods'])
                    elif category == 'gas':
                        amount = -random.uniform(30, 80)
                        description = 'Gas Station'
                    elif category == 'restaurants':
                        amount = -random.uniform(15, 60)
                        description = random.choice(['Restaurant', 'Fast Food', 'Coffee Shop'])
                    elif category == 'utilities':
                        amount = -random.uniform(50, 200)
                        description = random.choice(['Electric Bill', 'Internet', 'Phone'])
                    elif category == 'entertainment':
                        amount = -random.uniform(10, 50)
                        description = random.choice(['Movie Theater', 'Streaming Service', 'Concert'])
                    else:  # shopping
                        amount = -random.uniform(25, 200)
                        description = 'Online Purchase'
                
                if category != 'salary' or (category == 'salary' and date.day in [1, 15]):
                    transactions.append({
                        'transaction_id': f'txn_{len(transactions):04d}',
                        'date': date.strftime('%Y-%m-%d'),
                        'amount': round(amount, 2),
                        'category': category,
                        'description': description,
                        'account_id': 'acc_001'  # Most transactions from checking
                    })
        
        return sorted(transactions, key=lambda x: x['date'], reverse=True)
    
    async def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user financial data"""
        try:
            # Check cache first
            cache_key = f"user_{user_id}"
            if cache_key in self.user_cache:
                cached_data, timestamp = self.user_cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                    return cached_data
            
            if self.use_mock_data:
                # Return mock data
                user_data = self.mock_users.get(user_id, self.mock_users['default'])
            else:
                # Fetch from API
                user_data = await self._fetch_from_api(user_id)
            
            # Cache the result
            self.user_cache[cache_key] = (user_data, datetime.now())
            
            return user_data
        
        except Exception as e:
            logger.error(f"Error getting user data for {user_id}: {e}")
            # Return basic fallback data
            return {
                'user_id': user_id,
                'accounts': [],
                'transactions': [],
                'error': str(e)
            }
    
    async def _fetch_from_api(self, user_id: str) -> Dict[str, Any]:
        """Fetch user data from external API"""
        if not self.api_base_url:
            raise ValueError("API base URL not configured")
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(self.api_timeout)) as session:
            # Fetch user accounts
            accounts_url = f"{self.api_base_url}/users/{user_id}/accounts"
            async with session.get(accounts_url) as response:
                accounts_data = await response.json()
            
            # Fetch recent transactions
            transactions_url = f"{self.api_base_url}/users/{user_id}/transactions"
            params = {'limit': 100, 'days': 30}
            async with session.get(transactions_url, params=params) as response:
                transactions_data = await response.json()
            
            # Fetch user profile
            profile_url = f"{self.api_base_url}/users/{user_id}/profile"
            async with session.get(profile_url) as response:
                profile_data = await response.json()
            
            return {
                'user_id': user_id,
                'accounts': accounts_data.get('accounts', []),
                'transactions': transactions_data.get('transactions', []),
                'profile': profile_data,
                'last_updated': datetime.now().isoformat()
            }
    
    async def get_account_balance(self, user_id: str, account_type: Optional[str] = None) -> Dict[str, Any]:
        """Get current account balance(s)"""
        user_data = await self.get_user_data(user_id)
        accounts = user_data.get('accounts', [])
        
        if not accounts:
            return {'total_balance': 0.0, 'accounts': []}
        
        if account_type:
            # Filter by account type
            filtered_accounts = [acc for acc in accounts if acc.get('account_type') == account_type]
        else:
            # All accounts except credit cards (negative balances)
            filtered_accounts = [acc for acc in accounts if acc.get('account_type') != 'credit']
        
        total_balance = sum(acc.get('balance', 0) for acc in filtered_accounts)
        
        return {
            'total_balance': total_balance,
            'accounts': filtered_accounts,
            'account_count': len(filtered_accounts)
        }
    
    async def get_spending_analysis(self, user_id: str, category: Optional[str] = None, 
                                  days: int = 30) -> Dict[str, Any]:
        """Analyze spending patterns"""
        user_data = await self.get_user_data(user_id)
        transactions = user_data.get('transactions', [])
        
        # Filter transactions by date range
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_transactions = [
            txn for txn in transactions 
            if datetime.strptime(txn['date'], '%Y-%m-%d') >= cutoff_date
            and txn.get('amount', 0) < 0  # Only expenses
        ]
        
        # Group by category
        spending_by_category = {}
        total_spending = 0
        
        for txn in recent_transactions:
            cat = txn.get('category', 'other')
            amount = abs(txn.get('amount', 0))
            
            if cat not in spending_by_category:
                spending_by_category[cat] = {'total': 0, 'transactions': 0, 'average': 0}
            
            spending_by_category[cat]['total'] += amount
            spending_by_category[cat]['transactions'] += 1
            total_spending += amount
        
        # Calculate averages and percentages
        for cat, data in spending_by_category.items():
            data['average'] = data['total'] / data['transactions']
            data['percentage'] = (data['total'] / total_spending * 100) if total_spending > 0 else 0
        
        result = {
            'total_spending': total_spending,
            'spending_by_category': spending_by_category,
            'period_days': days,
            'transaction_count': len(recent_transactions)
        }
        
        # Filter by specific category if requested
        if category and category in spending_by_category:
            result['category_focus'] = {
                'category': category,
                'data': spending_by_category[category]
            }
        
        return result
    
    async def project_future_balance(self, user_id: str, days: int) -> Dict[str, Any]:
        """Project future account balance based on income/expense patterns"""
        user_data = await self.get_user_data(user_id)
        
        # Get current balance
        current_balance_data = await self.get_account_balance(user_id)
        current_balance = current_balance_data['total_balance']
        
        # Calculate average daily income and expenses
        spending_data = await self.get_spending_analysis(user_id, days=30)
        monthly_expenses = spending_data['total_spending']
        daily_expenses = monthly_expenses / 30
        
        # Get income data
        monthly_income = user_data.get('monthly_income', 0)
        daily_income = monthly_income / 30
        
        # Calculate net daily change
        daily_net = daily_income - daily_expenses
        
        # Project future balance
        projected_balance = current_balance + (daily_net * days)
        
        return {
            'current_balance': current_balance,
            'projected_balance': projected_balance,
            'projection_days': days,
            'daily_net_change': daily_net,
            'monthly_income': monthly_income,
            'monthly_expenses': monthly_expenses,
            'projection_date': (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
        }
    
    async def get_financial_goals(self, user_id: str) -> Dict[str, Any]:
        """Get user's financial goals and progress"""
        user_data = await self.get_user_data(user_id)
        goals = user_data.get('financial_goals', [])
        
        # Calculate progress for each goal
        for goal in goals:
            target = goal.get('target_amount', 0)
            current = goal.get('current_amount', 0)
            
            if target > 0:
                goal['progress_percentage'] = min(100, (current / target) * 100)
                goal['remaining_amount'] = max(0, target - current)
            else:
                goal['progress_percentage'] = 0
                goal['remaining_amount'] = 0
            
            # Calculate time to goal based on current savings rate
            target_date = datetime.strptime(goal.get('target_date', '2025-12-31'), '%Y-%m-%d')
            days_remaining = (target_date - datetime.now()).days
            goal['days_remaining'] = max(0, days_remaining)
        
        return {
            'goals': goals,
            'total_goals': len(goals),
            'completed_goals': len([g for g in goals if g.get('progress_percentage', 0) >= 100])
        }
    
    async def get_investment_summary(self, user_id: str) -> Dict[str, Any]:
        """Get investment portfolio summary"""
        user_data = await self.get_user_data(user_id)
        investments = user_data.get('investments', [])
        
        total_value = sum(inv.get('total_value', 0) for inv in investments)
        
        # Calculate portfolio allocation
        allocations = {}
        for inv in investments:
            symbol = inv.get('symbol', 'unknown')
            value = inv.get('total_value', 0)
            percentage = (value / total_value * 100) if total_value > 0 else 0
            
            allocations[symbol] = {
                'value': value,
                'percentage': percentage,
                'shares': inv.get('shares', 0),
                'current_price': inv.get('current_price', 0)
            }
        
        return {
            'total_portfolio_value': total_value,
            'number_of_positions': len(investments),
            'allocations': allocations,
            'investments': investments
        }
    
    def clear_cache(self, user_id: Optional[str] = None):
        """Clear user data cache"""
        if user_id:
            cache_key = f"user_{user_id}"
            if cache_key in self.user_cache:
                del self.user_cache[cache_key]
        else:
            self.user_cache.clear()
        
        logger.info(f"Cache cleared for {'user ' + user_id if user_id else 'all users'}")
    
    def add_mock_transaction(self, user_id: str, transaction_data: Dict[str, Any]):
        """Add a mock transaction for testing"""
        if user_id not in self.mock_users:
            self.mock_users[user_id] = self.mock_users['default'].copy()
        
        transaction_data['transaction_id'] = f"txn_mock_{len(self.mock_users[user_id]['transactions'])}"
        transaction_data['date'] = transaction_data.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        self.mock_users[user_id]['transactions'].insert(0, transaction_data)
        
        # Clear cache to reflect new data
        self.clear_cache(user_id)