# for bidder
SYSTEM_MESSAGE = """
You are {name}, who is attending an ascending-bid auction as a bidder. This auction will have some other bidders to compete with you in bidding wars. The price is gradually raised, bidders drop out until finally only one bidder remains, and that bidder wins the item at this final price. Remember: {desire_desc}.

Here are some must-know rules for this auction:

1. Item Values: The true value of an item means its resale value in the broader market, which you don't know. You will have a personal estimation of the item value. However, note that your estimated value could deviate from the true value, due to your potential overestimation or underestimation of this item.
2. Winning Bid: The highest bid wins the item. Your profit from winning an item is determined by the difference between the item's true value and your winning bid. You should try to win an item at a bid as minimal as possible to save your budget.
""".strip()


_LEARNING_STATEMENT = " and your learnings from previous auctions"


INSTRUCT_PLAN_TEMPLATE = """
As {bidder_name}, you have a total budget of ${budget}. This auction has a total of {item_num} items to be sequentially presented, they are:
{items_info}

---

Please plan for your bidding strategy for the auction based on the information{learning_statement}. A well-thought-out plan positions you advantageously against competitors, allowing you to allocate resources effectively. With a clear strategy, you can make decisions rapidly and confidently, especially under the pressure of the auction environment. Remember: {desire_desc}. 

After articulate your thinking, in you plan, assign a priority level to each item. Present the priorities for all items in a JSON format, each item should be represented as a key-value pair, where the key is the item name and the value is its priority on the scale from 1-3. An example output is: {{"Fixture Y": 3, "Module B": 2, "Product G": 2}}. The descriptions of the priority scale of items are as follows.
    * 1 - This item is the least important. Consider giving it up if necessary to save money for the rest of the auction.
    * 2 - This item holds value but isn't a top priority for the bidder. Could bid on it if you have enough budget.
    * 3 - This item is of utmost importance and is a top priority for the bidder in the rest of the auction.
""".strip()


INSTRUCT_BID_TEMPLATE = """
Now, the auctioneer says: "{auctioneer_msg}"

---

As {bidder_name}, you have to decide whether to bid on this item or withdraw and explain why, according to your plan{learning_statement}. Remember, {desire_desc}.

Here are some common practices of bidding:
1. Showing your interest by bidding with or slightly above the starting price of this item, then gradually increase your bid.
2. Think step by step of the pros and cons and the consequences of your action (e.g., remaining budget in future bidding) in order to achieve your primary objective.

Give your reasons first, then make your final decision clearly. You should either withdraw (saying "I'm out!") or make a higher bid for this item (saying "I bid $xxx!").
""".strip()


INSTRUCT_SUMMARIZE_TEMPLATE = """
Here is the history of the bidding war of {cur_item}:
"{bidding_history}"

The auctioneer concludes: "{hammer_msg}" 

---

{win_lose_msg} 
As {bidder_name}, you have to update the status of the auction based on this round of bidding. Here's your previous status:
```
{prev_status}
```

Summarize the notable behaviors of all bidders in this round of bidding for future reference. Then, update the status JSON regarding the following information:
- 'remaining_budget': The remaining budget of you, expressed as a numerical value.
- 'total_profits': The total profits achieved so far for each bidder, where a numerical value following a bidder's name. No equation is needed, just the numerical value.
- 'winning_bids': The winning bids for every item won by each bidder, listed as key-value pairs, for example, {{"bidder_name": {{"item_name_1": winning_bid}}, {{"item_name_2": winning_bid}}, ...}}. If a bidder hasn't won any item, then the value for this bidder should be an empty dictionary {{}}.
- Only include the bidders mentioned in the given text. If a bidder is not mentioned (e.g. Bidder 4 in the following example), then do not include it in the JSON object.

After summarizing the bidding history, you must output the current status in a parsible JSON format. An example output looks like:
```
{{"remaining_budget": 8000, "total_profits": {{"Bidder 1": 1300, "Bidder 2": 1800, "Bidder 3": 0}}, "winning_bids": {{"Bidder 1": {{"Item 2": 1200, "Item 3": 1000}}, "Bidder 2": {{"Item 1": 2000}}, "Bidder 3": {{}}}}}}
```
""".strip()


INSTRUCT_LEARNING_TEMPLATE = """
Review and reflect on the historical data provided from a past auction. 

{past_auction_log}

Here are your past learnings:

{past_learnings}

Based on the auction log, formulate or update your learning points that could be advantageous to your strategies in the future. Your learnings should be strategic, and of universal relevance and practical use for future auctions. Consolidate your learnings into a concise numbered list of sentences.
""".strip()


INSTRUCT_REPLAN_TEMPLATE = """
The current status of you and other bidders is as follows:
```
{status_quo}
```

Here are the remaining items in the rest of the auction:
"{remaining_items_info}"

As {bidder_name}, considering the current status{learning_statement}, review your strategies. Adjust your plans based on the outcomes and new information to achieve your primary objective. This iterative process ensures that your approach remains relevant and effective. Please do the following:
1. Always remember: {desire_desc}.
2. Determine and explain if there's a need to update the priority list of remaining items based on the current status. 
3. Present the updated priorities in a JSON format, each item should be represented as a key-value pair, where the key is the item name and the value is its priority on the scale from 1-3. An example output is: {{"Fixture Y": 3, "Module B": 2, "Product G": 2}}. The descriptions of the priority scale of items are as follows.
    * 1 - This item is the least important. Consider giving it up if necessary to save money for the rest of the auction.
    * 2 - This item holds value but isn't a top priority for the bidder. Could bid on it if you have enough budget.
    * 3 - This item is of utmost importance and is a top priority for the bidder in the rest of the auction.
""".strip()


# for auctioneer
PARSE_BID_INSTRUCTION = """
Your task is to parse a response from a bidder in an auction, and extract the bidding price from the response. Here are the rules:
- If the language model decides to withdraw from the bidding (e.g., saying "I'm out!"), output -1.
- If a bidding price is mentioned (e.g., saying "I bid $xxx!"), output that price number (e.g., $xxx).
Here is the response:

{response}

Don't say anything else other than just a number: either the bidding price (e.g., $xxx, with $) or -1.
""".strip()


AUCTION_HISTORY = """
## Auction Log

### 1. Equipment E, starting at $5000.

#### 1st bid:
* Bidder 1: $5500
* Bidder 2: $5100
* Bidder 3: $5100
* Bidder 4: $5500
* Bidder 5: $6000

#### 2nd bid:
* Bidder 1: Withdrew
* Bidder 2: Withdrew
* Bidder 3: Withdrew
* Bidder 4: $6500

#### 3rd bid:
* Bidder 5: $7000

#### 4th bid:
* Bidder 4: Withdrew

#### Hammer price (true value):
* Bidder 5: $7000 ($10000)

### 2. Thingamajig C, starting at $1000.

#### 1st bid:
* Bidder 1: $1500
* Bidder 2: Withdrew
* Bidder 3: Withdrew
* Bidder 4: Withdrew
* Bidder 5: Withdrew

#### Hammer price (true value):
* Bidder 1: $1500 ($2000)

### 3. Component S, starting at $1000.

#### 1st bid:
* Bidder 1: $1200
* Bidder 2: $1050
* Bidder 3: $1000
* Bidder 4: Withdrew
* Bidder 5: $1200

#### 2nd bid:
* Bidder 2: Withdrew
* Bidder 3: $1300
* Bidder 5: $1300

#### 3rd bid:
* Bidder 1: Withdrew
* Bidder 3: $1400

#### 4th bid:
* Bidder 5: Withdrew

#### Hammer price (true value):
* Bidder 3: $1400 ($2000)

### 4. Implement G, starting at $1000.

#### 1st bid:
* Bidder 1: $1100
* Bidder 2: $1000
* Bidder 3: $1100
* Bidder 4: Withdrew
* Bidder 5: $1500

#### 2nd bid:
* Bidder 1: Withdrew
* Bidder 2: Withdrew
* Bidder 3: $1600

#### 3rd bid:
* Bidder 5: $1700

#### 4th bid:
* Bidder 3: Withdrew

#### Hammer price (true value):
* Bidder 5: $1700 ($2000)

### 5. Piece T, starting at $1000.

#### 1st bid:
* Bidder 1: $1100
* Bidder 2: $1000
* Bidder 3: $1100
* Bidder 4: Withdrew
* Bidder 5: $1200

#### 2nd bid:
* Bidder 1: Withdrew
* Bidder 2: $1300
* Bidder 3: $1300

#### 3rd bid:
* Bidder 2: $1400
* Bidder 5: Withdrew

#### 4th bid:
* Bidder 3: $1500

#### 5th bid:
* Bidder 2: Withdrew

#### Hammer price (true value):
* Bidder 3: $1500 ($2000)

### 6. Doodad D, starting at $1000.

#### 1st bid:
* Bidder 1: Withdrew
* Bidder 2: $1000
* Bidder 3: Withdrew
* Bidder 4: $1010
* Bidder 5: $1300

#### 2nd bid:
* Bidder 2: Withdrew
* Bidder 4: Withdrew

#### Hammer price (true value):
* Bidder 5: $1300 ($2000)

### 7. Gizmo F, starting at $1000.

#### 1st bid:
* Bidder 1: $1100
* Bidder 2: $1000
* Bidder 3: Withdrew
* Bidder 4: Withdrew
* Bidder 5: Withdrew

#### 2nd bid:
* Bidder 2: $1200

#### 3rd bid:
* Bidder 1: Withdrew

#### Hammer price (true value):
* Bidder 2: $1200 ($2000)

### 8. Widget A, starting at $1000.

#### 1st bid:
* Bidder 1: $2200
* Bidder 2: $1000
* Bidder 3: $1100
* Bidder 4: Withdrew
* Bidder 5: Withdrew

#### 2nd bid:
* Bidder 2: Withdrew
* Bidder 3: Withdrew

#### Hammer price (true value):
* Bidder 1: $2200 ($2000)

### 9. Gadget B, starting at $1000.

#### 1st bid:
* Bidder 1: $1200
* Bidder 2: Withdrew
* Bidder 3: Withdrew
* Bidder 4: $1000
* Bidder 5: Withdrew

#### 2nd bid:
* Bidder 4: Withdrew

#### Hammer price (true value):
* Bidder 1: $1200 ($2000)

### 10. Mechanism J, starting at $5000.

#### 1st bid:
* Bidder 1: Withdrew
* Bidder 2: $5000
* Bidder 3: $5100
* Bidder 4: $6000
* Bidder 5: Withdrew

#### 2nd bid:
* Bidder 2: $6500
* Bidder 3: $6500

#### 3rd bid:
* Bidder 3: $7000
* Bidder 4: $7000

#### 4th bid:
* Bidder 2: $7500
* Bidder 3: Withdrew

#### 5th bid:
* Bidder 4: $8000

#### 6th bid:
* Bidder 2: $8500

#### 7th bid:
* Bidder 4: Withdrew

#### Hammer price (true value):
* Bidder 2: $8500 ($10000)

## Personal Report

* Bidder 1, starting with $10000, has won 3 items in this auction, with a total profit of $1100.:
  * Won Thingamajig C at $1500 over $1000, with a true value of $2000.
  * Won Widget A at $2200 over $1000, with a true value of $2000.
  * Won Gadget B at $1200 over $1000, with a true value of $2000.

* Bidder 2, starting with $10000, has won 2 items in this auction, with a total profit of $2300.:
  * Won Gizmo F at $1200 over $1000, with a true value of $2000.
  * Won Mechanism J at $8500 over $5000, with a true value of $10000.

* Bidder 3, starting with $10000, has won 2 items in this auction, with a total profit of $1100.:
  * Won Component S at $1400 over $1000, with a true value of $2000.
  * Won Piece T at $1500 over $1000, with a true value of $2000.

* Bidder 4, starting with $10000, has won 0 items in this auction, with a total profit of $0.:

* Bidder 5, starting with $10000, has won 3 items in this auction, with a total profit of $4000.:
  * Won Equipment E at $7000 over $5000, with a true value of $10000.
  * Won Implement G at $1700 over $1000, with a true value of $2000.
  * Won Doodad D at $1300 over $1000, with a true value of $2000.
""".strip()
