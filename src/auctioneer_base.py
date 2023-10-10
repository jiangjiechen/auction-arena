import re
from typing import List, Dict
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel
from collections import defaultdict
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import random
import inflect
from .bidder_base import Bidder
from .human_bidder import HumanBidder
from .item_base import Item
from .prompt_base import PARSE_BID_INSTRUCTION

p = inflect.engine()


class Auctioneer(BaseModel):
    enable_discount: bool = False
    items: List[Item] = []
    cur_item: Item = None
    highest_bidder: Bidder = None
    highest_bid: int = -1
    bidding_history = defaultdict(list) # history about the bidding war of one item
    items_queue: List[Item] = []   # updates when a item is taken.
    auction_logs = defaultdict(list)    # history about the bidding war of all items
    openai_cost = 0
    prev_round_max_bid: int = -1
    min_bid: int = 0
    fail_to_sell = False
    min_markup_pct = 0.1

    class Config:
        arbitrary_types_allowed = True
    
    def init_items(self, items: List[Item]):
        for item in items:
            # reset discounted price
            item.reset_price()
        self.items = items
        self.items_queue = items.copy()

    def summarize_items_info(self):
        desc = ''
        for item in self.items:
            desc += f"- {item.get_desc()}\n"
        return desc.strip()
    
    def present_item(self):
        cur_item = self.items_queue.pop(0)
        self.cur_item = cur_item
        return cur_item
    
    def shuffle_items(self):
        random.shuffle(self.items)
        self.items_queue = self.items.copy()
    
    def record_bid(self, bid_info: dict, bid_round: int):
        '''
        Save the bidding history for each round, log the highest bidder and highest bidding
        '''
        # bid_info: {'bidder': xxx, 'bid': xxx, 'raw_msg': xxx}
        self.bidding_history[bid_round].append(bid_info)
        for hist in self.bidding_history[bid_round]:
            if hist['bid'] > 0:
                if self.highest_bid < hist['bid']:
                    self.highest_bid = hist['bid']
                    self.highest_bidder = hist['bidder']
                elif self.highest_bid == hist['bid']:
                    # random if there's a tie
                    self.highest_bidder = random.choice([self.highest_bidder, hist['bidder']])
        self.auction_logs[f"{self.cur_item.get_desc()}"].append(
            {'bidder': bid_info['bidder'], 
             'bid': bid_info['bid'], 
             'bid_round': bid_round})

    def _biddings_to_string(self, bid_round: int):
        '''
        Return a string that summarizes the bidding history in a round
        '''
        # bid_hist_text = '' if bid_round == 0 else f'- {self.highest_bidder}: ${self.highest_bid}\n'
        bid_hist_text = ''
        for js in self.bidding_history[bid_round]:
            if js['bid'] < 0:
                bid_hist_text += f"- {js['bidder']} withdrew\n"
            else:
                bid_hist_text += f"- {js['bidder']}: ${js['bid']}\n"
        return bid_hist_text.strip()
    
    def all_bidding_history_to_string(self):
        bid_hist_text = ''
        for bid_round in self.bidding_history:
            bid_hist_text += f"Round {bid_round}:\n{self._biddings_to_string(bid_round)}\n\n"
        return bid_hist_text.strip()

    def ask_for_bid(self, bid_round: int):
        '''
        Ask for bid, return the message to be sent to bidders
        '''
        if self.highest_bidder is None:
            if bid_round > 0:
                msg = f"Seeing as we've had no takers at the initial price, we're going to lower the starting bid to ${self.cur_item.price} for {self.cur_item.name} to spark some interest! Do I have any takers?"
            else:
                remaining_items = [self.cur_item.name] + [item.name for item in self.items_queue]
                msg = f"Attention, bidders! {len(remaining_items)} item(s) left, they are: {', '.join(remaining_items)}.\n\nNow, please bid on {self.cur_item}. The starting price for bidding for {self.cur_item} is ${self.cur_item.price}. Anyone interested in this item?"
        else:
            bidding_history = self._biddings_to_string(bid_round - 1)
            msg = f"Thank you! This is the {p.ordinal(bid_round)} round of bidding for this item:\n{bidding_history}\n\nNow we have ${self.highest_bid} from {self.highest_bidder.name} for {self.cur_item.name}. The minimum increase over this highest bid is ${int(self.cur_item.price * self.min_markup_pct)}. Do I have any advance on ${self.highest_bid}?"
        return msg
    
    def ask_for_rebid(self, fail_msg: str, bid_price: int):
        return f"Your bid of ${bid_price} failed, because {fail_msg}: You must reconsider your bid."

    def get_hammer_msg(self):
        if self.highest_bidder is None:
            return f"Since no one bid on {self.cur_item.name}, we'll move on to the next item."
        else:
            return f"Sold! {self.cur_item} to {self.highest_bidder} at ${self.highest_bid}! The true value for {self.cur_item} is ${self.cur_item.true_value}."# Thus {self.highest_bidder}'s profit by winning this item is ${self.cur_item.true_value - self.highest_bid}."

    def check_hammer(self, bid_round: int):
        # check if the item is sold
        self.fail_to_sell = False
        num_bid = self._num_bids_in_round(bid_round)

        # highest_bidder has already been updated in record_bid().
        # so when num_bid == 0 & highest_bidder is None, it means no one bid on this item
        if self.highest_bidder is None:
            if num_bid == 0:
                # failed to sell, as there is no highest bidder
                self.fail_to_sell = True
                if self.enable_discount and bid_round < 3:
                    # lower the starting price by 50%. discoutn only applies to the first 3 rounds
                    self.cur_item.lower_price(0.5)
                    is_sold = False
                else:
                    is_sold = True
            else:
                # won't happen
                raise ValueError(f"highest_bidder is None but num_bid is {num_bid}")
        else:
            if self.prev_round_max_bid < 0 and num_bid == 1:
                # only one bidder in the first round 
                is_sold = True
            else:
                self.prev_round_max_bid = self.highest_bid
                is_sold = self._num_bids_in_round(bid_round) == 0
        return is_sold
    
    def _num_bids_in_round(self, bid_round: int):
        # check if there is no bid in the current round
        cnt = 0
        for hist in self.bidding_history[bid_round]:
            if hist['bid'] > 0:
                cnt += 1
        return cnt

    def hammer_fall(self):
        print(f'* Sold! {self.cur_item} (${self.cur_item.true_value}) goes to {self.highest_bidder} at ${self.highest_bid}.')
        self.auction_logs[f"{self.cur_item.get_desc()}"].append({
            'bidder': self.highest_bidder, 
            'bid': f"{self.highest_bid} (${self.cur_item.true_value})",     # no need for the first $, as it will be added in the self.log()
            'bid_round': 'Hammer price (true value)'})
        self.cur_item = None
        self.highest_bidder = None
        self.highest_bid = -1
        self.bidding_history = defaultdict(list)
        self.prev_round_max_bid = -1
        self.fail_to_sell = False

    def end_auction(self):
        return len(self.items_queue) == 0
    
    def gather_all_status(self, bidders: List[Bidder]):
        status = {}
        for bidder in bidders:
            status[bidder.name] = {
                'profit': bidder.profit, 
                'items_won': bidder.items_won
            }
        return status

    def parse_bid(self, text: str):
        prompt = PARSE_BID_INSTRUCTION.format(response=text)
        with get_openai_callback() as cb:
            llm = ChatOpenAI(model='gpt-3.5-turbo-0613', temperature=0)
            result = llm([HumanMessage(content=prompt)]).content
            self.openai_cost += cb.total_cost
        
        bid_number = re.findall(r'\$?\d+', result.replace(',', ''))
        # find number in the result
        if '-1' in result:
            return -1
        elif len(bid_number) > 0:
            return int(bid_number[-1].replace('$', ''))
        else:
            print('* Rebid:', text)
            return None

    def log(self, bidder_personal_reports: list = [], show_model_name=True):
        ''' example
        Apparatus H, starting at $1000.

        1st bid:
        Bidder 1 (gpt-3.5-turbo-16k-0613): $1200
        Bidder 2 (gpt-3.5-turbo-16k-0613): $1100
        Bidder 3 (gpt-3.5-turbo-16k-0613): Withdrawn
        Bidder 4 (gpt-3.5-turbo-16k-0613): $1200
        
        2nd bid:
        Bidder 1 (gpt-3.5-turbo-16k-0613): Withdrawn
        Bidder 2 (gpt-3.5-turbo-16k-0613): Withdrawn
        
        Hammer price:
        Bidder 4 (gpt-3.5-turbo-16k-0613): $1200
        '''
        markdown_output = "## Auction Log\n\n"
        for i, (item, bids) in enumerate(self.auction_logs.items()):
            markdown_output += f"### {i+1}. {item}\n\n"
            cur_bid_round = -1
            for i, bid in enumerate(bids):
                if bid['bid_round'] != cur_bid_round:
                    cur_bid_round = bid['bid_round']
                    if isinstance(bid['bid_round'], int):
                        markdown_output += f"\n#### {p.ordinal(bid['bid_round']+1)} bid:\n\n"
                    else:
                        markdown_output += f"\n#### {bid['bid_round']}:\n\n"
                bid_price = f"${bid['bid']}" if bid['bid'] != -1 else 'Withdrew'
                if isinstance(bid['bidder'], Bidder) or isinstance(bid['bidder'], HumanBidder):
                    if show_model_name:
                        markdown_output += f"* {bid['bidder']} ({bid['bidder'].model_name}): {bid_price}\n"
                    else:
                        markdown_output += f"* {bid['bidder']}: {bid_price}\n"
                else:
                    markdown_output += f"* None bid\n"
            markdown_output += "\n"
        
        if len(bidder_personal_reports) != 0:
            markdown_output += f"\n## Personal Report"
            for report in bidder_personal_reports:
                markdown_output += f"\n\n{report}"
        return markdown_output.strip()
    
    def finish_auction(self):
        self.auction_logs = defaultdict(list)
        self.cur_item = None
        self.highest_bidder = None
        self.highest_bid = -1
        self.bidding_history = defaultdict(list)
        self.items_queue = []
        self.items = []
        self.prev_round_max_bid = -1
        self.fail_to_sell = False
        self.min_bid = 0
        