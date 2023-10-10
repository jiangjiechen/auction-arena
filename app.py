import os
import gradio as gr
from app_modules.presets import *
from app_modules.overwrites import *
from app_modules.utils import *
from src.item_base import create_items
from src.bidder_base import Bidder
from src.human_bidder import HumanBidder
from src.auctioneer_base import Auctioneer
from auction_workflow import run_auction, make_auction_hash
from utils import chunks, reset_state_list


BIDDER_NUM = 4
items = create_items('data/items_demo.jsonl')

def auction_loop_app(*args):
    global items

    bidder_list = args[0]   # gr.State() -> session state
    items_id = args[1]
    os.environ['OPENAI_API_KEY'] = args[2] if args[2] != '' else os.environ.get('OPENAI_API_KEY', '')
    os.environ['ANTHROPIC_API_KEY'] = args[3] if args[3] != '' else os.environ.get('ANTHROPIC_API_KEY', '')
    thread_num = args[4]
    item_shuffle = args[5]
    enable_discount = args[6]
    min_markup_pct = args[7]
    args = args[8:]
    auction_hash = make_auction_hash()

    items_to_bid = [items[i] for i in items_id]

    auctioneer = Auctioneer(enable_discount=enable_discount, min_markup_pct=min_markup_pct)
    auctioneer.init_items(items_to_bid)
    if item_shuffle:
        auctioneer.shuffle_items()

    # must correspond to the order in app's parameters
    input_keys = [
        'chatbot', 
        'model_name', 
        'desire',
        'plan_strategy', 
        'budget', 
        'correct_belief', 
        'enable_learning',
        'temperature', 
        'overestimate_percent',
    ]
    
    # convert flatten list into a json list
    input_jsl = []
    for i, chunk in enumerate(chunks(args, len(input_keys))):
        js = {'name': f"Bidder {i+1}", 'auction_hash': auction_hash}
        for k, v in zip(input_keys, chunk):
            js[k] = v
        input_jsl.append(js)
    
    for js in input_jsl:
        js.pop('chatbot')
        if 'human' in js['model_name']:
            bidder_list.append(HumanBidder.create(**js))
        else:
            bidder_list.append(Bidder.create(**js))
    
    yield from run_auction(auction_hash, auctioneer, bidder_list, thread_num, yield_for_demo=True)


with open("assets/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()

with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
    with gr.Row():
        gr.HTML(title)

    # gr.Markdown(description_top)

    with gr.Row():
        with gr.Column(scale=6):
            # item_file = gr.File(label="Upload Item File", file_types=[".jsonl"])
            items_checkbox = gr.CheckboxGroup(
                choices=[item.info() for item in items[:20]],
                label="Items in Auction",
                info="Select the items you want to include in the auction.",
                value=[item.info() for item in items[:8]],
                type="index",
            )
        
        with gr.Column(scale=4):
            with gr.Row():
                openai_key = gr.Textbox(label="OpenAI API Key", value="", type="password", placeholder="sk-..")
                anthropic_key = gr.Textbox(label="Anthropic API Key", value="", type="password", placeholder="sk-ant-..")
            
            with gr.Row():
                with gr.Row():
                    item_shuffle = gr.Checkbox(
                        label="Shuffle Items", 
                        value=False, 
                        info='Shuffle the order of items in the auction.')
                    enable_discount = gr.Checkbox(
                        label="Enable Discount",
                        value=False,
                        info='When an item fails to sell at auction, it can be auctioned again at a reduced price.')
                
                with gr.Column():
                    min_markup_pct = gr.Slider(
                        minimum=0.1,
                        maximum=0.5,
                        value=0.1,
                        step=0.1,
                        interactive=True,
                        label='Min Increase',
                        info="The minimum percentage to increase a bid.",
                    )

                    thread_num = gr.Slider(
                        minimum=1,
                        maximum=BIDDER_NUM,
                        value=min(5, BIDDER_NUM),
                        step=1,
                        interactive=True,
                        label='Thread Number',
                        info="More threads, faster bidding, but will run into RateLimitError quicker."
                    )

    with gr.Row():
        bidder_info_gr = []
        chatbots = []
        monitors = []
        textbox_list = []
        for i in range(BIDDER_NUM):
            with gr.Tab(label=f"Bidder {i+1}"):
                with gr.Row().style(equal_height=True):
                    with gr.Column(scale=6):
                        with gr.Row():
                            chatbot = gr.Chatbot(elem_id="chuanhu_chatbot", height=600, label='Auction Log')
                        input_box = gr.Textbox(label="Human Bidder Input", interactive=False, placeholder="Please wait a moment before engaging in the auction.", visible=False)
                        chatbots.append(chatbot)
                        textbox_list.append(input_box)
                    with gr.Column(scale=4):
                        with gr.Tab(label=f'Parameters'):
                            model_name = gr.Dropdown(
                                choices=[
                                    'rule',
                                    'human',
                                    'gpt-3.5-turbo-0613',
                                    'gpt-3.5-turbo-16k-0613',
                                    'gpt-4-0613',
                                    # 'claude-instant-1.1',
                                    'claude-instant-1.2',
                                    # 'claude-1.3',
                                    'claude-2.0',
                                    # 'chat-bison-001',
                                ], 
                                value='gpt-3.5-turbo-16k-0613',
                                label="Model Selection",
                            )
                            budget = gr.Number(
                                value=10000, 
                                label='Budget ($)'
                            )
                            with gr.Row():
                                plan_strategy = gr.Dropdown(
                                    choices=[
                                        'none',
                                        'static',
                                        'adaptive',
                                    ],
                                    value='adaptive',
                                    label='Planning Strategy',
                                    info='None: no plan. Static: plan only once. Adaptive: replan for the remaining items.'
                                )
                                desire = gr.Dropdown(
                                    choices=[
                                        # 'default',
                                        'maximize_profit',
                                        'maximize_items',
                                        # 'specific_items', 
                                    ],
                                    value='maximize_profit',
                                    label='Desire',
                                    info='Default desires: spending all the budget, stay within budget. All desires include the default one.',
                                )
                            overestimate_percent = gr.Slider(
                                minimum=-100,
                                maximum=100,
                                value=10,
                                step=10,
                                interactive=True,
                                label='Overestimate Percent (%)',
                                info="Overestimate the true value of items by this percentage.",
                            )
                            with gr.Row():
                                correct_belief = gr.Checkbox(
                                    label='Correct Wrong Beliefs',
                                    value=True,
                                    info='Forceful beliefs correction about self and others.',
                                )
                                enable_learning = gr.Checkbox(
                                    label='Enable Learning',
                                    value=False,
                                    info='Learn from past auctions for future guidance. Only for adaptive bidder.',
                                    visible=False
                                )
                            temperature = gr.Slider(
                                minimum=0.,
                                maximum=2.0,
                                value=0.7,
                                step=0.1,
                                interactive=True,
                                label="Temperature",
                            )

                            # deprecated
                            # special_items = gr.CheckboxGroup(
                            #     value = [],
                            #     label='Special Items',
                            #     info='Special items add 20% value for you personally.',
                            #     visible=False,
                            # )
                            # hedge_percent = gr.Slider(
                            #     minimum=0,
                            #     maximum=100,
                            #     value=90,
                            #     step=1,
                            #     interactive=True,
                            #     label='Strategy (Hedging %)',
                            #     info="The maximum percentage of the estimated value to bid on an item.",
                            #     visible=False
                            # )

                        with gr.Tab(label='Monitors'):
                            with gr.Row():
                                budget_monitor = gr.Number(label='Budget Left ($)', interactive=False)
                                profit_monitor = gr.Number(label='Profit ($)', interactive=False)
                            
                            with gr.Row():
                                engagement_monitor = gr.Number(
                                    label='Engagement',
                                    interactive=False,
                                    info='The number of times the bidder has bid.'
                                )
                                failure_monitor = gr.Number(
                                    label='Failed Bids', 
                                    info='Out-of-budget, or less than the previous highest bid.',
                                    interactive=False
                                )

                            items_own_monitor = gr.DataFrame(
                                label='Items Owned',
                                headers=['Item', 'Bid ($)', 'Value ($)'],
                                datatype=['str', 'number', 'number'],
                                interactive=False,
                            )

                            with gr.Row():
                                tokens_monitor = gr.Number(
                                    label='Token Used', 
                                    interactive=False, 
                                    info='Tokens used in the last call.'
                                )
                                money_monitor = gr.Number(
                                    label='API Cost ($)', 
                                    info='Only OpenAI cost for now.',
                                    interactive=False
                                )
                                
                            plan_change_monitor = gr.DataFrame(
                                label='Plan Changes',
                                headers=['Round', 'Changed', 'New Plan'],
                                datatype=['str', 'bool', 'str'],
                                interactive=False,
                            )
                        
                            plot_monitor = gr.Plot(
                                label='Budget-Profit Plot', 
                                interactive=False
                            )

                        with gr.Tab(label='Belief Errors'):
                            with gr.Row():
                                self_belief_error_cnt_monitor = gr.Number(
                                    label='Wrong Beliefs of Self', 
                                    info='Not knowing its own budget, bid items, or won items.',
                                    interactive=False,
                                )
                                other_belief_error_cnt_monitor = gr.Number(
                                    label='Wrong Beliefs of Others',
                                    info='Not knowing other bidders\' profits.',
                                    interactive=False,
                                )
                            budget_belief_monitor = gr.DataFrame(
                                label='Wrong Belief of Budget ($)',
                                headers=['Round', 'Belief', 'Truth'],
                                datatype=['str', 'number', 'number'],
                                interactive=False,
                            )
                            profit_belief_monitor = gr.DataFrame(
                                label='Wrong Belief of Profit ($)',
                                headers=['Bidder (Round)', 'Belief', 'Truth'],
                                datatype=['str', 'number', 'number'],
                                interactive=False,
                            )
                            win_bid_belief_monitor = gr.DataFrame(
                                label='Wrong Belief of Items Won',
                                headers=['Bidder (Round)', 
                                'Belief', 'Truth'],
                                datatype=['str', 'str', 'str'],
                                interactive=False,
                            )

                        monitors += [
                            budget_monitor, 
                            profit_monitor, 
                            items_own_monitor, 
                            tokens_monitor, 
                            money_monitor,
                            failure_monitor,
                            self_belief_error_cnt_monitor,
                            other_belief_error_cnt_monitor,
                            engagement_monitor,
                            plot_monitor,
                            plan_change_monitor,
                            budget_belief_monitor,
                            profit_belief_monitor,
                            win_bid_belief_monitor,
                        ]

                bidder_info_gr += [
                    chatbot,
                    model_name,
                    desire,
                    plan_strategy,
                    budget,
                    correct_belief,
                    enable_learning,
                    temperature,
                    overestimate_percent,
                ]
    
    with gr.Row():
        with gr.Column():
            startBtn = gr.Button('Start Bidding', variant='primary', interactive=True)
        with gr.Column():
            clearBtn = gr.Button('New Auction', variant='secondary', interactive=False)
        btn_list = [startBtn, clearBtn]
    
    with gr.Accordion(label='Bidding Log (click to open)', open=True):
        with gr.Row():
            bidding_log = gr.Markdown(value="")

    gr.Markdown(description)
    
    bidder_list_state = gr.State([])    # session state

    start_args = dict(
        fn=auction_loop_app,
        inputs=[bidder_list_state, items_checkbox, openai_key, anthropic_key, thread_num, item_shuffle, enable_discount, min_markup_pct] + bidder_info_gr,
        outputs=[bidder_list_state] + chatbots + monitors + [bidding_log] + btn_list + textbox_list, # TODO: handle textbox_list interactivity
        show_progress=True,
    )
    start_event = startBtn.click(**start_args)
    
    def bot(user_message, bidder_list, id):
        if len(bidder_list) > 0:
            bidder = bidder_list[int(id)]
            if bidder.need_input:
                bidder.input_box = user_message
                bidder.semaphore += 1
        return '', bidder_list
    
    # handle user input from time to time
    for i in range(len(textbox_list)):
        _dummy_id = gr.Number(i, visible=False, interactive=False)
        textbox_list[i].submit(
            bot, 
            [textbox_list[i], bidder_list_state, _dummy_id], 
            [textbox_list[i], bidder_list_state])

    clearBtn.click(reset_state_list, 
                   inputs=[bidder_list_state] + chatbots + monitors + [bidding_log], 
                   outputs=[bidder_list_state] + chatbots + monitors + [bidding_log], 
                   show_progress=True).then(lambda: gr.update(interactive=True), outputs=[startBtn])
    
    demo.title = 'Auction Arena'


demo.queue(max_size=64, concurrency_count=16).launch(
    # server_name='0.0.0.0',
    # ssl_verify=False,
    # share=True, 
    # debug=True,
    show_api=False,
)

demo.close()