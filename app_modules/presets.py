import gradio as gr

title = """<img src="https://huggingface.co/spaces/jiangjiechen/Auction-Arena-Demo/resolve/main/assets/logo.png" style="float: left;" width="200" height="200"><h1> Auction Arena </h1>

An interactive demo for this paper: <a href="https://arxiv.org/abs/2310.05746">Put Your Money Where Your Mouth Is: Evaluating Strategic Planning and Execution of LLM Agents in an Auction Arena</a>. Details of this work can be found at <a href="https://auction-arena.github.io">this page</a>. 
<br>
<br>
After choosing items and setting the basic auction rules (like shuffle item order, setting minimal increase, enable discount if none bids, etc.), you can either watch AI vs AI in this auction arena by setting `model_name` as LLMs. Or if you like to participate in the competition yourself, you can set `model_name=human` to engage in the arena personally. Please <b>enter your API key</b> before start. <b>OpenAI API Key is a must</b>, others are not. Otherwise you will encounter errors (please refresh the page if you do). 
<br>
<br>
Feel free to <a href="mailto:jjchen19@fudan.edu.cn">contact us</a> if you have any questions!
"""

# description_top = """\
# <div align="center">
# <p>

# </p >
# </div>
# """
description = """\
<div align="center" style="margin:16px 0">
The demo is built on <a href="https://github.com/GaiZhenbiao/ChuanhuChatGPT">ChuanhuChatGPT</a>.
</div>
"""

small_and_beautiful_theme = gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="#02C160",
            c100="rgba(2, 193, 96, 0.2)",
            c200="#02C160",
            c300="rgba(2, 193, 96, 0.32)",
            c400="rgba(2, 193, 96, 0.32)",
            c500="rgba(2, 193, 96, 1.0)",
            c600="rgba(2, 193, 96, 1.0)",
            c700="rgba(2, 193, 96, 0.32)",
            c800="rgba(2, 193, 96, 0.32)",
            c900="#02C160",
            c950="#02C160",
        ),
        secondary_hue=gr.themes.Color(
            c50="#576b95",
            c100="#576b95",
            c200="#576b95",
            c300="#576b95",
            c400="#576b95",
            c500="#576b95",
            c600="#576b95",
            c700="#576b95",
            c800="#576b95",
            c900="#576b95",
            c950="#576b95",
        ),
        neutral_hue=gr.themes.Color(
            name="gray",
            c50="#f9fafb",
            c100="#f3f4f6",
            c200="#e5e7eb",
            c300="#d1d5db",
            c400="#B2B2B2",
            c500="#808080",
            c600="#636363",
            c700="#515151",
            c800="#393939",
            c900="#272727",
            c950="#171717",
        ),
        radius_size=gr.themes.sizes.radius_sm,
    ).set(
        button_primary_background_fill="#06AE56",
        button_primary_background_fill_dark="#06AE56",
        button_primary_background_fill_hover="#07C863",
        button_primary_border_color="#06AE56",
        button_primary_border_color_dark="#06AE56",
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#FFFFFF",
        button_secondary_background_fill="#F2F2F2",
        button_secondary_background_fill_dark="#2B2B2B",
        button_secondary_text_color="#393939",
        button_secondary_text_color_dark="#FFFFFF",
        # background_fill_primary="#F7F7F7",
        # background_fill_primary_dark="#1F1F1F",
        block_title_text_color="*primary_500",
        block_title_background_fill="*primary_100",
        input_background_fill="#F6F6F6",
    )
