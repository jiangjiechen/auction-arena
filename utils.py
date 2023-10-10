import ujson as json
import re
import traceback


def trace_back(error_msg):
    exc = traceback.format_exc()
    msg = f'[Error]: {error_msg}.\n[Traceback]: {exc}'
    return msg


def extract_numbered_list(paragraph):
    # Updated regular expression to match numbered list
    # It looks for:
    # - start of line
    # - one or more digits
    # - a period or parenthesis
    # - optional whitespace
    # - any character (captured in a group) until the end of line or a new number
    pattern = r"^\s*(\d+[.)]\s?.*?)(?=\s*\d+[.)]|$)"
    
    matches = re.findall(pattern, paragraph, re.DOTALL | re.MULTILINE)
    return [match.strip() for match in matches]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def reset_state_list(*states):
    empty = [None for _ in states[1:]]
    return [[]] + empty


def LoadJsonL(filename):
    if isinstance(filename, str):
        jsl = []
        with open(filename) as f:
            for line in f:
                jsl.append(json.loads(line))
        return jsl
    else:
        return filename


def extract_jsons_from_text(text):
    json_dicts = []
    stack = []
    start_index = None
    
    for i, char in enumerate(text):
        if char == '{':
            stack.append(char)
            if start_index is None:
                start_index = i
        elif char == '}':
            if stack:
                stack.pop()
            if not stack and start_index is not None:
                json_candidate = text[start_index:i+1]
                try:
                    parsed_json = json.loads(json_candidate)
                    json_dicts.append(parsed_json)
                    start_index = None
                except json.JSONDecodeError:
                    pass
                finally:
                    start_index = None
    
    if len(json_dicts) == 0: json_dicts = [{}]
    return json_dicts