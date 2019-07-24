textline_data can be downloaded from here: https://cinnamon-lab.app.box.com/folder/62761028117

# What we need?
### Vocabulary
File `configs/invoice_charset_v2.txt`
### Data
File `textline_data/train.txt`, `textline_data/val.txt`

# How to train?
1. Convert `.txt` to `json` by using `convert_txt_to_json.py`
- Change file path in `convert_txt_to_json.py`
2. Run `train.py`
- You can change file path in this file if you change in `convert_txt_to_json.py`