# Store all constants used by the data loader module
# =============================================================================

CONVERSION_TABLE = {
    '‐': '-', '–': '-', '−': '-',
    '─': '-', 
    '◆': '・',
    '○': '・',
    '♦': '・',
    '●': '・',
    
    '、': ',',
    
    '―': 'ー',       # 'ー': '-' (actual kata)
    '—': '-', '“': '"', '”': '"', '〜': '~',
    '。': '.', '〈': '(', '〉': ')', '《': '(', '》': ')', '〔': '(',
    '〕': ')',
    '‘': "'", '’': "'", '′': "'", "〃": '"', '　': ' ',
    '×': 'x', 
    'ɡ': 'g', 
    'Α': 'A', 
    'Χ': 'X',
    'Ε': 'E', 
    'Ζ': 'Z',
    'А': 'A', 
    'М': 'M', 
    'Н': 'H', 
    'О': 'O', 
    'Т': 'T',
    'а': 'a',
    'е': 'e', 
    'о': 'o',
    'р': 'p', 
    'с': 'c',
    'н': 'H',
#     'コト': 'ヿ',
    '／':'/',
    
    '☆':'',
    '★':'',
    
    #Daichi3
    
    '＊':'*',
    '＋':'+',
    '，':',',
    '－':'-',
    '．':'.',
    '／':'/',
    '０':'0',
    '１':'1',
    '２':'2',
    '３':'3',
    '４':'4',
    '５':'5',
    '６':'6',
    '７':'7',
    '８':'8',
    '９':'9',
    '：':':',
    '；':';',
    '＜':'<',
    '＝':'=',
    '＞':'>',
    '？':'?',
    '＠':'@',
    'Ａ':'A',
    'Ｂ':'B',
    'Ｃ':'C',
    'Ｄ':'D',
    'Ｅ':'E',
    'Ｆ':'F',
    'Ｇ':'G',
    'Ｈ':'H',
    'Ｉ':'I',
    'Ｊ':'J',
    'Ｋ':'K',
    'Ｌ':'L',
    'Ｍ':'M',
    'Ｎ':'N',
    'Ｏ':'O',
    'Ｐ':'P',
    'Ｑ':'Q',
    'Ｒ':'R',
    'Ｓ':'S',
    'Ｔ':'T',
    'Ｕ':'U',
    'Ｖ':'V',
    'Ｗ':'W',
    'Ｘ':'X',
    'Ｙ':'Y',
    'Ｚ':'Z',
    '［':'[',
    '＼':'\\',
    '］':']',
    '＾':'^',
    '＿':'_',
    '｀':'`',
    'ａ':'a',
    'ｂ':'b',
    'ｃ':'c',
    'ｄ':'d',
    'ｅ':'e',
    'ｆ':'f',
    'ｇ':'g',
    'ｈ':'h',
    'ｉ':'i',
    'ｊ':'j',
    'ｋ':'k',
    'ｌ':'l',
    'ｍ':'m',
    'ｎ':'n',
    'ｏ':'o',
    'ｐ':'p',
    'ｑ':'q',
    'ｒ':'r',
    'ｓ':'s',
    'ｔ':'t',
    'ｕ':'u',
    'ｖ':'v',
    'ｗ':'w',
    'ｘ':'x',
    'ｙ':'y',
    'ｚ':'z',
    '｛':'{',
    '｜':'|',
    '｝':'}',
    '～':'~',
    #---------
    
    
    
    '゛': '゙', 
    ' ゙': '゙',        # handakuten
    '゜': '゚', 
    ' ゚': '゚',        # dakuten

    # Maths
    '⊿': 'Δ', '△': 'Δ', '▵': 'Δ', '⇒': '→',

    # Only for Japanese purpose
    'á': 'a', 'ä': 'a', 'é': 'e', 'ö': 'o', 'ü': 'u', 'и': 'n',

    # REMOVE
    '´': '', ' ́': '', '\u0000': '',
    
    
}

DAKUTEN = '゙'
HANDAKUTEN = '゚'

TOP_CHARS = {'°', "'", '"', DAKUTEN, HANDAKUTEN}
MIDDLE_CHARS = {'+', '-', '=', '・', '一', '~'}
BOTTOM_CHARS = set('qypgj,')

NUMBERS = set('0123456789')

NOT_RESIZE = {'-', '一', '_'}
SMALL_CHARS = {'.', ',', '・', '°', "'", '"', DAKUTEN, HANDAKUTEN}    # max 0.3
MEDIUM_CHARS = {'+', '=', '~'}          # 0.3 - 0.5
SMALL_LATIN = set('weruioaszxcvnm&@')     # 0.6 - 0.75
SMALL_KATA_HIRA = set(
    'ァィゥェォヵヶッャュョ' + 'ぁぃぅぇぉゕゖっゃゅょゎ') # 0.5-0.7
NORMAL_FORCE_SMALLER = set(             # 0.7 - 0.75
    'ハバパムロコゴニ' +                # kata
    '二'                             # kanji
)
KATA_HIRA = set(        # 0.75 - 0.9 normal size
    'あいうゔえおかがきぎくぐけげこごさざしじすずせぜそぞただちぢつづてでとどなにぬねのは'
    'ばぱひびぴふぶぷへべぺほぼぽまみむめもやゆよらりるれろわゐゑをんーゝゞゟ' +
    'アイウヴエオカガキギクグケゲサザシジスズセゼソゾタダチヂツヅテデトドナヌネノ'
    'ヒビピフブプヘベペホボポマミメモヤユヨラリルレワヰヱヲンーヽヾ' # remove ニ
)
LARGER_THAN_NORMAL = set('@')

INVALID_KATAS = set('ヷヸヹヺ・ーヽヾヿヰヱヴ゠')
INVALID_HIGAS = set('ゝゞゟゐゑゔ')

TRAIN_MODE = 1
TEST_MODE = 2
VALIDATION_MODE = 3
INFER_MODE = 4
