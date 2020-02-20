from nltk.tag import StanfordPOSTagger
from lib.utils import POS_TAGGERS_JAR_PATH, POS_TAGGERS_EN_MODEL_PATH, POS_TAGGERS_DE_MODEL_PATH

POS_TAGGER_EN = StanfordPOSTagger(POS_TAGGERS_EN_MODEL_PATH, POS_TAGGERS_JAR_PATH)
POS_TAGGER_DE = StanfordPOSTagger(POS_TAGGERS_DE_MODEL_PATH, POS_TAGGERS_JAR_PATH)

def get_pos_tags(tokenised_sentence, lang):
    if lang == 'en':
        pos_tagger = POS_TAGGER_EN
    elif lang == 'de':
        pos_tagger = POS_TAGGER_DE
    else:
        raise NotImplementedError('Pass in either en or de as language')

    return pos_tagger.tag(tokenised_sentence)
