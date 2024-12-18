# Based on seminar materials

# Don't forget to support cases when target_text == ''
import editdistance

def calc_cer(target_text, predicted_text) -> float:
    # TODO
    
    debug = True
    # Check inputs
    if debug:
        print('target_text:', target_text)
        print('predicted_text:', predicted_text) # PRINT PREDICTED TEXT!!!
    
    # # Handle empty target case
    if not target_text:
        return 1.0 if predicted_text else 0.0

    
    # editdistance.eval computes the Levenshtein distance between two sequences of words
    cer = editdistance.eval(target_text, predicted_text) / len(target_text)
    
    return cer


def calc_wer(target_text, predicted_text) -> float:
    # TODO
    # pass
    
    # Handle empty target case
    if not target_text:
        return 1.0 if predicted_text else 0.0
    if predicted_text is None:
        predicted_text = ""
    
    target_words = target_text.lower().split()
    predicted_words = predicted_text.lower().split()
    
    # editdistance.eval computes the Levenshtein distance between two sequences of words
    wer = editdistance.eval(target_words, predicted_words) / len(target_words)
    
    return wer