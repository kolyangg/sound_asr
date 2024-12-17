# Based on seminar materials

# Don't forget to support cases when target_text == ''
import editdistance

def calc_cer(target_text, predicted_text) -> float:
    # TODO
    # pass
    # TO CHECK A LOT !!! #
    # # Handle empty target case
    # print('target_text:', target_text)
    # print('predicted_text:', predicted_text)
    # if not target_text:
    #     return 1.0 if predicted_text else 0.0
    
    # if predicted_text is None:
    #     predicted_text = ""
    # # Convert to lowercase and remove extra whitespace
    # target_text = " ".join(target_text.lower().split())
    # predicted_text = " ".join(predicted_text.lower().split())
    
    # # Initialize matrix of size (len(target) + 1) x (len(predicted) + 1)
    # dp = [[0] * (len(predicted_text) + 1) for _ in range(len(target_text) + 1)]
    
    # # Fill first row and column
    # for i in range(len(target_text) + 1):
    #     dp[i][0] = i
    # for j in range(len(predicted_text) + 1):
    #     dp[0][j] = j
        
    # # Fill the rest of the matrix
    # for i in range(1, len(target_text) + 1):
    #     for j in range(1, len(predicted_text) + 1):
    #         if target_text[i-1] == predicted_text[j-1]:
    #             dp[i][j] = dp[i-1][j-1]
    #         else:
    #             # minimum of substitution, insertion, deletion
    #             dp[i][j] = min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) + 1
                
    # # Return CER
    # return dp[-1][-1] / len(target_text)
    
    # TO CHECK A LOT !!! #
    
    
    # Check inputs
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
    # TO CHECK A LOT !!! #
    
    # Handle empty target case
    # if not target_text:
    #     return 1.0 if predicted_text else 0.0
    # if predicted_text is None:
    #     predicted_text = ""
    
    # # Convert to lowercase and split into words
    # target_words = target_text.lower().split()
    # predicted_words = predicted_text.lower().split()
    
    # # Handle case where target has words but prediction is empty
    # if not predicted_words and target_words:
    #     return 1.0
    
    # # Initialize matrix of size (len(target_words) + 1) x (len(predicted_words) + 1)
    # dp = [[0] * (len(predicted_words) + 1) for _ in range(len(target_words) + 1)]
    
    # # Fill first row and column
    # for i in range(len(target_words) + 1):
    #     dp[i][0] = i
    # for j in range(len(predicted_words) + 1):
    #     dp[0][j] = j
        
    # # Fill the rest of the matrix
    # for i in range(1, len(target_words) + 1):
    #     for j in range(1, len(predicted_words) + 1):
    #         if target_words[i-1] == predicted_words[j-1]:
    #             dp[i][j] = dp[i-1][j-1]
    #         else:
    #             # minimum of substitution, insertion, deletion
    #             dp[i][j] = min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) + 1
                
    # # Return WER
    # return dp[-1][-1] / len(target_words)
    
    # TO CHECK A LOT !!! #
    
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